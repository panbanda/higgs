use std::sync::{Arc, OnceLock};

use mlx_rs::{error::Exception, ops, ops::concatenate_axis, Array, Dtype, Stream};

use crate::turboquant::{
    KvCacheConfig, KvCacheMode, QuantizedKey, QuantizedValue, TurboQuantContext,
};

/// View over a KV cache after appending new tokens.
#[derive(Debug, Clone)]
pub enum KvCacheView {
    Dense { keys: Array, values: Array },
    TurboQuant(TurboQuantKvView),
}

static TURBOQUANT_ACTIVATE_AT: OnceLock<i32> = OnceLock::new();
const DEFAULT_TURBOQUANT_ACTIVATE_AT: i32 = 5000;

fn parse_turboquant_activate_at(raw: Option<&str>) -> i32 {
    match raw.and_then(|s| s.parse::<i32>().ok()) {
        Some(v) => v.max(0),
        None => DEFAULT_TURBOQUANT_ACTIVATE_AT,
    }
}

fn turboquant_activate_at() -> i32 {
    *TURBOQUANT_ACTIVATE_AT.get_or_init(|| {
        parse_turboquant_activate_at(
            std::env::var("HIGGS_TURBOQUANT_ACTIVATE_AT")
                .ok()
                .as_deref(),
        )
    })
}

const fn should_activate_turboquant(offset: i32, new_tokens: i32, activate_at: i32) -> bool {
    activate_at <= 0 || offset + new_tokens >= activate_at
}

impl KvCacheView {
    pub fn into_dense(self) -> Result<(Array, Array), Exception> {
        match self {
            Self::Dense { keys, values } => Ok((keys, values)),
            Self::TurboQuant(view) => view.materialize_dense(),
        }
    }

    pub fn turboquant(&self) -> Option<&TurboQuantKvView> {
        match self {
            Self::Dense { .. } => None,
            Self::TurboQuant(view) => Some(view),
        }
    }
}

/// Quantized cache view used by the TurboQuant decode path.
#[derive(Debug, Clone)]
pub struct TurboQuantKvView {
    pub context: Arc<TurboQuantContext>,
    pub key_codes: Array,
    pub key_norms: Array,
    pub key_gammas: Array,
    pub value_codes: Array,
    pub value_norms: Array,
    pub seq_len: i32,
}

impl TurboQuantKvView {
    pub fn materialize_dense(&self) -> Result<(Array, Array), Exception> {
        let num_kv_heads = usize_from_i32(self.context.num_kv_heads, "num_kv_heads")?;
        let head_dim = usize_from_i32(self.context.head_dim, "head_dim")?;
        let seq_len = usize_from_i32(self.seq_len, "seq_len")?;
        let key_code_bytes = usize_from_i32(self.context.key_code_bytes, "key_code_bytes")?;
        let key_code_words = usize_from_i32(self.context.key_code_words, "key_code_words")?;
        let value_code_bytes = usize_from_i32(self.context.value_code_bytes, "value_code_bytes")?;
        let value_code_words = usize_from_i32(self.context.value_code_words, "value_code_words")?;

        // Eval all view arrays — they may be lazy GPU results from the pack kernel.
        self.key_codes.eval()?;
        self.key_norms.eval()?;
        self.key_gammas.eval()?;
        self.value_codes.eval()?;
        self.value_norms.eval()?;

        // Code arrays are u32 words — reinterpret as bytes for CPU dequant
        let key_codes_u32 = self.key_codes.as_slice::<u32>();
        let key_codes_u8: Vec<u8> = key_codes_u32.iter().flat_map(|w| w.to_le_bytes()).collect();
        let key_norms = self.key_norms.as_slice::<f32>();
        let key_gammas = self.key_gammas.as_slice::<f32>();
        let value_codes_u32 = self.value_codes.as_slice::<u32>();
        let value_codes_u8: Vec<u8> = value_codes_u32
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let value_norms = self.value_norms.as_slice::<f32>();

        // Each row occupies key_code_words * 4 bytes in the reinterpreted buffer
        let key_row_bytes = checked_mul(key_code_words, 4, "key row bytes")?;
        let value_row_bytes = checked_mul(value_code_words, 4, "value row bytes")?;

        let total_values = checked_mul(num_kv_heads, seq_len, "cache size")?;
        let total_dense = checked_mul(total_values, head_dim, "dense cache size")?;
        let mut dense_keys = Vec::with_capacity(total_dense);
        let mut dense_values = Vec::with_capacity(total_dense);

        for head in 0..num_kv_heads {
            for pos in 0..seq_len {
                let scalar_index = checked_add(
                    checked_mul(head, seq_len, "scalar index")?,
                    pos,
                    "scalar index",
                )?;
                let key_byte_start = checked_mul(scalar_index, key_row_bytes, "key code index")?;
                let key_byte_end = checked_add(key_byte_start, key_code_bytes, "key code range")?;
                let value_byte_start =
                    checked_mul(scalar_index, value_row_bytes, "value code index")?;
                let value_byte_end =
                    checked_add(value_byte_start, value_code_bytes, "value code range")?;

                let key = QuantizedKey {
                    norm: key_norms[scalar_index],
                    gamma: key_gammas[scalar_index],
                    codes: key_codes_u8[key_byte_start..key_byte_end].to_vec(),
                };
                let value = QuantizedValue {
                    norm: value_norms[scalar_index],
                    codes: value_codes_u8[value_byte_start..value_byte_end].to_vec(),
                };

                dense_keys.extend(self.context.dequantize_key(&key)?);
                dense_values.extend(self.context.dequantize_value(&value)?);
            }
        }

        let shape = [
            1,
            self.context.num_kv_heads,
            self.seq_len,
            self.context.head_dim,
        ];
        let keys = Array::from_slice(&dense_keys, &shape);
        let values = Array::from_slice(&dense_values, &shape);
        Ok((keys, values))
    }

    pub fn decode_scores(&self, queries: &Array, num_heads: i32) -> Result<Array, Exception> {
        let query_shape = queries.shape();
        if query_shape != [1, num_heads, 1, self.context.head_dim] {
            return Err(Exception::custom(
                "TurboQuant decode expects [1, H, 1, D] queries",
            ));
        }

        let queries = queries
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.context.head_dim])?;
        let q_rot = self.context.rotate_queries(&queries)?;

        crate::turboquant::decode_scores(
            &q_rot,
            &self.key_codes,
            &self.key_norms,
            &self.context.key_centroids_array()?,
            num_heads,
            self.context.num_kv_heads,
            self.context.head_dim,
            self.seq_len,
            self.seq_len,
            self.context.config.key_bits(),
            self.context.key_code_words,
        )
    }

    pub fn decode_values(&self, weights: &Array, num_heads: i32) -> Result<Array, Exception> {
        let weights = weights
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.seq_len])?;
        let out_rot = crate::turboquant::decode_weighted_values(
            &weights,
            &self.value_codes,
            &self.value_norms,
            &self.context.value_centroids_array()?,
            num_heads,
            self.context.num_kv_heads,
            self.context.head_dim,
            self.seq_len,
            self.seq_len,
            self.context.config.bits,
            self.context.value_code_words,
        )?;

        out_rot
            .hadamard_transform(None)?
            .reshape(&[1, num_heads, 1, self.context.head_dim])
    }
}

/// Trait for key-value caches used in autoregressive generation.
pub trait KeyValueCache {
    /// Whether the cache stores quantized KV pairs.
    fn is_quantized(&self) -> bool {
        false
    }

    /// Group size for quantized cache. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Bit width for quantized cache. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    /// Current sequence offset (number of tokens already cached).
    fn offset(&self) -> i32;

    /// Maximum cache size, if bounded.
    fn max_size(&self) -> Option<i32>;

    /// Append new key/value tensors and return a cache view.
    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception>;

    /// Append new key/value tensors and return the full cached key/value.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        self.update_and_view(keys, values)?.into_dense()
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        T::update_and_view(self, keys, values)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// Simple KV cache that concatenates new keys/values with existing ones.
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        if let (Some(existing_keys), Some(existing_values)) = (self.keys.take(), self.values.take())
        {
            self.keys = Some(concatenate_axis(&[existing_keys, keys], -2)?);
            self.values = Some(concatenate_axis(&[existing_values, values], -2)?);
        } else {
            self.keys = Some(keys);
            self.values = Some(values);
        }

        let key_shape = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?
            .shape();
        let seq_dim_index = key_shape.len().wrapping_sub(2);
        self.offset = *key_shape
            .get(seq_dim_index)
            .ok_or_else(|| Exception::custom("Key shape has fewer than 2 dimensions"))?;

        let result_keys = self
            .keys
            .clone()
            .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?;
        let result_values = self
            .values
            .clone()
            .ok_or_else(|| Exception::custom("Values cannot be None after update"))?;

        Ok(KvCacheView::Dense {
            keys: result_keys,
            values: result_values,
        })
    }
}

/// Pre-allocated KV cache that grows in chunks, avoiding per-token allocation.
///
/// Matches Python `mlx_lm`'s `KVCache`: pre-allocates 256 slots at a time and
/// uses `mlx_slice_update` for writes instead of concatenation every token.
/// Keys/values have shape `[B, n_heads, seq_len, head_dim]` with sequence on axis 2.
#[derive(Debug, Clone)]
pub struct SteppingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    turbo: Option<TurboQuantStorage>,
    config: KvCacheConfig,
    offset: i32,
    step: i32,
}

#[derive(Debug, Clone)]
struct TurboQuantStorage {
    context: Arc<TurboQuantContext>,
    key_codes: Option<Array>,   // [H, capacity, key_code_words] u32
    key_norms: Option<Array>,   // [H, capacity] f32
    key_gammas: Option<Array>,  // [H, capacity] f32
    value_codes: Option<Array>, // [H, capacity, value_code_words] u32
    value_norms: Option<Array>, // [H, capacity] f32
    capacity: i32,
}

impl Default for SteppingKeyValueCache {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
            turbo: None,
            config: KvCacheConfig::default(),
            offset: 0,
            step: 256,
        }
    }
}

impl SteppingKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_turbo(
        config: KvCacheConfig,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, Exception> {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            ..config
        };
        let context = Arc::new(TurboQuantContext::new(config, head_dim, num_kv_heads)?);
        Ok(Self {
            keys: None,
            values: None,
            turbo: Some(TurboQuantStorage::new(context)),
            config,
            offset: 0,
            step: 256,
        })
    }

    pub const fn kv_cache_config(&self) -> KvCacheConfig {
        self.config
    }

    /// Roll back the cache offset by `n` positions.
    ///
    /// Used by MTP speculative decode to undo a rejected draft token's KV entry.
    /// The underlying storage is not deallocated — subsequent writes will overwrite.
    pub fn trim_by(&mut self, n: i32) {
        self.offset = (self.offset - n).max(0);
    }

    /// References to internal arrays that must be eval'd between chunked-prefill steps.
    pub fn eval_targets(&self) -> Vec<&Array> {
        let mut targets = Vec::with_capacity(8);
        if let Some(ref k) = self.keys {
            targets.push(k);
        }
        if let Some(ref v) = self.values {
            targets.push(v);
        }
        if let Some(ref turbo) = self.turbo {
            targets.extend(turbo.eval_targets());
        }
        targets
    }

    /// Read-only access to internal key array (includes allocated-but-unused slots).
    pub fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    /// Read-only access to internal value array (includes allocated-but-unused slots).
    pub fn values(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Create a pre-filled cache from existing K/V arrays.
    ///
    /// Sets `offset = keys.shape()[2]` so the next `update_dense` triggers a
    /// normal grow cycle. Dense mode only (no TurboQuant).
    pub fn from_arrays(keys: Array, values: Array) -> Self {
        let offset = keys.shape()[2];
        Self {
            keys: Some(keys),
            values: Some(values),
            turbo: None,
            config: KvCacheConfig::default(),
            offset,
            step: 256,
        }
    }

    // -- TurboQuant prefix-cache helpers ----------------------------------------

    /// Read-only access to internal TQ arrays (for prefix cache block slicing).
    /// Returns `(context, key_codes, key_norms, key_gammas, value_codes, value_norms)`.
    #[allow(clippy::type_complexity)]
    pub fn turbo_arrays(
        &self,
    ) -> Option<(
        &Arc<TurboQuantContext>,
        &Array,
        &Array,
        &Array,
        &Array,
        &Array,
    )> {
        let t = self.turbo.as_ref()?;
        Some((
            &t.context,
            t.key_codes.as_ref()?,
            t.key_norms.as_ref()?,
            t.key_gammas.as_ref()?,
            t.value_codes.as_ref()?,
            t.value_norms.as_ref()?,
        ))
    }

    /// Reconstruct a TQ cache from pre-gathered arrays (prefix cache materialization).
    pub fn from_turbo_arrays(
        context: Arc<TurboQuantContext>,
        key_codes: Array,
        key_norms: Array,
        key_gammas: Array,
        value_codes: Array,
        value_norms: Array,
        offset: i32,
    ) -> Self {
        let capacity = key_codes.shape().get(1).copied().unwrap_or(0);
        Self {
            keys: None,
            values: None,
            turbo: Some(TurboQuantStorage {
                context,
                key_codes: Some(key_codes),
                key_norms: Some(key_norms),
                key_gammas: Some(key_gammas),
                value_codes: Some(value_codes),
                value_norms: Some(value_norms),
                capacity,
            }),
            config: KvCacheConfig {
                mode: KvCacheMode::Turboquant,
                ..KvCacheConfig::default()
            },
            offset,
            step: 256,
        }
    }

    /// True when TQ storage has been populated (bulk quantization has happened).
    /// Distinct from `is_quantized()` which checks config only.
    pub fn is_turbo_active(&self) -> bool {
        self.turbo.as_ref().is_some_and(|t| t.capacity > 0)
    }

    fn update_dense(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        let prev = self.offset;
        let new_tokens = keys.shape()[2];

        let need_grow = self
            .keys
            .as_ref()
            .is_none_or(|k| (prev + new_tokens) > k.shape()[2]);

        if need_grow {
            let b = keys.shape()[0];
            let n_kv_heads = keys.shape()[1];
            let k_head_dim = keys.shape()[3];
            let v_head_dim = values.shape()[3];

            let n_steps = (self.step + new_tokens - 1) / self.step;
            let new_slots = n_steps * self.step;

            let new_k = ops::zeros_dtype(&[b, n_kv_heads, new_slots, k_head_dim], keys.dtype())?;
            let new_v = ops::zeros_dtype(&[b, n_kv_heads, new_slots, v_head_dim], values.dtype())?;

            let (grown_k, grown_v) = match (self.keys.as_ref(), self.values.as_ref()) {
                (Some(old_k), Some(old_v)) => {
                    let (trimmed_k, trimmed_v) = if prev % self.step != 0 {
                        (slice_axis2(old_k, 0, prev)?, slice_axis2(old_v, 0, prev)?)
                    } else {
                        (old_k.clone(), old_v.clone())
                    };
                    let cat_k = concatenate_axis(&[trimmed_k, new_k], 2)?;
                    let cat_v = concatenate_axis(&[trimmed_v, new_v], 2)?;
                    (cat_k, cat_v)
                }
                _ => (new_k, new_v),
            };
            self.keys = Some(grown_k);
            self.values = Some(grown_v);
        }

        let k = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("Keys cannot be None after grow"))?;
        let v = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("Values cannot be None after grow"))?;

        let updated_k = slice_update_axis2(k, &keys, prev, new_tokens)?;
        let updated_v = slice_update_axis2(v, &values, prev, new_tokens)?;
        self.keys = Some(updated_k);
        self.values = Some(updated_v);

        self.offset = prev + new_tokens;

        let result_k = slice_axis2(
            self.keys
                .as_ref()
                .ok_or_else(|| Exception::custom("Keys cannot be None after update"))?,
            0,
            self.offset,
        )?;
        let result_v = slice_axis2(
            self.values
                .as_ref()
                .ok_or_else(|| Exception::custom("Values cannot be None after update"))?,
            0,
            self.offset,
        )?;

        Ok(KvCacheView::Dense {
            keys: result_k,
            values: result_v,
        })
    }

    fn update_and_view_with_activation_threshold(
        &mut self,
        keys: Array,
        values: Array,
        activate_at: i32,
    ) -> Result<KvCacheView, Exception> {
        let new_tokens = keys.shape()[2];

        let view = if let Some(turbo) = self.turbo.as_mut() {
            if new_tokens > 1 && turbo.capacity == 0 {
                // First prefill: accumulate in dense fp16 storage so attention
                // uses native SDPA (single batched GPU op). Quantization is
                // deferred until the first decode token.
                self.update_dense(keys, values)?
            } else {
                let should_activate =
                    should_activate_turboquant(self.offset, new_tokens, activate_at);

                // Decode (or subsequent multi-token after first decode).
                // If dense KV was accumulated during prefill, bulk-quantize it
                // into TurboQuant storage before appending the new token.
                if turbo.capacity == 0 && self.offset > 0 && should_activate {
                    if let (Some(dense_k), Some(dense_v)) = (&self.keys, &self.values) {
                        let k = slice_axis2(dense_k, 0, self.offset)?;
                        let v = slice_axis2(dense_v, 0, self.offset)?;
                        turbo.append(k, v, 0, self.step)?;
                        self.keys = None;
                        self.values = None;
                    }
                }

                if turbo.capacity == 0 && !should_activate {
                    self.update_dense(keys, values)?
                } else {
                    KvCacheView::TurboQuant(turbo.append(keys, values, self.offset, self.step)?)
                }
            }
        } else {
            self.update_dense(keys, values)?
        };
        self.offset = match &view {
            KvCacheView::Dense { keys, .. } => keys.shape()[2],
            KvCacheView::TurboQuant(view) => view.seq_len,
        };
        Ok(view)
    }
}

impl TurboQuantStorage {
    fn new(context: Arc<TurboQuantContext>) -> Self {
        Self {
            context,
            key_codes: None,
            key_norms: None,
            key_gammas: None,
            value_codes: None,
            value_norms: None,
            capacity: 0,
        }
    }

    fn ensure_capacity(&mut self, required: i32, step: i32) -> Result<(), Exception> {
        if required <= self.capacity {
            return Ok(());
        }
        let new_cap = ((required + step - 1) / step) * step;
        let h = self.context.num_kv_heads;
        let old_cap = self.capacity;

        self.key_codes = Some(grow_array(
            self.key_codes.take(),
            old_cap,
            &[h, new_cap, self.context.key_code_words],
            Dtype::Uint32,
        )?);
        self.key_norms = Some(grow_array(
            self.key_norms.take(),
            old_cap,
            &[h, new_cap],
            Dtype::Float32,
        )?);
        self.key_gammas = Some(grow_array(
            self.key_gammas.take(),
            old_cap,
            &[h, new_cap],
            Dtype::Float32,
        )?);
        self.value_codes = Some(grow_array(
            self.value_codes.take(),
            old_cap,
            &[h, new_cap, self.context.value_code_words],
            Dtype::Uint32,
        )?);
        self.value_norms = Some(grow_array(
            self.value_norms.take(),
            old_cap,
            &[h, new_cap],
            Dtype::Float32,
        )?);

        self.capacity = new_cap;
        Ok(())
    }

    fn append(
        &mut self,
        keys: Array,
        values: Array,
        prev: i32,
        step: i32,
    ) -> Result<TurboQuantKvView, Exception> {
        validate_turboquant_shapes(&keys, &values, &self.context)?;

        let new_tokens = keys.shape()[2];
        self.ensure_capacity(prev + new_tokens, step)?;

        // Force contiguous layout matching the logical [B, H, T, D] shape.
        let key_shape = keys.shape().to_vec();
        let value_shape = values.shape().to_vec();
        let keys = keys
            .as_dtype(Dtype::Float32)?
            .flatten(None, None)?
            .reshape(&key_shape)?;
        let values = values
            .as_dtype(Dtype::Float32)?
            .flatten(None, None)?
            .reshape(&value_shape)?;

        // Squeeze batch dim: [1, H, T, D] → [H, T, D] for GPU quantization
        let keys_3d =
            keys.reshape(&[self.context.num_kv_heads, new_tokens, self.context.head_dim])?;
        let values_3d =
            values.reshape(&[self.context.num_kv_heads, new_tokens, self.context.head_dim])?;

        // GPU quantize → lazy Arrays (no eval, no CPU readback)
        let (v_norms, v_codes) = self.context.quantize_values_gpu(&values_3d)?;
        let (k_norms, k_gammas, k_codes) = self.context.quantize_keys_gpu(&keys_3d)?;

        // slice_update into pre-allocated storage (all lazy GPU ops)
        let err = || Exception::custom("TurboQuant storage not allocated");
        self.value_norms = Some(slice_update_axis(
            self.value_norms.as_ref().ok_or_else(err)?,
            &v_norms,
            1,
            prev,
            new_tokens,
        )?);
        self.value_codes = Some(slice_update_axis(
            self.value_codes.as_ref().ok_or_else(err)?,
            &v_codes,
            1,
            prev,
            new_tokens,
        )?);
        self.key_norms = Some(slice_update_axis(
            self.key_norms.as_ref().ok_or_else(err)?,
            &k_norms,
            1,
            prev,
            new_tokens,
        )?);
        self.key_gammas = Some(slice_update_axis(
            self.key_gammas.as_ref().ok_or_else(err)?,
            &k_gammas,
            1,
            prev,
            new_tokens,
        )?);
        self.key_codes = Some(slice_update_axis(
            self.key_codes.as_ref().ok_or_else(err)?,
            &k_codes,
            1,
            prev,
            new_tokens,
        )?);

        self.view(prev + new_tokens)
    }

    fn view(&self, seq_len: i32) -> Result<TurboQuantKvView, Exception> {
        let err = || Exception::custom("TurboQuant storage not allocated");
        Ok(TurboQuantKvView {
            context: Arc::clone(&self.context),
            key_codes: slice_axis(self.key_codes.as_ref().ok_or_else(err)?, 1, 0, seq_len)?,
            key_norms: slice_axis(self.key_norms.as_ref().ok_or_else(err)?, 1, 0, seq_len)?,
            key_gammas: slice_axis(self.key_gammas.as_ref().ok_or_else(err)?, 1, 0, seq_len)?,
            value_codes: slice_axis(self.value_codes.as_ref().ok_or_else(err)?, 1, 0, seq_len)?,
            value_norms: slice_axis(self.value_norms.as_ref().ok_or_else(err)?, 1, 0, seq_len)?,
            seq_len,
        })
    }

    fn eval_targets(&self) -> Vec<&Array> {
        let mut targets = Vec::with_capacity(5);
        if let Some(ref a) = self.key_codes {
            targets.push(a);
        }
        if let Some(ref a) = self.key_norms {
            targets.push(a);
        }
        if let Some(ref a) = self.key_gammas {
            targets.push(a);
        }
        if let Some(ref a) = self.value_codes {
            targets.push(a);
        }
        if let Some(ref a) = self.value_norms {
            targets.push(a);
        }
        targets
    }
}

impl KeyValueCache for SteppingKeyValueCache {
    fn is_quantized(&self) -> bool {
        self.config.is_turboquant()
    }

    fn bits(&self) -> Option<i32> {
        self.config
            .is_turboquant()
            .then_some(i32::from(self.config.bits))
    }

    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_view(&mut self, keys: Array, values: Array) -> Result<KvCacheView, Exception> {
        self.update_and_view_with_activation_threshold(keys, values, turboquant_activate_at())
    }
}

fn validate_turboquant_shapes(
    keys: &Array,
    values: &Array,
    context: &TurboQuantContext,
) -> Result<(), Exception> {
    let key_shape = keys.shape();
    let value_shape = values.shape();
    if key_shape.len() != 4 || value_shape.len() != 4 {
        return Err(Exception::custom(
            "TurboQuant cache expects 4D [B, H, T, D] tensors",
        ));
    }
    if key_shape[0] != 1 || value_shape[0] != 1 {
        return Err(Exception::custom(
            "TurboQuant cache currently only supports batch size 1",
        ));
    }
    if key_shape[1] != context.num_kv_heads || value_shape[1] != context.num_kv_heads {
        return Err(Exception::custom("TurboQuant KV head count mismatch"));
    }
    if key_shape[2] != value_shape[2] {
        return Err(Exception::custom(
            "TurboQuant keys/values token count mismatch",
        ));
    }
    if key_shape[3] != context.head_dim || value_shape[3] != context.head_dim {
        return Err(Exception::custom("TurboQuant head_dim mismatch"));
    }
    Ok(())
}

fn usize_from_i32(value: i32, label: &str) -> Result<usize, Exception> {
    usize::try_from(value).map_err(|_| Exception::custom(format!("{label} conversion overflow")))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, Exception> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Exception::custom(format!("{label} overflow")))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize, Exception> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Exception::custom(format!("{label} overflow")))
}

/// Slice an array along axis 2: `arr[..., start:end, ...]`
#[allow(unsafe_code, clippy::indexing_slicing)]
pub fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    let ndim = arr.ndim();
    debug_assert!(ndim >= 3, "slice_axis2 requires ndim >= 3, got {ndim}");
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = arr.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[2] = start;
    ends[2] = end;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice(
            &raw mut result,
            arr.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Slice an array along axis 1: `arr[:, start:end, ...]`
///
/// Used for TQ arrays with shape `[H, capacity, ...]`.
#[allow(unsafe_code, clippy::indexing_slicing)]
pub fn slice_axis1(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    let ndim = arr.ndim();
    debug_assert!(ndim >= 2, "slice_axis1 requires ndim >= 2, got {ndim}");
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = arr.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[1] = start;
    ends[1] = end;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice(
            &raw mut result,
            arr.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Write `update` into `target` at `[..., start:start+n, ...]` on axis 2.
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_update_axis2(
    target: &Array,
    update: &Array,
    start: i32,
    n: i32,
) -> Result<Array, Exception> {
    let ndim = target.ndim();
    debug_assert!(
        ndim >= 3,
        "slice_update_axis2 requires ndim >= 3, got {ndim}"
    );
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = target.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[2] = start;
    ends[2] = start + n;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice_update(
            &raw mut result,
            target.as_ptr(),
            update.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice_update failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Slice an array along an arbitrary axis: `arr[..., start:end, ...]`.
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_axis(arr: &Array, axis: usize, start: i32, end: i32) -> Result<Array, Exception> {
    let ndim = arr.ndim();
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = arr.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[axis] = start;
    ends[axis] = end;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice(
            &raw mut result,
            arr.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Write `update` into `target` at `[..., start:start+n, ...]` on an arbitrary axis.
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_update_axis(
    target: &Array,
    update: &Array,
    axis: usize,
    start: i32,
    n: i32,
) -> Result<Array, Exception> {
    let ndim = target.ndim();
    let mut starts = vec![0i32; ndim];
    let mut ends: Vec<i32> = target.shape().to_vec();
    let strides = vec![1i32; ndim];
    starts[axis] = start;
    ends[axis] = start + n;

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        let status = mlx_sys::mlx_slice_update(
            &raw mut result,
            target.as_ptr(),
            update.as_ptr(),
            starts.as_ptr(),
            starts.len(),
            ends.as_ptr(),
            ends.len(),
            strides.as_ptr(),
            strides.len(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom("mlx_slice_update failed"));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Grow an Array buffer along axis 1 to a new capacity, preserving old data.
fn grow_array(
    old: Option<Array>,
    old_cap: i32,
    new_shape: &[i32],
    dtype: Dtype,
) -> Result<Array, Exception> {
    let new_buf = ops::zeros_dtype(new_shape, dtype)?;
    if old_cap > 0 {
        if let Some(old_arr) = old {
            return slice_update_axis(&new_buf, &old_arr, 1, 0, old_cap);
        }
    }
    Ok(new_buf)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    /// Create a zero-filled KV pair with shape `[1, n_heads, seq_len, head_dim]`.
    fn make_kv_pair(seq_len: i32, head_dim: i32) -> (Array, Array) {
        let shape = [1, 2, seq_len, head_dim];
        (
            Array::zeros::<f32>(&shape).unwrap(),
            Array::zeros::<f32>(&shape).unwrap(),
        )
    }

    #[test]
    fn test_concat_cache_initial_update() {
        let mut cache = ConcatKeyValueCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());

        let (keys, values) = make_kv_pair(4, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 4, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.offset(), 4);
    }

    #[test]
    fn test_concat_cache_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();
        assert_eq!(cache.offset(), 4);

        let (keys2, values2) = make_kv_pair(1, 8);
        let (result_keys, result_values) = cache.update_and_fetch(keys2, values2).unwrap();
        assert_eq!(result_keys.shape(), &[1, 2, 5, 8]);
        assert_eq!(result_values.shape(), &[1, 2, 5, 8]);
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn test_concat_cache_many_sequential_updates() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys, values) = make_kv_pair(3, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), 3);

        for i in 0..5 {
            let (k, v) = make_kv_pair(1, 8);
            let (rk, rv) = cache.update_and_fetch(k, v).unwrap();
            let expected_seq = 3 + i + 1;
            assert_eq!(cache.offset(), expected_seq);
            assert_eq!(rk.shape(), &[1, 2, expected_seq, 8]);
            assert_eq!(rv.shape(), &[1, 2, expected_seq, 8]);
        }

        assert_eq!(cache.offset(), 8);
    }

    #[test]
    fn test_concat_cache_default_values() {
        let cache = ConcatKeyValueCache::default();
        assert_eq!(cache.offset(), 0);
        assert!(cache.max_size().is_none());
        assert!(!cache.is_quantized());
        assert!(cache.group_size().is_none());
        assert!(cache.bits().is_none());
    }

    #[test]
    fn test_concat_cache_mismatched_shapes_error() {
        let mut cache = ConcatKeyValueCache::new();

        let (keys1, values1) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys1, values1).unwrap();

        // Mismatched head_dim (16 instead of 8)
        let (keys2, values2) = make_kv_pair(1, 16);
        let result = cache.update_and_fetch(keys2, values2);
        assert!(
            result.is_err(),
            "Mismatched head_dim should fail concatenation"
        );
    }

    #[test]
    fn test_concat_cache_1d_keys_error() {
        let mut cache = ConcatKeyValueCache::new();
        let keys = Array::zeros::<f32>(&[4]).unwrap();
        let values = Array::zeros::<f32>(&[4]).unwrap();
        let result = cache.update_and_fetch(keys, values);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_cache_ref_mut_delegation() {
        let mut cache = ConcatKeyValueCache::new();
        let cache_ref: &mut ConcatKeyValueCache = &mut cache;

        assert_eq!(KeyValueCache::offset(&cache_ref), 0);
        assert!(KeyValueCache::max_size(&cache_ref).is_none());
        assert!(!KeyValueCache::is_quantized(&cache_ref));
        assert!(KeyValueCache::group_size(&cache_ref).is_none());
        assert!(KeyValueCache::bits(&cache_ref).is_none());

        let (keys, values) = make_kv_pair(3, 8);
        let (rk, rv) = cache_ref.update_and_fetch(keys, values).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 3, 8]);
        assert_eq!(rv.shape(), &[1, 2, 3, 8]);
        assert_eq!(KeyValueCache::offset(&cache_ref), 3);
    }

    // --- SteppingKeyValueCache tests ---

    #[test]
    fn test_stepping_cache_initial_update() {
        let mut cache = SteppingKeyValueCache::new();
        assert_eq!(cache.offset(), 0);

        let (keys, values) = make_kv_pair(4, 8);
        let (rk, rv) = cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 4, 8]);
        assert_eq!(rv.shape(), &[1, 2, 4, 8]);
        assert_eq!(cache.offset(), 4);
        // Internal buffer should be 256 slots
        assert_eq!(cache.keys.as_ref().unwrap().shape()[2], 256);
    }

    #[test]
    fn test_stepping_cache_sequential_decode() {
        let mut cache = SteppingKeyValueCache::new();

        // Prefill with 4 tokens
        let (keys, values) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), 4);

        // Decode 5 single tokens
        for i in 0..5 {
            let (k, v) = make_kv_pair(1, 8);
            let (rk, rv) = cache.update_and_fetch(k, v).unwrap();
            let expected_seq = 4 + i + 1;
            assert_eq!(cache.offset(), expected_seq);
            assert_eq!(rk.shape(), &[1, 2, expected_seq, 8]);
            assert_eq!(rv.shape(), &[1, 2, expected_seq, 8]);
        }
        // Should still be using the initial 256-slot buffer (no regrowth)
        assert_eq!(cache.keys.as_ref().unwrap().shape()[2], 256);
    }

    #[test]
    fn test_stepping_cache_values_preserved() {
        let mut cache = SteppingKeyValueCache::new();

        // Write ones
        let ones_k = Array::ones::<f32>(&[1, 1, 2, 4]).unwrap();
        let ones_v = Array::ones::<f32>(&[1, 1, 2, 4]).unwrap();
        cache.update_and_fetch(ones_k, ones_v).unwrap();

        // Write twos
        let two = Array::from_f32(2.0);
        let twos_k = Array::full::<f32>(&[1, 1, 1, 4], &two).unwrap();
        let twos_v = Array::full::<f32>(&[1, 1, 1, 4], &two).unwrap();
        let (rk, rv) = cache.update_and_fetch(twos_k, twos_v).unwrap();

        rk.eval().unwrap();
        rv.eval().unwrap();

        assert_eq!(rk.shape(), &[1, 1, 3, 4]);
        // First 2 tokens should be 1.0, third should be 2.0
        let k_data: Vec<f32> = rk.as_slice().to_vec();
        assert!((k_data[0] - 1.0).abs() < 1e-6);
        assert!((k_data[4] - 1.0).abs() < 1e-6);
        assert!((k_data[8] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_turboquant_cache_round_trips_dense_fetch() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 7,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        let (keys, values) = make_kv_pair(3, 8);
        let (dense_keys, dense_values) = cache.update_and_fetch(keys, values).unwrap();

        assert_eq!(dense_keys.shape(), &[1, 2, 3, 8]);
        assert_eq!(dense_values.shape(), &[1, 2, 3, 8]);
        assert!(cache.is_quantized());
        assert_eq!(cache.bits(), Some(3));
    }

    #[test]
    fn test_as_slice_after_transpose_order() {
        // Verify whether as_slice returns logical (transposed) or storage order
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let arr = Array::from_slice(&data, &[1, 3, 2, 4]); // [B=1, L=3, H=2, D=4]
        let transposed = arr.transpose_axes(&[0, 2, 1, 3]).unwrap(); // [B=1, H=2, L=3, D=4]
        assert_eq!(transposed.shape(), &[1, 2, 3, 4]);
        transposed.eval().unwrap();
        let slice = transposed.as_slice::<f32>();

        // If LOGICAL order (transpose respected): slice[4..8] = [8,9,10,11] (h=0, t=1)
        // If STORAGE order (transpose ignored): slice[4..8] = [4,5,6,7] (original layout)
        let is_logical = slice[4] == 8.0;
        let is_storage = slice[4] == 4.0;
        eprintln!(
            "as_slice order: logical={is_logical}, storage={is_storage}, slice[4]={}, slice={:?}",
            slice[4],
            &slice[..24]
        );
        // This test documents the actual behavior — whichever assertion passes
        // tells us whether TurboQuantStorage::append is correct.
        assert!(
            is_logical || is_storage,
            "unexpected as_slice order: slice[4] = {}",
            slice[4]
        );
        // as_slice returns storage order (confirmed), so we must flatten+reshape
        // to make arrays contiguous before calling as_slice.
        assert!(is_storage, "expected storage order from as_slice");

        // Verify the fix: flatten+reshape forces contiguous layout
        let fixed = transposed
            .flatten(None, None)
            .unwrap()
            .reshape(&[1, 2, 3, 4])
            .unwrap();
        fixed.eval().unwrap();
        let fixed_slice = fixed.as_slice::<f32>();
        // After flatten+reshape, slice[4..8] should be [8,9,10,11] (h=0, t=1 in logical order)
        assert_eq!(
            fixed_slice[4],
            8.0,
            "flatten+reshape must produce contiguous logical order, got {:?}",
            &fixed_slice[..24]
        );
    }

    #[test]
    fn test_turboquant_cache_deferred_quantization() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 11,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();

        // Multi-token prefill: returns Dense (quantization deferred)
        let (keys, values) = make_kv_pair(2, 8);
        let view = cache
            .update_and_view_with_activation_threshold(keys, values, 0)
            .unwrap();
        assert!(
            view.turboquant().is_none(),
            "prefill should return Dense view"
        );
        assert_eq!(cache.offset(), 2);

        // First decode token with an immediate threshold: triggers bulk quantize.
        let (k1, v1) = make_kv_pair(1, 8);
        let view = cache
            .update_and_view_with_activation_threshold(k1, v1, 0)
            .unwrap();
        let turbo = view.turboquant().unwrap();
        assert_eq!(turbo.seq_len, 3); // 2 prefill + 1 decode
                                      // head_dim=8, key_bits=2: ceil(8*2/32) = 1 u32 word
        assert_eq!(turbo.key_codes.shape(), &[2, 3, 1]);
        // head_dim=8, bits=3: ceil(8*3/32) = 1 u32 word
        assert_eq!(turbo.value_codes.shape(), &[2, 3, 1]);
        // Dense storage cleared after bulk quantize
        assert!(cache.keys.is_none());
    }

    #[test]
    fn test_turboquant_cache_threshold_keeps_dense_until_limit() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 11,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();

        let (prefill_k, prefill_v) = make_kv_pair(2, 8);
        let view = cache
            .update_and_view_with_activation_threshold(prefill_k, prefill_v, 4)
            .unwrap();
        assert!(view.turboquant().is_none());
        assert_eq!(cache.offset(), 2);

        let (k1, v1) = make_kv_pair(1, 8);
        let view = cache
            .update_and_view_with_activation_threshold(k1, v1, 4)
            .unwrap();
        assert!(
            view.turboquant().is_none(),
            "decode below threshold should stay dense"
        );
        assert_eq!(cache.offset(), 3);
        assert!(
            cache.keys.is_some(),
            "dense storage should be retained below threshold"
        );

        let (k2, v2) = make_kv_pair(1, 8);
        let view = cache
            .update_and_view_with_activation_threshold(k2, v2, 4)
            .unwrap();
        let turbo = view
            .turboquant()
            .expect("threshold-crossing decode should activate TurboQuant");
        assert_eq!(turbo.seq_len, 4);
        assert!(
            cache.keys.is_none(),
            "dense storage should clear after activation"
        );
    }

    #[test]
    fn parse_turboquant_activate_at_clamps_invalid_values() {
        assert_eq!(
            parse_turboquant_activate_at(None),
            DEFAULT_TURBOQUANT_ACTIVATE_AT
        );
        assert_eq!(
            parse_turboquant_activate_at(Some("bad")),
            DEFAULT_TURBOQUANT_ACTIVATE_AT
        );
        assert_eq!(parse_turboquant_activate_at(Some("-5")), 0);
        assert_eq!(parse_turboquant_activate_at(Some("8192")), 8192);
    }

    #[test]
    fn should_activate_turboquant_respects_threshold() {
        assert!(should_activate_turboquant(10, 1, 0));
        assert!(!should_activate_turboquant(3, 1, 8));
        assert!(should_activate_turboquant(7, 1, 8));
        assert!(should_activate_turboquant(8, 1, 8));
    }
}
