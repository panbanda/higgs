use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use mlx_rs::{Array, Dtype, Stream, error::Exception, ops, ops::concatenate_axis};

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
static TURBOQUANT_INACTIVE_LOGGED: AtomicBool = AtomicBool::new(false);
const DEFAULT_TURBOQUANT_ACTIVATE_AT: i32 = 2048;
static MLA_LATENT_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
const DEFAULT_MLA_LATENT_CACHE_ENABLED: bool = false;

fn parse_mla_latent_cache_enabled(raw: Option<&str>) -> bool {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => true,
        Some("0" | "false" | "off" | "no") => false,
        _ => DEFAULT_MLA_LATENT_CACHE_ENABLED,
    }
}

/// Whether `DeepSeek` MLA caches should store compressed latent rows.
///
/// Env-only decision used by the standalone lazy-init cache path
/// ([`crate::deepseek_v2::DeepSeekV2::make_cache`] and the cache-recreation
/// branch of `forward_hidden`), which has no `KvCacheConfig` to consult. The
/// config-aware path is [`resolve_mla_latent_cache`].
pub fn mla_latent_cache_enabled() -> bool {
    *MLA_LATENT_CACHE_ENABLED.get_or_init(|| {
        parse_mla_latent_cache_enabled(std::env::var("HIGGS_MLA_LATENT_CACHE").ok().as_deref())
    })
}

fn parse_mla_latent_cache_override(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

/// Resolve the effective MLA latent-cache decision for engine-managed caches
/// built from a [`KvCacheConfig`].
///
/// Precedence: `HIGGS_MLA_LATENT_CACHE`, when set to a recognized value,
/// always wins (forcing latent caching on or off); otherwise
/// `config_enabled` (typically `KvCacheConfig::mla_latent`, derived from
/// `ModelConfig::mla_latent_cache`) decides; if neither is set, latent
/// caching stays off.
///
/// Unlike [`mla_latent_cache_enabled`], this re-reads the env var on every
/// call rather than caching it, since the config value can differ per model.
pub fn resolve_mla_latent_cache(config_enabled: bool) -> bool {
    parse_mla_latent_cache_override(std::env::var("HIGGS_MLA_LATENT_CACHE").ok().as_deref())
        .unwrap_or(config_enabled)
}

fn parse_turboquant_activate_at(raw: Option<&str>) -> i32 {
    raw.and_then(|s| s.parse::<i32>().ok())
        .map_or(DEFAULT_TURBOQUANT_ACTIVATE_AT, |v| v.max(0))
}

fn turboquant_activate_at() -> i32 {
    *TURBOQUANT_ACTIVATE_AT.get_or_init(|| {
        let raw = std::env::var("HIGGS_TURBOQUANT_MIN_TOKENS")
            .ok()
            .or_else(|| std::env::var("HIGGS_TURBOQUANT_ACTIVATE_AT").ok());
        parse_turboquant_activate_at(raw.as_deref())
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

    pub const fn turboquant(&self) -> Option<&TurboQuantKvView> {
        match self {
            Self::Dense { .. } => None,
            Self::TurboQuant(view) => Some(view),
        }
    }
}

/// Quantized cache view used by the `TurboQuant` decode path.
#[derive(Debug, Clone)]
pub struct TurboQuantKvView {
    /// Shared quantization metadata for this cache view.
    pub context: Arc<TurboQuantContext>,
    /// Packed key-code rows shaped `[H, T, key_code_words]`.
    pub key_codes: Array,
    /// Per-key vector norms shaped `[H, T]`.
    pub key_norms: Array,
    /// Per-key gamma factors shaped `[H, T]`.
    pub key_gammas: Array,
    /// Packed value-code rows shaped `[H, T, value_code_words]`.
    pub value_codes: Array,
    /// Per-value vector norms shaped `[H, T]`.
    pub value_norms: Array,
    /// Number of valid cached timesteps in this view.
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
                    norm: *key_norms
                        .get(scalar_index)
                        .ok_or_else(|| Exception::custom("key_norms index out of bounds"))?,
                    gamma: *key_gammas
                        .get(scalar_index)
                        .ok_or_else(|| Exception::custom("key_gammas index out of bounds"))?,
                    codes: key_codes_u8
                        .get(key_byte_start..key_byte_end)
                        .ok_or_else(|| Exception::custom("key_codes range out of bounds"))?
                        .to_vec(),
                };
                let value = QuantizedValue {
                    norm: *value_norms
                        .get(scalar_index)
                        .ok_or_else(|| Exception::custom("value_norms index out of bounds"))?,
                    codes: value_codes_u8
                        .get(value_byte_start..value_byte_end)
                        .ok_or_else(|| Exception::custom("value_codes range out of bounds"))?
                        .to_vec(),
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

        let queries_flat = queries
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.context.head_dim])?;
        let q_rot = self.context.rotate_queries(&queries_flat)?;

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
        let weights_flat = weights
            .as_dtype(Dtype::Float32)?
            .reshape(&[num_heads, self.seq_len])?;
        let out_rot = crate::turboquant::decode_weighted_values(
            &weights_flat,
            &self.value_codes,
            &self.value_norms,
            &self.context.value_centroids_array()?,
            num_heads,
            self.context.num_kv_heads,
            self.context.head_dim,
            self.seq_len,
            self.seq_len,
            self.context.config.value_bits(),
            self.context.value_code_words,
        )?;

        out_rot
            .hadamard_transform(None)?
            .reshape(&[1, num_heads, 1, self.context.head_dim])
    }
}

/// Trait for key-value caches used in autoregressive generation.
pub trait KeyValueCache {
    /// Whether this cache stores MLA compressed rows.
    fn is_mla(&self) -> bool {
        false
    }
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

    /// Append MLA latent rows and return the full latent cache plus prior offset.
    fn update_latent_and_fetch(&mut self, _rows: Array) -> Result<(Array, i32), Exception> {
        Err(Exception::custom(
            "cache does not support MLA latent storage",
        ))
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_mla(&self) -> bool {
        T::is_mla(self)
    }

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

    fn update_latent_and_fetch(&mut self, rows: Array) -> Result<(Array, i32), Exception> {
        T::update_latent_and_fetch(self, rows)
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
    mla: Option<MlaStorage>,
    config: KvCacheConfig,
    offset: i32,
    step: i32,
}

#[derive(Debug, Clone)]
struct MlaStorage {
    latent: Option<Array>, // [1, 1, capacity, kv_lora_rank + rope_dim]
    kv_lora_rank: i32,
    rope_dim: i32,
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
            mla: None,
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
        let turbo_config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            ..config
        };
        let context = Arc::new(TurboQuantContext::new(
            turbo_config,
            head_dim,
            num_kv_heads,
        )?);
        Ok(Self {
            keys: None,
            values: None,
            turbo: Some(TurboQuantStorage::new(context)),
            mla: None,
            config: turbo_config,
            offset: 0,
            step: 256,
        })
    }

    /// Construct an MLA cache that stores compressed latent plus shared `RoPE` K.
    pub fn new_mla(
        kv_lora_rank: i32,
        rope_dim: i32,
        config: KvCacheConfig,
    ) -> Result<Self, Exception> {
        if config.is_turboquant() {
            return Err(Exception::custom(
                "MLA latent cache cannot be combined with TurboQuant",
            ));
        }
        if kv_lora_rank <= 0 || rope_dim <= 0 {
            return Err(Exception::custom("MLA latent dimensions must be positive"));
        }
        Ok(Self {
            keys: None,
            values: None,
            turbo: None,
            mla: Some(MlaStorage {
                latent: None,
                kv_lora_rank,
                rope_dim,
            }),
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
    pub fn trim_by(&mut self, n: usize) {
        let trim = i32::try_from(n).unwrap_or(i32::MAX);
        self.offset = self.offset.saturating_sub(trim).max(0);
    }

    /// An **independent** deep copy: every MLX buffer is materialized into a
    /// fresh buffer (`Array::deep_clone`), so the result shares no storage with
    /// `self`. The derived `Clone` only bumps MLX refcounts (shared buffers),
    /// which is unsafe as a speculative-decode rollback checkpoint: the live
    /// cache's in-place `slice_update` lets MLX donate (reuse/free) a buffer the
    /// checkpoint still references, double-freeing it (the MTP `malloc: pointer
    /// being freed was not allocated` abort). Use this for any checkpoint that
    /// will outlive an in-place update of the live cache.
    #[must_use]
    pub fn deep_clone(&self) -> Self {
        Self {
            keys: self.keys.as_ref().map(eval_deep_clone),
            values: self.values.as_ref().map(eval_deep_clone),
            turbo: self.turbo.as_ref().map(TurboQuantStorage::deep_clone),
            mla: self.mla.as_ref().map(|m| MlaStorage {
                latent: m.latent.as_ref().map(eval_deep_clone),
                kv_lora_rank: m.kv_lora_rank,
                rope_dim: m.rope_dim,
            }),
            config: self.config,
            offset: self.offset,
            step: self.step,
        }
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
        if let Some(Some(latent)) = self.mla.as_ref().map(|m| m.latent.as_ref()) {
            targets.push(latent);
        }
        targets
    }

    /// Read-only access to internal key array (includes allocated-but-unused slots).
    pub const fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    /// Read-only access to internal value array (includes allocated-but-unused slots).
    pub const fn values(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Read-only access to the MLA latent buffer (including unused capacity).
    pub const fn latent_array(&self) -> Option<&Array> {
        match &self.mla {
            Some(storage) => storage.latent.as_ref(),
            None => None,
        }
    }

    /// Whether this cache stores MLA compressed latents instead of dense K/V.
    pub const fn is_mla(&self) -> bool {
        self.mla.is_some()
    }

    /// MLA latent and shared-RoPE widths, when this is an MLA cache.
    pub const fn mla_dimensions(&self) -> Option<(i32, i32)> {
        match &self.mla {
            Some(storage) => Some((storage.kv_lora_rank, storage.rope_dim)),
            None => None,
        }
    }

    /// Simultaneous mutable access to the key and value arrays.
    ///
    /// Re-borrows both optional fields from a single `&mut` split to satisfy
    /// the borrow checker when both must yield from one iterator.
    pub const fn key_value_arrays_mut(&mut self) -> (Option<&mut Array>, Option<&mut Array>) {
        (self.keys.as_mut(), self.values.as_mut())
    }

    /// Create a pre-filled cache from existing K/V arrays.
    ///
    /// Sets `offset = keys.shape()[2]` so the next `update_dense` triggers a
    /// normal grow cycle. Dense mode only (no `TurboQuant`).
    pub fn from_arrays(keys: Array, values: Array) -> Result<Self, Exception> {
        let offset = validate_dense_restore_shapes(&keys, &values)?;
        Ok(Self {
            keys: Some(keys),
            values: Some(values),
            turbo: None,
            mla: None,
            config: KvCacheConfig::default(),
            offset,
            step: 256,
        })
    }

    /// Restore a pre-filled MLA latent cache from `[1, 1, T, r + rope]` rows.
    pub fn from_latent_array(
        latent: Array,
        kv_lora_rank: i32,
        rope_dim: i32,
    ) -> Result<Self, Exception> {
        let [b, h, offset, width] = <[i32; 4]>::try_from(latent.shape()).map_err(|_| {
            Exception::custom("from_latent_array: latent must be [1, 1, T, r + rope]")
        })?;
        if b != 1
            || h != 1
            || width != kv_lora_rank + rope_dim
            || kv_lora_rank <= 0
            || rope_dim <= 0
        {
            return Err(Exception::custom(
                "from_latent_array: invalid MLA latent shape or dimensions",
            ));
        }
        Ok(Self {
            keys: None,
            values: None,
            turbo: None,
            mla: Some(MlaStorage {
                latent: Some(latent),
                kv_lora_rank,
                rope_dim,
            }),
            config: KvCacheConfig::default(),
            offset,
            step: 256,
        })
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

    /// Quantization context, including before the `TurboQuant` arrays have been
    /// populated. Prefix-cache restoration uses this to validate disk payloads.
    pub fn turbo_context(&self) -> Option<&Arc<TurboQuantContext>> {
        self.turbo.as_ref().map(|storage| &storage.context)
    }

    /// Restore dense arrays while retaining this cache's configured storage
    /// mode. This matters for `TurboQuant`'s dense-tail layers.
    pub fn from_arrays_with_config(
        keys: Array,
        values: Array,
        config: KvCacheConfig,
    ) -> Result<Self, Exception> {
        let offset = validate_dense_restore_shapes(&keys, &values)?;
        Ok(Self {
            keys: Some(keys),
            values: Some(values),
            turbo: None,
            mla: None,
            config,
            offset,
            step: 256,
        })
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
    ) -> Result<Self, Exception> {
        let capacity = validate_turbo_restore_shapes(
            &context,
            &key_codes,
            &key_norms,
            &key_gammas,
            &value_codes,
            &value_norms,
            offset,
        )?;
        let config = context.config;
        Ok(Self {
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
            mla: None,
            config,
            offset,
            step: 256,
        })
    }

    /// True when TQ storage has been populated (bulk quantization has happened).
    /// Distinct from `is_quantized()` which checks config only.
    pub fn is_turbo_active(&self) -> bool {
        self.turbo.as_ref().is_some_and(|t| t.capacity > 0)
    }

    fn update_dense(&mut self, keys: &Array, values: &Array) -> Result<KvCacheView, Exception> {
        let prev = self.offset;
        let k_shape = keys.shape();
        let v_shape = values.shape();
        let dim = |s: &[i32], i: usize, label: &'static str| -> Result<i32, Exception> {
            s.get(i).copied().ok_or_else(|| {
                Exception::custom(format!("update_dense: missing dim {i} ({label})"))
            })
        };
        let new_tokens = dim(k_shape, 2, "keys T")?;

        let key_cap = self
            .keys
            .as_ref()
            .map(|k| dim(k.shape(), 2, "cached keys T"))
            .transpose()?;
        let need_grow = key_cap.is_none_or(|cap| (prev + new_tokens) > cap);

        if need_grow {
            let b = dim(k_shape, 0, "keys B")?;
            let n_kv_heads = dim(k_shape, 1, "keys H")?;
            let k_head_dim = dim(k_shape, 3, "keys D")?;
            let v_head_dim = dim(v_shape, 3, "values D")?;

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

        let updated_k = slice_update_axis2(k, keys, prev, new_tokens)?;
        let updated_v = slice_update_axis2(v, values, prev, new_tokens)?;
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
        keys: &Array,
        values: &Array,
        activate_at: i32,
    ) -> Result<KvCacheView, Exception> {
        if keys.ndim() != 4 {
            return Err(Exception::custom(format!(
                "update_and_view: keys must be [B, H, T, D], got ndim {}",
                keys.ndim()
            )));
        }
        if values.ndim() != 4 {
            return Err(Exception::custom(format!(
                "update_and_view: values must be [B, H, T, D], got ndim {}",
                values.ndim()
            )));
        }
        let new_tokens = *keys.shape().get(2).ok_or_else(|| {
            Exception::custom("update_and_view: keys must have a token dim at axis 2")
        })?;

        let new_view = if let Some(turbo) = self.turbo.as_mut() {
            if new_tokens > 1 && turbo.capacity == 0 {
                // First prefill: accumulate in dense fp16 storage so attention
                // uses native SDPA (single batched GPU op). Quantization is
                // deferred until the first decode token.
                self.update_dense(keys, values)?
            } else {
                let should_activate =
                    should_activate_turboquant(self.offset, new_tokens, activate_at);
                if !should_activate
                    && TURBOQUANT_INACTIVE_LOGGED
                        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                {
                    tracing::info!(
                        activate_at,
                        current_tokens = self.offset + new_tokens,
                        "TurboQuant configured but inactive; using dense KV until the activation threshold is reached"
                    );
                }

                // Decode (or subsequent multi-token after first decode).
                // If dense KV was accumulated during prefill, bulk-quantize it
                // into TurboQuant storage before appending the new token.
                if turbo.capacity == 0 && self.offset > 0 && should_activate {
                    if let (Some(dense_k), Some(dense_v)) = (&self.keys, &self.values) {
                        let k = slice_axis2(dense_k, 0, self.offset)?;
                        let v = slice_axis2(dense_v, 0, self.offset)?;
                        turbo.append(&k, &v, 0, self.step)?;
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
        self.offset = match &new_view {
            KvCacheView::Dense {
                keys: dense_keys, ..
            } => *dense_keys.shape().get(2).ok_or_else(|| {
                Exception::custom("update_and_view: dense result missing token dim")
            })?,
            KvCacheView::TurboQuant(turbo_view) => turbo_view.seq_len,
        };
        Ok(new_view)
    }

    fn update_latent(&mut self, rows: &Array) -> Result<(Array, i32), Exception> {
        let storage = self
            .mla
            .as_mut()
            .ok_or_else(|| Exception::custom("cache does not support MLA latent storage"))?;
        let [b, h, new_tokens, width] = <[i32; 4]>::try_from(rows.shape())
            .map_err(|_| Exception::custom("MLA rows must be [1, 1, T, r + rope]"))?;
        if b != 1 || h != 1 || width != storage.kv_lora_rank + storage.rope_dim {
            return Err(Exception::custom(
                "MLA rows shape does not match cache dimensions",
            ));
        }
        let prev = self.offset;
        let capacity = storage
            .latent
            .as_ref()
            .and_then(|a| a.shape().get(2).copied())
            .unwrap_or(0);
        if prev + new_tokens > capacity {
            let slots = ((new_tokens + self.step - 1) / self.step) * self.step;
            let grown = ops::zeros_dtype(&[1, 1, slots, width], rows.dtype())?;
            storage.latent = Some(match storage.latent.take() {
                Some(old) => concatenate_axis(&[slice_axis2(&old, 0, prev)?, grown], 2)?,
                None => grown,
            });
        }
        let latent = storage
            .latent
            .as_ref()
            .ok_or_else(|| Exception::custom("MLA latent missing after grow"))?;
        storage.latent = Some(slice_update_axis2(latent, rows, prev, new_tokens)?);
        self.offset = prev + new_tokens;
        let full = slice_axis2(
            storage
                .latent
                .as_ref()
                .ok_or_else(|| Exception::custom("MLA latent missing after update"))?,
            0,
            self.offset,
        )?;
        Ok((full, self.offset))
    }
}

fn validate_dense_restore_shapes(keys: &Array, values: &Array) -> Result<i32, Exception> {
    let [k_b, k_h, k_t, _k_d] = <[i32; 4]>::try_from(keys.shape())
        .map_err(|_| Exception::custom("from_arrays: keys must be 4D [B, H, T, D]"))?;
    let [v_b, v_h, v_t, _v_d] = <[i32; 4]>::try_from(values.shape())
        .map_err(|_| Exception::custom("from_arrays: values must be 4D [B, H, T, D]"))?;
    if k_b != v_b || k_h != v_h || k_t != v_t {
        return Err(Exception::custom(
            "from_arrays: keys/values batch, head, and token dimensions must match",
        ));
    }
    Ok(k_t)
}

fn validate_turbo_restore_shapes(
    context: &TurboQuantContext,
    key_codes: &Array,
    key_norms: &Array,
    key_gammas: &Array,
    value_codes: &Array,
    value_norms: &Array,
    offset: i32,
) -> Result<i32, Exception> {
    let [kc_h, kc_t, kc_w] = <[i32; 3]>::try_from(key_codes.shape())
        .map_err(|_| Exception::custom("from_turbo_arrays: key_codes must be 3D [H, T, W]"))?;
    let [kn_h, kn_t] = <[i32; 2]>::try_from(key_norms.shape())
        .map_err(|_| Exception::custom("from_turbo_arrays: key_norms must be 2D [H, T]"))?;
    let [kg_h, kg_t] = <[i32; 2]>::try_from(key_gammas.shape())
        .map_err(|_| Exception::custom("from_turbo_arrays: key_gammas must be 2D [H, T]"))?;
    let [vc_h, vc_t, vc_w] = <[i32; 3]>::try_from(value_codes.shape())
        .map_err(|_| Exception::custom("from_turbo_arrays: value_codes must be 3D [H, T, W]"))?;
    let [vn_h, vn_t] = <[i32; 2]>::try_from(value_norms.shape())
        .map_err(|_| Exception::custom("from_turbo_arrays: value_norms must be 2D [H, T]"))?;

    if kc_h != context.num_kv_heads
        || kn_h != context.num_kv_heads
        || kg_h != context.num_kv_heads
        || vc_h != context.num_kv_heads
        || vn_h != context.num_kv_heads
    {
        return Err(Exception::custom(
            "from_turbo_arrays: KV head count mismatch",
        ));
    }
    if kc_t != kn_t || kc_t != kg_t || kc_t != vc_t || kc_t != vn_t {
        return Err(Exception::custom(
            "from_turbo_arrays: all TurboQuant arrays must have the same token count",
        ));
    }
    if kc_w != context.key_code_words || vc_w != context.value_code_words {
        return Err(Exception::custom(
            "from_turbo_arrays: packed code width mismatch for TurboQuant context",
        ));
    }
    if offset < 0 || offset > kc_t {
        return Err(Exception::custom(format!(
            "from_turbo_arrays: offset {offset} out of bounds for capacity {kc_t}"
        )));
    }

    Ok(kc_t)
}

impl TurboQuantStorage {
    const fn new(context: Arc<TurboQuantContext>) -> Self {
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

    /// Independent deep copy (see [`SteppingKeyValueCache::deep_clone`]). The
    /// shared read-only `context` is refcounted (safe to share); every packed
    /// array is materialized into its own buffer so an in-place update of the
    /// live cache cannot donate/free a buffer this snapshot holds.
    fn deep_clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            key_codes: self.key_codes.as_ref().map(eval_deep_clone),
            key_norms: self.key_norms.as_ref().map(eval_deep_clone),
            key_gammas: self.key_gammas.as_ref().map(eval_deep_clone),
            value_codes: self.value_codes.as_ref().map(eval_deep_clone),
            value_norms: self.value_norms.as_ref().map(eval_deep_clone),
            capacity: self.capacity,
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
        keys: &Array,
        values: &Array,
        prev: i32,
        step: i32,
    ) -> Result<TurboQuantKvView, Exception> {
        validate_turboquant_shapes(keys, values, &self.context)?;

        let new_tokens = *keys
            .shape()
            .get(2)
            .ok_or_else(|| Exception::custom("TurboQuantStorage::append: keys missing dim 2"))?;
        self.ensure_capacity(prev + new_tokens, step)?;

        // Force contiguous layout matching the logical [B, H, T, D] shape.
        let key_shape = keys.shape().to_vec();
        let value_shape = values.shape().to_vec();
        let keys_cont = keys
            .as_dtype(Dtype::Float32)?
            .flatten(None, None)?
            .reshape(&key_shape)?;
        let values_cont = values
            .as_dtype(Dtype::Float32)?
            .flatten(None, None)?
            .reshape(&value_shape)?;

        // Squeeze batch dim: [1, H, T, D] → [H, T, D] for GPU quantization
        let keys_3d =
            keys_cont.reshape(&[self.context.num_kv_heads, new_tokens, self.context.head_dim])?;
        let values_3d =
            values_cont.reshape(&[self.context.num_kv_heads, new_tokens, self.context.head_dim])?;

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
    fn is_mla(&self) -> bool {
        self.is_mla()
    }
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
        self.update_and_view_with_activation_threshold(&keys, &values, turboquant_activate_at())
    }

    fn update_latent_and_fetch(&mut self, rows: Array) -> Result<(Array, i32), Exception> {
        self.update_latent(&rows)
    }
}

fn validate_turboquant_shapes(
    keys: &Array,
    values: &Array,
    context: &TurboQuantContext,
) -> Result<(), Exception> {
    let key_shape = keys.shape();
    let value_shape = values.shape();
    let [k_b, k_h, k_t, k_d] = <[i32; 4]>::try_from(key_shape).map_err(|_| {
        Exception::custom("TurboQuant cache expects 4D [B, H, T, D] tensors (keys)")
    })?;
    let [v_b, v_h, v_t, v_d] = <[i32; 4]>::try_from(value_shape).map_err(|_| {
        Exception::custom("TurboQuant cache expects 4D [B, H, T, D] tensors (values)")
    })?;
    if k_b != 1 || v_b != 1 {
        return Err(Exception::custom(
            "TurboQuant cache currently only supports batch size 1",
        ));
    }
    if k_h != context.num_kv_heads || v_h != context.num_kv_heads {
        return Err(Exception::custom("TurboQuant KV head count mismatch"));
    }
    if k_t != v_t {
        return Err(Exception::custom(
            "TurboQuant keys/values token count mismatch",
        ));
    }
    if k_d != context.head_dim || v_d != context.head_dim {
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
pub fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    slice_axis(arr, 2, start, end)
}

/// Slice an array along axis 1: `arr[:, start:end, ...]`
///
/// Used for TQ arrays with shape `[H, capacity, ...]`.
pub fn slice_axis1(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
    slice_axis(arr, 1, start, end)
}

/// Evaluate, then independently deep-copy an MLX array.
///
/// `Array::deep_clone` copies bytes straight from the buffer pointer
/// (`mlx_array_data_*`), which is only valid once the array is evaluated. The
/// cache stores *lazy* `slice_update` results (see `update_dense`), so cloning a
/// checkpoint without forcing evaluation first reads an unmaterialized pointer
/// and segfaults. Eval makes the snapshot both valid and buffer-independent of
/// the live cache (no shared buffer for a later in-place update to donate/free).
//
// `expect`: a failed `eval` on a live cache array leaves nothing safe to copy —
// a loud panic here is strictly better than the segfault a lazy `deep_clone`
// would cause, and the infallible signature keeps every checkpoint call site
// (`AnyCache::deep_clone`, `deep_clone_mtp_cache`, the MTP cycle) clone-and-go.
#[allow(clippy::expect_used)]
fn eval_deep_clone(a: &Array) -> Array {
    a.eval().expect("eval before deep_clone checkpoint");
    a.deep_clone()
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
    if axis >= ndim {
        return Err(Exception::custom(format!(
            "slice_axis: axis {axis} out of bounds for ndim {ndim}"
        )));
    }
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
    if axis >= ndim {
        return Err(Exception::custom(format!(
            "slice_update_axis: axis {axis} out of bounds for ndim {ndim}"
        )));
    }
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

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
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
    fn deep_clone_preserves_contents_and_offset() {
        // deep_clone must be a faithful, independent copy.
        let mut cache = SteppingKeyValueCache::new();
        let ones_k = Array::ones::<f32>(&[1, 2, 2, 8]).unwrap();
        let ones_v = Array::ones::<f32>(&[1, 2, 2, 8]).unwrap();
        cache.update_and_fetch(ones_k, ones_v).unwrap();

        let copy = cache.deep_clone();
        assert_eq!(copy.offset(), cache.offset());

        let orig_k: Vec<f32> = {
            let k = cache.keys.as_ref().unwrap();
            k.eval().unwrap();
            k.as_slice().to_vec()
        };
        let copy_k: Vec<f32> = {
            let k = copy.keys.as_ref().unwrap();
            k.eval().unwrap();
            k.as_slice().to_vec()
        };
        assert_eq!(orig_k, copy_k, "deep_clone must copy contents faithfully");
    }

    #[test]
    fn deep_clone_checkpoint_survives_live_in_place_update() {
        // The speculative-decode invariant: a checkpoint captured before the
        // live cache is advanced must NOT change when the live cache does an
        // in-place `slice_update`. A shallow `clone()` shares the KV buffer, so
        // MLX can donate/free it under the checkpoint (the double-free abort);
        // `deep_clone()` is independent.
        let mut cache = SteppingKeyValueCache::new();
        let ones_k = Array::ones::<f32>(&[1, 2, 2, 8]).unwrap();
        let ones_v = Array::ones::<f32>(&[1, 2, 2, 8]).unwrap();
        cache.update_and_fetch(ones_k, ones_v).unwrap();

        let checkpoint = cache.deep_clone();
        let before: Vec<f32> = {
            let k = checkpoint.keys.as_ref().unwrap();
            k.eval().unwrap();
            k.as_slice().to_vec()
        };

        // Advance the LIVE cache in place with a token of value 2.0; force eval
        // so any buffer donation would fire.
        let two = Array::from_f32(2.0);
        let twos_k = Array::full::<f32>(&[1, 2, 1, 8], &two).unwrap();
        let twos_v = Array::full::<f32>(&[1, 2, 1, 8], &two).unwrap();
        let (rk, _) = cache.update_and_fetch(twos_k, twos_v).unwrap();
        rk.eval().unwrap();

        assert_eq!(
            checkpoint.offset(),
            2,
            "checkpoint offset must be unchanged"
        );
        let after: Vec<f32> = {
            let k = checkpoint.keys.as_ref().unwrap();
            k.eval().unwrap();
            k.as_slice().to_vec()
        };
        assert_eq!(
            before, after,
            "deep_clone checkpoint must survive the live cache's in-place update"
        );
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
        let data: Vec<f32> = (0..24)
            .map(|i| f32::from(i8::try_from(i).unwrap()))
            .collect();
        let arr = Array::from_slice(&data, &[1, 3, 2, 4]); // [B=1, L=3, H=2, D=4]
        let transposed = arr.transpose_axes(&[0, 2, 1, 3]).unwrap(); // [B=1, H=2, L=3, D=4]
        assert_eq!(transposed.shape(), &[1, 2, 3, 4]);
        transposed.eval().unwrap();
        let slice = transposed.as_slice::<f32>();

        // If LOGICAL order (transpose respected): slice[4..8] = [8,9,10,11] (h=0, t=1)
        // If STORAGE order (transpose ignored): slice[4..8] = [4,5,6,7] (original layout)
        let slice_4 = *slice.get(4).unwrap();
        let is_logical = (slice_4 - 8.0).abs() < f32::EPSILON;
        let is_storage = (slice_4 - 4.0).abs() < f32::EPSILON;
        // This test documents the actual behavior — whichever assertion passes
        // tells us whether TurboQuantStorage::append is correct.
        assert!(
            is_logical || is_storage,
            "unexpected as_slice order: slice[4] = {slice_4}"
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
        let fixed_4 = *fixed_slice.get(4).unwrap();
        // After flatten+reshape, slice[4..8] should be [8,9,10,11] (h=0, t=1 in logical order)
        assert!(
            (fixed_4 - 8.0).abs() < f32::EPSILON,
            "flatten+reshape must produce contiguous logical order, got {fixed_4}"
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
        let prefill_view = cache
            .update_and_view_with_activation_threshold(&keys, &values, 0)
            .unwrap();
        assert!(
            prefill_view.turboquant().is_none(),
            "prefill should return Dense view"
        );
        assert_eq!(cache.offset(), 2);

        // First decode token with an immediate threshold: triggers bulk quantize.
        let (k1, v1) = make_kv_pair(1, 8);
        let decode_view = cache
            .update_and_view_with_activation_threshold(&k1, &v1, 0)
            .unwrap();
        let turbo = decode_view.turboquant().unwrap();
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
        let prefill_view = cache
            .update_and_view_with_activation_threshold(&prefill_k, &prefill_v, 4)
            .unwrap();
        assert!(prefill_view.turboquant().is_none());
        assert_eq!(cache.offset(), 2);

        let (k1, v1) = make_kv_pair(1, 8);
        let below_view = cache
            .update_and_view_with_activation_threshold(&k1, &v1, 4)
            .unwrap();
        assert!(
            below_view.turboquant().is_none(),
            "decode below threshold should stay dense"
        );
        assert_eq!(cache.offset(), 3);
        assert!(
            cache.keys.is_some(),
            "dense storage should be retained below threshold"
        );

        let (k2, v2) = make_kv_pair(1, 8);
        let cross_view = cache
            .update_and_view_with_activation_threshold(&k2, &v2, 4)
            .unwrap();
        let turbo = cross_view
            .turboquant()
            .unwrap_or_else(|| panic!("threshold-crossing decode should activate TurboQuant"));
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
    fn parse_mla_latent_cache_enabled_accepts_common_flag_values() {
        assert!(parse_mla_latent_cache_enabled(Some("1")));
        assert!(!parse_mla_latent_cache_enabled(Some("0")));
        assert!(parse_mla_latent_cache_enabled(Some("on")));
        assert!(parse_mla_latent_cache_enabled(Some("TRUE")));
        assert!(!parse_mla_latent_cache_enabled(Some("no")));
        assert!(!parse_mla_latent_cache_enabled(Some("garbage")));
    }

    #[test]
    fn parse_mla_latent_cache_override_recognizes_explicit_values_only() {
        assert_eq!(parse_mla_latent_cache_override(Some("1")), Some(true));
        assert_eq!(parse_mla_latent_cache_override(Some("true")), Some(true));
        assert_eq!(parse_mla_latent_cache_override(Some("0")), Some(false));
        assert_eq!(parse_mla_latent_cache_override(Some("off")), Some(false));
        assert_eq!(parse_mla_latent_cache_override(Some("garbage")), None);
        assert_eq!(parse_mla_latent_cache_override(None), None);
    }

    #[test]
    #[allow(unsafe_code)]
    fn resolve_mla_latent_cache_prefers_config_when_no_env_override() {
        // SAFETY: test-only env mutation; higgs-models tests run single-threaded
        // (--test-threads=1) so there is no cross-test interleaving.
        unsafe {
            std::env::remove_var("HIGGS_MLA_LATENT_CACHE");
        }
        assert!(resolve_mla_latent_cache(true));
        assert!(!resolve_mla_latent_cache(false));
    }

    #[test]
    #[allow(unsafe_code)]
    fn resolve_mla_latent_cache_env_override_wins_over_config() {
        // SAFETY: test-only env mutation; see note above.
        unsafe {
            std::env::set_var("HIGGS_MLA_LATENT_CACHE", "0");
        }
        assert!(!resolve_mla_latent_cache(true));
        unsafe {
            std::env::set_var("HIGGS_MLA_LATENT_CACHE", "1");
        }
        assert!(resolve_mla_latent_cache(false));
        unsafe {
            std::env::remove_var("HIGGS_MLA_LATENT_CACHE");
        }
    }

    #[test]
    fn should_activate_turboquant_respects_threshold() {
        assert!(should_activate_turboquant(10, 1, 0));
        assert!(!should_activate_turboquant(3, 1, 8));
        assert!(should_activate_turboquant(7, 1, 8));
        assert!(should_activate_turboquant(8, 1, 8));
    }

    /// Regression: `update_and_view_with_activation_threshold` must reject
    /// 3D inputs because `update_dense` indexes `shape()[3]`. Previously the
    /// guard was `ndim() < 3`, which let 3D arrays through and panicked.
    #[test]
    fn test_update_and_view_rejects_3d_input() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        // 3D shape [B, T, D] — missing the head axis.
        let bad_keys = Array::zeros::<f32>(&[1, 4, 8]).unwrap();
        let bad_values = Array::zeros::<f32>(&[1, 4, 8]).unwrap();
        let result = cache.update_and_view_with_activation_threshold(&bad_keys, &bad_values, 0);
        assert!(result.is_err(), "3D input should be rejected, not panic");
    }

    /// Regression: `from_turbo_arrays` must propagate the parent context's
    /// `KvCacheConfig` (`key_bits`, seed, etc.) instead of hard-coding defaults.
    /// Previously the constructor wrote `KvCacheConfig { mode: Turboquant,
    /// ..default() }`, silently dropping all user-configured fields.
    #[test]
    fn test_from_turbo_arrays_propagates_config() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 4,
            seed: 99,
            ..Default::default()
        };
        let context = Arc::new(TurboQuantContext::new(config, 8, 2).unwrap());

        // Build minimal placeholder arrays — shapes don't matter for this test;
        // we only verify the config was carried through.
        let key_codes = Array::zeros::<u32>(&[2, 0, 1]).unwrap();
        let key_norms = Array::zeros::<f32>(&[2, 0]).unwrap();
        let key_gammas = Array::zeros::<f32>(&[2, 0]).unwrap();
        let value_codes = Array::zeros::<u32>(&[2, 0, 1]).unwrap();
        let value_norms = Array::zeros::<f32>(&[2, 0]).unwrap();

        let cache = SteppingKeyValueCache::from_turbo_arrays(
            context,
            key_codes,
            key_norms,
            key_gammas,
            value_codes,
            value_norms,
            0,
        )
        .unwrap();

        let propagated = cache.kv_cache_config();
        assert_eq!(propagated.bits, 4, "bits must be carried from context");
        assert_eq!(propagated.seed, 99, "seed must be carried from context");
        assert!(matches!(propagated.mode, KvCacheMode::Turboquant));
    }

    #[test]
    fn mla_storage_appends_trims_and_clones_independently() {
        let mut cache = SteppingKeyValueCache::new_mla(4, 2, KvCacheConfig::default()).unwrap();
        let first = Array::ones::<f32>(&[1, 1, 3, 6]).unwrap();
        let (stored, offset) = cache.update_latent_and_fetch(first).unwrap();
        assert_eq!(stored.shape(), &[1, 1, 3, 6]);
        assert_eq!(offset, 3);
        assert!(cache.is_mla());
        assert_eq!(cache.eval_targets().len(), 1);

        cache.trim_by(1);
        let replacement = Array::zeros::<f32>(&[1, 1, 1, 6]).unwrap();
        let (rewritten, rewritten_offset) = cache.update_latent_and_fetch(replacement).unwrap();
        assert_eq!(rewritten.shape(), &[1, 1, 3, 6]);
        assert_eq!(rewritten_offset, 3);

        let checkpoint = cache.deep_clone();
        let update = Array::full::<f32>(&[1, 1, 1, 6], &Array::from_f32(2.0)).unwrap();
        cache.update_latent_and_fetch(update).unwrap();
        assert_eq!(checkpoint.offset(), 3);
        assert_eq!(checkpoint.latent_array().unwrap().shape(), &[1, 1, 256, 6]);
    }

    #[test]
    fn mla_from_latent_array_round_trips() {
        let latent = Array::ones::<f32>(&[1, 1, 5, 6]).unwrap();
        let cache = SteppingKeyValueCache::from_latent_array(latent, 4, 2).unwrap();
        assert!(cache.is_mla());
        assert_eq!(cache.offset(), 5);
        assert_eq!(cache.latent_array().unwrap().shape(), &[1, 1, 5, 6]);
    }
}
