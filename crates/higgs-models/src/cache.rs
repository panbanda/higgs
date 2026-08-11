use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use mlx_rs::{Array, Dtype, Stream, error::Exception, ops, ops::concatenate_axis};

/// RoPE parameters needed to re-rotate cached keys when token positions are
/// renumbered after a prune. Mirrors the `nn::Rope` fields the model builds
/// (`base`, `dimensions`, `scale`, `traditional`). Only non-traditional
/// (NeoX half-split) RoPE is supported — the Qwen3 path uses `traditional(false)`.
#[allow(clippy::doc_markdown, clippy::too_long_first_doc_paragraph)]
#[derive(Debug, Clone, Copy)]
pub struct RopeShift {
    pub base: f32,
    pub dims: i32,
    pub scale: f32,
    pub traditional: bool,
}

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
/// KV token count at which TurboQuant quantization activates for decode.
///
/// Default high because TurboQuant's custom Metal decode kernels are slower
/// than MLX's dense SDPA. Benchmarked on Qwen3.5-9B-4bit (M4 32 GB), forcing
/// activation via `HIGGS_TURBOQUANT_MIN_TOKENS=0`:
///
/// | context | dense | turboquant | Δ decode |
/// |---------|-------|------------|----------|
/// | ~15 tok | 15.6 tok/s | 11.7 tok/s | −25% |
/// | ~7K tok | 12.5 tok/s | 10.1 tok/s | −19% |
///
/// (plus a multi-second first-token stall at 7K when the prefilled KV is
/// bulk-quantized: TTFT 0.4 s → 6.4 s.) Dense KV costs only ~10 KB/token, so
/// ~100K tokens fit in ~1 GB — TurboQuant's memory saving only pays for its
/// decode tax near that scale. Lower via `HIGGS_TURBOQUANT_MIN_TOKENS` when
/// context length genuinely threatens memory.
// ponytail: fixed default; the real fix is a faster fused decode kernel
// (turboquant.rs values kernel is O(T)/thread, no SIMD reduction) — until then
// dense wins below ~100K, so gate TurboQuant off by default.
#[allow(clippy::doc_markdown)]
pub const DEFAULT_TURBOQUANT_ACTIVATE_AT: i32 = 100_000;

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
            config: turbo_config,
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

    /// Prune the half-open token span `[a, b)` from a dense cache, compacting the
    /// survivors and renumbering positions so they stay dense (TIM-style KV prune
    /// + positional reuse).
    ///
    /// Keys are stored *post-RoPE* at their insertion position, so dropping a span
    /// and shifting the suffix down by `Δ = b - a` requires left-multiplying the
    /// surviving suffix keys by `R(-Δ)` — one uniform rotation (values are not
    /// roped). The result is bit-equivalent to a cache built as if the pruned
    /// tokens never existed; see `prune_span_equiv_never_inserted`.
    ///
    /// Dense path only. Returns an error on a TurboQuant cache: re-roping quantized
    /// keys would need dequant→re-rope→requant.
    // ponytail: dense KV only; add TurboQuant prune when long-context + prune are
    // both needed at once.
    #[allow(clippy::doc_markdown)]
    pub fn prune_span(&mut self, a: i32, b: i32, rope: RopeShift) -> Result<(), Exception> {
        if self.turbo.is_some() {
            return Err(Exception::custom(
                "prune_span: dense KV only (TurboQuant unsupported)",
            ));
        }
        let off = self.offset;
        if a < 0 || a > b || b > off {
            return Err(Exception::custom(format!(
                "prune_span: invalid span [{a}, {b}) for offset {off}"
            )));
        }
        let delta = b - a;
        if delta == 0 {
            return Ok(());
        }
        if a == 0 && b == off {
            return Err(Exception::custom(
                "prune_span: refusing to prune the entire cache",
            ));
        }

        let keys = self
            .keys
            .as_ref()
            .ok_or_else(|| Exception::custom("prune_span: dense keys missing"))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| Exception::custom("prune_span: dense values missing"))?;

        let mut key_parts: Vec<Array> = Vec::with_capacity(2);
        let mut value_parts: Vec<Array> = Vec::with_capacity(2);
        if a > 0 {
            key_parts.push(slice_axis2(keys, 0, a)?);
            value_parts.push(slice_axis2(values, 0, a)?);
        }
        if off > b {
            let key_tail = slice_axis2(keys, b, off)?;
            key_parts.push(rope_shift_back(&key_tail, delta, rope)?);
            value_parts.push(slice_axis2(values, b, off)?);
        }

        let new_keys = if key_parts.len() == 1 {
            key_parts.remove(0)
        } else {
            concatenate_axis(&key_parts, 2)?
        };
        let new_values = if value_parts.len() == 1 {
            value_parts.remove(0)
        } else {
            concatenate_axis(&value_parts, 2)?
        };

        self.keys = Some(new_keys);
        self.values = Some(new_values);
        self.offset = off - delta;
        Ok(())
    }

    /// An **independent** deep copy: every MLX buffer is copied and materialized
    /// on-device, so the result shares no storage with `self`. The derived
    /// `Clone` only bumps MLX refcounts (shared buffers), which is unsafe as a
    /// speculative-decode rollback checkpoint: the live cache's in-place
    /// `slice_update` lets MLX donate (reuse/free) a buffer the checkpoint still
    /// references, double-freeing it (the MTP `malloc: pointer being freed was
    /// not allocated` abort). Use this for any checkpoint that will outlive an
    /// in-place update of the live cache.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn deep_clone(&self) -> Self {
        self.try_deep_clone()
            .expect("device copy failed for KV cache checkpoint")
    }

    /// Fallible independent device-side copy of every retained KV buffer.
    pub fn try_deep_clone(&self) -> Result<Self, Exception> {
        Ok(Self {
            keys: self.keys.as_ref().map(try_eval_deep_clone).transpose()?,
            values: self.values.as_ref().map(try_eval_deep_clone).transpose()?,
            turbo: self
                .turbo
                .as_ref()
                .map(TurboQuantStorage::try_deep_clone)
                .transpose()?,
            config: self.config,
            offset: self.offset,
            step: self.step,
        })
    }

    /// Estimated device bytes retained by this cache's allocated arrays.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let dense = self
            .keys
            .as_ref()
            .map_or(0, Array::nbytes)
            .saturating_add(self.values.as_ref().map_or(0, Array::nbytes));
        self.turbo
            .as_ref()
            .map_or(dense, |turbo| dense.saturating_add(turbo.estimated_bytes()))
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
    pub const fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    /// Read-only access to internal value array (includes allocated-but-unused slots).
    pub const fn values(&self) -> Option<&Array> {
        self.values.as_ref()
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

    /// Bulk-compress a dense cache's resident KV into `TurboQuant` storage for
    /// cheap between-turn retention, reusing the same dense→TQ path that the
    /// activation-threshold decode transition uses (slice `[0, offset)` → GPU
    /// `quantize_*_gpu` → packed storage, then drop the fp16 buffers).
    ///
    /// Returns `Ok(true)` when the cache was compressed, `Ok(false)` when it was
    /// left dense (already TQ-active, empty, or a non-power-of-2 `head_dim` that
    /// the FWHT can't handle). Idempotent and safe to call on any cache.
    ///
    /// Correctness: after compression the cache is structurally identical to one
    /// that activated TurboQuant naturally during decode — `self.turbo` holds the
    /// populated storage and `self.keys/values` are cleared — so a continuation
    /// (`update_and_view` appending the next turn's tokens) takes the ordinary
    /// `turbo.append` decode-append path. The reused tokens are read back through
    /// the same dequantize path as any other TQ cache, so they carry exactly the
    /// TurboQuant reconstruction error and nothing more (no second quantization,
    /// no positional/RoPE renumbering). Keys are stored post-RoPE at their
    /// original positions and are not renumbered, so continuation positions stay
    /// consistent.
    ///
    /// If the cache was created dense (config `mode == Off`), `config` supplies
    /// the TurboQuant bit-widths/seed to use; an already-TurboQuant `self.config`
    /// is reused as-is and `config` is ignored.
    #[allow(clippy::doc_markdown)]
    pub fn quantize_for_retention(&mut self, config: KvCacheConfig) -> Result<bool, Exception> {
        // Already TQ-active, or nothing dense to compress → leave as-is.
        if self.is_turbo_active() || self.offset == 0 {
            return Ok(false);
        }
        let Some(dense_k) = self.keys.as_ref() else {
            return Ok(false);
        };

        // Geometry from the dense buffer: shape is [B, H, capacity, D].
        let shape = dense_k.shape();
        let dim = |i: usize, label: &'static str| -> Result<i32, Exception> {
            shape.get(i).copied().ok_or_else(|| {
                Exception::custom(format!("quantize_for_retention: missing dim {i} ({label})"))
            })
        };
        let num_kv_heads = dim(1, "H")?;
        let head_dim = dim(3, "D")?;

        // FWHT requires a power-of-two head_dim; anything else cannot be TQ-packed.
        // Leave the cache dense rather than ship a wrong (or erroring) compression.
        if head_dim < 2 || (head_dim & (head_dim - 1)) != 0 {
            return Ok(false);
        }

        // Reuse an existing TurboQuant config if the cache already carries one
        // (its centroids/seed must match a later decode); otherwise adopt the
        // caller-supplied config, forced into TurboQuant mode.
        let turbo_config = if self.config.is_turboquant() {
            self.config
        } else {
            KvCacheConfig {
                mode: KvCacheMode::Turboquant,
                ..config
            }
        };

        let context = Arc::new(TurboQuantContext::new(
            turbo_config,
            head_dim,
            num_kv_heads,
        )?);
        let mut storage = TurboQuantStorage::new(context);

        // Bulk-quantize exactly the valid span [0, offset) — the same slice the
        // activation-threshold transition feeds to `append`.
        let k = slice_axis2(dense_k, 0, self.offset)?;
        let v = slice_axis2(
            self.values
                .as_ref()
                .ok_or_else(|| Exception::custom("quantize_for_retention: dense values missing"))?,
            0,
            self.offset,
        )?;
        storage.append(&k, &v, 0, self.step)?;

        self.turbo = Some(storage);
        self.config = turbo_config;
        self.keys = None;
        self.values = None;
        Ok(true)
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
        // DIAGNOSTIC (HIGGS_DIAG_EVAL_CACHE=1): force-eval the slice_update result
        // so the stored cache is concrete (independent of later layer temporaries).
        // Tests the lazy-recycling corruption hypothesis.
        if std::env::var("HIGGS_DIAG_EVAL_CACHE").is_ok_and(|flag| flag == "1") {
            mlx_rs::transforms::eval([&updated_k, &updated_v])?;
        }
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
    fn try_deep_clone(&self) -> Result<Self, Exception> {
        Ok(Self {
            context: Arc::clone(&self.context),
            key_codes: self
                .key_codes
                .as_ref()
                .map(try_eval_deep_clone)
                .transpose()?,
            key_norms: self
                .key_norms
                .as_ref()
                .map(try_eval_deep_clone)
                .transpose()?,
            key_gammas: self
                .key_gammas
                .as_ref()
                .map(try_eval_deep_clone)
                .transpose()?,
            value_codes: self
                .value_codes
                .as_ref()
                .map(try_eval_deep_clone)
                .transpose()?,
            value_norms: self
                .value_norms
                .as_ref()
                .map(try_eval_deep_clone)
                .transpose()?,
            capacity: self.capacity,
        })
    }

    fn estimated_bytes(&self) -> usize {
        self.key_codes
            .as_ref()
            .map_or(0, Array::nbytes)
            .saturating_add(self.key_norms.as_ref().map_or(0, Array::nbytes))
            .saturating_add(self.key_gammas.as_ref().map_or(0, Array::nbytes))
            .saturating_add(self.value_codes.as_ref().map_or(0, Array::nbytes))
            .saturating_add(self.value_norms.as_ref().map_or(0, Array::nbytes))
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

/// Re-rotate a `[1, H, T, D]` block of post-RoPE keys by `R(-delta)`, shifting
/// every token's effective position down by `delta`.
///
/// For non-traditional (NeoX) RoPE the head dim splits into halves
/// `x1 = x[..:dims/2]`, `x2 = x[dims/2:dims]`, rotated per frequency
/// `f_i = base^(-2i/dims)` by angle `delta * scale * f_i`. Applying `R(-delta)`:
///
/// ```text
/// out1 =  x1*cos + x2*sin
/// out2 = -x1*sin + x2*cos
/// ```
///
/// Dims beyond `dims` (partial rotary) pass through unrotated. The angles are
/// constant across tokens and heads, so this is one broadcast multiply — cheap.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown
)]
fn rope_shift_back(block: &Array, delta: i32, rope: RopeShift) -> Result<Array, Exception> {
    if rope.traditional {
        return Err(Exception::custom(
            "rope_shift_back: traditional (interleaved) RoPE unsupported",
        ));
    }
    let dims = rope.dims;
    if dims < 2 || dims % 2 != 0 {
        return Err(Exception::custom(format!(
            "rope_shift_back: rope dims must be even and >= 2, got {dims}"
        )));
    }
    let full_dim = *block
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("rope_shift_back: block has no last dim"))?;
    if dims > full_dim {
        return Err(Exception::custom(format!(
            "rope_shift_back: rope dims {dims} exceed head dim {full_dim}"
        )));
    }

    let half = dims / 2;
    let half_usize = usize::try_from(half).unwrap_or(0);
    let ln_base = rope.base.ln();
    let mut cos_vec = Vec::with_capacity(half_usize);
    let mut sin_vec = Vec::with_capacity(half_usize);
    for i in 0..half_usize {
        let inv_freq = (-(2.0 * i as f32) / dims as f32 * ln_base).exp();
        let angle = delta as f32 * rope.scale * inv_freq;
        cos_vec.push(angle.cos());
        sin_vec.push(angle.sin());
    }
    let cos = Array::from_slice(&cos_vec, &[1, 1, 1, half]);
    let sin = Array::from_slice(&sin_vec, &[1, 1, 1, half]);

    let x1 = slice_axis(block, 3, 0, half)?;
    let x2 = slice_axis(block, 3, half, dims)?;
    let out1 = x1.multiply(&cos)?.add(x2.multiply(&sin)?)?;
    let out2 = x2.multiply(&cos)?.subtract(x1.multiply(&sin)?)?;
    let rotated = concatenate_axis(&[out1, out2], 3)?;

    if dims < full_dim {
        let passthrough = slice_axis(block, 3, dims, full_dim)?;
        concatenate_axis(&[rotated, passthrough], 3)
    } else {
        Ok(rotated)
    }
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

/// Independently deep-copy an MLX array, staying on device.
///
/// `Array::deep_clone` round-trips the buffer through **host** memory
/// (`mlx_array_data_*` → CPU copy → re-upload on next use). For a multi-GB
/// hybrid cache snapshot that is seconds of wall time per store/materialize —
/// it dominated warm-turn TTFT. Pinned MLX 0.30.6's `mlx_copy` is not enough:
/// evaluation only creates a copy-on-write alias, so a small recurrent-state
/// view can retain its much larger prompt backing allocation.
///
/// [`crate::metal_kernel::materialized_device_copy`] uses an identity Metal
/// kernel whose output allocation is unconditionally fresh and exactly
/// `out.nbytes()`. The immediate evaluation therefore gives the checkpoint
/// independent, compact device storage before the live cache can mutate.
///
/// The fallible form lets publication paths reject a snapshot atomically when
/// the device copy or evaluation fails. Public clone-and-go APIs add their
/// infallible `expect` adapter at the whole-cache boundary.
pub(crate) fn try_eval_deep_clone(a: &Array) -> Result<Array, Exception> {
    crate::metal_kernel::materialized_device_copy(a)
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

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::doc_markdown,
    clippy::suboptimal_flops
)]
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

    /// The core KV-prune invariant: pruning span `[a, b)` and re-roping the
    /// survivors must leave the cache bit-equivalent (within f32 tolerance) to a
    /// cache built as if those tokens never existed and the rest were renumbered
    /// densely. Proves the compaction + `R(-Δ)` geometry generally, against the
    /// model's real `apply_rope`, not a hand-picked expected tensor.
    #[test]
    fn prune_span_equiv_never_inserted() {
        use crate::utils::apply_rope;
        use mlx_rs::builder::Builder;
        use mlx_rs::nn;

        let (h, d, n) = (2i32, 8i32, 10i32);
        let (a, b) = (3i32, 7i32); // prune [3, 7), Δ = 4
        let base = 1_000_000.0_f32;
        let rope = nn::RopeBuilder::new(d)
            .traditional(false)
            .base(base)
            .scale(1.0)
            .build()
            .unwrap();
        let shift = RopeShift {
            base,
            dims: d,
            scale: 1.0,
            traditional: false,
        };

        // Deterministic, varied raw (pre-RoPE) keys/values, distinct per element.
        let count = usize::try_from(h * n * d).unwrap();
        let raw_k: Vec<f32> = (0..count).map(|i| (i as f32 * 0.7).sin()).collect();
        let raw_v: Vec<f32> = (0..count).map(|i| (i as f32 * 0.9 + 1.0).cos()).collect();
        let raw_k = Array::from_slice(&raw_k, &[1, h, n, d]);
        let raw_v = Array::from_slice(&raw_v, &[1, h, n, d]);

        // LIVE: rope all tokens at positions 0..n, insert, then prune [a, b).
        let roped_all = apply_rope(&raw_k, &rope, 0).unwrap();
        let mut live = SteppingKeyValueCache::new();
        live.update_and_fetch(roped_all, raw_v.clone()).unwrap();
        live.prune_span(a, b, shift).unwrap();
        assert_eq!(live.offset(), n - (b - a));

        // REFERENCE: survivors only (raw rows [0,a) ++ [b,n)) roped at compacted
        // positions 0..n-Δ, inserted into a fresh cache.
        let surv_k = concatenate_axis(
            &[
                slice_axis2(&raw_k, 0, a).unwrap(),
                slice_axis2(&raw_k, b, n).unwrap(),
            ],
            2,
        )
        .unwrap();
        let surv_v = concatenate_axis(
            &[
                slice_axis2(&raw_v, 0, a).unwrap(),
                slice_axis2(&raw_v, b, n).unwrap(),
            ],
            2,
        )
        .unwrap();
        let ref_roped = apply_rope(&surv_k, &rope, 0).unwrap();
        let mut reference = SteppingKeyValueCache::new();
        reference.update_and_fetch(ref_roped, surv_v).unwrap();

        // Compare the valid regions [0:offset). `as_slice` returns *storage*
        // order (see `test_as_slice_after_transpose_order`); the reference keys
        // are a non-contiguous slice of the 256-slot buffer, so flatten both to
        // logical order before comparing.
        let off = live.offset();
        let logical = |a: &Array| -> Vec<f32> {
            let v = slice_axis2(a, 0, off).unwrap().flatten(None, None).unwrap();
            v.eval().unwrap();
            v.as_slice::<f32>().to_vec()
        };
        let live_k = logical(live.keys().unwrap());
        let ref_k = logical(reference.keys().unwrap());
        let live_v = logical(live.values().unwrap());
        let ref_v = logical(reference.values().unwrap());

        let max_abs = |x: &[f32], y: &[f32]| -> f32 {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0_f32, f32::max)
        };
        let key_err = max_abs(&live_k, &ref_k);
        let val_err = max_abs(&live_v, &ref_v);
        assert!(
            key_err < 2e-3,
            "keys diverge after prune+re-rope: {key_err}"
        );
        assert!(
            val_err < 1e-6,
            "values must match exactly (not roped): {val_err}"
        );
    }

    #[test]
    fn prune_span_rejects_turboquant() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        let (keys, values) = make_kv_pair(4, 8);
        cache.update_and_fetch(keys, values).unwrap();
        let shift = RopeShift {
            base: 1_000_000.0,
            dims: 8,
            scale: 1.0,
            traditional: false,
        };
        assert!(cache.prune_span(1, 2, shift).is_err());
    }

    #[test]
    fn deep_clone_preserves_contents_and_offset() {
        let _exec = crate::mlx_exec::acquire();
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
    fn device_deep_clone_compacts_large_backing_view() {
        let _exec = crate::mlx_exec::acquire();
        const BACKING_TOKENS: i32 = 262_144;
        const RETAINED_TOKENS: i32 = 4;
        let values: Vec<f32> = (0..BACKING_TOKENS * 4).map(|index| index as f32).collect();
        let backing = Array::from_slice(&values, &[1, 1, BACKING_TOKENS, 4]);
        backing.eval().unwrap();
        let retained =
            slice_axis2(&backing, BACKING_TOKENS - RETAINED_TOKENS, BACKING_TOKENS).unwrap();
        retained.eval().unwrap();

        assert!(
            backing.nbytes() > retained.nbytes() * 1_000,
            "regression requires a tiny view over a much larger allocation"
        );
        let expected = retained.as_slice::<f32>().to_vec();
        let retained_ptr = retained.as_slice::<f32>().as_ptr();

        let checkpoint = try_eval_deep_clone(&retained).unwrap();
        assert_eq!(checkpoint.shape(), retained.shape());
        assert_eq!(checkpoint.dtype(), retained.dtype());
        assert_eq!(checkpoint.nbytes(), retained.nbytes());
        assert_ne!(
            checkpoint.as_slice::<f32>().as_ptr(),
            retained_ptr,
            "checkpoint must own a distinct evaluated device allocation"
        );

        drop(retained);
        drop(backing);
        let pressure = Array::zeros::<f32>(&[1, 1, BACKING_TOKENS, 4]).unwrap();
        pressure.eval().unwrap();
        drop(pressure);

        assert_eq!(
            checkpoint.as_slice::<f32>(),
            expected,
            "compact checkpoint must remain valid after its large backing is released"
        );
    }

    #[test]
    fn device_deep_clone_preserves_cache_dtypes_shapes_and_bits() {
        let _exec = crate::mlx_exec::acquire();
        let values = Array::from_slice(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 2, 2]);
        for dtype in [
            Dtype::Float16,
            Dtype::Bfloat16,
            Dtype::Float32,
            Dtype::Uint32,
        ] {
            let typed = values.as_dtype(dtype).unwrap();
            let copy = try_eval_deep_clone(&typed).unwrap();
            assert_eq!(copy.shape(), &[2, 2, 2]);
            assert_eq!(copy.dtype(), dtype);
            assert_eq!(
                copy.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>(),
                typed.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>()
            );
        }

        let bit_patterns = [
            0.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            f32::from_bits(0x7fc1_2345).to_bits(),
            f32::NEG_INFINITY.to_bits(),
        ];
        let bit_values: Vec<f32> = bit_patterns.iter().copied().map(f32::from_bits).collect();
        let bit_source = Array::from_slice(&bit_values, &[2, 2]);
        let bit_copy = try_eval_deep_clone(&bit_source).unwrap();
        let copied_bits: Vec<u32> = bit_copy
            .as_slice::<f32>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(copied_bits, bit_patterns);

        let f16_patterns = [0x0000_u16, 0x8000, 0x7e01, 0xfc00];
        let f16_values: Vec<half::f16> = f16_patterns
            .iter()
            .copied()
            .map(half::f16::from_bits)
            .collect();
        let f16_source = Array::from_slice(&f16_values, &[2, 2]);
        let f16_copy = try_eval_deep_clone(&f16_source).unwrap();
        let copied_f16_bits: Vec<u16> = f16_copy
            .as_slice::<half::f16>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(copied_f16_bits, f16_patterns);

        let bf16_patterns = [0x0000_u16, 0x8000, 0x7fc1, 0xff80];
        let bf16_values: Vec<half::bf16> = bf16_patterns
            .iter()
            .copied()
            .map(half::bf16::from_bits)
            .collect();
        let bf16_source = Array::from_slice(&bf16_values, &[2, 2]);
        let bf16_copy = try_eval_deep_clone(&bf16_source).unwrap();
        let copied_bf16_bits: Vec<u16> = bf16_copy
            .as_slice::<half::bf16>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(copied_bf16_bits, bf16_patterns);

        let empty = Array::zeros::<u32>(&[2, 0, 3]).unwrap();
        let empty_copy = try_eval_deep_clone(&empty).unwrap();
        assert_eq!(empty_copy.shape(), &[2, 0, 3]);
        assert_eq!(empty_copy.dtype(), Dtype::Uint32);
        assert_eq!(empty_copy.nbytes(), 0);
    }

    #[test]
    fn device_deep_clone_materializes_noncontiguous_logical_order() {
        let _exec = crate::mlx_exec::acquire();
        let values: Vec<f32> = (0_u8..24).map(f32::from).collect();
        let backing = Array::from_slice(&values, &[1, 3, 2, 4]);
        let transposed = backing.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let expected = transposed
            .flatten(None, None)
            .unwrap()
            .reshape(transposed.shape())
            .unwrap();
        expected.eval().unwrap();

        let checkpoint = try_eval_deep_clone(&transposed).unwrap();
        assert_eq!(checkpoint.shape(), transposed.shape());
        assert_eq!(checkpoint.dtype(), Dtype::Float32);
        assert_eq!(
            checkpoint.as_slice::<f32>(),
            expected.as_slice::<f32>(),
            "materialization must preserve the logical order of a strided view"
        );
        assert_ne!(
            checkpoint.as_slice::<f32>().as_ptr(),
            backing.as_slice::<f32>().as_ptr(),
            "materialized output must not alias the transposed backing allocation"
        );
    }

    #[test]
    fn turboquant_deep_clone_materializes_all_mutable_arrays() {
        let _exec = crate::mlx_exec::acquire();
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 17,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new_turbo(config, 2, 8).unwrap();
        let (keys, values) = make_kv_pair(4, 8);
        cache
            .update_and_view_with_activation_threshold(&keys, &values, 0)
            .unwrap();
        let (decode_key, decode_value) = make_kv_pair(1, 8);
        cache
            .update_and_view_with_activation_threshold(&decode_key, &decode_value, 0)
            .unwrap();
        let checkpoint = cache.try_deep_clone().unwrap();
        let live = cache.turbo.as_ref().unwrap();
        let copied = checkpoint.turbo.as_ref().unwrap();

        assert!(Arc::ptr_eq(&live.context, &copied.context));
        let assert_distinct = |left: &Array, right: &Array| {
            left.eval().unwrap();
            right.eval().unwrap();
            assert_eq!(left.shape(), right.shape());
            assert_eq!(left.dtype(), right.dtype());
            if left.dtype() == Dtype::Uint32 {
                assert_eq!(left.as_slice::<u32>(), right.as_slice::<u32>());
                assert_ne!(
                    left.as_slice::<u32>().as_ptr(),
                    right.as_slice::<u32>().as_ptr()
                );
            } else {
                assert_eq!(left.dtype(), Dtype::Float32);
                assert_eq!(left.as_slice::<f32>(), right.as_slice::<f32>());
                assert_ne!(
                    left.as_slice::<f32>().as_ptr(),
                    right.as_slice::<f32>().as_ptr()
                );
            }
        };

        assert_distinct(
            live.key_codes.as_ref().unwrap(),
            copied.key_codes.as_ref().unwrap(),
        );
        assert_distinct(
            live.key_norms.as_ref().unwrap(),
            copied.key_norms.as_ref().unwrap(),
        );
        assert_distinct(
            live.key_gammas.as_ref().unwrap(),
            copied.key_gammas.as_ref().unwrap(),
        );
        assert_distinct(
            live.value_codes.as_ref().unwrap(),
            copied.value_codes.as_ref().unwrap(),
        );
        assert_distinct(
            live.value_norms.as_ref().unwrap(),
            copied.value_norms.as_ref().unwrap(),
        );
    }

    #[test]
    fn deep_clone_checkpoint_survives_live_in_place_update() {
        let _exec = crate::mlx_exec::acquire();
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

    /// Deterministic standard-normal sample (Box–Muller). TurboQuant's centroids
    /// are tuned for roughly-Gaussian post-rotation coordinates, so the synthetic
    /// KV used to exercise it must be Gaussian — not smooth/correlated — or the
    /// reconstruction quality is artificially low.
    fn gaussian(seed: u64) -> f32 {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32) / ((1u64 << 31) as f32)
        };
        let u1 = next().max(1.0e-7);
        let u2 = next();
        (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Build a `[1, H, T, D]` KV pair with deterministic Gaussian values so the
    /// TurboQuant path is exercised on data it was designed for.
    fn make_varied_kv_pair(h: i32, t: i32, d: i32) -> (Array, Array) {
        let count = usize::try_from(h * t * d).unwrap();
        let k: Vec<f32> = (0..count).map(|i| gaussian(i as u64 + 1)).collect();
        let v: Vec<f32> = (0..count).map(|i| gaussian(i as u64 + 1_000_003)).collect();
        (
            Array::from_slice(&k, &[1, h, t, d]),
            Array::from_slice(&v, &[1, h, t, d]),
        )
    }

    /// Logical dense KV footprint in bytes for `[H, T, D]`, fp16 (2 B/value),
    /// keys + values. This is the between-turn footprint a dense retained cache
    /// would pin (production KV is fp16).
    fn dense_kv_bytes(h: i32, t: i32, d: i32) -> usize {
        let per = usize::try_from(h * t * d).unwrap() * 2; // 2 bytes/value
        per * 2 // keys + values
    }

    /// Packed TurboQuant footprint in bytes for a compressed cache, using the
    /// code-word sizing in this module: codes are u32 words, norms/gammas f32.
    fn turbo_kv_bytes(cache: &SteppingKeyValueCache, h: i32, t: i32) -> usize {
        let (ctx, ..) = cache.turbo_arrays().expect("cache must be TQ-active");
        let ht = usize::try_from(h * t).unwrap();
        let key_code = usize::try_from(ctx.key_code_words).unwrap() * ht * 4;
        let value_code = usize::try_from(ctx.value_code_words).unwrap() * ht * 4;
        // key_norms + key_gammas + value_norms, each [H, T] f32.
        let norms = ht * 4 * 3;
        key_code + value_code + norms
    }

    /// Compressing a dense cache on demand must shrink its between-turn KV
    /// footprint by roughly 4–6× (a few bits/value packed codes vs fp16 dense),
    /// reusing the existing dense→TQ bulk-quantize path. CPU-only, no model load.
    #[test]
    fn quantize_for_retention_shrinks_footprint_4x_to_6x() {
        let (h, t, d) = (4i32, 200i32, 128i32);
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3, // key=2, value=3
            seed: 7,
            ..Default::default()
        };

        // Dense cache (default Off mode) filled by a single prefill.
        let mut cache = SteppingKeyValueCache::new();
        let (keys, values) = make_varied_kv_pair(h, t, d);
        cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(cache.offset(), t);
        assert!(!cache.is_turbo_active(), "starts dense");

        let dense_bytes = dense_kv_bytes(h, t, d);

        // Compress on demand using the existing bulk-quantize machinery.
        let compressed = cache.quantize_for_retention(config).unwrap();
        assert!(compressed, "dense power-of-2 cache must compress");
        assert!(cache.is_turbo_active(), "now TQ-active");
        assert!(cache.keys().is_none(), "dense buffers dropped");
        assert_eq!(cache.offset(), t, "offset preserved across compression");

        let turbo_bytes = turbo_kv_bytes(&cache, h, t);
        let ratio = dense_bytes as f32 / turbo_bytes as f32;
        assert!(
            (4.0..=6.0).contains(&ratio),
            "compression ratio {ratio:.2}x out of [4, 6] (dense={dense_bytes} B, turbo={turbo_bytes} B)"
        );
    }

    /// Compressing a dense cache then reconstructing it must preserve the KV to
    /// TurboQuant tolerance, AND continuation (appending the next turn's token)
    /// must keep working on the now-TQ cache. The on-demand compression must be
    /// bit-identical to a cache that activated TurboQuant naturally during decode
    /// (same context/seed), so we check the reconstruction against that reference.
    #[test]
    fn quantize_for_retention_round_trips_and_continues() {
        let (h, t, d) = (2i32, 40i32, 128i32);
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 7,
            ..Default::default()
        };
        let (keys, values) = make_varied_kv_pair(h, t, d);

        // --- ON-DEMAND: dense cache, then compress for retention. ---
        let mut on_demand = SteppingKeyValueCache::new();
        on_demand
            .update_and_fetch(keys.clone(), values.clone())
            .unwrap();
        assert!(on_demand.quantize_for_retention(config).unwrap());

        // Reconstruct the stored span via the TQ dense materialization. The
        // stored code/norm arrays are slices of an over-allocated `[H, capacity,
        // …]` buffer; `materialize_dense` indexes a packed `[H, seq_len, …]`
        // layout, so force each slice contiguous (flatten+reshape) first — the
        // same contiguity discipline `TurboQuantStorage::append` uses.
        let (ctx, kc, kn, kg, vc, vn) = on_demand.turbo_arrays().expect("TQ-active");
        let kw = ctx.key_code_words;
        let vw = ctx.value_code_words;
        let contig3 = |a: &Array, words: i32| -> Array {
            slice_axis1(a, 0, t)
                .unwrap()
                .flatten(None, None)
                .unwrap()
                .reshape(&[h, t, words])
                .unwrap()
        };
        let contig2 = |a: &Array| -> Array {
            slice_axis1(a, 0, t)
                .unwrap()
                .flatten(None, None)
                .unwrap()
                .reshape(&[h, t])
                .unwrap()
        };
        let view = TurboQuantKvView {
            context: Arc::clone(ctx),
            key_codes: contig3(kc, kw),
            key_norms: contig2(kn),
            key_gammas: contig2(kg),
            value_codes: contig3(vc, vw),
            value_norms: contig2(vn),
            seq_len: t,
        };
        let (rk, rv) = view.materialize_dense().unwrap();
        let od_k = rk.flatten(None, None).unwrap();
        let od_v = rv.flatten(None, None).unwrap();
        let od_k = od_k.as_slice::<f32>();
        let od_v = od_v.as_slice::<f32>();

        // Reconstruction must track the ORIGINAL input within TQ tolerance.
        // Compare per-vector cosine similarity (norm-correction makes magnitudes
        // faithful but discrete codes perturb direction); values use 3 bits, keys 2.
        let orig_k = keys.flatten(None, None).unwrap();
        let orig_v = values.flatten(None, None).unwrap();
        let orig_k = orig_k.as_slice::<f32>();
        let orig_v = orig_v.as_slice::<f32>();
        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na < f32::EPSILON || nb < f32::EPSILON {
                0.0
            } else {
                dot / (na * nb)
            }
        };
        let dim = usize::try_from(d).unwrap();
        let vec_count = usize::try_from(h * t).unwrap();
        let mut sum_k = 0.0f32;
        let mut sum_v = 0.0f32;
        for i in 0..vec_count {
            let s = i * dim;
            let e = s + dim;
            sum_k += cos(&orig_k[s..e], &od_k[s..e]);
            sum_v += cos(&orig_v[s..e], &od_v[s..e]);
        }
        let avg_k = sum_k / vec_count as f32;
        let avg_v = sum_v / vec_count as f32;
        // 2-bit keys and 3-bit values: the same per-bit-width floors the existing
        // TQ roundtrip tests (`test_key/value_roundtrip_cosine_similarity`) use.
        assert!(avg_k > 0.50, "key reconstruction cos {avg_k:.4} too low");
        assert!(avg_v > 0.90, "value reconstruction cos {avg_v:.4} too low");

        // --- CONTINUATION: appending the next turn's token must work on the
        // now-TQ cache and keep it TQ (the production multi-turn continuation). ---
        let (nk, nv) = make_varied_kv_pair(h, 1, d);
        let cont_view = on_demand.update_and_view(nk, nv).unwrap();
        assert!(
            cont_view.turboquant().is_some(),
            "continuation must append via the TQ path"
        );
        assert_eq!(on_demand.offset(), t + 1, "continuation advances offset");
    }

    /// A non-power-of-2 `head_dim` can't be FWHT-packed, so retention compression
    /// must leave the cache dense (correctness over footprint) and still continue.
    #[test]
    fn quantize_for_retention_leaves_non_pow2_head_dim_dense() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            ..Default::default()
        };
        let mut cache = SteppingKeyValueCache::new();
        // head_dim = 6 (not a power of 2).
        let (keys, values) = make_kv_pair(4, 6);
        cache.update_and_fetch(keys, values).unwrap();
        let compressed = cache.quantize_for_retention(config).unwrap();
        assert!(!compressed, "non-pow2 head_dim must not compress");
        assert!(!cache.is_turbo_active());
        assert!(cache.keys().is_some(), "dense buffers retained");
        // Continuation still works (dense append).
        let (k1, v1) = make_kv_pair(1, 6);
        cache.update_and_fetch(k1, v1).unwrap();
        assert_eq!(cache.offset(), 5);
    }

    /// Idempotent / no-op cases: empty cache and already-TQ-active cache.
    #[test]
    fn quantize_for_retention_noops_when_nothing_to_compress() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            ..Default::default()
        };
        // Empty dense cache → nothing to compress.
        let mut empty = SteppingKeyValueCache::new();
        assert!(!empty.quantize_for_retention(config).unwrap());

        // Already TQ-active cache → no-op (second call returns false).
        let mut cache = SteppingKeyValueCache::new();
        let (keys, values) = make_varied_kv_pair(2, 16, 128);
        cache.update_and_fetch(keys, values).unwrap();
        assert!(cache.quantize_for_retention(config).unwrap());
        assert!(
            !cache.quantize_for_retention(config).unwrap(),
            "second compression is a no-op"
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
}
