//! Cached self-attention shared by the dense decoder architectures.
//!
//! Every architecture projects, splits heads, optionally normalizes queries
//! and keys, rotates them at the cache position, appends to the KV cache,
//! attends and merges the heads again. What differs between architectures is
//! the projection layout (separate or fused), the normalization, the `RoPE`
//! parameters, the scale and, for Gemma 2, how scores are computed. Those are
//! the arguments of the functions below; the order of operations is fixed.
//!
//! Positions come from the cache ([`apply_rope_at_cache`]) and masks may be
//! any boolean array that broadcasts against the scores, so the same code
//! runs batched over a [`crate::batched_cache::BatchedLayerCache`].

use mlx_rs::{
    Array, Dtype, array,
    error::Exception,
    fast::{self, ScaledDotProductAttentionMask},
    module::Module,
    nn, ops,
};

use crate::{
    cache::{KeyValueCache, KvCacheView},
    utils::{apply_rope, apply_rope_at_cache, cached_scaled_dot_product_attention},
};

/// Query, key and value projections of one layer, each `[B, L, heads * head_dim]`.
pub struct Projections {
    pub queries: Array,
    pub keys: Array,
    pub values: Array,
}

/// Number of query heads and of key/value heads (fewer when grouped).
#[derive(Debug, Clone, Copy)]
pub struct HeadCounts {
    pub query: i32,
    pub key_value: i32,
}

/// Queries, keys and values as `[B, heads, L, head_dim]`.
pub struct Heads {
    pub queries: Array,
    pub keys: Array,
    pub values: Array,
}

/// Per-head normalizations applied over `head_dim` before the heads are
/// moved ahead of the positions.
#[derive(Default)]
pub struct HeadNorms<'a> {
    pub query: Option<&'a mut nn::RmsNorm>,
    pub key: Option<&'a mut nn::RmsNorm>,
    pub value: Option<&'a ScaleFreeRmsNorm>,
}

/// `RMSNorm` without a learnable scale (Gemma 4's value norm, matching
/// mlx-lm's `RMSNormNoScale`).
pub struct ScaleFreeRmsNorm {
    epsilon: f32,
}

impl ScaleFreeRmsNorm {
    pub const fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let mean_square = x.square()?.mean_axes(&[-1], true)?;
        let inverse_rms = mean_square.add(array!(self.epsilon))?.rsqrt()?;
        x.multiply(inverse_rms)
    }
}

/// Where a layer's attention comes from: how its heads are laid out, which
/// `RoPE` rotates them and how scores are scaled.
pub struct SelfAttention<'a> {
    pub head_counts: HeadCounts,
    pub scale: f32,
    pub rope: &'a nn::Rope,
    pub norms: HeadNorms<'a>,
    pub decode: DecodeKernel,
}

/// How single-token decode reads a `TurboQuant` cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeKernel {
    /// Score against the quantized keys directly when the cache offers it.
    CacheNative,
    /// Always materialize dense keys and values first.
    Dense,
}

impl SelfAttention<'_> {
    /// Attend over the cache (or over the input alone without one) and
    /// return `[B, L, heads * head_dim]`, ready for the output projection.
    pub fn forward<C: KeyValueCache>(
        self,
        projections: &Projections,
        mask: Option<ScaledDotProductAttentionMask<'_>>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let Self {
            head_counts,
            scale,
            rope,
            norms,
            decode,
        } = self;
        let heads = split_heads(projections, head_counts, norms)?;
        let attended = attend(heads, rope, scale, mask, decode, cache)?;
        merge_heads(&attended)
    }
}

/// Reshape projections to `[B, L, heads, head_dim]`, normalize them as
/// `norms` says, then move heads before positions.
pub fn split_heads(
    projections: &Projections,
    head_counts: HeadCounts,
    norms: HeadNorms<'_>,
) -> Result<Heads, Exception> {
    let shape = projections.queries.shape();
    let (Some(&batch), Some(&length)) = (shape.first(), shape.get(1)) else {
        return Err(Exception::custom("Input must have >= 2 dims"));
    };

    let queries = split_queries(&projections.queries, head_counts.query, norms.query)?;
    let mut keys = projections
        .keys
        .reshape(&[batch, length, head_counts.key_value, -1])?;
    let mut values = projections
        .values
        .reshape(&[batch, length, head_counts.key_value, -1])?;
    if let Some(norm) = norms.key {
        keys = norm.forward(&keys)?;
    }
    if let Some(norm) = norms.value {
        values = norm.forward(&values)?;
    }
    Ok(Heads {
        queries,
        keys: keys.transpose_axes(&[0, 2, 1, 3])?,
        values: values.transpose_axes(&[0, 2, 1, 3])?,
    })
}

/// Queries alone, as [`split_heads`] makes them: `[B, heads, L, head_dim]`.
pub fn split_queries(
    projected: &Array,
    heads: i32,
    norm: Option<&mut nn::RmsNorm>,
) -> Result<Array, Exception> {
    let shape = projected.shape();
    let (Some(&batch), Some(&length)) = (shape.first(), shape.get(1)) else {
        return Err(Exception::custom("Input must have >= 2 dims"));
    };
    let mut queries = projected.reshape(&[batch, length, heads, -1])?;
    if let Some(rms_norm) = norm {
        queries = rms_norm.forward(&queries)?;
    }
    queries.transpose_axes(&[0, 2, 1, 3])
}

/// `[B, heads, L, head_dim]` back to `[B, L, heads * head_dim]`.
pub fn merge_heads(attended: &Array) -> Result<Array, Exception> {
    let shape = attended.shape();
    let (Some(&batch), Some(&length)) = (shape.first(), shape.get(2)) else {
        return Err(Exception::custom("attention output must be 4D"));
    };
    attended
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, length, -1])
}

/// Rotate queries and keys at the cache position (position 0 without a cache).
pub fn rotate<C: KeyValueCache>(
    queries: &Array,
    keys: &Array,
    rope: &nn::Rope,
    cache: Option<&C>,
) -> Result<(Array, Array), Exception> {
    match cache {
        Some(kv_cache) => Ok((
            apply_rope_at_cache(queries, rope, kv_cache)?,
            apply_rope_at_cache(keys, rope, kv_cache)?,
        )),
        None => Ok((apply_rope(queries, rope, 0)?, apply_rope(keys, rope, 0)?)),
    }
}

/// Rotate, append to the cache and run fused scaled-dot-product attention.
/// Returns `[B, heads, L, head_dim]`.
///
/// A causal mask, or [`DecodeKernel::Dense`], skips the `TurboQuant` decode
/// path: dense keys are fetched and MLX applies the mask. Otherwise an array
/// mask goes through [`cached_scaled_dot_product_attention`].
pub fn attend<C: KeyValueCache>(
    heads: Heads,
    rope: &nn::Rope,
    scale: f32,
    mask: Option<ScaledDotProductAttentionMask<'_>>,
    decode: DecodeKernel,
    cache: Option<&mut C>,
) -> Result<Array, Exception> {
    let (queries, keys) = rotate(&heads.queries, &heads.keys, rope, cache.as_deref())?;
    let values = heads.values;
    let Some(kv_cache) = cache else {
        return fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            mask,
            None::<&Array>,
        );
    };
    let fetch_dense = decode == DecodeKernel::Dense
        || matches!(mask, Some(ScaledDotProductAttentionMask::Causal));
    if fetch_dense {
        let (cached_keys, cached_values) = kv_cache.update_and_fetch(keys, values)?;
        return fast::scaled_dot_product_attention(
            queries,
            cached_keys,
            cached_values,
            scale,
            mask,
            None::<&Array>,
        );
    }
    let array_mask = match mask {
        Some(ScaledDotProductAttentionMask::Array(array_mask)) => Some(array_mask),
        Some(ScaledDotProductAttentionMask::Causal) | None => None,
    };
    cached_scaled_dot_product_attention(queries, kv_cache, keys, values, scale, array_mask)
}

/// Attention computed from explicit scores instead of the fused kernel, for
/// architectures that transform the scores before the softmax.
///
/// Scores are `[B, key_value_heads, group_size, L, S]`: queries are grouped
/// by the key/value head they share, so keys and values broadcast over the
/// group without being copied. A 4D mask (`[B, 1, 1, S]` or `[B, 1, L, S]`,
/// as batched decode and prefill supply) gains a unit axis to line up with
/// the group axis; 2D masks broadcast unchanged.
///
/// Compared with the fused kernel, this one applies, in this order: the scale,
/// logit soft-capping (`tanh(score / cap) * cap`), a sliding-window lower
/// bound, then the mask.
#[derive(Debug, Clone)]
pub struct GroupedScoreAttention {
    scale: f32,
    softcap: Option<f32>,
    sliding_window: Option<i32>,
    constants: ScoreConstants,
}

/// Scalars cast to the scores' dtype once, so half-precision scores are not
/// promoted to f32 and no cast is repeated per token.
#[derive(Debug, Clone, Default)]
struct ScoreConstants {
    scale: Option<Array>,
    inverse_cap: Option<Array>,
    cap: Option<Array>,
    negative_infinity: Option<Array>,
}

fn scalar_as(slot: &mut Option<Array>, value: f32, dtype: Dtype) -> Result<&Array, Exception> {
    if slot.as_ref().is_none_or(|cached| cached.dtype() != dtype) {
        *slot = Some(array!(value).as_dtype(dtype)?);
    }
    slot.as_ref()
        .ok_or_else(|| Exception::custom("scalar cache not initialized"))
}

impl GroupedScoreAttention {
    pub fn new(scale: f32, softcap: Option<f32>, sliding_window: Option<i32>) -> Self {
        Self {
            scale,
            softcap,
            sliding_window,
            constants: ScoreConstants::default(),
        }
    }

    /// Attend over the cache (or the input alone without one). Queries and
    /// keys are rotated here. Returns `[B, L, heads * head_dim]`.
    pub fn forward<C: KeyValueCache>(
        &mut self,
        heads: Heads,
        head_counts: HeadCounts,
        rope: &nn::Rope,
        mask: Option<&Array>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let (queries, keys) = rotate(&heads.queries, &heads.keys, rope, cache.as_deref())?;
        let values = heads.values;
        let cache_view = match cache {
            Some(kv_cache) => kv_cache.update_and_view(keys, values)?,
            None => KvCacheView::Dense { keys, values },
        };
        match cache_view {
            KvCacheView::TurboQuant(quantized) if is_single_sequence_decode(&queries) => {
                let scores = quantized.decode_scores(&queries, head_counts.query)?;
                let weights = self.softmax_weights(&scores, queries_length(&queries)?, mask)?;
                merge_heads(&quantized.decode_values(&weights, head_counts.query)?)
            }
            other @ (KvCacheView::Dense { .. } | KvCacheView::TurboQuant(_)) => {
                let (cached_keys, cached_values) = other.into_dense()?;
                self.dense_forward(&queries, &cached_keys, &cached_values, head_counts, mask)
            }
        }
    }

    fn dense_forward(
        &mut self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        head_counts: HeadCounts,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = queries.shape();
        let (Some(&batch), Some(&length), Some(&head_dim)) =
            (shape.first(), shape.get(2), shape.get(3))
        else {
            return Err(Exception::custom("queries must be 4D"));
        };
        let key_length = *keys
            .shape()
            .get(2)
            .ok_or_else(|| Exception::custom("keys must be 4D"))?;
        let group_size = head_counts.query / head_counts.key_value;
        let grouped_queries =
            queries.reshape(&[batch, head_counts.key_value, group_size, length, head_dim])?;
        let grouped_keys =
            keys.reshape(&[batch, head_counts.key_value, 1, key_length, head_dim])?;
        let grouped_values =
            values.reshape(&[batch, head_counts.key_value, 1, key_length, head_dim])?;

        let scale = scalar_as(
            &mut self.constants.scale,
            self.scale,
            grouped_queries.dtype(),
        )?;
        let scores = grouped_queries
            .matmul(&grouped_keys.transpose_axes(&[0, 1, 2, 4, 3])?)?
            .multiply(scale)?;
        let weights = self.finish_scores(&scores, length, mask, true)?;
        let attended = weights.matmul(&grouped_values)?.reshape(&[
            batch,
            head_counts.query,
            length,
            head_dim,
        ])?;
        merge_heads(&attended)
    }

    fn softmax_weights(
        &mut self,
        scores: &Array,
        length: i32,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let scale = scalar_as(&mut self.constants.scale, self.scale, scores.dtype())?;
        let scaled = scores.multiply(scale)?;
        self.finish_scores(&scaled, length, mask, false)
    }

    /// Soft-cap, window, mask and softmax of already scaled scores.
    fn finish_scores(
        &mut self,
        scaled: &Array,
        length: i32,
        mask: Option<&Array>,
        grouped: bool,
    ) -> Result<Array, Exception> {
        let mut scores = scaled.clone();
        if let Some(cap) = self.softcap {
            let inverse_cap =
                scalar_as(&mut self.constants.inverse_cap, 1.0 / cap, scores.dtype())?.clone();
            let cap_array = scalar_as(&mut self.constants.cap, cap, scores.dtype())?;
            scores = ops::tanh(&scores.multiply(&inverse_cap)?)?.multiply(cap_array)?;
        }

        if let Some(window) = self.sliding_window {
            let key_length = *scores
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("scores must have >= 1 dim"))?;
            if key_length > window {
                let window_mask = sliding_window_mask(length, key_length, window)?;
                let negative_infinity = scalar_as(
                    &mut self.constants.negative_infinity,
                    f32::NEG_INFINITY,
                    scores.dtype(),
                )?;
                scores = ops::select(&window_mask, &scores, negative_infinity)?;
            }
        }

        if let Some(visible) = mask {
            let negative_infinity = scalar_as(
                &mut self.constants.negative_infinity,
                f32::NEG_INFINITY,
                scores.dtype(),
            )?;
            scores = if grouped && visible.ndim() == 4 {
                ops::select(&visible.expand_dims(1)?, &scores, negative_infinity)?
            } else {
                ops::select(visible, &scores, negative_infinity)?
            };
        }
        ops::softmax_axis(&scores, -1, None)
    }
}

fn is_single_sequence_decode(queries: &Array) -> bool {
    matches!(queries.shape(), [1, _, 1, _])
}

fn queries_length(queries: &Array) -> Result<i32, Exception> {
    queries
        .shape()
        .get(2)
        .copied()
        .ok_or_else(|| Exception::custom("queries must be 4D"))
}

/// Boolean `[L, S]` mask for sliding-window attention: query `i` (the last
/// `L` of `S` positions) sees keys no further back than `window - 1` before it.
/// The causal upper bound is the caller's mask.
pub fn sliding_window_mask(length: i32, key_length: i32, window: i32) -> Result<Array, Exception> {
    let offset = key_length - length;
    let query_positions = ops::arange::<_, f32>(offset, offset + length, None)?;
    let key_positions = ops::arange::<_, f32>(None, key_length, None)?;

    let lower_bounds = query_positions.subtract(array!(window - 1))?;
    let lower_expanded = lower_bounds.reshape(&[length, 1])?;
    let key_expanded = key_positions.reshape(&[1, key_length])?;
    key_expanded.ge(&lower_expanded)
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use mlx_rs::{builder::Builder, ops::indexing::IndexOp, random};

    use super::*;
    use crate::cache::SteppingKeyValueCache;

    fn random_array(shape: &[i32]) -> Array {
        random::normal::<f32>(shape, None, None, None).unwrap()
    }

    fn largest_difference(left: &Array, right: &Array) -> f32 {
        let difference = left.subtract(right).unwrap().abs().unwrap();
        ops::max(&difference, None).unwrap().item_cast::<f32>()
    }

    #[test]
    fn sliding_window_mask_shape() {
        let mask = sliding_window_mask(4, 10, 3).unwrap();
        assert_eq!(mask.shape(), &[4, 10]);
    }

    #[test]
    fn sliding_window_mask_values() {
        // L=3, S=5, window=2: query positions 2, 3, 4 see keys at or after
        // `position - window + 1`. The causal upper bound is the caller's mask.
        let mask = sliding_window_mask(3, 5, 2).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [
            false, true, true, true, true, false, false, true, true, true, false, false, false,
            true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn split_heads_moves_heads_before_positions() {
        let heads = split_heads(
            &Projections {
                queries: random_array(&[2, 3, 4 * 8]),
                keys: random_array(&[2, 3, 2 * 8]),
                values: random_array(&[2, 3, 2 * 8]),
            },
            HeadCounts {
                query: 4,
                key_value: 2,
            },
            HeadNorms::default(),
        )
        .unwrap();
        assert_eq!(heads.queries.shape(), &[2, 4, 3, 8]);
        assert_eq!(heads.keys.shape(), &[2, 2, 3, 8]);
        assert_eq!(heads.values.shape(), &[2, 2, 3, 8]);
        assert_eq!(merge_heads(&heads.queries).unwrap().shape(), &[2, 3, 32]);
    }

    #[test]
    fn split_heads_normalizes_queries_and_keys_over_head_dimension() {
        let mut norm = nn::RmsNormBuilder::new(8).eps(1e-6).build().unwrap();
        let projections = || Projections {
            queries: random_array(&[1, 2, 2 * 8])
                .multiply(array!(5.0_f32))
                .unwrap(),
            keys: random_array(&[1, 2, 8]).multiply(array!(5.0_f32)).unwrap(),
            values: random_array(&[1, 2, 8]),
        };
        let counts = HeadCounts {
            query: 2,
            key_value: 1,
        };
        let plain = split_heads(&projections(), counts, HeadNorms::default()).unwrap();
        let mut query_norm = norm.clone();
        let normed = split_heads(
            &projections(),
            counts,
            HeadNorms {
                query: Some(&mut query_norm),
                key: Some(&mut norm),
                value: None,
            },
        )
        .unwrap();
        let mean_square = |array: &Array| {
            array
                .square()
                .unwrap()
                .mean_axes(&[-1], false)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        assert!(mean_square(&plain.queries).iter().any(|value| *value > 2.0));
        for value in mean_square(&normed.queries) {
            assert!((value - 1.0).abs() < 0.05, "mean square {value}");
        }
    }

    /// A batch whose second row has two left-padding slots must give each
    /// row what it gets alone, for the grouped `[B, kv, group, L, S]` scores.
    #[test]
    fn grouped_scores_honor_batched_padding_mask() {
        let counts = HeadCounts {
            query: 4,
            key_value: 2,
        };
        let queries = random_array(&[2, 4, 1, 8]);
        let keys = random_array(&[2, 2, 5, 8]);
        let values = random_array(&[2, 2, 5, 8]);
        let mut attention = GroupedScoreAttention::new(0.35, Some(50.0), None);

        let padding = Array::from_slice(
            &[
                true, true, true, true, true, //
                false, false, true, true, true,
            ],
            &[2, 1, 1, 5],
        );
        let batched = attention
            .dense_forward(&queries, &keys, &values, counts, Some(&padding))
            .unwrap();

        let unpadded_row = attention
            .dense_forward(
                &queries.index(1..2),
                &keys.index((1..2, .., 2.., ..)),
                &values.index((1..2, .., 2.., ..)),
                counts,
                None,
            )
            .unwrap();
        let full_row = attention
            .dense_forward(
                &queries.index(0..1),
                &keys.index(0..1),
                &values.index(0..1),
                counts,
                None,
            )
            .unwrap();

        assert!(largest_difference(&batched.index(1..2), &unpadded_row) < 1e-5);
        assert!(largest_difference(&batched.index(0..1), &full_row) < 1e-5);
    }

    #[test]
    fn grouped_scores_honor_batched_prefill_mask() {
        let counts = HeadCounts {
            query: 4,
            key_value: 2,
        };
        let queries = random_array(&[1, 4, 3, 8]);
        let keys = random_array(&[1, 2, 3, 8]);
        let values = random_array(&[1, 2, 3, 8]);
        let causal: Vec<bool> = (0..3)
            .flat_map(|row| (0..3).map(move |column| column <= row))
            .collect();
        let two_dimensional = Array::from_slice(&causal, &[3, 3]);
        let four_dimensional = Array::from_slice(&causal, &[1, 1, 3, 3]);

        let mut attention = GroupedScoreAttention::new(0.35, Some(50.0), Some(2));
        let from_two = attention
            .dense_forward(&queries, &keys, &values, counts, Some(&two_dimensional))
            .unwrap();
        let from_four = attention
            .dense_forward(&queries, &keys, &values, counts, Some(&four_dimensional))
            .unwrap();
        assert_eq!(from_two.shape(), &[1, 3, 32]);
        assert!(largest_difference(&from_two, &from_four) < 1e-6);
    }

    #[test]
    fn soft_capping_bounds_scores() {
        let counts = HeadCounts {
            query: 1,
            key_value: 1,
        };
        let queries = random_array(&[1, 1, 1, 8])
            .multiply(array!(100.0_f32))
            .unwrap();
        let keys = random_array(&[1, 1, 4, 8])
            .multiply(array!(100.0_f32))
            .unwrap();
        let values = random_array(&[1, 1, 4, 8]);
        let capped = GroupedScoreAttention::new(1.0, Some(1.0), None)
            .dense_forward(&queries, &keys, &values, counts, None)
            .unwrap();
        let uncapped = GroupedScoreAttention::new(1.0, None, None)
            .dense_forward(&queries, &keys, &values, counts, None)
            .unwrap();
        assert!(largest_difference(&capped, &uncapped) > 1e-3);
    }

    #[test]
    fn grouped_scores_match_fused_attention_without_transforms() {
        let counts = HeadCounts {
            query: 4,
            key_value: 2,
        };
        let queries = random_array(&[1, 4, 3, 8]);
        let keys = random_array(&[1, 2, 3, 8]);
        let values = random_array(&[1, 2, 3, 8]);
        let grouped = GroupedScoreAttention::new(0.35, None, None)
            .dense_forward(&queries, &keys, &values, counts, None)
            .unwrap();
        let fused =
            fast::scaled_dot_product_attention(queries, keys, values, 0.35, None, None::<&Array>)
                .unwrap();
        assert!(largest_difference(&grouped, &merge_heads(&fused).unwrap()) < 1e-5);
    }

    #[test]
    fn causal_mode_matches_explicit_causal_array_with_cache() {
        let rope = nn::RopeBuilder::new(8).build().unwrap();
        let heads = || Heads {
            queries: random_array(&[1, 2, 4, 8]),
            keys: random_array(&[1, 2, 4, 8]),
            values: random_array(&[1, 2, 4, 8]),
        };
        let cached = heads();
        let rerun = Heads {
            queries: cached.queries.clone(),
            keys: cached.keys.clone(),
            values: cached.values.clone(),
        };
        let explicit: Vec<bool> = (0..4)
            .flat_map(|row| (0..4).map(move |column| column <= row))
            .collect();
        let explicit_mask = Array::from_slice(&explicit, &[4, 4]);

        let mut causal_cache = SteppingKeyValueCache::new();
        let causal = attend(
            cached,
            &rope,
            0.35,
            Some(ScaledDotProductAttentionMask::Causal),
            DecodeKernel::CacheNative,
            Some(&mut causal_cache),
        )
        .unwrap();
        let mut array_cache = SteppingKeyValueCache::new();
        let from_array = attend(
            rerun,
            &rope,
            0.35,
            Some(ScaledDotProductAttentionMask::Array(&explicit_mask)),
            DecodeKernel::Dense,
            Some(&mut array_cache),
        )
        .unwrap();
        assert!(largest_difference(&causal, &from_array) < 1e-5);
        assert_eq!(causal_cache.offset(), 4);
    }
}
