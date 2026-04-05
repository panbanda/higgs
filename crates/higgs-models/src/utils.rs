use mlx_rs::{
    Array, arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    nn, ops,
    ops::indexing::{IndexOp, NewAxis},
};

use crate::cache::{KeyValueCache, KvCacheView};

/// Apply `RoPE` directly without the 3D reshape in `nn::Rope::forward`.
///
/// The mlx-rs `Rope` wrapper reshapes to 3D before calling `mlx_fast_rope`,
/// which triggers a bug in MLX where batch elements beyond the first are
/// zeroed when `seq_len=1`. Calling `mlx_fast_rope` on the original shape avoids this.
pub(crate) fn apply_rope(x: &Array, rope: &nn::Rope, offset: i32) -> Result<Array, Exception> {
    mlx_rs::fast::rope(
        x,
        rope.dimensions,
        rope.traditional,
        rope.base,
        rope.scale,
        offset,
        None,
    )
}

/// Attention mask variant.
#[derive(Debug, Clone)]
pub(crate) enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

/// Non-quantized scaled dot product attention using MLX fast path.
pub(crate) fn scaled_dot_product_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array, Exception> {
    mlx_rs::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
        None::<&Array>,
    )
}

/// Append K/V to the cache and run attention, using the TurboQuant decode path
/// for single-token decode when the cache view exposes quantized storage.
pub(crate) fn cached_scaled_dot_product_attention<C>(
    queries: Array,
    kv_cache: &mut C,
    keys: Array,
    values: Array,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    match kv_cache.update_and_view(keys, values)? {
        KvCacheView::TurboQuant(view)
            if mask.is_none() && is_single_token_decode(&queries, view.context.head_dim) =>
        {
            let num_heads = queries
                .shape()
                .get(1)
                .copied()
                .ok_or_else(|| Exception::custom("queries must be 4D"))?;
            let scores = view.decode_scores(&queries, num_heads)?;
            let scale_arr = Array::from_f32(scale).as_dtype(scores.dtype())?;
            let scaled_scores = scores.multiply(&scale_arr)?;
            let weights = ops::softmax_axis(&scaled_scores, -1, true)?;
            view.decode_values(&weights, num_heads)
        }
        view => {
            // Multi-token prefill (and any other non-decode path) materializes
            // dense KV and uses native SDPA — single batched GPU op for all T.
            let (keys, values) = view.into_dense()?;
            scaled_dot_product_attention(queries, keys, values, scale, mask)
        }
    }
}

fn is_single_token_decode(queries: &Array, head_dim: i32) -> bool {
    let shape = queries.shape();
    shape.len() == 4 && shape[0] == 1 && shape[2] == 1 && shape[3] == head_dim
}

/// Create a causal attention mask.
#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(N: i32, raw_offset: Option<i32>) -> Result<Array, Exception> {
    let offset = raw_offset.unwrap_or(0);

    let row_indices = arange!(stop = offset + N)?;
    let col_indices = arange!(start = offset, stop = offset + N)?;
    let col_expanded = col_indices.index((.., NewAxis));
    let row_expanded = row_indices.index(NewAxis);

    col_expanded.ge(&row_expanded)
}

/// Create an attention mask from the hidden state and cache.
#[allow(non_snake_case)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    as_array: Option<bool>,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let use_array = as_array.unwrap_or(false);
    let shape = h.shape();
    let T = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("Hidden state must have at least 2 dimensions"))?;

    if T > 1 {
        let offset = cache
            .first()
            .and_then(|c| c.as_ref())
            .map_or(0, KeyValueCache::offset);

        if use_array {
            create_causal_mask(T, Some(offset))
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}

/// Create a boolean attention mask for batched decode.
///
/// Each request attends only to its own valid KV positions (not padding).
/// Returns shape `[N, 1, 1, max_kv_len]` where `mask[i, 0, 0, j]` is true when
/// `j < kv_lengths[i]`.
pub(crate) fn create_batched_decode_mask(
    kv_lengths: &[i32],
    max_kv_len: i32,
) -> Result<Array, Exception> {
    let n = i32::try_from(kv_lengths.len())
        .map_err(|_| Exception::custom("too many requests for batched mask"))?;
    let lengths = Array::from_slice(kv_lengths, &[n]).reshape(&[n, 1])?;
    let positions = arange!(stop = max_kv_len)?.reshape(&[1, max_kv_len])?;
    let mask = lengths.gt(positions)?;
    // 4D so it broadcasts correctly with SDPA's [N, n_heads, 1, max_kv_len]
    mask.reshape(&[n, 1, 1, max_kv_len])
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::{
        cache::SteppingKeyValueCache,
        turboquant::{KvCacheConfig, KvCacheMode},
    };

    #[test]
    fn test_create_causal_mask_n4() {
        // N=4, no offset: should produce a 4x4 lower-triangular bool mask
        // Row i, col j: mask[i,j] = (j >= i) => upper triangular if comparing col >= row
        // Actually the code does col_expanded.ge(row_expanded) where
        //   col = [offset..offset+N] (the token positions) expanded as column
        //   row = [0..offset+N] expanded as row
        // For offset=0, N=4:
        //   col_indices = [0,1,2,3] -> shape [4,1]
        //   row_indices = [0,1,2,3] -> shape [1,4]
        //   result[i,j] = col[i] >= row[j]
        let mask = create_causal_mask(4, None).unwrap();
        assert_eq!(mask.shape(), &[4, 4]);

        // Evaluate the mask to get concrete values
        let flat: Vec<bool> = mask.as_slice().to_vec();
        // Expected: col[i] >= row[j]
        // i=0: [0>=0, 0>=1, 0>=2, 0>=3] = [T, F, F, F]
        // i=1: [1>=0, 1>=1, 1>=2, 1>=3] = [T, T, F, F]
        // i=2: [2>=0, 2>=1, 2>=2, 2>=3] = [T, T, T, F]
        // i=3: [3>=0, 3>=1, 3>=2, 3>=3] = [T, T, T, T]
        let expected = [
            true, false, false, false, true, true, false, false, true, true, true, false, true,
            true, true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_create_causal_mask_n1() {
        // N=1, no offset: single token, should be [1, 1] with value true
        let mask = create_causal_mask(1, None).unwrap();
        assert_eq!(mask.shape(), &[1, 1]);
        let val: bool = mask.item();
        assert!(val);
    }

    #[test]
    fn test_create_causal_mask_with_offset() {
        // N=2, offset=3: new tokens at positions 3,4; need to attend to all 5 positions
        let mask = create_causal_mask(2, Some(3)).unwrap();
        // col_indices = [3, 4] -> [2, 1]
        // row_indices = [0, 1, 2, 3, 4] -> [1, 5]
        // shape: [2, 5]
        assert_eq!(mask.shape(), &[2, 5]);

        let flat: Vec<bool> = mask.as_slice().to_vec();
        // i=0 (col=3): [3>=0, 3>=1, 3>=2, 3>=3, 3>=4] = [T, T, T, T, F]
        // i=1 (col=4): [4>=0, 4>=1, 4>=2, 4>=3, 4>=4] = [T, T, T, T, T]
        let expected = [true, true, true, true, false, true, true, true, true, true];
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_create_attention_mask_single_token() {
        // T=1: no mask needed
        let h = Array::zeros::<f32>(&[1, 1, 64]).unwrap();
        let cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let result = create_attention_mask(&h, &cache, Some(true)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_create_attention_mask_multi_token_as_array() {
        // T=3, no cache: should produce an array mask
        let h = Array::zeros::<f32>(&[1, 3, 64]).unwrap();
        let cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let result = create_attention_mask(&h, &cache, Some(true)).unwrap();
        assert!(result.is_some());
        match result.unwrap() {
            AttentionMask::Array(a) => assert_eq!(a.shape(), &[3, 3]),
            AttentionMask::Causal => panic!("Expected Array mask"),
        }
    }

    #[test]
    fn test_create_attention_mask_multi_token_as_causal() {
        // T=3, as_array=false: should return Causal variant
        let h = Array::zeros::<f32>(&[1, 3, 64]).unwrap();
        let cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let result = create_attention_mask(&h, &cache, Some(false)).unwrap();
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), AttentionMask::Causal));
    }

    #[test]
    fn test_create_attention_mask_default_is_causal() {
        // as_array=None defaults to false (Causal)
        let h = Array::zeros::<f32>(&[1, 4, 64]).unwrap();
        let cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let result = create_attention_mask(&h, &cache, None).unwrap();
        assert!(matches!(result.unwrap(), AttentionMask::Causal));
    }

    #[test]
    fn test_create_attention_mask_with_cache_offset() {
        // Pre-populate cache so offset > 0
        let h = Array::zeros::<f32>(&[1, 2, 64]).unwrap();
        let mut kv_cache = SteppingKeyValueCache::new();
        let keys = Array::zeros::<f32>(&[1, 2, 5, 8]).unwrap();
        let values = Array::zeros::<f32>(&[1, 2, 5, 8]).unwrap();
        kv_cache.update_and_fetch(keys, values).unwrap();
        assert_eq!(kv_cache.offset(), 5);

        let cache: Vec<Option<SteppingKeyValueCache>> = vec![Some(kv_cache)];
        let result = create_attention_mask(&h, &cache, Some(true)).unwrap();
        match result.unwrap() {
            AttentionMask::Array(a) => {
                // N=2 tokens, offset=5: mask shape [2, 7]
                assert_eq!(a.shape(), &[2, 7]);
            }
            AttentionMask::Causal => panic!("Expected Array mask"),
        }
    }

    #[test]
    fn test_scaled_dot_product_attention_output_shape() {
        // B=1, H=2, L=3, D=4
        let queries = Array::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let keys = Array::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let values = Array::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let scale = (4.0_f32).sqrt().recip();
        let result = scaled_dot_product_attention(queries, keys, values, scale, None).unwrap();
        assert_eq!(result.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_scaled_dot_product_attention_single_head_single_token() {
        // B=1, H=1, L=1, D=2
        let queries = Array::ones::<f32>(&[1, 1, 1, 2]).unwrap();
        let keys = Array::ones::<f32>(&[1, 1, 1, 2]).unwrap();
        let values = Array::from_slice(&[3.0_f32, 7.0], &[1, 1, 1, 2]);
        let scale = (2.0_f32).sqrt().recip();
        let result = scaled_dot_product_attention(queries, keys, values, scale, None).unwrap();
        assert_eq!(result.shape(), &[1, 1, 1, 2]);
        // With single KV pair, softmax(score) = [1.0], so output = values
        let v0: f32 = result.index((.., .., .., 0..1)).item();
        let v1: f32 = result.index((.., .., .., 1..2)).item();
        assert!((v0 - 3.0).abs() < 1e-4);
        assert!((v1 - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_cached_scaled_dot_product_attention_matches_dense_turbo_adapter() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 13,
            ..Default::default()
        };
        let queries = Array::from_slice(
            &[0.1_f32, -0.3, 0.8, 0.2, -0.4, 0.6, 0.9, -0.1],
            &[1, 1, 1, 8],
        );
        let keys = Array::from_slice(
            &[0.3_f32, 0.7, -0.1, 0.4, -0.2, 0.5, 0.8, 0.6],
            &[1, 1, 1, 8],
        );
        let values = Array::from_slice(
            &[1.0_f32, 2.0, 3.5, 4.0, -1.0, -0.5, 0.25, 0.75],
            &[1, 1, 1, 8],
        );
        let scale = (8.0_f32).sqrt().recip();

        let mut fast_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        let fast = cached_scaled_dot_product_attention(
            queries.clone(),
            &mut fast_cache,
            keys.clone(),
            values.clone(),
            scale,
            None,
        )
        .unwrap();

        let mut dense_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        let (dense_keys, dense_values) = dense_cache.update_and_fetch(keys, values).unwrap();
        let dense =
            scaled_dot_product_attention(queries, dense_keys, dense_values, scale, None).unwrap();

        assert_eq!(fast.shape(), dense.shape());
        let fast_vals = fast.as_slice::<f32>();
        let dense_vals = dense.as_slice::<f32>();
        for (lhs, rhs) in fast_vals.iter().zip(dense_vals.iter()) {
            assert!((lhs - rhs).abs() < 1e-4, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn test_cached_scaled_dot_product_attention_matches_dense_turbo_prefill_masked() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 17,
            ..Default::default()
        };
        let queries = Array::from_slice(
            &[
                0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3,
                0.2, -0.1,
            ],
            &[1, 1, 2, 8],
        );
        let keys = Array::from_slice(
            &[
                0.3_f32, 0.1, -0.2, 0.4, -0.5, 0.6, -0.7, 0.8, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3,
                -0.2, 0.1,
            ],
            &[1, 1, 2, 8],
        );
        let values = Array::from_slice(
            &[
                1.0_f32, 0.5, -0.25, 0.75, -1.0, 0.25, -0.5, 0.125, -0.75, 1.25, 0.5, -0.125,
                0.375, -0.625, 0.875, -1.125,
            ],
            &[1, 1, 2, 8],
        );
        let mask = create_causal_mask(2, None).unwrap();
        let scale = (8.0_f32).sqrt().recip();

        let mut fast_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        let fast = cached_scaled_dot_product_attention(
            queries.clone(),
            &mut fast_cache,
            keys.clone(),
            values.clone(),
            scale,
            Some(&mask),
        )
        .unwrap();

        let mut dense_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        let (dense_keys, dense_values) = dense_cache.update_and_fetch(keys, values).unwrap();
        let dense =
            scaled_dot_product_attention(queries, dense_keys, dense_values, scale, Some(&mask))
                .unwrap();

        assert_eq!(fast.shape(), dense.shape());
        let fast_vals = fast.as_slice::<f32>();
        let dense_vals = dense.as_slice::<f32>();
        for (lhs, rhs) in fast_vals.iter().zip(dense_vals.iter()) {
            assert!((lhs - rhs).abs() < 3e-2, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn test_cached_scaled_dot_product_attention_matches_dense_turbo_mask_with_offset() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 19,
            ..Default::default()
        };
        let prefix_keys = Array::from_slice(
            &[
                0.2_f32, 0.1, -0.1, 0.3, -0.3, 0.5, -0.5, 0.7, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8,
                -0.8, 1.0,
            ],
            &[1, 1, 2, 8],
        );
        let prefix_values = Array::from_slice(
            &[
                0.5_f32, -0.5, 0.75, -0.75, 1.0, -1.0, 1.25, -1.25, -0.25, 0.25, -0.375, 0.375,
                -0.5, 0.5, -0.625, 0.625,
            ],
            &[1, 1, 2, 8],
        );
        let queries = Array::from_slice(
            &[
                0.15_f32, -0.25, 0.35, -0.45, 0.55, -0.65, 0.75, -0.85, -0.15, 0.25, -0.35, 0.45,
                -0.55, 0.65, -0.75, 0.85,
            ],
            &[1, 1, 2, 8],
        );
        let keys = Array::from_slice(
            &[
                0.9_f32, -0.7, 0.5, -0.3, 0.1, 0.2, -0.4, 0.6, -0.9, 0.7, -0.5, 0.3, -0.1, -0.2,
                0.4, -0.6,
            ],
            &[1, 1, 2, 8],
        );
        let values = Array::from_slice(
            &[
                1.5_f32, -1.25, 1.0, -0.75, 0.5, -0.25, 0.125, -0.0625, -1.5, 1.25, -1.0, 0.75,
                -0.5, 0.25, -0.125, 0.0625,
            ],
            &[1, 1, 2, 8],
        );
        let mask = create_causal_mask(2, Some(2)).unwrap();
        let scale = (8.0_f32).sqrt().recip();

        let mut fast_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        fast_cache
            .update_and_fetch(prefix_keys.clone(), prefix_values.clone())
            .unwrap();
        let fast = cached_scaled_dot_product_attention(
            queries.clone(),
            &mut fast_cache,
            keys.clone(),
            values.clone(),
            scale,
            Some(&mask),
        )
        .unwrap();

        let mut dense_cache = SteppingKeyValueCache::new_turbo(config, 1, 8).unwrap();
        dense_cache
            .update_and_fetch(prefix_keys, prefix_values)
            .unwrap();
        let (dense_keys, dense_values) = dense_cache.update_and_fetch(keys, values).unwrap();
        let dense =
            scaled_dot_product_attention(queries, dense_keys, dense_values, scale, Some(&mask))
                .unwrap();

        assert_eq!(fast.shape(), dense.shape());
        let fast_vals = fast.as_slice::<f32>();
        let dense_vals = dense.as_slice::<f32>();
        for (lhs, rhs) in fast_vals.iter().zip(dense_vals.iter()) {
            assert!((lhs - rhs).abs() < 4e-2, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn test_attention_mask_conversion_array() {
        let arr = Array::ones::<f32>(&[3, 3]).unwrap();
        let mask = AttentionMask::Array(arr);
        let sdpa_mask: ScaledDotProductAttentionMask = (&mask).into();
        assert!(matches!(sdpa_mask, ScaledDotProductAttentionMask::Array(_)));
    }

    #[test]
    fn test_attention_mask_conversion_causal() {
        let mask = AttentionMask::Causal;
        let sdpa_mask: ScaledDotProductAttentionMask = (&mask).into();
        assert!(matches!(sdpa_mask, ScaledDotProductAttentionMask::Causal));
    }

    // -----------------------------------------------------------------------
    // SDPA Causal enum vs materialized array
    // -----------------------------------------------------------------------

    /// Causal enum with GQA (more Q heads than KV heads) — reproduces
    /// the real model shape (e.g. Qwen3.5-0.8B: 8 Q heads, 2 KV heads).
    #[test]
    fn test_sdpa_causal_enum_gqa() {
        use mlx_rs::fast;

        let t_q: i32 = 5;
        let offset: i32 = 3;
        let t_kv = offset + t_q; // 8
        let n_q_heads: i32 = 8;
        let n_kv_heads: i32 = 2;
        let head_dim: i32 = 16;
        let scale = (head_dim as f32).sqrt().recip();

        let q_data: Vec<f32> = (0..n_q_heads * t_q * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let k_data: Vec<f32> = (0..n_kv_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let v_data: Vec<f32> = (0..n_kv_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.13 + 1.0).sin())
            .collect();

        let queries = Array::from_slice(&q_data, &[1, n_q_heads, t_q, head_dim]);
        let keys = Array::from_slice(&k_data, &[1, n_kv_heads, t_kv, head_dim]);
        let values = Array::from_slice(&v_data, &[1, n_kv_heads, t_kv, head_dim]);

        // Path 1: Materialized causal mask array
        let mask_array = create_causal_mask(t_q, Some(offset)).unwrap();
        let result_array = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask_array)),
            None::<&Array>,
        )
        .unwrap();

        // Path 2: Causal enum
        let result_causal = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None::<&Array>,
        )
        .unwrap();

        assert_eq!(result_array.shape(), result_causal.shape());
        let arr_vals = result_array.as_slice::<f32>();
        let cau_vals = result_causal.as_slice::<f32>();
        for (a, c) in arr_vals.iter().zip(cau_vals.iter()) {
            assert!(
                (a - c).abs() < 1e-4,
                "GQA causal mismatch: array={a}, causal={c} (diff={})",
                (a - c).abs()
            );
        }
    }

    /// Causal enum with GQA + float16 + head_dim=128 — matches real Qwen3.5 shapes.
    #[test]
    fn test_sdpa_causal_enum_gqa_f16_large() {
        use mlx_rs::{Dtype, fast};

        let t_q: i32 = 12; // typical chat prompt length
        let n_q_heads: i32 = 8;
        let n_kv_heads: i32 = 2;
        let head_dim: i32 = 128;
        let t_kv = t_q; // first prefill, no offset
        let scale = (head_dim as f32).sqrt().recip();

        let q_data: Vec<f32> = (0..n_q_heads * t_q * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k_data: Vec<f32> = (0..n_kv_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let v_data: Vec<f32> = (0..n_kv_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.013 + 1.0).sin())
            .collect();

        let queries = Array::from_slice(&q_data, &[1, n_q_heads, t_q, head_dim])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let keys = Array::from_slice(&k_data, &[1, n_kv_heads, t_kv, head_dim])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let values = Array::from_slice(&v_data, &[1, n_kv_heads, t_kv, head_dim])
            .as_dtype(Dtype::Float16)
            .unwrap();

        // This is the exact call path used in real inference
        let result = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None::<&Array>,
        )
        .unwrap();

        assert_eq!(result.shape(), &[1, n_q_heads, t_q, head_dim]);
    }

    #[test]
    fn test_sdpa_causal_enum_matches_array_with_offset() {
        use mlx_rs::fast;

        let t_q: i32 = 3;
        let offset: i32 = 5;
        let t_kv = offset + t_q; // 8
        let n_heads: i32 = 2;
        let head_dim: i32 = 8;
        let scale = (head_dim as f32).sqrt().recip();

        // Varying values so attention isn't degenerate
        let q_data: Vec<f32> = (0..n_heads * t_q * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let kv_data: Vec<f32> = (0..n_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let v_data: Vec<f32> = (0..n_heads * t_kv * head_dim)
            .map(|i| (i as f32 * 0.13 + 1.0).sin())
            .collect();

        let queries = Array::from_slice(&q_data, &[1, n_heads, t_q, head_dim]);
        let keys = Array::from_slice(&kv_data, &[1, n_heads, t_kv, head_dim]);
        let values = Array::from_slice(&v_data, &[1, n_heads, t_kv, head_dim]);

        // Path 1: Materialized causal mask array
        let mask_array = create_causal_mask(t_q, Some(offset)).unwrap();
        let result_array = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask_array)),
            None::<&Array>,
        )
        .unwrap();

        // Path 2: Causal enum (MLX fast path)
        let result_causal = fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None::<&Array>,
        )
        .unwrap();

        assert_eq!(result_array.shape(), result_causal.shape());
        let arr_vals = result_array.as_slice::<f32>();
        let cau_vals = result_causal.as_slice::<f32>();
        for (a, c) in arr_vals.iter().zip(cau_vals.iter()) {
            assert!(
                (a - c).abs() < 1e-4,
                "array={a}, causal={c} (diff={})",
                (a - c).abs()
            );
        }
    }

    #[test]
    fn test_intermediate_eval_preserves_output() {
        // Verify that mid-graph eval() doesn't change the computed result.
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Array::from_slice(&[0.5_f32, -0.5, 1.0, -1.0, 0.25, -0.25], &[3, 2]);
        let bias = Array::from_slice(&[0.1_f32, -0.2], &[1, 2]);

        // Without intermediate eval
        let y1 = ops::matmul(&x, &w).unwrap();
        let z1 = y1.add(&bias).unwrap();
        let r1 = z1.multiply(Array::from_f32(2.0)).unwrap();

        // With intermediate eval after each step
        let y2 = ops::matmul(&x, &w).unwrap();
        mlx_rs::transforms::eval([&y2]).unwrap();
        let z2 = y2.add(&bias).unwrap();
        mlx_rs::transforms::eval([&z2]).unwrap();
        let r2 = z2.multiply(Array::from_f32(2.0)).unwrap();

        let v1 = r1.as_slice::<f32>();
        let v2 = r2.as_slice::<f32>();
        assert_eq!(v1.len(), v2.len());
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6, "no_eval={a}, with_eval={b}");
        }
    }

    // -----------------------------------------------------------------------
    // create_batched_decode_mask
    // -----------------------------------------------------------------------

    #[test]
    fn test_batched_decode_mask_shape() {
        let mask = create_batched_decode_mask(&[5, 3, 7], 7).unwrap();
        assert_eq!(mask.shape(), &[3, 1, 1, 7]);
    }

    #[test]
    fn test_batched_decode_mask_single_request() {
        let mask = create_batched_decode_mask(&[4], 4).unwrap();
        assert_eq!(mask.shape(), &[1, 1, 1, 4]);
        let flat: Vec<bool> = mask.as_slice().to_vec();
        // All 4 positions valid
        assert_eq!(flat, vec![true, true, true, true]);
    }

    #[test]
    fn test_batched_decode_mask_values() {
        // Request 0 has 3 tokens, request 1 has 5 tokens, max_kv_len=5
        let mask = create_batched_decode_mask(&[3, 5], 5).unwrap();
        let flat: Vec<bool> = mask.reshape(&[2, 5]).unwrap().as_slice().to_vec();
        // Request 0: positions 0,1,2 valid (< 3), positions 3,4 invalid
        assert_eq!(&flat[0..5], &[true, true, true, false, false]);
        // Request 1: all 5 positions valid
        assert_eq!(&flat[5..10], &[true, true, true, true, true]);
    }

    #[test]
    fn test_batched_decode_mask_equal_lengths() {
        let mask = create_batched_decode_mask(&[4, 4], 4).unwrap();
        let flat: Vec<bool> = mask.reshape(&[2, 4]).unwrap().as_slice().to_vec();
        // Both requests: all 4 positions valid
        assert!(flat.iter().all(|&v| v));
    }

    #[test]
    fn test_batched_decode_mask_broadcasts_with_sdpa() {
        // Verify shape [N, 1, 1, max_kv_len] broadcasts with [N, n_heads, 1, max_kv_len]
        let mask = create_batched_decode_mask(&[3, 5], 5).unwrap();
        assert_eq!(mask.shape(), &[2, 1, 1, 5]);
        // This shape broadcasts correctly: the 1s expand to n_heads and seq_len
    }

    /// Ground-truth test: TurboQuant attention vs unquantized dense attention.
    /// Uses random KV vectors with realistic dimensions. Measures cosine similarity
    /// between quantized and unquantized attention outputs.
    #[test]
    fn test_turboquant_attention_vs_unquantized_ground_truth() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 42,
            ..Default::default()
        };
        let num_kv_heads = 2;
        let num_q_heads = 8; // GQA: 4 Q heads per KV head
        let head_dim = 64;
        let seq_len = 16;
        let scale = (head_dim as f32).sqrt().recip();

        // Generate random KV with some structure (sin/cos patterns)
        let k_data: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| (i as f32 * 0.037).sin() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| (i as f32 * 0.053 + 1.0).cos() * 0.3)
            .collect();
        let q_data: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.071 + 0.5).sin() * 0.4)
            .collect();

        let keys = Array::from_slice(
            &k_data,
            &[1, num_kv_heads as i32, seq_len as i32, head_dim as i32],
        );
        let values = Array::from_slice(
            &v_data,
            &[1, num_kv_heads as i32, seq_len as i32, head_dim as i32],
        );
        let queries = Array::from_slice(&q_data, &[1, num_q_heads as i32, 1, head_dim as i32]);

        // Path 1: Dense (unquantized) attention — ground truth
        let dense_out = scaled_dot_product_attention(
            queries.clone(),
            keys.clone(),
            values.clone(),
            scale,
            None,
        )
        .unwrap();

        // Path 2: TurboQuant attention via cache
        let mut turbo_cache =
            SteppingKeyValueCache::new_turbo(config, num_kv_heads as i32, head_dim as i32).unwrap();
        let turbo_out = cached_scaled_dot_product_attention(
            queries,
            &mut turbo_cache,
            keys,
            values,
            scale,
            None,
        )
        .unwrap();

        assert_eq!(dense_out.shape(), turbo_out.shape());
        let dense_vals = dense_out.as_slice::<f32>();
        let turbo_vals = turbo_out.as_slice::<f32>();

        // Cosine similarity between full output vectors
        let dot: f64 = dense_vals
            .iter()
            .zip(turbo_vals)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm_d: f64 = dense_vals
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_t: f64 = turbo_vals
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (norm_d * norm_t);

        // 3-bit values + 2-bit keys on head_dim=64 introduces meaningful quantization
        // noise. cos > 0.85 confirms the algorithm is working correctly — higher
        // quality emerges at larger head_dim (128) and averages out across many heads/layers.
        assert!(
            cos > 0.85,
            "TurboQuant 3-bit attention vs dense: cos={cos:.6} (need > 0.85)"
        );
    }

    /// Verify TurboQuant sequential decode quality over 20 autoregressive steps.
    /// Uses cosine similarity per step — more stable than argmax comparison.
    #[test]
    fn test_turboquant_sequential_decode_quality() {
        let config = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 77,
            ..Default::default()
        };
        let head_dim = 64;
        let scale = (head_dim as f32).sqrt().recip();

        let mut turbo_cache = SteppingKeyValueCache::new_turbo(config, 1, head_dim).unwrap();
        let mut dense_cache = SteppingKeyValueCache::new();

        let mut total_cos = 0.0_f64;
        let steps = 20;
        for step in 0..steps {
            let q_data: Vec<f32> = (0..head_dim)
                .map(|i| ((step * head_dim + i) as f32 * 0.11).sin())
                .collect();
            let k_data: Vec<f32> = (0..head_dim)
                .map(|i| ((step * head_dim + i) as f32 * 0.07).cos())
                .collect();
            let v_data: Vec<f32> = (0..head_dim)
                .map(|i| ((step * head_dim + i) as f32 * 0.13 + 1.0).sin())
                .collect();

            let queries = Array::from_slice(&q_data, &[1, 1, 1, head_dim]);
            let keys = Array::from_slice(&k_data, &[1, 1, 1, head_dim]);
            let values = Array::from_slice(&v_data, &[1, 1, 1, head_dim]);

            let turbo_out = cached_scaled_dot_product_attention(
                queries.clone(),
                &mut turbo_cache,
                keys.clone(),
                values.clone(),
                scale,
                None,
            )
            .unwrap();
            let dense_out = cached_scaled_dot_product_attention(
                queries,
                &mut dense_cache,
                keys,
                values,
                scale,
                None,
            )
            .unwrap();

            let t_vals = turbo_out.as_slice::<f32>();
            let d_vals = dense_out.as_slice::<f32>();
            let dot: f64 = t_vals
                .iter()
                .zip(d_vals)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let n_t: f64 = t_vals
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            let n_d: f64 = d_vals
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            if n_t > 1e-12 && n_d > 1e-12 {
                total_cos += dot / (n_t * n_d);
            }
        }
        let avg_cos = total_cos / steps as f64;
        assert!(
            avg_cos > 0.80,
            "sequential 3-bit decode avg cos={avg_cos:.4} (need > 0.80)"
        );
    }
}
