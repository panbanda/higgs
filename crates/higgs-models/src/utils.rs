use mlx_rs::{
    Array, arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    nn,
    ops::indexing::{IndexOp, NewAxis},
};

use crate::cache::KeyValueCache;

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
    use crate::cache::SteppingKeyValueCache;

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
}
