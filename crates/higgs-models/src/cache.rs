use mlx_rs::{Array, Stream, error::Exception, ops, ops::concatenate_axis};

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

    /// Append new key/value tensors and return the full cached key/value.
    fn update_and_fetch(&mut self, keys: Array, values: Array)
    -> Result<(Array, Array), Exception>;
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

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
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

        Ok((result_keys, result_values))
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
    offset: i32,
    step: i32,
}

impl Default for SteppingKeyValueCache {
    fn default() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step: 256,
        }
    }
}

impl SteppingKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Slice an array along axis 2: `arr[..., start:end, ...]`
#[allow(unsafe_code, clippy::indexing_slicing)]
fn slice_axis2(arr: &Array, start: i32, end: i32) -> Result<Array, Exception> {
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

impl KeyValueCache for SteppingKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    #[allow(clippy::indexing_slicing)]
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
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

        Ok((result_k, result_v))
    }
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
}
