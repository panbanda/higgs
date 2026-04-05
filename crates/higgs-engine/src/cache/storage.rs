// SPDX-License-Identifier: Apache-2.0
//! CPU storage for KV cache in f16 format.

use crate::cache::CacheError;
use half::f16;

/// Layout specification for KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheLayout {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCacheLayout {
    /// Calculate elements per token (num_kv_heads * head_dim).
    pub fn elems_per_token(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Calculate total elements in cache.
    pub fn total_elems(&self) -> usize {
        self.num_blocks * self.block_size * self.elems_per_token()
    }
}

/// CPU storage for KV cache in f16 format.
#[derive(Debug)]
pub struct CpuKvStorage {
    k: Vec<f16>,
    v: Vec<f16>,
    layout: KvCacheLayout,
}

impl CpuKvStorage {
    /// Create new KV storage with specified capacity.
    pub fn new(num_blocks: usize, block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layout = KvCacheLayout {
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        };
        let total_elems = layout.total_elems();
        Self {
            k: vec![f16::ZERO; total_elems],
            v: vec![f16::ZERO; total_elems],
            layout,
        }
    }

    /// Get the cache layout.
    pub fn layout(&self) -> &KvCacheLayout {
        &self.layout
    }

    /// Get capacity in tokens.
    pub fn capacity_tokens(&self) -> usize {
        self.layout.num_blocks * self.layout.block_size
    }

    /// Write a single token's KV data in f16 format.
    ///
    /// # Arguments
    /// * `base` - Base index in the cache (token position * elems_per_token)
    /// * `k_src` - Key data [num_kv_heads * head_dim]
    /// * `v_src` - Value data [num_kv_heads * head_dim]
    pub fn write_token_f16(
        &mut self,
        base: usize,
        k_src: &[f16],
        v_src: &[f16],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_src.len() != kv_dim || v_src.len() != kv_dim {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_src.len(),
                v_len: v_src.len(),
            });
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CacheError::WriteOutOfBounds {
                base,
                len: kv_dim,
                cap: self.k.len(),
            });
        }
        self.k[base..end].copy_from_slice(k_src);
        self.v[base..end].copy_from_slice(v_src);
        Ok(())
    }

    /// Write a single token's KV data in f32 format (converted to f16).
    pub fn write_token_f32(
        &mut self,
        base: usize,
        k_src: &[f32],
        v_src: &[f32],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_src.len() != kv_dim || v_src.len() != kv_dim {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_src.len(),
                v_len: v_src.len(),
            });
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CacheError::WriteOutOfBounds {
                base,
                len: kv_dim,
                cap: self.k.len(),
            });
        }
        for i in 0..kv_dim {
            self.k[base + i] = f16::from_f32(k_src[i]);
            self.v[base + i] = f16::from_f32(v_src[i]);
        }
        Ok(())
    }

    /// Read a single token's KV data in f16 format.
    pub fn read_token_f16(
        &self,
        base: usize,
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: v_out.len(),
            });
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CacheError::ReadOutOfBounds {
                base,
                len: kv_dim,
                cap: self.k.len(),
            });
        }
        k_out.copy_from_slice(&self.k[base..end]);
        v_out.copy_from_slice(&self.v[base..end]);
        Ok(())
    }

    /// Read a single token's KV data in f32 format (converted from f16).
    pub fn read_token_f32(
        &self,
        base: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_out.len() != kv_dim || v_out.len() != kv_dim {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: v_out.len(),
            });
        }
        let end = base.saturating_add(kv_dim);
        if end > self.k.len() || end > self.v.len() {
            return Err(CacheError::ReadOutOfBounds {
                base,
                len: kv_dim,
                cap: self.k.len(),
            });
        }
        for i in 0..kv_dim {
            k_out[i] = self.k[base + i].to_f32();
            v_out[i] = self.v[base + i].to_f32();
        }
        Ok(())
    }

    /// Gather tokens from non-contiguous positions in f16 format.
    ///
    /// # Arguments
    /// * `bases` - Base indices for each token to gather
    /// * `k_out` - Output buffer [bases.len() * kv_dim]
    /// * `v_out` - Output buffer [bases.len() * kv_dim]
    pub fn gather_tokens_f16(
        &self,
        bases: &[usize],
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_out.len() != v_out.len() {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: v_out.len(),
            });
        }
        let need = bases.len().saturating_mul(kv_dim);
        if k_out.len() != need {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: need,
            });
        }
        for (t, &base) in bases.iter().enumerate() {
            let end = base.saturating_add(kv_dim);
            if end > self.k.len() || end > self.v.len() {
                return Err(CacheError::ReadOutOfBounds {
                    base,
                    len: kv_dim,
                    cap: self.k.len(),
                });
            }
            let dst = t * kv_dim;
            k_out[dst..dst + kv_dim].copy_from_slice(&self.k[base..end]);
            v_out[dst..dst + kv_dim].copy_from_slice(&self.v[base..end]);
        }
        Ok(())
    }

    /// Gather tokens from non-contiguous positions in f32 format.
    pub fn gather_tokens_f32(
        &self,
        bases: &[usize],
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<(), CacheError> {
        let kv_dim = self.layout.elems_per_token();
        if k_out.len() != v_out.len() {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: v_out.len(),
            });
        }
        let need = bases.len().saturating_mul(kv_dim);
        if k_out.len() != need {
            return Err(CacheError::GatherLengthMismatch {
                k_len: k_out.len(),
                v_len: need,
            });
        }
        for (t, &base) in bases.iter().enumerate() {
            let end = base.saturating_add(kv_dim);
            if end > self.k.len() || end > self.v.len() {
                return Err(CacheError::ReadOutOfBounds {
                    base,
                    len: kv_dim,
                    cap: self.k.len(),
                });
            }
            let dst = t * kv_dim;
            for i in 0..kv_dim {
                k_out[dst + i] = self.k[base + i].to_f32();
                v_out[dst + i] = self.v[base + i].to_f32();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_storage_write_read_f16() {
        let num_blocks = 1024;
        let block_size = 64;
        let num_kv_heads = 2;
        let head_dim = 128;

        let mut storage = CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim);

        // Write token at position 0
        let k_data: Vec<f16> = (0..(num_kv_heads * head_dim))
            .map(|i| f16::from_f32(i as f32))
            .collect();
        let v_data: Vec<f16> = (0..(num_kv_heads * head_dim))
            .map(|i| f16::from_f32(i as f32 + 1000.0))
            .collect();

        storage.write_token_f16(0, &k_data, &v_data).unwrap();

        // Read back
        let mut k_out = vec![f16::ZERO; num_kv_heads * head_dim];
        let mut v_out = vec![f16::ZERO; num_kv_heads * head_dim];
        storage.read_token_f16(0, &mut k_out, &mut v_out).unwrap();

        assert_eq!(k_out, k_data);
        assert_eq!(v_out, v_data);
    }

    #[test]
    fn test_kv_storage_write_read_f32() {
        let num_blocks = 1024;
        let block_size = 64;
        let num_kv_heads = 2;
        let head_dim = 128;

        let mut storage = CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim);

        // Write token at position 0 in f32
        let k_data: Vec<f32> = (0..(num_kv_heads * head_dim)).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (0..(num_kv_heads * head_dim))
            .map(|i| i as f32 + 1000.0)
            .collect();

        storage.write_token_f32(0, &k_data, &v_data).unwrap();

        // Read back in f32
        let mut k_out = vec![0.0f32; num_kv_heads * head_dim];
        let mut v_out = vec![0.0f32; num_kv_heads * head_dim];
        storage.read_token_f32(0, &mut k_out, &mut v_out).unwrap();

        // Allow for f16 conversion error
        for (k_exp, k_act) in k_data.iter().zip(k_out.iter()) {
            assert!((k_exp - k_act).abs() < 0.001);
        }
        for (v_exp, v_act) in v_data.iter().zip(v_out.iter()) {
            assert!((v_exp - v_act).abs() < 0.001);
        }
    }

    #[test]
    fn test_kv_storage_gather_f16() {
        let num_blocks = 1024;
        let block_size = 64;
        let num_kv_heads = 2;
        let head_dim = 128;
        let kv_dim = num_kv_heads * head_dim;

        let mut storage = CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim);

        // Write tokens at positions 0, 10, 25
        for (pos, val) in [(0, 1.0f32), (10, 2.0), (25, 3.0)] {
            let k_data = vec![f16::from_f32(val); kv_dim];
            let v_data = vec![f16::from_f32(val + 100.0); kv_dim];
            storage
                .write_token_f16(pos * kv_dim, &k_data, &v_data)
                .unwrap();
        }

        // Gather from positions [0, 25]
        let bases = vec![0, 25 * kv_dim];
        let mut k_out = vec![f16::ZERO; 2 * kv_dim];
        let mut v_out = vec![f16::ZERO; 2 * kv_dim];

        storage
            .gather_tokens_f16(&bases, &mut k_out, &mut v_out)
            .unwrap();

        // Verify gathered values
        assert_eq!(k_out[0], f16::from_f32(1.0)); // Position 0
        assert_eq!(k_out[kv_dim], f16::from_f32(3.0)); // Position 25
    }

    #[test]
    fn test_kv_storage_gather_f32() {
        let num_blocks = 1024;
        let block_size = 64;
        let num_kv_heads = 2;
        let head_dim = 128;
        let kv_dim = num_kv_heads * head_dim;

        let mut storage = CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim);

        // Write tokens at positions 0, 10, 25
        for (pos, val) in [(0, 1.0f32), (10, 2.0), (25, 3.0)] {
            let k_data = vec![f16::from_f32(val); kv_dim];
            let v_data = vec![f16::from_f32(val + 100.0); kv_dim];
            storage
                .write_token_f16(pos * kv_dim, &k_data, &v_data)
                .unwrap();
        }

        // Gather from positions [0, 25]
        let bases = vec![0, 25 * kv_dim];
        let mut k_out = vec![0.0f32; 2 * kv_dim];
        let mut v_out = vec![0.0f32; 2 * kv_dim];

        storage
            .gather_tokens_f32(&bases, &mut k_out, &mut v_out)
            .unwrap();

        // Verify gathered values
        assert!((k_out[0] - 1.0).abs() < 0.001); // Position 0
        assert!((k_out[kv_dim] - 3.0).abs() < 0.001); // Position 25
    }

    #[test]
    fn test_kv_storage_bounds_checking() {
        let mut storage = CpuKvStorage::new(10, 64, 2, 128);
        let kv_dim = 256;

        // Write out of bounds
        let k_data = vec![f16::ZERO; kv_dim];
        let v_data = vec![f16::ZERO; kv_dim];
        let err = storage
            .write_token_f16(storage.k.len(), &k_data, &v_data)
            .unwrap_err();
        assert!(matches!(err, CacheError::WriteOutOfBounds { .. }));

        // Read out of bounds
        let mut k_out = vec![f16::ZERO; kv_dim];
        let mut v_out = vec![f16::ZERO; kv_dim];
        let err = storage
            .read_token_f16(storage.k.len(), &mut k_out, &mut v_out)
            .unwrap_err();
        assert!(matches!(err, CacheError::ReadOutOfBounds { .. }));
    }
}
