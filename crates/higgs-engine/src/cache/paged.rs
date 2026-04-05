// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache integrating allocator, page table, and storage.

use crate::cache::{BlockAllocator, CacheError, CpuKvStorage, KvCacheLayout, PageTable};
use half::f16;

/// View into a session's KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheView<'a> {
    pub session_id: u64,
    pub blocks: &'a [u32],
    pub num_tokens: usize,
    pub layout: &'a KvCacheLayout,
}

impl<'a> KvCacheView<'a> {
    /// Get number of tokens in this session.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get block ID for a token position.
    pub fn block_for_token(&self, token_idx: usize) -> u32 {
        let block_size = self.layout.block_size;
        let block_idx = token_idx / block_size;
        self.blocks[block_idx]
    }

    /// Get offset within block for a token position.
    pub fn offset_in_block(&self, token_idx: usize) -> usize {
        token_idx % self.layout.block_size
    }
}

/// Paged KV cache with block-based memory management.
#[derive(Debug)]
pub struct PagedKvCache {
    allocator: BlockAllocator,
    page_table: PageTable,
    storage: CpuKvStorage,
    session_tokens: std::collections::HashMap<u64, usize>,
    block_size: usize,
}

impl PagedKvCache {
    /// Create new paged KV cache.
    ///
    /// # Arguments
    /// * `num_blocks` - Total number of blocks to allocate
    /// * `block_size` - Tokens per block
    /// * `num_kv_heads` - Number of KV heads
    /// * `head_dim` - Dimension per head
    pub fn new(num_blocks: usize, block_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            allocator: BlockAllocator::new(num_blocks),
            page_table: PageTable::new(),
            storage: CpuKvStorage::new(num_blocks, block_size, num_kv_heads, head_dim),
            session_tokens: std::collections::HashMap::new(),
            block_size,
        }
    }

    /// Get cache layout.
    pub fn layout(&self) -> &KvCacheLayout {
        self.storage.layout()
    }

    /// Get total capacity in tokens.
    pub fn capacity_tokens(&self) -> usize {
        self.storage.capacity_tokens()
    }

    /// Get number of free blocks.
    pub fn free_blocks(&self) -> usize {
        self.allocator.free_count()
    }

    /// Get number of blocks in use.
    pub fn in_use_blocks(&self) -> usize {
        self.allocator.in_use_count()
    }

    /// Create a new session.
    pub fn create_session(&mut self, session_id: u64) -> Result<(), CacheError> {
        if self.page_table.has_session(session_id) {
            return Err(CacheError::SessionNotFound(session_id));
        }
        self.session_tokens.insert(session_id, 0);
        Ok(())
    }

    /// Get current token count for a session.
    pub fn session_token_count(&self, session_id: u64) -> Option<usize> {
        self.session_tokens.get(&session_id).copied()
    }

    /// Append a token to a session.
    ///
    /// Allocates new blocks as needed.
    pub fn append_token(
        &mut self,
        session_id: u64,
        k: &[f16],
        v: &[f16],
    ) -> Result<(), CacheError> {
        let current_tokens = self
            .session_tokens
            .get(&session_id)
            .copied()
            .ok_or(CacheError::SessionNotFound(session_id))?;

        let kv_dim = self.storage.layout().elems_per_token();
        let current_block_idx = current_tokens / self.block_size;
        let offset_in_block = current_tokens % self.block_size;

        // Get or allocate blocks for this session
        let blocks = self.ensure_blocks_for_session(session_id, current_block_idx + 1)?;

        // Calculate write position
        let block_id = blocks[current_block_idx];
        let base = (block_id as usize * self.block_size + offset_in_block) * kv_dim;

        // Write token
        self.storage.write_token_f16(base, k, v)?;

        // Update token count
        self.session_tokens.insert(session_id, current_tokens + 1);

        Ok(())
    }

    /// Append tokens in f32 format.
    pub fn append_token_f32(
        &mut self,
        session_id: u64,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), CacheError> {
        let current_tokens = self
            .session_tokens
            .get(&session_id)
            .copied()
            .ok_or(CacheError::SessionNotFound(session_id))?;

        let kv_dim = self.storage.layout().elems_per_token();
        let current_block_idx = current_tokens / self.block_size;
        let offset_in_block = current_tokens % self.block_size;

        let blocks = self.ensure_blocks_for_session(session_id, current_block_idx + 1)?;

        let block_id = blocks[current_block_idx];
        let base = (block_id as usize * self.block_size + offset_in_block) * kv_dim;

        self.storage.write_token_f32(base, k, v)?;

        self.session_tokens.insert(session_id, current_tokens + 1);

        Ok(())
    }

    /// Get view of a session's KV cache.
    pub fn get_session_view(&self, session_id: u64) -> Option<KvCacheView<'_>> {
        let blocks = self.page_table.get_blocks(session_id)?;
        let num_tokens = self.session_tokens.get(&session_id).copied()?;

        Some(KvCacheView {
            session_id,
            blocks,
            num_tokens,
            layout: self.storage.layout(),
        })
    }

    /// Remove a session and free its blocks.
    pub fn remove_session(&mut self, session_id: u64) -> Result<(), CacheError> {
        // Get blocks to free
        let blocks = self.page_table.remove_session(session_id)?;

        // Free blocks
        for block_id in blocks {
            self.allocator.free(block_id)?;
        }

        // Remove token count
        self.session_tokens.remove(&session_id);

        Ok(())
    }

    /// Gather tokens from a session at specified indices.
    pub fn gather_session_tokens_f16(
        &self,
        session_id: u64,
        token_indices: &[usize],
        k_out: &mut [f16],
        v_out: &mut [f16],
    ) -> Result<(), CacheError> {
        let view = self
            .get_session_view(session_id)
            .ok_or(CacheError::SessionNotFound(session_id))?;

        let kv_dim = self.storage.layout().elems_per_token();

        // Convert token indices to cache positions
        let bases: Vec<usize> = token_indices
            .iter()
            .map(|&token_idx| {
                let block_idx = token_idx / self.block_size;
                let offset = token_idx % self.block_size;
                let block_id = view.blocks[block_idx] as usize;
                (block_id * self.block_size + offset) * kv_dim
            })
            .collect();

        self.storage.gather_tokens_f16(&bases, k_out, v_out)
    }

    /// Ensure enough blocks are allocated for a session.
    fn ensure_blocks_for_session(
        &mut self,
        session_id: u64,
        needed_blocks: usize,
    ) -> Result<&[u32], CacheError> {
        // Check if session exists
        if !self.page_table.has_session(session_id) {
            // Allocate initial block
            let block_id = self.allocator.alloc().ok_or(CacheError::OutOfBlocks {
                requested: 1,
                free: self.allocator.free_count(),
            })?;
            self.page_table.assign_blocks(session_id, &[block_id])?;
            if needed_blocks == 1 {
                return Ok(self.page_table.get_blocks(session_id).unwrap());
            }
        }

        // Get current blocks
        let current_blocks = self.page_table.get_blocks(session_id).unwrap().to_vec();

        if current_blocks.len() >= needed_blocks {
            return Ok(self.page_table.get_blocks(session_id).unwrap());
        }

        // Allocate additional blocks
        let additional = needed_blocks - current_blocks.len();
        let new_blocks = self.allocator.alloc_n(additional)?;

        // Merge with existing blocks
        let mut all_blocks = current_blocks;
        all_blocks.extend(new_blocks);
        self.page_table.assign_blocks(session_id, &all_blocks)?;

        Ok(self.page_table.get_blocks(session_id).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_cache_session_lifecycle() {
        let mut cache = PagedKvCache::new(1024, 64, 2, 128);
        let kv_dim = 256;

        // Create session
        let session_id = 1u64;
        cache.create_session(session_id).unwrap();

        // Append tokens
        let k_data = vec![f16::ZERO; kv_dim];
        let v_data = vec![f16::ZERO; kv_dim];
        cache.append_token(session_id, &k_data, &v_data).unwrap();
        cache.append_token(session_id, &k_data, &v_data).unwrap();

        // Check token count
        assert_eq!(cache.session_token_count(session_id), Some(2));

        // Get view
        let view = cache.get_session_view(session_id).unwrap();
        assert_eq!(view.num_tokens(), 2);

        // Remove session
        cache.remove_session(session_id).unwrap();
        assert!(cache.get_session_view(session_id).is_none());
    }

    #[test]
    fn test_paged_cache_block_allocation() {
        let mut cache = PagedKvCache::new(10, 4, 2, 128);

        // Create session and append 10 tokens (needs 3 blocks with block_size=4)
        let session_id = 1u64;
        cache.create_session(session_id).unwrap();

        let kv_dim = 256;
        let k_data = vec![f16::ZERO; kv_dim];
        let v_data = vec![f16::ZERO; kv_dim];

        for _ in 0..10 {
            cache.append_token(session_id, &k_data, &v_data).unwrap();
        }

        // Should have allocated 3 blocks (10 tokens / 4 per block = 2.5 -> 3)
        let view = cache.get_session_view(session_id).unwrap();
        assert_eq!(view.blocks.len(), 3);
        assert_eq!(cache.in_use_blocks(), 3);
    }

    #[test]
    fn test_paged_cache_gather() {
        let mut cache = PagedKvCache::new(1024, 64, 2, 128);
        let kv_dim = 256;

        // Create session
        let session_id = 1u64;
        cache.create_session(session_id).unwrap();

        // Append tokens with distinct values
        for i in 0..5 {
            let k_data = vec![f16::from_f32(i as f32); kv_dim];
            let v_data = vec![f16::from_f32(i as f32 + 100.0); kv_dim];
            cache.append_token(session_id, &k_data, &v_data).unwrap();
        }

        // Gather tokens [0, 2, 4]
        let mut k_out = vec![f16::ZERO; 3 * kv_dim];
        let mut v_out = vec![f16::ZERO; 3 * kv_dim];
        cache
            .gather_session_tokens_f16(session_id, &[0, 2, 4], &mut k_out, &mut v_out)
            .unwrap();

        // Verify gathered values
        assert_eq!(k_out[0], f16::from_f32(0.0)); // Token 0
        assert_eq!(k_out[kv_dim], f16::from_f32(2.0)); // Token 2
        assert_eq!(k_out[2 * kv_dim], f16::from_f32(4.0)); // Token 4
    }

    #[test]
    fn test_paged_cache_out_of_blocks() {
        let mut cache = PagedKvCache::new(2, 4, 2, 128);
        let kv_dim = 256;

        // Create session
        let session_id = 1u64;
        cache.create_session(session_id).unwrap();

        let k_data = vec![f16::ZERO; kv_dim];
        let v_data = vec![f16::ZERO; kv_dim];

        // Append 9 tokens (needs 3 blocks, but only 2 available)
        for _ in 0..8 {
            cache.append_token(session_id, &k_data, &v_data).unwrap();
        }

        // 9th token should fail
        let err = cache
            .append_token(session_id, &k_data, &v_data)
            .unwrap_err();
        assert!(matches!(err, CacheError::OutOfBlocks { .. }));
    }
}
