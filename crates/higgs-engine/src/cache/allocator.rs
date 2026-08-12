// SPDX-License-Identifier: Apache-2.0
//! Fixed-size block allocator for paged KV cache.
//!
//! The allocator manages block IDs (u32). The actual KV bytes live in a
//! separate storage owned by the cache implementation.
//!
//! Implementation inspired by cellm's `BlockAllocator`:
//! <https://github.com/jeffasante/cellm/blob/main/crates/cellm-cache/src/allocator.rs>

use crate::cache::CacheError;
use std::collections::VecDeque;

/// Fixed-size block allocator for paged KV cache.
#[derive(Debug)]
pub struct BlockAllocator {
    total: usize,
    in_use: usize,
    free_list: VecDeque<u32>,
    is_free: Vec<bool>,
}

impl BlockAllocator {
    /// Create an allocator with `total_blocks` IDs in the free list.
    ///
    /// Returns [`CacheError::BlockCountOverflow`] when `total_blocks` exceeds
    /// `u32::MAX`, since block IDs are `u32`.
    pub fn new(total_blocks: usize) -> Result<Self, CacheError> {
        let total_u32 = u32::try_from(total_blocks)
            .map_err(|_| CacheError::BlockCountOverflow(total_blocks))?;

        let mut free_list = VecDeque::with_capacity(total_blocks);
        free_list.extend(0..total_u32);

        Ok(Self {
            total: total_blocks,
            in_use: 0,
            free_list,
            is_free: vec![true; total_blocks],
        })
    }

    /// Get total number of blocks.
    pub const fn total_count(&self) -> usize {
        self.total
    }

    /// Get number of blocks currently in use.
    pub const fn in_use_count(&self) -> usize {
        self.in_use
    }

    /// Get number of free blocks available.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Allocate one block ID.
    pub fn alloc(&mut self) -> Option<u32> {
        let id = self.free_list.pop_front()?;
        let idx = usize::try_from(id).ok()?;
        let slot = self.is_free.get_mut(idx)?;
        debug_assert!(*slot, "free_list and is_free out of sync");
        *slot = false;
        self.in_use += 1;
        Some(id)
    }

    /// Allocate `n` block IDs or return an error without changing state.
    pub fn alloc_n(&mut self, n: usize) -> Result<Vec<u32>, CacheError> {
        if n > self.free_count() {
            return Err(CacheError::OutOfBlocks {
                requested: n,
                free: self.free_count(),
            });
        }

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            // free_count was checked above, so this loop must succeed; the
            // let-else is a defensive fallback rather than a real error path.
            let Some(id) = self.alloc() else {
                return Err(CacheError::OutOfBlocks {
                    requested: n,
                    free: out.len(),
                });
            };
            out.push(id);
        }
        Ok(out)
    }

    /// Free a previously allocated block ID.
    pub fn free(&mut self, block_id: u32) -> Result<(), CacheError> {
        let idx = usize::try_from(block_id).map_err(|_| CacheError::InvalidBlockId(block_id))?;
        if idx >= self.total {
            return Err(CacheError::InvalidBlockId(block_id));
        }
        let slot = self
            .is_free
            .get_mut(idx)
            .ok_or(CacheError::InvalidBlockId(block_id))?;
        if *slot {
            return Err(CacheError::DoubleFree(block_id));
        }
        *slot = true;
        self.free_list.push_back(block_id);
        self.in_use = self.in_use.saturating_sub(1);
        Ok(())
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_free_roundtrip() {
        let mut allocator = BlockAllocator::new(3).unwrap();
        assert_eq!(allocator.total_count(), 3);
        assert_eq!(allocator.free_count(), 3);
        assert_eq!(allocator.in_use_count(), 0);

        // Allocate a block
        let block_id = allocator.alloc().unwrap();
        assert!(block_id < 3);
        assert_eq!(allocator.free_count(), 2);
        assert_eq!(allocator.in_use_count(), 1);

        // Free the block
        allocator.free(block_id).unwrap();
        assert_eq!(allocator.free_count(), 3);
        assert_eq!(allocator.in_use_count(), 0);
    }

    #[test]
    fn test_alloc_exhaustion() {
        let mut allocator = BlockAllocator::new(2).unwrap();
        assert!(allocator.alloc().is_some());
        assert!(allocator.alloc().is_some());
        assert!(allocator.alloc().is_none()); // Exhausted
    }

    #[test]
    fn test_double_free_errors() {
        let mut allocator = BlockAllocator::new(1).unwrap();
        let block_id = allocator.alloc().unwrap();
        allocator.free(block_id).unwrap();

        let err = allocator.free(block_id).unwrap_err();
        assert!(matches!(err, CacheError::DoubleFree(_)));
    }

    #[test]
    fn test_invalid_block_id_errors() {
        let mut allocator = BlockAllocator::new(1).unwrap();
        let err = allocator.free(99).unwrap_err();
        assert!(matches!(err, CacheError::InvalidBlockId(_)));
    }

    #[test]
    fn test_alloc_n_is_atomic() {
        let mut allocator = BlockAllocator::new(2).unwrap();
        let err = allocator.alloc_n(3).unwrap_err();
        assert!(matches!(
            err,
            CacheError::OutOfBlocks {
                requested: 3,
                free: 2
            }
        ));
        assert_eq!(allocator.free_count(), 2); // No blocks allocated
        assert_eq!(allocator.in_use_count(), 0);
    }

    #[test]
    fn test_alloc_n_succeeds() {
        let mut allocator = BlockAllocator::new(5).unwrap();
        let blocks = allocator.alloc_n(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(allocator.free_count(), 2);
        assert_eq!(allocator.in_use_count(), 3);
    }
}
