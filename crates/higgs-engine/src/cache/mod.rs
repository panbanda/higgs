// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache for Higgs.
//!
//! Inspired by cellm's `BlockAllocator`:
//! <https://github.com/jeffasante/cellm/blob/main/crates/cellm-cache/src/allocator.rs>

pub mod allocator;
pub mod disk_prefix_cache;
pub mod disk_storage;
pub mod paged;
pub mod pagetable;
pub(crate) mod paired;
pub mod storage;

pub use allocator::BlockAllocator;
pub use disk_prefix_cache::{
    DEFAULT_MAX_DISK_BLOCKS, DEFAULT_MIN_TOKENS_TO_PERSIST, DiskPrefixCache, DiskPrefixCacheConfig,
};
pub use paged::PagedKvCache;
pub use pagetable::PageTable;
pub use storage::{CpuKvStorage, KvCacheLayout};

/// Cache operation errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CacheError {
    #[error("Out of blocks: requested {requested}, free {free}")]
    OutOfBlocks { requested: usize, free: usize },

    #[error("Invalid block ID: {0}")]
    InvalidBlockId(u32),

    #[error("Double free of block ID: {0}")]
    DoubleFree(u32),

    #[error("Session not found: {0}")]
    SessionNotFound(u64),

    #[error("Session already exists: {0}")]
    SessionAlreadyExists(u64),

    #[error("Cache write out of bounds: base={base}, len={len}, cap={cap}")]
    WriteOutOfBounds { base: usize, len: usize, cap: usize },

    #[error("Cache read out of bounds: base={base}, len={len}, cap={cap}")]
    ReadOutOfBounds { base: usize, len: usize, cap: usize },

    #[error("Gather output length mismatch: k={k_len}, v={v_len}")]
    GatherLengthMismatch { k_len: usize, v_len: usize },

    #[error("Invalid cache argument: {0}")]
    InvalidArgument(&'static str),

    #[error("Block count {0} exceeds u32::MAX")]
    BlockCountOverflow(usize),
}
