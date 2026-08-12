// SPDX-License-Identifier: Apache-2.0
//! Page table for mapping session IDs to block IDs.

use crate::cache::CacheError;
use std::collections::HashMap;

/// Maps session IDs to allocated block IDs.
#[derive(Debug, Default)]
pub struct PageTable {
    sessions: HashMap<u64, Vec<u32>>,
}

impl PageTable {
    /// Create a new empty page table.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Assign blocks to a session.
    pub fn assign_blocks(&mut self, session_id: u64, blocks: &[u32]) -> Result<(), CacheError> {
        self.sessions.insert(session_id, blocks.to_vec());
        Ok(())
    }

    /// Get blocks assigned to a session.
    pub fn get_blocks(&self, session_id: u64) -> Option<&[u32]> {
        self.sessions.get(&session_id).map(std::vec::Vec::as_slice)
    }

    /// Get mutable reference to blocks for a session.
    pub fn get_blocks_mut(&mut self, session_id: u64) -> Option<&mut Vec<u32>> {
        self.sessions.get_mut(&session_id)
    }

    /// Remove a session and return its blocks.
    pub fn remove_session(&mut self, session_id: u64) -> Result<Vec<u32>, CacheError> {
        self.sessions
            .remove(&session_id)
            .ok_or(CacheError::SessionNotFound(session_id))
    }

    /// Check if a session exists.
    pub fn has_session(&self, session_id: u64) -> bool {
        self.sessions.contains_key(&session_id)
    }

    /// Get number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::needless_for_each,
    clippy::needless_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_range_loop,
    clippy::manual_flatten,
    clippy::unnecessary_map_or,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    clippy::explicit_iter_loop,
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr,
    clippy::str_to_string,
    clippy::if_then_some_else_none,
    clippy::redundant_type_annotations
)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagetable_session_blocks() {
        let mut table = PageTable::new();

        // Assign blocks to session
        let session_id = 1u64;
        let blocks = vec![10u32, 11, 12];
        table.assign_blocks(session_id, &blocks).unwrap();

        // Retrieve blocks
        let retrieved = table.get_blocks(session_id).unwrap();
        assert_eq!(retrieved, blocks);

        // Remove session
        let removed = table.remove_session(session_id).unwrap();
        assert_eq!(removed, blocks);
        assert!(table.get_blocks(session_id).is_none());
    }

    #[test]
    fn test_pagetable_remove_not_found() {
        let mut table = PageTable::new();
        let err = table.remove_session(999).unwrap_err();
        assert!(matches!(err, CacheError::SessionNotFound(_)));
    }

    #[test]
    fn test_pagetable_has_session() {
        let mut table = PageTable::new();
        assert!(!table.has_session(1));

        table.assign_blocks(1, &[1, 2, 3]).unwrap();
        assert!(table.has_session(1));

        table.remove_session(1).unwrap();
        assert!(!table.has_session(1));
    }

    #[test]
    fn test_pagetable_session_count() {
        let mut table = PageTable::new();
        assert_eq!(table.session_count(), 0);

        table.assign_blocks(1, &[1]).unwrap();
        table.assign_blocks(2, &[2]).unwrap();
        table.assign_blocks(3, &[3]).unwrap();
        assert_eq!(table.session_count(), 3);
    }
}
