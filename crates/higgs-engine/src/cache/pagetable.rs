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
        if blocks.is_empty() {
            return Ok(());
        }
        self.sessions.insert(session_id, blocks.to_vec());
        Ok(())
    }

    /// Get blocks assigned to a session.
    pub fn get_blocks(&self, session_id: u64) -> Option<&[u32]> {
        self.sessions.get(&session_id).map(|v| v.as_slice())
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
