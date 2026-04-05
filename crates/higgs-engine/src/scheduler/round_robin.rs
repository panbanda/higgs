// SPDX-License-Identifier: Apache-2.0
//! Round-robin scheduler for continuous batching.
//!
//! Implementation inspired by cellm's RoundRobinScheduler:
//! https://github.com/jeffasante/cellm/blob/main/crates/cellm-scheduler/src/rr.rs

use std::collections::VecDeque;

/// Session identifier.
pub type SessionId = u64;

/// Round-robin scheduler for decode steps.
#[derive(Debug, Default)]
pub struct RoundRobinScheduler {
    q: VecDeque<SessionId>,
}

impl RoundRobinScheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Self { q: VecDeque::new() }
    }

    /// Check if scheduler is empty.
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }

    /// Get number of sessions in queue.
    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// Add a session to the scheduler.
    ///
    /// Does nothing if session is already in queue.
    pub fn add(&mut self, id: SessionId) {
        if self.q.contains(&id) {
            return;
        }
        self.q.push_back(id);
    }

    /// Remove a session from the scheduler.
    pub fn remove(&mut self, id: SessionId) {
        if let Some(idx) = self.q.iter().position(|&x| x == id) {
            self.q.remove(idx);
        }
    }

    /// Get next session ID and rotate to back (round-robin).
    pub fn next(&mut self) -> Option<SessionId> {
        let id = self.q.pop_front()?;
        self.q.push_back(id); // Rotate to back
        Some(id)
    }

    /// Get all session IDs without modifying queue.
    pub fn sessions(&self) -> impl Iterator<Item = &SessionId> {
        self.q.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_rotates() {
        let mut scheduler = RoundRobinScheduler::new();

        scheduler.add(1);
        scheduler.add(2);
        scheduler.add(3);

        assert_eq!(scheduler.next(), Some(1));
        assert_eq!(scheduler.next(), Some(2));
        assert_eq!(scheduler.next(), Some(3));
        assert_eq!(scheduler.next(), Some(1)); // Wraps around
    }

    #[test]
    fn test_scheduler_remove_works() {
        let mut scheduler = RoundRobinScheduler::new();
        scheduler.add(1);
        scheduler.add(2);
        scheduler.remove(1);

        assert_eq!(scheduler.next(), Some(2));
        assert_eq!(scheduler.next(), Some(2)); // Only session 2 remains
    }

    #[test]
    fn test_scheduler_add_duplicate() {
        let mut scheduler = RoundRobinScheduler::new();
        scheduler.add(1);
        scheduler.add(1); // Duplicate
        scheduler.add(1); // Duplicate

        assert_eq!(scheduler.len(), 1);
        assert_eq!(scheduler.next(), Some(1));
        assert_eq!(scheduler.next(), Some(1));
    }

    #[test]
    fn test_scheduler_is_empty() {
        let mut scheduler = RoundRobinScheduler::new();
        assert!(scheduler.is_empty());
        assert_eq!(scheduler.len(), 0);

        scheduler.add(1);
        assert!(!scheduler.is_empty());
        assert_eq!(scheduler.len(), 1);

        scheduler.next();
        assert!(!scheduler.is_empty()); // Still has session

        scheduler.remove(1);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_scheduler_sessions() {
        let mut scheduler = RoundRobinScheduler::new();
        scheduler.add(1);
        scheduler.add(2);
        scheduler.add(3);

        let sessions: Vec<_> = scheduler.sessions().copied().collect();
        assert_eq!(sessions, vec![1, 2, 3]);
    }

    #[test]
    fn test_scheduler_empty_next() {
        let mut scheduler = RoundRobinScheduler::new();
        assert_eq!(scheduler.next(), None);
    }
}
