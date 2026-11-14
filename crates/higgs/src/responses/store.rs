//! Bounded in-memory store backing `previous_response_id`, GET and DELETE.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::types::responses::{InputItem, ResponseObject};

/// Entry cap.
///
/// Each entry holds the whole conversation up to that turn, so long agent
/// sessions are the memory driver; 256 keeps a busy local server in the tens
/// of megabytes while covering many concurrent sessions.
pub const MAX_STORED_RESPONSES: usize = 256;

/// Entries expire this long after being stored. Long enough for an agent to
/// pause between turns, short enough that abandoned sessions free their memory.
pub const STORED_RESPONSE_TTL: Duration = Duration::from_secs(60 * 60);

/// A response together with the full conversation that produced it.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: ResponseObject,
    /// Every input item up to and including this turn's output, ready to be
    /// prepended when a later request names this response as its parent.
    pub conversation: Vec<InputItem>,
}

struct Entry {
    stored: StoredResponse,
    stored_at: Instant,
    last_used: u64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<String, Entry>,
    use_counter: u64,
}

/// Least-recently-used, time-limited map from response id to [`StoredResponse`].
pub struct ResponseStore {
    inner: Mutex<Inner>,
    capacity: usize,
    ttl: Duration,
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::with_limits(MAX_STORED_RESPONSES, STORED_RESPONSE_TTL)
    }
}

impl ResponseStore {
    pub fn with_limits(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            capacity,
            ttl,
        }
    }

    pub fn insert(&self, stored: StoredResponse) {
        self.insert_at(stored, Instant::now());
    }

    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        self.get_at(id, Instant::now())
    }

    pub fn remove(&self, id: &str) -> bool {
        let mut inner = self.lock();
        inner.entries.remove(id).is_some()
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn insert_at(&self, stored: StoredResponse, now: Instant) {
        let mut inner = self.lock();
        let ttl = self.ttl;
        inner
            .entries
            .retain(|_, entry| now.saturating_duration_since(entry.stored_at) < ttl);
        inner.use_counter += 1;
        let last_used = inner.use_counter;
        inner.entries.insert(
            stored.response.id.clone(),
            Entry {
                stored,
                stored_at: now,
                last_used,
            },
        );
        while inner.entries.len() > self.capacity {
            let Some(oldest) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            inner.entries.remove(&oldest);
        }
        drop(inner);
    }

    fn get_at(&self, id: &str, now: Instant) -> Option<StoredResponse> {
        let mut inner = self.lock();
        let expired = inner
            .entries
            .get(id)
            .is_some_and(|entry| now.saturating_duration_since(entry.stored_at) >= self.ttl);
        if expired {
            inner.entries.remove(id);
            return None;
        }
        inner.use_counter += 1;
        let tick = inner.use_counter;
        let entry = inner.entries.get_mut(id)?;
        entry.last_used = tick;
        let stored = entry.stored.clone();
        drop(inner);
        Some(stored)
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::responses::ResponseStatus;

    fn stored(id: &str) -> StoredResponse {
        StoredResponse {
            response: ResponseObject {
                id: id.to_owned(),
                object: "response",
                created_at: 0,
                status: ResponseStatus::Completed,
                model: "m".to_owned(),
                output: vec![],
                output_text: String::new(),
                usage: None,
                incomplete_details: None,
                error: None,
                instructions: None,
                metadata: serde_json::Map::new(),
                previous_response_id: None,
                tools: vec![],
                tool_choice: serde_json::json!("auto"),
                parallel_tool_calls: true,
                temperature: None,
                top_p: None,
                max_output_tokens: None,
            },
            conversation: vec![],
        }
    }

    #[test]
    fn stores_and_returns_entries() {
        let store = ResponseStore::default();
        store.insert(stored("resp_a"));
        assert_eq!(store.get("resp_a").unwrap().response.id, "resp_a");
        assert!(store.get("resp_missing").is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn evicts_least_recently_used_beyond_capacity() {
        let store = ResponseStore::with_limits(2, Duration::from_secs(3600));
        store.insert(stored("resp_a"));
        store.insert(stored("resp_b"));
        assert!(store.get("resp_a").is_some());
        store.insert(stored("resp_c"));

        assert_eq!(store.len(), 2);
        assert!(store.get("resp_a").is_some());
        assert!(store.get("resp_b").is_none(), "b was least recently used");
        assert!(store.get("resp_c").is_some());
    }

    #[test]
    fn entries_expire_after_the_ttl() {
        let store = ResponseStore::with_limits(8, Duration::from_secs(60));
        let start = Instant::now();
        store.insert_at(stored("resp_a"), start);

        assert!(
            store
                .get_at("resp_a", start + Duration::from_secs(59))
                .is_some()
        );
        assert!(
            store
                .get_at("resp_a", start + Duration::from_secs(60))
                .is_none()
        );
        assert_eq!(store.len(), 0, "expired entry is dropped on access");
    }

    #[test]
    fn insert_purges_expired_entries() {
        let store = ResponseStore::with_limits(8, Duration::from_secs(60));
        let start = Instant::now();
        store.insert_at(stored("resp_old"), start);
        store.insert_at(stored("resp_new"), start + Duration::from_secs(120));

        assert_eq!(store.len(), 1);
        assert!(
            store
                .get_at("resp_new", start + Duration::from_secs(121))
                .is_some()
        );
    }

    #[test]
    fn remove_deletes_and_reports_presence() {
        let store = ResponseStore::default();
        store.insert(stored("resp_a"));
        assert!(store.remove("resp_a"));
        assert!(!store.remove("resp_a"));
        assert!(store.get("resp_a").is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn reinserting_an_id_replaces_it_without_growing() {
        let store = ResponseStore::with_limits(2, Duration::from_secs(3600));
        store.insert(stored("resp_a"));
        store.insert(stored("resp_a"));
        assert_eq!(store.len(), 1);
    }
}
