//! Bounded in-memory store backing `previous_response_id`, GET and DELETE.

use std::collections::HashMap;
use std::io;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::types::responses::{InputContent, InputItem, ResponseObject};

/// Entry cap.
///
/// Each entry holds the whole conversation up to that turn, so long agent
/// sessions are the memory driver; 256 keeps a busy local server in the tens
/// of megabytes while covering many concurrent sessions.
pub const MAX_STORED_RESPONSES: usize = 256;

/// Default memory for all stored entries together.
pub const DEFAULT_STORE_BYTES: usize = 256 * 1024 * 1024;

/// Default size of the largest single entry.
pub const DEFAULT_ENTRY_BYTES: usize = 10 * 1024 * 1024;

/// Entries expire this long after being stored. Long enough for an agent to
/// pause between turns, short enough that abandoned sessions free their memory.
pub const STORED_RESPONSE_TTL: Duration = Duration::from_hours(1);

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
    /// Serialized size, measured once on insert.
    bytes: usize,
}

/// The entry is above the size limit of one entry, so it was not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryTooLarge {
    pub bytes: usize,
    pub limit: usize,
}

/// Counts the bytes that a serializer writes and keeps none of them.
struct CountingWriter(usize);

impl io::Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0 += buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl StoredResponse {
    /// The JSON size of the response plus the text of the conversation, a cheap and stable proxy
    /// for the memory of the entry. Each conversation item counts a fixed overhead too.
    fn estimated_bytes(&self) -> usize {
        let mut counter = CountingWriter(0);
        let _ = serde_json::to_writer(&mut counter, &self.response);
        counter.0
            + self
                .conversation
                .iter()
                .map(input_item_bytes)
                .sum::<usize>()
    }
}

const ITEM_OVERHEAD_BYTES: usize = 64;

fn input_item_bytes(item: &InputItem) -> usize {
    ITEM_OVERHEAD_BYTES
        + match item {
            InputItem::Message { role, content } => {
                role.len()
                    + content
                        .iter()
                        .map(|part| match part {
                            InputContent::Text(text) | InputContent::ImageUrl(text) => text.len(),
                        })
                        .sum::<usize>()
            }
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => call_id.len() + name.len() + arguments.len(),
            InputItem::FunctionCallOutput { call_id, output } => call_id.len() + output.len(),
        }
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
    budget_bytes: usize,
    entry_limit_bytes: usize,
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
            budget_bytes: DEFAULT_STORE_BYTES,
            entry_limit_bytes: DEFAULT_ENTRY_BYTES,
        }
    }

    /// Set the memory for all entries together and the largest single entry.
    #[must_use]
    pub const fn with_byte_limits(mut self, budget_bytes: usize, entry_limit_bytes: usize) -> Self {
        self.budget_bytes = budget_bytes;
        self.entry_limit_bytes = entry_limit_bytes;
        self
    }

    /// Store the entry, dropping the least recently used ones while the count or the total
    /// size is over its limit. An entry above the limit for one entry, or above the whole
    /// budget, is refused and the store stays as it was.
    pub fn insert(&self, stored: StoredResponse) -> Result<(), EntryTooLarge> {
        self.insert_at(stored, Instant::now())
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

    fn insert_at(&self, stored: StoredResponse, now: Instant) -> Result<(), EntryTooLarge> {
        let bytes = stored.estimated_bytes();
        let limit = self.entry_limit_bytes.min(self.budget_bytes);
        if bytes > limit {
            return Err(EntryTooLarge { bytes, limit });
        }
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
                bytes,
            },
        );
        while inner.entries.len() > self.capacity
            || inner
                .entries
                .values()
                .map(|entry| entry.bytes)
                .sum::<usize>()
                > self.budget_bytes
        {
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
        Ok(())
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
        store.insert(stored("resp_a")).unwrap();
        assert_eq!(store.get("resp_a").unwrap().response.id, "resp_a");
        assert!(store.get("resp_missing").is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn evicts_least_recently_used_beyond_capacity() {
        let store = ResponseStore::with_limits(2, Duration::from_hours(1));
        store.insert(stored("resp_a")).unwrap();
        store.insert(stored("resp_b")).unwrap();
        assert!(store.get("resp_a").is_some());
        store.insert(stored("resp_c")).unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.get("resp_a").is_some());
        assert!(store.get("resp_b").is_none(), "b was least recently used");
        assert!(store.get("resp_c").is_some());
    }

    #[test]
    fn entries_expire_after_the_ttl() {
        let store = ResponseStore::with_limits(8, Duration::from_mins(1));
        let start = Instant::now();
        store.insert_at(stored("resp_a"), start).unwrap();

        assert!(
            store
                .get_at("resp_a", start + Duration::from_secs(59))
                .is_some()
        );
        assert!(
            store
                .get_at("resp_a", start + Duration::from_mins(1))
                .is_none()
        );
        assert_eq!(store.len(), 0, "expired entry is dropped on access");
    }

    #[test]
    fn insert_purges_expired_entries() {
        let store = ResponseStore::with_limits(8, Duration::from_mins(1));
        let start = Instant::now();
        store.insert_at(stored("resp_old"), start).unwrap();
        store
            .insert_at(stored("resp_new"), start + Duration::from_mins(2))
            .unwrap();

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
        store.insert(stored("resp_a")).unwrap();
        assert!(store.remove("resp_a"));
        assert!(!store.remove("resp_a"));
        assert!(store.get("resp_a").is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn reinserting_an_id_replaces_it_without_growing() {
        let store = ResponseStore::with_limits(2, Duration::from_hours(1));
        store.insert(stored("resp_a")).unwrap();
        store.insert(stored("resp_a")).unwrap();
        assert_eq!(store.len(), 1);
    }

    /// An entry whose conversation holds a text item of about `bytes` bytes.
    fn sized(id: &str, bytes: usize) -> StoredResponse {
        let mut entry = stored(id);
        entry.conversation = vec![InputItem::Message {
            role: "user".to_owned(),
            content: vec![InputContent::Text("x".repeat(bytes))],
        }];
        entry
    }

    #[test]
    fn the_size_of_an_entry_follows_its_content() {
        let small = sized("resp_a", 10).estimated_bytes();
        let large = sized("resp_a", 10_000).estimated_bytes();
        assert_eq!(large - small, 9_990);
    }

    #[test]
    fn the_byte_budget_evicts_the_least_recently_used_entries() {
        let each = sized("resp_a", 1_000).estimated_bytes();
        let store = ResponseStore::with_limits(100, Duration::from_hours(1))
            .with_byte_limits(each * 2 + each / 2, each * 2);
        store.insert(sized("resp_a", 1_000)).unwrap();
        store.insert(sized("resp_b", 1_000)).unwrap();
        assert!(store.get("resp_a").is_some());
        store.insert(sized("resp_c", 1_000)).unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.get("resp_a").is_some());
        assert!(store.get("resp_b").is_none(), "b was least recently used");
        assert!(store.get("resp_c").is_some());
    }

    #[test]
    fn many_small_entries_cannot_pass_the_byte_budget() {
        let each = sized("resp_000", 500).estimated_bytes();
        let store = ResponseStore::with_limits(1_000, Duration::from_hours(1))
            .with_byte_limits(each * 10, each * 10);
        for index in 0..200 {
            store
                .insert(sized(&format!("resp_{index:03}"), 500))
                .unwrap();
        }
        assert!(store.len() <= 10, "{}", store.len());
        assert!(store.get("resp_199").is_some());
    }

    #[test]
    fn an_entry_above_the_limit_is_refused_and_evicts_nothing() {
        let each = sized("resp_a", 1_000).estimated_bytes();
        let store = ResponseStore::with_limits(100, Duration::from_hours(1))
            .with_byte_limits(each * 10, each * 2);
        store.insert(sized("resp_a", 1_000)).unwrap();

        let refused = store.insert(sized("resp_big", 5_000)).unwrap_err();
        assert_eq!(refused.limit, each * 2);
        assert!(refused.bytes > refused.limit);
        assert!(store.get("resp_big").is_none());
        assert!(store.get("resp_a").is_some());
    }

    #[test]
    fn an_entry_above_the_whole_budget_is_refused_even_under_the_entry_limit() {
        let store = ResponseStore::with_limits(100, Duration::from_hours(1))
            .with_byte_limits(500, 1_000_000);
        assert!(store.insert(sized("resp_a", 1_000)).is_err());
        assert!(store.is_empty());
    }
}
