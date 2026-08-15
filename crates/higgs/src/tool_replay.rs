//! Exact tool-call replay for preserving prompt-prefix cache identity.
//!
//! Clients return normalized `OpenAI` tool calls, while models often generated a
//! differently formatted JSON payload. A replay entry is used only after the
//! name and parsed JSON arguments match, so this module never changes the
//! semantic content presented to the model; it restores only that model's own
//! serialization.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use higgs_engine::tool_parser::ParsedToolCall;

const DEFAULT_CAPACITY: usize = 256;
const MAX_ENTRY_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone)]
struct ReplayEntry {
    name: String,
    arguments: serde_json::Value,
    raw_text: String,
}

/// FIFO-bounded cache of raw model output, scoped to one model.
#[derive(Debug)]
pub struct ToolReplayStore {
    capacity: usize,
    entries: HashMap<String, ReplayEntry>,
    order: VecDeque<String>,
}

impl ToolReplayStore {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, id: String, call: &ParsedToolCall) {
        if self.capacity == 0 || call.raw_text.len() > MAX_ENTRY_BYTES {
            return;
        }
        if self.entries.remove(&id).is_some() {
            self.order.retain(|existing| existing != &id);
        }
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(id.clone());
        self.entries.insert(
            id,
            ReplayEntry {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                raw_text: call.raw_text.clone(),
            },
        );
    }

    fn matching_raw(&self, id: &str, name: &str, arguments: &serde_json::Value) -> Option<String> {
        let entry = self.entries.get(id)?;
        (entry.name == name && entry.arguments == *arguments).then(|| entry.raw_text.clone())
    }
}

/// Thread-safe per-model stores plus replay hit/miss counters.
#[derive(Debug)]
pub struct ToolReplayRegistry {
    capacity: usize,
    stores: Mutex<HashMap<String, ToolReplayStore>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ToolReplayRegistry {
    #[must_use]
    pub fn from_env() -> Self {
        let capacity = std::env::var("HIGGS_TOOL_REPLAY_CAPACITY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CAPACITY);
        Self::new(capacity)
    }

    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            stores: Mutex::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn insert(&self, model: &str, id: String, call: &ParsedToolCall) {
        self.stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(model.to_owned())
            .or_insert_with(|| ToolReplayStore::new(self.capacity))
            .insert(id, call);
    }

    /// Return replay text only when the normalized client call is semantically
    /// identical to the captured model call.
    pub fn matching_raw(
        &self,
        model: &str,
        id: &str,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Option<String> {
        let found = self
            .stores
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(model)
            .and_then(|store| store.matching_raw(id, name, arguments));
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    #[must_use]
    pub fn counters(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(raw_text: &str) -> ParsedToolCall {
        ParsedToolCall {
            name: "weather".to_owned(),
            arguments: serde_json::json!({"city": "Denver"}),
            raw_text: raw_text.to_owned(),
        }
    }

    #[test]
    fn evicts_oldest_and_skips_oversized_entries() {
        let mut store = ToolReplayStore::new(1);
        store.insert("first".to_owned(), &call("one"));
        store.insert("second".to_owned(), &call("two"));
        assert!(
            store
                .matching_raw("first", "weather", &serde_json::json!({"city":"Denver"}))
                .is_none()
        );
        assert_eq!(
            store.matching_raw("second", "weather", &serde_json::json!({"city":"Denver"})),
            Some("two".to_owned())
        );

        store.insert("large".to_owned(), &call(&"x".repeat(MAX_ENTRY_BYTES + 1)));
        assert!(
            store
                .matching_raw("large", "weather", &serde_json::json!({"city":"Denver"}))
                .is_none()
        );
    }

    #[test]
    fn zero_capacity_disables_storage() {
        let mut store = ToolReplayStore::new(0);
        store.insert("call".to_owned(), &call("raw"));
        assert!(
            store
                .matching_raw("call", "weather", &serde_json::json!({"city":"Denver"}))
                .is_none()
        );
    }
}
