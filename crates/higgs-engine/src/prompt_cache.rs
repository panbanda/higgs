use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use higgs_models::AnyCache;

/// Minimum prefix length (in tokens) to consider a cache match useful.
const MIN_PREFIX_LEN: usize = 16;

/// KV cache state stored at a radix tree node.
struct CachedState {
    cache: AnyCache,
    /// Interior mutability so `find_longest_prefix` can update access time
    /// through a shared reference during tree traversal.
    last_accessed: Cell<Instant>,
}

/// Node in the compressed radix trie.
///
/// Each node stores the token chunk along its incoming edge. The root has an
/// empty edge. Children are keyed by the first token of their edge.
struct RadixNode {
    edge: Vec<u32>,
    cached: Option<CachedState>,
    children: HashMap<u32, Self>,
}

/// Result of a prefix cache lookup.
pub struct PrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Cloned cache state for the matched prefix.
    pub cache: AnyCache,
}

/// Radix-tree prefix KV cache with LRU eviction.
///
/// Stores KV states for token prefixes in a compressed trie. Common prefixes
/// are shared in the tree structure, and lookup is `O(query_length)` rather
/// than `O(entries * prefix_length)` as with a flat `HashMap` approach.
pub struct PrefixCache {
    root: RadixNode,
    num_cached: usize,
    max_cached: usize,
}

impl RadixNode {
    fn empty() -> Self {
        Self {
            edge: Vec::new(),
            cached: None,
            children: HashMap::new(),
        }
    }

    fn leaf(edge: Vec<u32>, cache: AnyCache) -> Self {
        Self {
            edge,
            cached: Some(CachedState {
                cache,
                last_accessed: Cell::new(Instant::now()),
            }),
            children: HashMap::new(),
        }
    }

    /// Walk the trie, returning the deepest node whose cache is at
    /// depth >= `MIN_PREFIX_LEN`.
    fn find_deepest_match(&self, tokens: &[u32], depth: usize) -> Option<(usize, &CachedState)> {
        let mut best = self
            .cached
            .as_ref()
            .filter(|_| depth >= MIN_PREFIX_LEN)
            .map(|cs| (depth, cs));

        let Some(&next_token) = tokens.get(depth) else {
            return best;
        };

        let Some(child) = self.children.get(&next_token) else {
            return best;
        };

        // Count how far the child's edge matches the remaining query tokens
        let remaining = tokens.get(depth..).unwrap_or_default();
        let common = child
            .edge
            .iter()
            .zip(remaining.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common == child.edge.len() {
            if let Some(deeper) = child.find_deepest_match(tokens, depth + common) {
                best = Some(deeper);
            }
        }

        best
    }

    /// Return the oldest `last_accessed` time among all cached nodes in this subtree.
    fn oldest_cached_time(&self) -> Option<Instant> {
        let mut oldest: Option<Instant> = self.cached.as_ref().map(|cs| cs.last_accessed.get());

        for child in self.children.values() {
            if let Some(child_time) = child.oldest_cached_time() {
                oldest = Some(oldest.map_or(child_time, |o| o.min(child_time)));
            }
        }

        oldest
    }

    /// Remove the first cached entry matching `target` time. Returns `true` if removed.
    fn remove_cached_with_time(&mut self, target: Instant) -> bool {
        if self
            .cached
            .as_ref()
            .is_some_and(|cs| cs.last_accessed.get() == target)
        {
            self.cached = None;
            return true;
        }

        for child in self.children.values_mut() {
            if child.remove_cached_with_time(target) {
                return true;
            }
        }

        false
    }

    /// Remove empty leaf nodes and compress single-child routing nodes.
    fn prune(&mut self) {
        for child in self.children.values_mut() {
            child.prune();
        }
        self.children
            .retain(|_, child| child.cached.is_some() || !child.children.is_empty());

        // Compress: if this node has no cache and exactly one child,
        // merge the child into this node. Skip root (empty edge) because
        // find_deepest_match does not check the root's edge.
        if self.cached.is_none() && self.children.len() == 1 && !self.edge.is_empty() {
            let Some(key) = self.children.keys().next().copied() else {
                return;
            };
            let Some(mut only_child) = self.children.remove(&key) else {
                return;
            };
            self.edge.append(&mut only_child.edge);
            self.cached = only_child.cached;
            self.children = only_child.children;
        }
    }
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            root: RadixNode::empty(),
            num_cached: 0,
            max_cached: max_entries,
        }
    }

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Returns `None` if no prefix matches or if the match is too short to
    /// be worth reusing (less than `MIN_PREFIX_LEN` tokens).
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        self.root
            .find_deepest_match(tokens, 0)
            .map(|(prefix_len, cs)| {
                cs.last_accessed.set(Instant::now());
                PrefixMatch {
                    prefix_len,
                    cache: cs.cache.clone(),
                }
            })
    }

    /// Store a prefix and its cache state.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: AnyCache) {
        if self.max_cached == 0 {
            return;
        }

        let added = Self::insert(&mut self.root, prefix_tokens, 0, cache);

        if added {
            self.num_cached += 1;

            while self.num_cached > self.max_cached {
                self.evict_lru();
            }
        }
    }

    /// Insert a prefix into the radix tree. Returns `true` if a new cache slot
    /// was created (as opposed to overwriting an existing one).
    fn insert(node: &mut RadixNode, tokens: &[u32], pos: usize, cache: AnyCache) -> bool {
        if pos >= tokens.len() {
            let is_new = node.cached.is_none();
            node.cached = Some(CachedState {
                cache,
                last_accessed: Cell::new(Instant::now()),
            });
            return is_new;
        }

        let Some(&next_token) = tokens.get(pos) else {
            return false;
        };

        if node.children.contains_key(&next_token) {
            let Some(child) = node.children.get(&next_token) else {
                return false;
            };

            let remaining = tokens.get(pos..).unwrap_or_default();
            let common = child
                .edge
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == child.edge.len() {
                let Some(child_mut) = node.children.get_mut(&next_token) else {
                    return false;
                };
                return Self::insert(child_mut, tokens, pos + common, cache);
            }

            // Partial match -- split the edge at `common`
            let Some(mut old_child) = node.children.remove(&next_token) else {
                return false;
            };

            let common_edge = old_child.edge.get(..common).unwrap_or_default().to_vec();
            let leftover_edge = old_child.edge.get(common..).unwrap_or_default().to_vec();

            let Some(&leftover_key) = leftover_edge.first() else {
                return false;
            };
            old_child.edge = leftover_edge;

            let mut split = RadixNode {
                edge: common_edge,
                cached: None,
                children: HashMap::new(),
            };
            split.children.insert(leftover_key, old_child);

            if pos + common >= tokens.len() {
                split.cached = Some(CachedState {
                    cache,
                    last_accessed: Cell::new(Instant::now()),
                });
                node.children.insert(next_token, split);
                return true;
            }

            let new_edge = tokens.get(pos + common..).unwrap_or_default().to_vec();
            let Some(&new_key) = new_edge.first() else {
                node.children.insert(next_token, split);
                return false;
            };
            let new_leaf = RadixNode::leaf(new_edge, cache);
            split.children.insert(new_key, new_leaf);

            node.children.insert(next_token, split);
            return true;
        }

        // No matching child -- create a new leaf
        let new_edge = tokens.get(pos..).unwrap_or_default().to_vec();
        let new_leaf = RadixNode::leaf(new_edge, cache);
        node.children.insert(next_token, new_leaf);
        true
    }

    /// Evict the least recently used cached entry.
    fn evict_lru(&mut self) {
        if let Some(oldest) = self.root.oldest_cached_time() {
            if self.root.remove_cached_with_time(oldest) {
                self.num_cached -= 1;
                self.root.prune();
            }
        }
    }

    /// Number of cached entries.
    pub const fn len(&self) -> usize {
        self.num_cached
    }

    /// Whether the cache is empty.
    pub const fn is_empty(&self) -> bool {
        self.num_cached == 0
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.root = RadixNode::empty();
        self.num_cached = 0;
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use higgs_models::AnyCache;
    use higgs_models::cache::SteppingKeyValueCache;

    fn make_dummy_cache(num_layers: usize) -> AnyCache {
        let kv: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| Some(SteppingKeyValueCache::new()))
            .collect();
        AnyCache::KV(kv)
    }

    fn cache_layer_count(cache: &AnyCache) -> usize {
        match cache {
            AnyCache::KV(v) => v.len(),
            AnyCache::Hybrid(v) => v.len(),
        }
    }

    #[test]
    fn test_empty_cache_returns_none() {
        let mut cache = PrefixCache::new(10);
        assert!(cache.find_longest_prefix(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_store_and_find_exact_match() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        let kv = make_dummy_cache(4);

        cache.store(&prefix, kv);
        assert_eq!(cache.len(), 1);

        // Search with the same prefix followed by more tokens
        let mut query: Vec<u32> = prefix;
        query.extend_from_slice(&[100, 101, 102]);

        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.prefix_len, 32);
        assert_eq!(cache_layer_count(&matched.cache), 4);
    }

    fn assert_no_prefix_match(
        stored_range: std::ops::Range<u32>,
        query_range: std::ops::Range<u32>,
    ) {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = stored_range.collect();
        cache.store(&prefix, make_dummy_cache(4));

        let query: Vec<u32> = query_range.collect();
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_no_match_for_different_prefix() {
        assert_no_prefix_match(0..32, 100..132);
    }

    #[test]
    fn test_prefix_too_short_ignored() {
        assert_no_prefix_match(0..8, 0..32);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PrefixCache::new(2);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();
        let prefix_c: Vec<u32> = (200..232).collect();

        cache.store(&prefix_a, make_dummy_cache(4));
        cache.store(&prefix_b, make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        // Adding a third should evict the LRU (prefix_a, since it was stored first)
        cache.store(&prefix_c, make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        // prefix_c should still be found
        let mut query_c: Vec<u32> = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }

    #[test]
    fn test_zero_capacity_never_stores() {
        let mut cache = PrefixCache::new(0);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(&prefix, make_dummy_cache(4));
        assert!(cache.is_empty());

        let mut query = prefix;
        query.push(999);
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..32).collect();
        cache.store(&prefix, make_dummy_cache(4));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_longest_prefix_wins() {
        let mut cache = PrefixCache::new(10);

        let short_prefix: Vec<u32> = (0..20).collect();
        cache.store(&short_prefix, make_dummy_cache(4));

        let long_prefix: Vec<u32> = (0..50).collect();
        cache.store(&long_prefix, make_dummy_cache(4));

        let query: Vec<u32> = (0..64).collect();
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().prefix_len, 50);
    }

    fn assert_overwrite_uses_latest(capacity: usize, first_layers: usize, second_layers: usize) {
        let mut cache = PrefixCache::new(capacity);
        let prefix: Vec<u32> = (0..32).collect();

        cache.store(&prefix, make_dummy_cache(first_layers));
        assert_eq!(cache.len(), 1);

        cache.store(&prefix, make_dummy_cache(second_layers));
        assert_eq!(cache.len(), 1);

        let mut query = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(cache_layer_count(&result.cache), second_layers);
    }

    #[test]
    fn test_store_same_tokens_overwrites() {
        assert_overwrite_uses_latest(10, 2, 8);
    }

    #[test]
    fn test_overwrite_does_not_trigger_eviction() {
        assert_overwrite_uses_latest(1, 2, 4);
    }

    #[test]
    fn test_collision_guard_skips_different_tokens_same_hash() {
        let mut cache = PrefixCache::new(10);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();

        cache.store(&prefix_a, make_dummy_cache(2));
        cache.store(&prefix_b, make_dummy_cache(4));

        assert_eq!(cache.len(), 2);

        let mut query_a = prefix_a;
        query_a.push(999);
        let result_a = cache.find_longest_prefix(&query_a).unwrap();
        assert_eq!(cache_layer_count(&result_a.cache), 2);

        let mut query_b = prefix_b;
        query_b.push(999);
        let result_b = cache.find_longest_prefix(&query_b).unwrap();
        assert_eq!(cache_layer_count(&result_b.cache), 4);
    }

    #[test]
    fn test_eviction_only_when_new_key() {
        let mut cache = PrefixCache::new(2);

        let prefix_a: Vec<u32> = (0..32).collect();
        let prefix_b: Vec<u32> = (100..132).collect();
        let prefix_c: Vec<u32> = (200..232).collect();

        cache.store(&prefix_a, make_dummy_cache(1));
        cache.store(&prefix_b, make_dummy_cache(2));
        assert_eq!(cache.len(), 2);

        // Overwrite prefix_a: no eviction, still 2 entries
        cache.store(&prefix_a, make_dummy_cache(3));
        assert_eq!(cache.len(), 2);

        let mut query_a = prefix_a;
        query_a.push(999);
        assert!(cache.find_longest_prefix(&query_a).is_some());

        let mut query_b = prefix_b;
        query_b.push(999);
        assert!(cache.find_longest_prefix(&query_b).is_some());

        // New prefix_c triggers eviction
        cache.store(&prefix_c, make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        let mut query_c = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }

    #[test]
    fn test_find_longest_prefix_exactly_at_minimum_length() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..16).collect();
        cache.store(&prefix, make_dummy_cache(4));

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        assert_eq!(result.unwrap().prefix_len, 16);
    }

    #[test]
    fn test_find_longest_prefix_just_under_minimum_length() {
        let mut cache = PrefixCache::new(10);
        let prefix: Vec<u32> = (0..15).collect();
        cache.store(&prefix, make_dummy_cache(4));
        assert_eq!(cache.len(), 1); // stored, but won't match

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_none());
    }

    #[test]
    fn test_multiple_overlapping_prefixes_correct_longest() {
        let mut cache = PrefixCache::new(10);

        let short: Vec<u32> = (0..16).collect();
        let medium: Vec<u32> = (0..32).collect();
        let long: Vec<u32> = (0..64).collect();

        cache.store(&short, make_dummy_cache(1));
        cache.store(&medium, make_dummy_cache(2));
        cache.store(&long, make_dummy_cache(3));
        assert_eq!(cache.len(), 3);

        let query: Vec<u32> = (0..100).collect();
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(result.prefix_len, 64);
        assert_eq!(cache_layer_count(&result.cache), 3);

        let shorter_query: Vec<u32> = (0..40).collect();
        let result2 = cache.find_longest_prefix(&shorter_query).unwrap();
        assert_eq!(result2.prefix_len, 32);
        assert_eq!(cache_layer_count(&result2.cache), 2);

        let shortest_query: Vec<u32> = (0..20).collect();
        let result3 = cache.find_longest_prefix(&shortest_query).unwrap();
        assert_eq!(result3.prefix_len, 16);
        assert_eq!(cache_layer_count(&result3.cache), 1);
    }

    #[test]
    fn test_len_and_is_empty_after_operations() {
        let mut cache = PrefixCache::new(5);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let p1: Vec<u32> = (0..32).collect();
        cache.store(&p1, make_dummy_cache(1));
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        let p2: Vec<u32> = (100..132).collect();
        cache.store(&p2, make_dummy_cache(1));
        assert_eq!(cache.len(), 2);

        let p3: Vec<u32> = (200..232).collect();
        cache.store(&p3, make_dummy_cache(1));
        assert_eq!(cache.len(), 3);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // --- Radix tree specific tests ---

    #[test]
    fn test_shared_prefix_partial_match() {
        let mut cache = PrefixCache::new(10);

        let system_prefix: Vec<u32> = (0..64).collect();
        cache.store(&system_prefix, make_dummy_cache(2));

        let full_prompt: Vec<u32> = (0..128).collect();
        cache.store(&full_prompt, make_dummy_cache(4));
        assert_eq!(cache.len(), 2);

        // Query with same system prefix but different user message
        let mut different_suffix: Vec<u32> = (0..64).collect();
        different_suffix.extend(500..564);
        let result = cache.find_longest_prefix(&different_suffix).unwrap();
        assert_eq!(result.prefix_len, 64);
        assert_eq!(cache_layer_count(&result.cache), 2);
    }

    #[test]
    fn test_diverging_prefixes_both_stored() {
        let mut cache = PrefixCache::new(10);

        let mut prefix_a: Vec<u32> = (0..20).collect();
        prefix_a.extend(100..120);
        cache.store(&prefix_a, make_dummy_cache(1));

        let mut prefix_b: Vec<u32> = (0..20).collect();
        prefix_b.extend(200..220);
        cache.store(&prefix_b, make_dummy_cache(2));
        assert_eq!(cache.len(), 2);

        let mut query_a = prefix_a;
        query_a.push(999);
        let result_a = cache.find_longest_prefix(&query_a).unwrap();
        assert_eq!(result_a.prefix_len, 40);
        assert_eq!(cache_layer_count(&result_a.cache), 1);

        let mut query_b = prefix_b;
        query_b.push(999);
        let result_b = cache.find_longest_prefix(&query_b).unwrap();
        assert_eq!(result_b.prefix_len, 40);
        assert_eq!(cache_layer_count(&result_b.cache), 2);
    }

    #[test]
    fn test_edge_split_preserves_existing_cache() {
        let mut cache = PrefixCache::new(10);

        let long: Vec<u32> = (0..50).collect();
        cache.store(&long, make_dummy_cache(3));
        assert_eq!(cache.len(), 1);

        // Shorter prefix forces an edge split
        let short: Vec<u32> = (0..25).collect();
        cache.store(&short, make_dummy_cache(1));
        assert_eq!(cache.len(), 2);

        // The long prefix should still be found
        let mut query_long = long;
        query_long.push(999);
        let result = cache.find_longest_prefix(&query_long).unwrap();
        assert_eq!(result.prefix_len, 50);
        assert_eq!(cache_layer_count(&result.cache), 3);
    }
}
