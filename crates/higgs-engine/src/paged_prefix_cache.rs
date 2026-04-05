use std::cell::Cell;
use std::collections::HashMap;
use std::time::Instant;

use std::sync::Arc;

use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache, slice_axis1, slice_axis2};
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::turboquant::TurboQuantContext;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;

/// Default block size in tokens for paged caching.
pub const DEFAULT_BLOCK_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Block data structures
// ---------------------------------------------------------------------------

/// Per-layer, per-block: K and V array slices with shape `[1, H, block_size, D]`.
///
/// MLX arrays use internal ref-counting, so cloning blocks shares the
/// underlying data without copying.
#[derive(Debug, Clone)]
struct KvBlock {
    keys: Array,
    values: Array,
}

/// GDN state snapshot at a block boundary (Hybrid models only).
#[derive(Debug, Clone)]
struct GdnSnapshot {
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    conv_pos: i32,
    offset: i32,
}

/// Per-layer block for TurboQuant KV cache.
///
/// Each block holds the 5 quantized arrays for `block_size` tokens:
/// key/value codes (packed u32), norms, gammas.
#[derive(Debug, Clone)]
struct TqBlock {
    key_codes: Array,
    key_norms: Array,
    key_gammas: Array,
    value_codes: Array,
    value_norms: Array,
}

/// Per-layer cached data.
#[derive(Debug, Clone)]
enum CachedLayerData {
    /// Attention layer: sequence of dense K/V blocks.
    Kv(Vec<KvBlock>),
    /// Attention layer: sequence of TurboQuant blocks.
    TurboQuantKv(Vec<TqBlock>),
    /// GDN/SSM layer: state snapshot at block boundary.
    Gdn(GdnSnapshot),
    /// Layer had no cache data.
    Empty,
}

// ---------------------------------------------------------------------------
// Cache entry stored in radix trie
// ---------------------------------------------------------------------------

/// What's stored at each radix trie node.
enum CachedData {
    /// Block-paged cache -- shared block references (dense KV).
    Paged {
        layers: Vec<CachedLayerData>,
        total_tokens: usize,
        is_hybrid: bool,
    },
    /// Block-paged TurboQuant cache with shared quantization context.
    TurboQuantPaged {
        layers: Vec<CachedLayerData>,
        context: Arc<TurboQuantContext>,
        total_tokens: usize,
        is_hybrid: bool,
    },
    /// Full clone fallback (cache too short for paging).
    Cloned(AnyCache),
}

struct CachedState {
    data: CachedData,
    last_accessed: Cell<Instant>,
}

// ---------------------------------------------------------------------------
// Radix trie
// ---------------------------------------------------------------------------

struct RadixNode {
    edge: Vec<u32>,
    cached: Option<CachedState>,
    children: HashMap<u32, Self>,
}

/// Result of a paged prefix cache lookup.
pub struct PagedPrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Materialized cache state for the matched prefix.
    pub cache: AnyCache,
}

/// Paged prefix cache with block-level storage and LRU eviction.
///
/// Instead of cloning entire `AnyCache` objects (which pins a full KV slab per
/// layer per entry), this cache stores block-sized array slices. MLX arrays use
/// internal ref-counting, so blocks from shared prefixes only store data once.
/// On lookup, blocks are gathered into a contiguous cache via
/// `concatenate_axis` (one-time cost per request).
pub struct PagedPrefixCache {
    root: RadixNode,
    num_cached: usize,
    max_cached: usize,
    block_size: usize,
}

// ---------------------------------------------------------------------------
// RadixNode impl (mirrors prompt_cache.rs but stores CachedState)
// ---------------------------------------------------------------------------

impl RadixNode {
    fn empty() -> Self {
        Self {
            edge: Vec::new(),
            cached: None,
            children: HashMap::new(),
        }
    }

    fn leaf(edge: Vec<u32>, data: CachedData) -> Self {
        Self {
            edge,
            cached: Some(CachedState {
                data,
                last_accessed: Cell::new(Instant::now()),
            }),
            children: HashMap::new(),
        }
    }

    fn find_deepest_match(
        &self,
        tokens: &[u32],
        depth: usize,
        min_prefix: usize,
    ) -> Option<(usize, &CachedState)> {
        let mut best = self
            .cached
            .as_ref()
            .filter(|cs| {
                // Cloned entries are valid at any depth > 0 (no block alignment needed).
                // Paged entries require at least min_prefix tokens for block alignment.
                match &cs.data {
                    CachedData::Cloned(_) => depth > 0,
                    _ => depth >= min_prefix,
                }
            })
            .map(|cs| (depth, cs));

        let Some(&next_token) = tokens.get(depth) else {
            return best;
        };

        let Some(child) = self.children.get(&next_token) else {
            return best;
        };

        let remaining = tokens.get(depth..).unwrap_or_default();
        let common = child
            .edge
            .iter()
            .zip(remaining.iter())
            .take_while(|(a, b)| a == b)
            .count();

        if common == child.edge.len() {
            if let Some(deeper) = child.find_deepest_match(tokens, depth + common, min_prefix) {
                best = Some(deeper);
            }
        }

        best
    }

    fn oldest_cached_time(&self) -> Option<Instant> {
        let mut oldest: Option<Instant> = self.cached.as_ref().map(|cs| cs.last_accessed.get());

        for child in self.children.values() {
            if let Some(child_time) = child.oldest_cached_time() {
                oldest = Some(oldest.map_or(child_time, |o| o.min(child_time)));
            }
        }

        oldest
    }

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

    fn prune(&mut self) {
        for child in self.children.values_mut() {
            child.prune();
        }
        self.children
            .retain(|_, child| child.cached.is_some() || !child.children.is_empty());

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

// ---------------------------------------------------------------------------
// PagedPrefixCache impl
// ---------------------------------------------------------------------------

impl PagedPrefixCache {
    pub fn new(max_entries: usize, block_size: usize) -> Self {
        Self {
            root: RadixNode::empty(),
            num_cached: 0,
            max_cached: max_entries,
            block_size,
        }
    }

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Returns `None` if no prefix matches or if the match is shorter than one
    /// block. On hit, blocks are gathered into a contiguous `AnyCache`.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<PagedPrefixMatch> {
        self.root
            .find_deepest_match(tokens, 0, self.block_size)
            .and_then(|(prefix_len, cs)| match materialize(&cs.data) {
                Ok(cache) => {
                    tracing::debug!(prefix_len, "Prefix cache hit");
                    cs.last_accessed.set(Instant::now());
                    Some(PagedPrefixMatch { prefix_len, cache })
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Prefix cache materialize failed");
                    None
                }
            })
    }

    /// Store a prefix and its cache state as paged blocks.
    ///
    /// For dense KV caches, the K/V arrays are sliced into block-sized views
    /// (lazy, nearly free). For TurboQuant caches, a full clone fallback is
    /// used. Only block-aligned tokens are stored in the trie.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: &AnyCache) {
        if self.max_cached == 0 {
            return;
        }

        // TurboQuant caches with deferred quantization are stored as dense
        // blocks until TQ activates. Full TQ block paging is implemented in the
        // CachedData::TurboQuantPaged variant but requires the TQ arrays to be
        // populated (post-activation). For now, use clone fallback when TQ config
        // is set but arrays aren't yet quantized to avoid cache corruption.
        let data = match slice_into_blocks(cache, self.block_size) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to page cache, using clone fallback");
                CachedData::Cloned(cache.clone())
            }
        };

        let aligned_len = match &data {
            CachedData::Paged { total_tokens, .. }
            | CachedData::TurboQuantPaged { total_tokens, .. } => *total_tokens,
            CachedData::Cloned(_) => prefix_tokens.len(),
        };
        let tokens_to_store = prefix_tokens.get(..aligned_len).unwrap_or(prefix_tokens);

        let added = Self::insert(&mut self.root, tokens_to_store, 0, data);

        if added {
            self.num_cached += 1;
            while self.num_cached > self.max_cached {
                self.evict_lru();
            }
        }
    }

    fn insert(node: &mut RadixNode, tokens: &[u32], pos: usize, data: CachedData) -> bool {
        if pos >= tokens.len() {
            let is_new = node.cached.is_none();
            node.cached = Some(CachedState {
                data,
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
                return Self::insert(child_mut, tokens, pos + common, data);
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
                    data,
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
            let new_leaf = RadixNode::leaf(new_edge, data);
            split.children.insert(new_key, new_leaf);

            node.children.insert(next_token, split);
            return true;
        }

        // No matching child -- create a new leaf
        let new_edge = tokens.get(pos..).unwrap_or_default().to_vec();
        let new_leaf = RadixNode::leaf(new_edge, data);
        node.children.insert(next_token, new_leaf);
        true
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self.root.oldest_cached_time() {
            if self.root.remove_cached_with_time(oldest) {
                self.num_cached -= 1;
                self.root.prune();
            }
        }
    }

    pub const fn len(&self) -> usize {
        self.num_cached
    }

    pub const fn is_empty(&self) -> bool {
        self.num_cached == 0
    }

    pub fn clear(&mut self) {
        self.root = RadixNode::empty();
        self.num_cached = 0;
    }
}

// ---------------------------------------------------------------------------
// Slice & materialize helpers
// ---------------------------------------------------------------------------

/// Check if any layer in the cache uses TurboQuant.
fn is_turboquant(cache: &AnyCache) -> bool {
    match cache {
        AnyCache::KV(layers) => layers.iter().any(|l| {
            l.as_ref()
                .is_some_and(|c| c.kv_cache_config().is_turboquant())
        }),
        AnyCache::Hybrid(layers) => layers
            .iter()
            .any(|l| matches!(l, Some(LayerCache::KV(c)) if c.kv_cache_config().is_turboquant())),
    }
}

/// Get the KV offset from the first non-empty KV layer.
fn kv_offset(cache: &AnyCache) -> Option<i32> {
    match cache {
        AnyCache::KV(layers) => layers
            .iter()
            .find_map(|l| l.as_ref())
            .map(KeyValueCache::offset),
        AnyCache::Hybrid(layers) => layers.iter().find_map(|l| match l {
            Some(LayerCache::KV(c)) => Some(KeyValueCache::offset(c)),
            _ => None,
        }),
    }
}

/// Slice a cache into block-aligned paged data.
fn slice_into_blocks(cache: &AnyCache, block_size: usize) -> Result<CachedData, Exception> {
    // Hybrid caches (GDN+KV) can't be block-paged because GDN sequential state
    // doesn't align to block boundaries. The KV offset would mismatch the GDN
    // offset after materialization, producing corrupt attention. Use clone instead.
    if matches!(cache, AnyCache::Hybrid(_)) {
        return Ok(CachedData::Cloned(cache.clone()));
    }

    let offset = kv_offset(cache).unwrap_or(0);
    let num_blocks = offset as usize / block_size;
    if num_blocks == 0 {
        return Err(Exception::custom("Cache too short for paging"));
    }
    let total_tokens = num_blocks * block_size;
    let block_size_i32 =
        i32::try_from(block_size).map_err(|_| Exception::custom("block_size overflow"))?;

    // Slice KV layers as TQ blocks when actually quantized, dense otherwise.
    // Hybrid caches are handled by the early-return clone above.
    let (layers, tq_context) = match cache {
        AnyCache::KV(kv_layers) => {
            let mut ctx: Option<Arc<TurboQuantContext>> = None;
            let layers: Result<Vec<_>, _> = kv_layers
                .iter()
                .map(|layer_opt| {
                    let Some(kv) = layer_opt.as_ref() else {
                        return Ok(CachedLayerData::Empty);
                    };
                    if kv.is_quantized() {
                        if ctx.is_none() {
                            ctx = kv.turbo_arrays().map(|(c, ..)| Arc::clone(c));
                        }
                        slice_tq_layer(kv, num_blocks, block_size_i32)
                    } else {
                        slice_kv_layer(Some(kv), num_blocks, block_size_i32)
                    }
                })
                .collect();
            (layers?, ctx)
        }
        // Hybrid caches returned Cloned above; this arm is unreachable.
        AnyCache::Hybrid(_) => unreachable!("Hybrid caches use clone fallback"),
    };

    if let Some(context) = tq_context {
        Ok(CachedData::TurboQuantPaged {
            layers,
            context,
            total_tokens,
            is_hybrid: false,
        })
    } else {
        Ok(CachedData::Paged {
            layers,
            total_tokens,
            is_hybrid: false,
        })
    }
}

/// Slice a single TurboQuant KV layer into blocks along axis 1.
fn slice_tq_layer(
    kv: &SteppingKeyValueCache,
    num_blocks: usize,
    block_size: i32,
) -> Result<CachedLayerData, Exception> {
    let Some((_ctx, key_codes, key_norms, key_gammas, value_codes, value_norms)) =
        kv.turbo_arrays()
    else {
        return Ok(CachedLayerData::Empty);
    };

    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let start = i32::try_from(i)
            .map_err(|_| Exception::custom("block index overflow"))?
            .checked_mul(block_size)
            .ok_or_else(|| Exception::custom("block start overflow"))?;
        let end = start
            .checked_add(block_size)
            .ok_or_else(|| Exception::custom("block end overflow"))?;
        blocks.push(TqBlock {
            key_codes: slice_axis1(key_codes, start, end)?,
            key_norms: slice_axis1(key_norms, start, end)?,
            key_gammas: slice_axis1(key_gammas, start, end)?,
            value_codes: slice_axis1(value_codes, start, end)?,
            value_norms: slice_axis1(value_norms, start, end)?,
        });
    }

    Ok(CachedLayerData::TurboQuantKv(blocks))
}

/// Slice a single KV layer into blocks.
fn slice_kv_layer(
    kv: Option<&SteppingKeyValueCache>,
    num_blocks: usize,
    block_size: i32,
) -> Result<CachedLayerData, Exception> {
    let Some(kv) = kv else {
        return Ok(CachedLayerData::Empty);
    };
    let (Some(keys), Some(values)) = (kv.keys(), kv.values()) else {
        return Ok(CachedLayerData::Empty);
    };

    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let start = i32::try_from(i)
            .map_err(|_| Exception::custom("block index overflow"))?
            .checked_mul(block_size)
            .ok_or_else(|| Exception::custom("block start overflow"))?;
        let end = start
            .checked_add(block_size)
            .ok_or_else(|| Exception::custom("block end overflow"))?;
        let k = slice_axis2(keys, start, end)?;
        let v = slice_axis2(values, start, end)?;
        blocks.push(KvBlock { keys: k, values: v });
    }

    Ok(CachedLayerData::Kv(blocks))
}

/// Materialize block references into a contiguous `AnyCache`.
fn materialize(data: &CachedData) -> Result<AnyCache, Exception> {
    match data {
        CachedData::Cloned(cache) => Ok(cache.clone()),
        CachedData::Paged {
            layers, is_hybrid, ..
        } => {
            if *is_hybrid {
                materialize_hybrid(layers)
            } else {
                materialize_kv(layers)
            }
        }
        CachedData::TurboQuantPaged {
            layers,
            context,
            is_hybrid,
            ..
        } => {
            if *is_hybrid {
                materialize_tq_hybrid(layers, context)
            } else {
                materialize_tq_kv(layers, context)
            }
        }
    }
}

fn materialize_kv(layers: &[CachedLayerData]) -> Result<AnyCache, Exception> {
    let kv_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(Some),
            CachedLayerData::TurboQuantKv(_) => {
                Err(Exception::custom("TQ layer in non-TQ materialize"))
            }
            CachedLayerData::Empty => Ok(Some(SteppingKeyValueCache::new())),
            CachedLayerData::Gdn(_) => Err(Exception::custom("Unexpected GDN layer in KV cache")),
        })
        .collect();
    Ok(AnyCache::KV(kv_layers?))
}

fn materialize_tq_kv(
    layers: &[CachedLayerData],
    context: &Arc<TurboQuantContext>,
) -> Result<AnyCache, Exception> {
    let kv_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::TurboQuantKv(blocks) => gather_tq_blocks(blocks, context).map(Some),
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(Some),
            CachedLayerData::Empty => Ok(Some(SteppingKeyValueCache::new())),
            CachedLayerData::Gdn(_) => {
                Err(Exception::custom("Unexpected GDN layer in TQ KV cache"))
            }
        })
        .collect();
    Ok(AnyCache::KV(kv_layers?))
}

fn materialize_hybrid(layers: &[CachedLayerData]) -> Result<AnyCache, Exception> {
    let hybrid_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(|kv| Some(LayerCache::KV(kv))),
            CachedLayerData::TurboQuantKv(_) => {
                Err(Exception::custom("TQ layer in non-TQ hybrid materialize"))
            }
            CachedLayerData::Gdn(snap) => Ok(Some(LayerCache::Arrays(ArraysCache {
                conv_state: snap.conv_state.clone(),
                ssm_state: snap.ssm_state.clone(),
                conv_pos: snap.conv_pos,
                offset: snap.offset,
            }))),
            CachedLayerData::Empty => Ok(None),
        })
        .collect();
    Ok(AnyCache::Hybrid(hybrid_layers?))
}

fn materialize_tq_hybrid(
    layers: &[CachedLayerData],
    context: &Arc<TurboQuantContext>,
) -> Result<AnyCache, Exception> {
    let hybrid_layers: Result<Vec<_>, _> = layers
        .iter()
        .map(|layer| match layer {
            CachedLayerData::TurboQuantKv(blocks) => {
                gather_tq_blocks(blocks, context).map(|kv| Some(LayerCache::KV(kv)))
            }
            CachedLayerData::Kv(blocks) => gather_blocks(blocks).map(|kv| Some(LayerCache::KV(kv))),
            CachedLayerData::Gdn(snap) => Ok(Some(LayerCache::Arrays(ArraysCache {
                conv_state: snap.conv_state.clone(),
                ssm_state: snap.ssm_state.clone(),
                conv_pos: snap.conv_pos,
                offset: snap.offset,
            }))),
            CachedLayerData::Empty => Ok(None),
        })
        .collect();
    Ok(AnyCache::Hybrid(hybrid_layers?))
}

/// Gather KV blocks into a single contiguous `SteppingKeyValueCache`.
fn gather_blocks(blocks: &[KvBlock]) -> Result<SteppingKeyValueCache, Exception> {
    if blocks.is_empty() {
        return Ok(SteppingKeyValueCache::new());
    }

    if blocks.len() == 1 {
        return Ok(SteppingKeyValueCache::from_arrays(
            blocks[0].keys.clone(),
            blocks[0].values.clone(),
        ));
    }

    let key_arrays: Vec<Array> = blocks.iter().map(|b| b.keys.clone()).collect();
    let value_arrays: Vec<Array> = blocks.iter().map(|b| b.values.clone()).collect();
    let keys = concatenate_axis(&key_arrays, 2)?;
    let values = concatenate_axis(&value_arrays, 2)?;

    Ok(SteppingKeyValueCache::from_arrays(keys, values))
}

/// Gather TQ blocks into a single `SteppingKeyValueCache` with TQ storage.
fn gather_tq_blocks(
    blocks: &[TqBlock],
    context: &Arc<TurboQuantContext>,
) -> Result<SteppingKeyValueCache, Exception> {
    if blocks.is_empty() {
        return Ok(SteppingKeyValueCache::new());
    }

    // Concatenate all block arrays along axis 1 (the sequence dimension).
    let concat1 = |arrays: Vec<Array>| -> Result<Array, Exception> {
        if arrays.len() == 1 {
            Ok(arrays.into_iter().next().unwrap())
        } else {
            concatenate_axis(&arrays, 1)
        }
    };

    let key_codes = concat1(blocks.iter().map(|b| b.key_codes.clone()).collect())?;
    let key_norms = concat1(blocks.iter().map(|b| b.key_norms.clone()).collect())?;
    let key_gammas = concat1(blocks.iter().map(|b| b.key_gammas.clone()).collect())?;
    let value_codes = concat1(blocks.iter().map(|b| b.value_codes.clone()).collect())?;
    let value_norms = concat1(blocks.iter().map(|b| b.value_norms.clone()).collect())?;

    // Total tokens = sum of block sizes along axis 1.
    let total = key_norms.shape().get(1).copied().unwrap_or(0);

    Ok(SteppingKeyValueCache::from_turbo_arrays(
        Arc::clone(context),
        key_codes,
        key_norms,
        key_gammas,
        value_codes,
        value_norms,
        total,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use higgs_models::cache::KeyValueCache;

    /// Create a KV cache with `num_layers` layers, each containing `seq_len`
    /// tokens of shape `[1, 2, seq_len, 8]`.
    fn make_kv_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| {
                let keys = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                let values = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                Some(SteppingKeyValueCache::from_arrays(keys, values))
            })
            .collect();
        AnyCache::KV(layers)
    }

    /// Create a Hybrid cache with interleaved KV and GDN layers.
    fn make_hybrid_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<LayerCache>> = (0..num_layers)
            .map(|i| {
                if i % 4 == 0 {
                    Some(LayerCache::Arrays(ArraysCache {
                        conv_state: Some(Array::zeros::<f32>(&[1, 4, 4]).unwrap()),
                        ssm_state: Some(Array::zeros::<f32>(&[1, 16]).unwrap()),
                        conv_pos: 3,
                        offset: seq_len,
                    }))
                } else {
                    let keys = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                    let values = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                    Some(LayerCache::KV(SteppingKeyValueCache::from_arrays(
                        keys, values,
                    )))
                }
            })
            .collect();
        AnyCache::Hybrid(layers)
    }

    fn kv_layer_count(cache: &AnyCache) -> usize {
        match cache {
            AnyCache::KV(v) => v.len(),
            AnyCache::Hybrid(v) => v.len(),
        }
    }

    fn kv_cache_offset(cache: &AnyCache) -> i32 {
        match cache {
            AnyCache::KV(layers) => layers
                .iter()
                .find_map(|l| l.as_ref())
                .map(KeyValueCache::offset)
                .unwrap_or(0),
            AnyCache::Hybrid(layers) => layers
                .iter()
                .find_map(|l| match l {
                    Some(LayerCache::KV(c)) => Some(KeyValueCache::offset(c)),
                    _ => None,
                })
                .unwrap_or(0),
        }
    }

    #[test]
    fn test_empty_cache_returns_none() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        assert!(cache.find_longest_prefix(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_store_and_find_exact_match() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        let kv = make_kv_cache(4, 64);

        cache.store(&prefix, &kv);
        assert_eq!(cache.len(), 1);

        let mut query: Vec<u32> = prefix;
        query.extend_from_slice(&[100, 101, 102]);

        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.prefix_len, 64);
        assert_eq!(kv_layer_count(&matched.cache), 4);
    }

    #[test]
    fn test_block_aligned_prefix_len() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Store 50 tokens of data with 50 token prefix
        let prefix: Vec<u32> = (0..50).collect();
        let kv = make_kv_cache(4, 50);
        cache.store(&prefix, &kv);
        assert_eq!(cache.len(), 1);

        // Query with all 50 tokens + extra
        let mut query: Vec<u32> = (0..50).collect();
        query.push(999);
        let result = cache.find_longest_prefix(&query);
        assert!(result.is_some());

        let matched = result.unwrap();
        // Should be block-aligned: floor(50/32)*32 = 32
        assert_eq!(matched.prefix_len, 32);
        assert_eq!(kv_cache_offset(&matched.cache), 32);
    }

    #[test]
    fn test_materialize_correct_shapes() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..96).collect();
        let kv = make_kv_cache(4, 96);
        cache.store(&prefix, &kv);

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache.find_longest_prefix(&query).unwrap();

        // 96 tokens / 32 block_size = 3 blocks, materialized to 96 tokens
        assert_eq!(matched.prefix_len, 96);

        match &matched.cache {
            AnyCache::KV(layers) => {
                assert_eq!(layers.len(), 4);
                for layer in layers {
                    let kv = layer.as_ref().unwrap();
                    assert_eq!(KeyValueCache::offset(kv), 96);
                    assert_eq!(kv.keys().unwrap().shape(), &[1, 2, 96, 8]);
                    assert_eq!(kv.values().unwrap().shape(), &[1, 2, 96, 8]);
                }
            }
            _ => panic!("Expected KV cache"),
        }
    }

    #[test]
    fn test_hybrid_cache_roundtrip() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        let hybrid = make_hybrid_cache(8, 64);
        cache.store(&prefix, &hybrid);
        assert_eq!(cache.len(), 1);

        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(matched.prefix_len, 64);

        match &matched.cache {
            AnyCache::Hybrid(layers) => {
                assert_eq!(layers.len(), 8);
                for (i, layer) in layers.iter().enumerate() {
                    match layer.as_ref().unwrap() {
                        LayerCache::KV(kv) => {
                            assert_ne!(i % 4, 0, "Layer {i} should be KV");
                            assert_eq!(KeyValueCache::offset(kv), 64);
                        }
                        LayerCache::Arrays(ac) => {
                            assert_eq!(i % 4, 0, "Layer {i} should be GDN");
                            assert_eq!(ac.offset, 64);
                            assert_eq!(ac.conv_pos, 3);
                            assert!(ac.conv_state.is_some());
                            assert!(ac.ssm_state.is_some());
                        }
                    }
                }
            }
            _ => panic!("Expected Hybrid cache"),
        }
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PagedPrefixCache::new(2, DEFAULT_BLOCK_SIZE);

        let prefix_a: Vec<u32> = (0..64).collect();
        let prefix_b: Vec<u32> = (100..164).collect();
        let prefix_c: Vec<u32> = (200..264).collect();

        cache.store(&prefix_a, &make_kv_cache(4, 64));
        cache.store(&prefix_b, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 2);

        cache.store(&prefix_c, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 2);

        let mut query_c: Vec<u32> = prefix_c;
        query_c.push(999);
        assert!(cache.find_longest_prefix(&query_c).is_some());
    }

    #[test]
    fn test_zero_capacity_never_stores() {
        let mut cache = PagedPrefixCache::new(0, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        cache.store(&prefix, &make_kv_cache(4, 64));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_longest_prefix_wins() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let short_prefix: Vec<u32> = (0..32).collect();
        cache.store(&short_prefix, &make_kv_cache(4, 32));

        let long_prefix: Vec<u32> = (0..96).collect();
        cache.store(&long_prefix, &make_kv_cache(4, 96));

        let query: Vec<u32> = (0..128).collect();
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(result.prefix_len, 96);
    }

    #[test]
    fn test_prefix_shorter_than_block_ignored() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..16).collect();
        let kv = make_kv_cache(4, 16);
        cache.store(&prefix, &kv);
        // Stored (via clone fallback since too short for paging),
        // but find won't match because prefix_len < block_size
        let mut query: Vec<u32> = prefix;
        query.push(999);
        assert!(cache.find_longest_prefix(&query).is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();
        cache.store(&prefix, &make_kv_cache(4, 64));
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_overwrite_same_prefix() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..64).collect();

        cache.store(&prefix, &make_kv_cache(2, 64));
        assert_eq!(cache.len(), 1);

        cache.store(&prefix, &make_kv_cache(8, 64));
        assert_eq!(cache.len(), 1);

        let mut query = prefix;
        query.push(999);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(kv_layer_count(&result.cache), 8);
    }

    #[test]
    fn test_shared_prefix_partial_match() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let system_prefix: Vec<u32> = (0..64).collect();
        cache.store(&system_prefix, &make_kv_cache(2, 64));

        let full_prompt: Vec<u32> = (0..128).collect();
        cache.store(&full_prompt, &make_kv_cache(4, 128));
        assert_eq!(cache.len(), 2);

        // Query with same system prefix but different user message
        let mut different_suffix: Vec<u32> = (0..64).collect();
        different_suffix.extend(500..564);
        let result = cache.find_longest_prefix(&different_suffix).unwrap();
        assert_eq!(result.prefix_len, 64);
        assert_eq!(kv_layer_count(&result.cache), 2);
    }

    #[test]
    fn test_from_arrays_enables_decode() {
        // Verify that from_arrays produces a cache that can accept new tokens.
        let keys = Array::ones::<f32>(&[1, 2, 32, 8]).unwrap();
        let values = Array::ones::<f32>(&[1, 2, 32, 8]).unwrap();
        let mut kv = SteppingKeyValueCache::from_arrays(keys, values);
        assert_eq!(KeyValueCache::offset(&kv), 32);

        // Simulate a decode step
        let new_k = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let new_v = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let (rk, rv) = kv.update_and_fetch(new_k, new_v).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 33, 8]);
        assert_eq!(rv.shape(), &[1, 2, 33, 8]);
        assert_eq!(KeyValueCache::offset(&kv), 33);
    }
}
