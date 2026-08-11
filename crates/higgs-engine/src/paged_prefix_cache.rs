use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use higgs_models::cache::{KeyValueCache, SteppingKeyValueCache, slice_axis1, slice_axis2};
#[cfg(test)]
use higgs_models::dflash::{DFlashCache, DFlashSnapshot};
use higgs_models::qwen3_next::ArraysCache;
use higgs_models::turboquant::TurboQuantContext;
use higgs_models::{AnyCache, LayerCache};
use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;

use crate::cache::paired::{
    LivePair, PairedCacheError, RadixDFlashForkPlan, RadixDFlashSnapshot, RadixPairCheckpoint,
};

/// Default block size in tokens for paged caching.
pub const DEFAULT_BLOCK_SIZE: usize = 32;
/// Conservative first-release limit for retained target/dFlash radix pairs.
pub const MAX_PAIRED_RADIX_ENTRIES: usize = 2;

static NEXT_CACHE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
std::thread_local! {
    static FAIL_NEXT_CLONED_TARGET_MATERIALIZATION: Cell<bool> = const { Cell::new(false) };
}

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

impl KvBlock {
    /// Build a block whose K/V arrays are fully EVALUATED. Radix blocks are
    /// `Arc`-shared and reconstructed from different server threads, so a block
    /// must hold a concrete, immutable MLX buffer — never a pending lazy slice
    /// graph. Evaluating here is the soundness invariant the `unsafe impl Sync`
    /// below relies on; it costs nothing net (the slice would materialize on
    /// first use anyway) and removes the cross-thread data race on shared graphs.
    #[allow(clippy::disallowed_methods)] // eval-before-share soundness; on-gate in prod, off-gate only in single-threaded radix tests
    fn new(keys: Array, values: Array) -> Result<Arc<Self>, Exception> {
        // Construction-time soundness eval (eval-before-share). Always reached
        // via `store()` → `run_prefill` under the MLX gate in production; uses
        // the raw transform directly so the radix unit tests (single-threaded,
        // no gate) can construct blocks without tripping the gate's debug_assert.
        mlx_rs::transforms::eval([&keys, &values])?;
        Ok(Arc::new(Self { keys, values }))
    }
}

/// GDN state snapshot at a block boundary (Hybrid models only).
#[derive(Debug, Clone)]
struct GdnSnapshot {
    conv_state: Option<Array>,
    ssm_state: Option<Array>,
    conv_pos: i32,
    offset: i32,
}

/// Per-layer block for `TurboQuant` KV cache.
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

impl TqBlock {
    /// Build a `TurboQuant` block with all 5 arrays EVALUATED — same soundness
    /// invariant as [`KvBlock::new`]: shared across threads, so never lazy.
    #[allow(clippy::disallowed_methods)] // eval-before-share soundness (see KvBlock::new)
    fn new(
        key_codes: Array,
        key_norms: Array,
        key_gammas: Array,
        value_codes: Array,
        value_norms: Array,
    ) -> Result<Arc<Self>, Exception> {
        // Construction-time soundness eval; see KvBlock::new for why this uses
        // the raw transform rather than the gated wrapper.
        mlx_rs::transforms::eval([
            &key_codes,
            &key_norms,
            &key_gammas,
            &value_codes,
            &value_norms,
        ])?;
        Ok(Arc::new(Self {
            key_codes,
            key_norms,
            key_gammas,
            value_codes,
            value_norms,
        }))
    }
}

// MLX `Array` is `Send` but `!Sync` (it holds a `*mut c_void` into the MLX
// runtime). Radix blocks are `Arc`-shared and reconstructed from DIFFERENT
// server threads (each chat request runs in its own `spawn_blocking`), so the
// pointee must be `Send + Sync`. SAFETY rests on one invariant: a block holds
// ONLY fully-evaluated, immutable MLX buffers — never a pending lazy graph.
// That is enforced by construction — `KvBlock::new` / `TqBlock::new` `eval`
// their arrays before the block exists — so concurrent reconstruction touches
// no mutable MLX graph state, only read-only buffers. A *lazy* block would be
// unsound: concurrent build+eval of shared graphs is a data race (SIGSEGV) — the
// regression is pinned by `radix_blocks_reconstruct_safely_across_threads`.
#[allow(unsafe_code)]
unsafe impl Sync for KvBlock {}
#[allow(unsafe_code)]
unsafe impl Sync for TqBlock {}

/// Per-layer cached data covering a single radix edge's token run.
///
/// Blocks are wrapped in `Arc` so that, after an edge split, the shared leading
/// blocks live on a single parent edge and are physically referenced (not
/// copied) by every descendant path. `Arc::strong_count` therefore reflects how
/// many stored prefixes share a given block.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum CachedLayerData {
    /// Attention layer: sequence of dense K/V blocks.
    Kv(Vec<Arc<KvBlock>>),
    /// Attention layer: sequence of `TurboQuant` blocks.
    TurboQuantKv(Vec<Arc<TqBlock>>),
    /// GDN/SSM layer: state snapshot at block boundary.
    Gdn(GdnSnapshot),
    /// Layer had no cache data.
    Empty,
}

// ---------------------------------------------------------------------------
// Cache entry stored in radix trie
// ---------------------------------------------------------------------------

/// Per-edge block payload for a paged prefix.
///
/// One `EdgeBlocks` describes exactly the tokens spanned by the radix edge it
/// sits on; the full cache for a node is the concatenation of the `EdgeBlocks`
/// of every edge on the root -> node path (see `gather_path`). Because the
/// vectors hold `Arc`-wrapped blocks, splitting an edge moves the shared leading
/// blocks onto the parent edge and both children reference them through the
/// same `Arc`s -- block storage is deduplicated across overlapping prefixes.
struct EdgeBlocks {
    layers: Vec<CachedLayerData>,
    tokens: usize,
    /// `TurboQuant` context when these blocks are quantized; `None` for dense.
    /// Carried on the edge so a block-aligned match that lands *inside* an edge
    /// (not on a stored endpoint) can still reconstruct correctly.
    context: Option<Arc<TurboQuantContext>>,
}

/// Block payload carried by a radix edge.
enum EdgeData {
    /// Dense / `TurboQuant` paged blocks for this edge's tokens.
    Paged(EdgeBlocks),
    /// No paged payload (edges that only carry tokens for a `Cloned` endpoint,
    /// or purely structural internal edges).
    None,
}

/// Endpoint metadata for a stored prefix.
///
/// The actual KV blocks live on the path's edges (`EdgeData::Paged`); this only
/// records how to interpret them and any non-paged fallback.
enum CachedData {
    /// Block-paged cache (dense KV). Blocks are reconstructed from the path.
    Paged { is_hybrid: bool },
    /// Block-paged `TurboQuant` cache with shared quantization context.
    TurboQuantPaged {
        context: Arc<TurboQuantContext>,
        is_hybrid: bool,
    },
    /// Full clone fallback (cache too short for paging).
    Cloned(AnyCache),
}

/// One exact stored endpoint in the shared target radix.
///
/// The existing target payload and optional immutable dFlash sidecar move and
/// evict together. Arbitrary target replacement demotes the endpoint; a
/// trusted exact-key disk refresh leaves an existing whole pair untouched.
enum CachedEndpoint {
    TargetOnly(CachedData),
    TargetAndDflash {
        target: CachedData,
        dflash: Arc<RadixDFlashSnapshot>,
    },
}

impl CachedEndpoint {
    const fn target(&self) -> &CachedData {
        match self {
            Self::TargetOnly(target) | Self::TargetAndDflash { target, .. } => target,
        }
    }

    const fn dflash(&self) -> Option<&Arc<RadixDFlashSnapshot>> {
        match self {
            Self::TargetOnly(_) => None,
            Self::TargetAndDflash { dflash, .. } => Some(dflash),
        }
    }

    const fn is_paired(&self) -> bool {
        matches!(self, Self::TargetAndDflash { .. })
    }
}

struct CachedState {
    endpoint: CachedEndpoint,
    entry_id: u64,
    last_accessed: Cell<u64>,
    last_accessed_at: Cell<Instant>,
}

// ---------------------------------------------------------------------------
// Radix trie
// ---------------------------------------------------------------------------

struct RadixNode {
    edge: Vec<u32>,
    /// Blocks covering exactly `edge`'s tokens. Shared across descendant paths.
    edge_blocks: EdgeData,
    cached: Option<CachedState>,
    children: HashMap<u32, Self>,
}

/// Endpoint kind for a lookup match.
enum MatchEndpoint<'a> {
    /// A stored trie endpoint (full metadata available).
    Stored(&'a CachedEndpoint),
    /// A block-aligned position inside an edge (no stored endpoint). Paged
    /// caches are never hybrid; only the optional TQ context is needed.
    PartialPaged {
        context: Option<Arc<TurboQuantContext>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LookupPolicy {
    TargetAny,
    TargetAndPairExact,
}

/// A candidate lookup match: how deep it reaches, how to interpret its blocks,
/// the path of full edges, and an optional partially-matched final edge.
struct MatchResult<'a> {
    prefix_len: usize,
    kind: MatchEndpoint<'a>,
    full_path: Vec<&'a EdgeBlocks>,
    partial_tail: Option<EdgeBlocks>,
    entry_id: Option<u64>,
}

/// Pick the deeper of two candidate matches.
fn deeper_of<'a>(
    a: Option<MatchResult<'a>>,
    b: Option<MatchResult<'a>>,
) -> Option<MatchResult<'a>> {
    match (a, b) {
        (Some(am), Some(bm)) => Some(if bm.prefix_len > am.prefix_len {
            bm
        } else {
            am
        }),
        (left, right) => left.or(right),
    }
}

/// Result of a paged prefix cache lookup.
pub struct PagedPrefixMatch {
    /// Number of tokens from the beginning that matched the cached prefix.
    pub prefix_len: usize,
    /// Materialized cache state for the matched prefix.
    pub cache: AnyCache,
}

/// Legacy raw paired materialization retained only for unit fixtures.
#[cfg(test)]
pub(crate) struct MaterializedPairedPrefix {
    pub(crate) prefix_len: usize,
    pub(crate) cache: AnyCache,
    pub(crate) dflash_cache: DFlashCache,
}

/// Successful proven pair fork plus its one-shot retained-LRU authority.
pub(crate) struct PagedPairedPrefixMatch {
    pair: LivePair,
    touch: PairedTouchToken,
}

impl PagedPairedPrefixMatch {
    pub(crate) fn into_pair_and_touch(self) -> (LivePair, PairedTouchToken) {
        (self.pair, self.touch)
    }
}

/// Epoch captured before off-lock paired preparation or materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PairedCacheEpoch(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheInstanceId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachePublicationRevision(u64);

/// Configuration-bound authority for one off-lock paired preparation.
///
/// Callers can copy and move this ticket but cannot construct or alter it, so a
/// prepared pair is tied to one cache instance, epoch, publication revision,
/// and block layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PairedPrepareTicket {
    instance_id: CacheInstanceId,
    block_size: usize,
    epoch: PairedCacheEpoch,
    revision: CachePublicationRevision,
}

impl PairedPrepareTicket {
    /// Boundary representable by this ticket's captured target-radix layout.
    ///
    /// Dense targets publish only complete paged blocks. Hybrid targets use
    /// the existing exact cloned-endpoint path.
    #[must_use]
    pub(crate) const fn store_boundary(self, requested: usize, is_hybrid: bool) -> usize {
        if is_hybrid {
            requested
        } else {
            requested / self.block_size * self.block_size
        }
    }
}

/// Opaque token for refreshing paired LRU only after a successful fork.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PairedTouchToken {
    instance_id: CacheInstanceId,
    epoch: PairedCacheEpoch,
    entry_id: u64,
}

/// Conservative accounting for paired radix endpoint memory.
///
/// Target bytes describe the logical target state reachable at each paired
/// endpoint; shared paged blocks may therefore be counted more than once.
/// dFlash bytes are the sidecar's actual retained snapshot estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairedPrefixCacheStats {
    pub entries: usize,
    pub target_bytes: usize,
    pub dflash_bytes: usize,
}

impl PairedPrefixCacheStats {
    #[must_use]
    pub const fn total_bytes(self) -> usize {
        self.target_bytes.saturating_add(self.dflash_bytes)
    }
}

/// Owned target reconstruction selected while the radix is locked.
///
/// This contains only immutable array handles / `Arc` bumps. Device copies,
/// concatenation, and evaluation are deferred until the caller holds the
/// process MLX execution gate.
enum TargetMaterializationPlan {
    Cloned(AnyCache),
    DensePaged {
        layers: Vec<CachedLayerData>,
        is_hybrid: bool,
    },
    TurboQuantPaged {
        layers: Vec<CachedLayerData>,
        context: Arc<TurboQuantContext>,
        is_hybrid: bool,
    },
}

/// Exact paired radix hit whose resources are owned independently of the trie.
///
/// Constructing this plan is CPU-only and safe while the prefix mutex is held.
/// Materialization requires the process MLX gate and must happen after that
/// mutex is released.
pub(crate) struct PagedPairedLookupPlan {
    prefix_len: usize,
    target: TargetMaterializationPlan,
    dflash: RadixDFlashForkPlan,
    touch: PairedTouchToken,
}

impl PagedPairedLookupPlan {
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    pub(crate) fn materialize(
        self,
        expected_taps: usize,
    ) -> Result<PagedPairedPrefixMatch, PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "paired radix materialization requires the process MLX execution gate"
        );
        let cache =
            self.target
                .materialize()
                .map_err(|error| PairedCacheError::TargetMaterialization {
                    details: error.to_string(),
                })?;
        let expected =
            i32::try_from(self.prefix_len).map_err(|_| PairedCacheError::PrefixLengthOverflow {
                len: self.prefix_len,
            })?;
        cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        if self.dflash.prefix_len() != self.prefix_len {
            return Err(PairedCacheError::DFlashBoundary {
                expected,
                actual: i32::try_from(self.dflash.prefix_len()).unwrap_or(i32::MAX),
            });
        }
        let pair = self.dflash.materialize_pair(cache, expected_taps)?;
        Ok(PagedPairedPrefixMatch {
            pair,
            touch: self.touch,
        })
    }

    /// Legacy raw-half materialization for independently-labelled unit
    /// fixtures. Production callers can receive only [`LivePair`].
    #[cfg(test)]
    pub(crate) fn materialize_unproven_for_test(
        self,
    ) -> Result<(MaterializedPairedPrefix, PairedTouchToken), PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "paired radix materialization requires the process MLX execution gate"
        );
        let cache =
            self.target
                .materialize()
                .map_err(|error| PairedCacheError::TargetMaterialization {
                    details: error.to_string(),
                })?;
        let expected =
            i32::try_from(self.prefix_len).map_err(|_| PairedCacheError::PrefixLengthOverflow {
                len: self.prefix_len,
            })?;
        cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        if self.dflash.prefix_len() != self.prefix_len {
            return Err(PairedCacheError::DFlashBoundary {
                expected,
                actual: i32::try_from(self.dflash.prefix_len()).unwrap_or(i32::MAX),
            });
        }
        let dflash_cache = self.dflash.materialize()?;
        Ok((
            MaterializedPairedPrefix {
                prefix_len: self.prefix_len,
                cache,
                dflash_cache,
            },
            self.touch,
        ))
    }

    #[cfg(test)]
    fn fail_materialization_for_test(&self) {
        self.dflash.fail_materialization_for_test();
    }
}

impl TargetMaterializationPlan {
    fn materialize(self) -> Result<AnyCache, Exception> {
        match self {
            Self::Cloned(cache) => try_clone_target_for_materialization(&cache),
            Self::DensePaged { layers, is_hybrid } => {
                if is_hybrid {
                    materialize_hybrid(&layers)
                } else {
                    materialize_kv(&layers)
                }
            }
            Self::TurboQuantPaged {
                layers,
                context,
                is_hybrid,
            } => {
                if is_hybrid {
                    materialize_tq_hybrid(&layers, &context)
                } else {
                    materialize_tq_kv(&layers, &context)
                }
            }
        }
    }
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
    num_paired: usize,
    max_cached: usize,
    max_paired: usize,
    block_size: usize,
    instance_id: CacheInstanceId,
    epoch: PairedCacheEpoch,
    revision: CachePublicationRevision,
    next_entry_id: u64,
    access_clock: u64,
    paired_idle_ttl: Option<Duration>,
}

// ---------------------------------------------------------------------------
// RadixNode impl (mirrors prompt_cache.rs but stores CachedState)
// ---------------------------------------------------------------------------

impl RadixNode {
    fn empty() -> Self {
        Self {
            edge: Vec::new(),
            edge_blocks: EdgeData::None,
            cached: None,
            children: HashMap::new(),
        }
    }

    fn leaf(
        edge: Vec<u32>,
        edge_blocks: EdgeData,
        endpoint: CachedEndpoint,
        entry_id: u64,
        last_accessed: u64,
        last_accessed_at: Instant,
    ) -> Self {
        Self {
            edge,
            edge_blocks,
            cached: Some(CachedState {
                endpoint,
                entry_id,
                last_accessed: Cell::new(last_accessed),
                last_accessed_at: Cell::new(last_accessed_at),
            }),
            children: HashMap::new(),
        }
    }

    /// Walk the trie matching `tokens`, accumulating the path's edge blocks, and
    /// return the DEEPEST valid match.
    ///
    /// A match is valid at:
    /// - a stored endpoint (`cached`) reached at this `depth`, or
    /// - a block-aligned position *inside* a partially-matched child edge (true
    ///   `RadixAttention` sub-prefix sharing): if the query and an edge share the
    ///   first `k` whole blocks but then diverge, those `k` blocks form a valid
    ///   reusable prefix even though no endpoint was stored there.
    ///
    /// `full_path` references the `EdgeBlocks` of every fully-traversed edge from
    /// the root to the matched node; `partial_tail` (owned, cheap `Arc` clones)
    /// holds the leading whole blocks of a partially-matched final edge. The
    /// caller concatenates `full_path` then `partial_tail` to rebuild a
    /// byte-identical KV cache for the matched prefix.
    fn find_deepest_match<'a>(
        &'a self,
        tokens: &[u32],
        depth: usize,
        min_prefix: usize,
        block_size: usize,
        policy: LookupPolicy,
        path: &mut Vec<&'a EdgeBlocks>,
    ) -> Option<MatchResult<'a>> {
        // Record this edge's blocks on the running path (root edge is empty).
        if let EdgeData::Paged(blocks) = &self.edge_blocks {
            path.push(blocks);
        }

        // Block-token depth reachable here from the path's paged edges. May be
        // less than the edge-token `depth` after a non-block-aligned ancestor
        // split; it is exactly the reconstructable prefix length.
        let block_depth: usize = path.iter().map(|e| e.tokens).sum();

        // Candidate 1: a stored endpoint at this node (gives a `touch` handle and
        // handles the Cloned fallback).
        let mut deepest: Option<MatchResult<'a>> = self
            .cached
            .as_ref()
            .filter(|cs| {
                let target = cs.endpoint.target();
                let prefix_len = match target {
                    CachedData::Cloned(_) => depth,
                    CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => block_depth,
                };
                let target_qualifies = match target {
                    CachedData::Cloned(_) => depth > 0,
                    // Filter on the BLOCK-token sum we actually report as
                    // prefix_len, not raw edge depth.
                    CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => {
                        block_depth >= min_prefix
                    }
                };
                let policy_qualifies = match policy {
                    LookupPolicy::TargetAny => true,
                    LookupPolicy::TargetAndPairExact => {
                        cs.endpoint.dflash().is_some_and(|dflash| {
                            prefix_len == depth
                                && dflash.prefix_len() == prefix_len
                                && tokens
                                    .get(..prefix_len)
                                    .is_some_and(|prefix| dflash.matches_prefix(prefix))
                        })
                    }
                };
                target_qualifies && policy_qualifies
            })
            .map(|cs| MatchResult {
                prefix_len: match cs.endpoint.target() {
                    CachedData::Cloned(_) => depth,
                    CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => block_depth,
                },
                kind: MatchEndpoint::Stored(&cs.endpoint),
                full_path: path.clone(),
                partial_tail: None,
                entry_id: Some(cs.entry_id),
            });

        // Candidate 2: this node itself sits at a reconstructable block-aligned
        // prefix (e.g. a shared split node with no stored endpoint). The full
        // path's blocks reconstruct it exactly -- true RadixAttention prefix
        // sharing even when no endpoint was stored at this boundary.
        if policy == LookupPolicy::TargetAny
            && block_depth >= min_prefix
            && matches!(&self.edge_blocks, EdgeData::Paged(_))
        {
            let node_match = MatchResult {
                prefix_len: block_depth,
                kind: MatchEndpoint::PartialPaged {
                    context: path.last().and_then(|e| e.context.clone()),
                },
                full_path: path.clone(),
                partial_tail: None,
                entry_id: None,
            };
            deepest = deeper_of(deepest, Some(node_match));
        }

        if let Some(&next_token) = tokens.get(depth) {
            if let Some(child) = self.children.get(&next_token) {
                let remaining = tokens.get(depth..).unwrap_or_default();
                let common = child
                    .edge
                    .iter()
                    .zip(remaining.iter())
                    .take_while(|(a, b)| a == b)
                    .count();

                if common == child.edge.len() {
                    // Whole edge matched: descend.
                    if let Some(found) = child.find_deepest_match(
                        tokens,
                        depth + common,
                        min_prefix,
                        block_size,
                        policy,
                        path,
                    ) {
                        deepest = deeper_of(deepest, Some(found));
                    }
                } else {
                    // Partial edge match: reuse the leading whole blocks that the
                    // query shares with this child edge (RadixAttention sub-prefix).
                    if policy == LookupPolicy::TargetAny
                        && let Some(found) = child.partial_edge_match(
                            common,
                            min_prefix,
                            block_size,
                            path.as_slice(),
                        )
                    {
                        deepest = deeper_of(deepest, Some(found));
                    }
                }
            }
        }

        // Pop this edge's blocks so siblings explored by the caller don't inherit them.
        if matches!(&self.edge_blocks, EdgeData::Paged(_)) {
            path.pop();
        }

        deepest
    }

    /// Build a match from the leading whole blocks of THIS edge that the query
    /// shares (`common` tokens matched before divergence). Returns `None` when
    /// fewer than one block is shared or the edge carries no paged blocks.
    fn partial_edge_match<'a>(
        &'a self,
        common: usize,
        min_prefix: usize,
        block_size: usize,
        path: &[&'a EdgeBlocks],
    ) -> Option<MatchResult<'a>> {
        let EdgeData::Paged(blocks) = &self.edge_blocks else {
            return None;
        };
        let n_blocks = common / block_size;
        if n_blocks == 0 {
            return None;
        }
        let matched_tokens = n_blocks * block_size;
        // Derive the reachable prefix length from the ACTUAL block-token sum of
        // the path so far (not the edge-token depth, which can exceed the block
        // sum after a non-block-aligned ancestor split). This keeps `prefix_len`
        // exactly equal to the reconstructed cache's token count.
        let base: usize = path.iter().map(|e| e.tokens).sum();
        let prefix_len = base + matched_tokens;
        if prefix_len < min_prefix {
            return None;
        }
        // Take the leading `n_blocks` of every layer on this edge (Arc clones).
        let tail_layers: Vec<CachedLayerData> = blocks
            .layers
            .iter()
            .map(|l| l.split_at_blocks(n_blocks).0)
            .collect();
        let partial = EdgeBlocks {
            layers: tail_layers,
            tokens: matched_tokens,
            context: blocks.context.clone(),
        };
        Some(MatchResult {
            prefix_len,
            kind: MatchEndpoint::PartialPaged {
                context: blocks.context.clone(),
            },
            // The path up to (but excluding) this edge -- this edge's blocks are
            // not on `path` yet (the parent pushes its own edge; this child edge
            // is represented by `partial_tail`).
            full_path: path.to_vec(),
            partial_tail: Some(partial),
            entry_id: None,
        })
    }

    fn oldest_cached(&self) -> Option<(u64, u64)> {
        let mut oldest = self
            .cached
            .as_ref()
            .map(|cached| (cached.last_accessed.get(), cached.entry_id));

        for child in self.children.values() {
            if let Some(candidate) = child.oldest_cached() {
                oldest = Some(oldest.map_or(candidate, |current| current.min(candidate)));
            }
        }

        oldest
    }

    fn oldest_paired(&self) -> Option<(u64, u64)> {
        let mut oldest = self
            .cached
            .as_ref()
            .filter(|cached| cached.endpoint.is_paired())
            .map(|cached| (cached.last_accessed.get(), cached.entry_id));

        for child in self.children.values() {
            if let Some(candidate) = child.oldest_paired() {
                oldest = Some(oldest.map_or(candidate, |current| current.min(candidate)));
            }
        }

        oldest
    }

    /// Remove the endpoint with `entry_id` and report whether it was paired.
    ///
    /// Entry IDs are stable across access-clock ties and are replaced on every
    /// publication, so stale lookup tokens cannot affect a refreshed endpoint.
    fn remove_cached_by_id(&mut self, entry_id: u64) -> Option<bool> {
        if self
            .cached
            .as_ref()
            .is_some_and(|cached| cached.entry_id == entry_id)
        {
            return self.cached.take().map(|cached| cached.endpoint.is_paired());
        }

        for child in self.children.values_mut() {
            if let Some(was_paired) = child.remove_cached_by_id(entry_id) {
                return Some(was_paired);
            }
        }

        None
    }

    /// Remove a paired endpoint stored at exactly `tokens`.
    ///
    /// This deliberately does not remove target-only endpoints or endpoints
    /// at a shorter block-aligned prefix. Ordinary dense stores may floor their
    /// publication boundary, so they use this before publishing to invalidate a
    /// stale exact pair at the full request key.
    fn remove_exact_paired_endpoint(&mut self, tokens: &[u32]) -> bool {
        if tokens.is_empty() {
            if self
                .cached
                .as_ref()
                .is_some_and(|cached| cached.endpoint.is_paired())
            {
                self.cached = None;
                return true;
            }
            return false;
        }

        let Some(child) = tokens
            .first()
            .and_then(|first| self.children.get_mut(first))
        else {
            return false;
        };
        if !tokens.starts_with(&child.edge) {
            return false;
        }
        let remaining = tokens.get(child.edge.len()..).unwrap_or_default();
        child.remove_exact_paired_endpoint(remaining)
    }

    /// Whether `tokens` names one exact stored paired endpoint.
    fn has_exact_paired_endpoint(&self, tokens: &[u32]) -> bool {
        if tokens.is_empty() {
            return self
                .cached
                .as_ref()
                .is_some_and(|cached| cached.endpoint.is_paired());
        }

        let Some(child) = tokens.first().and_then(|first| self.children.get(first)) else {
            return false;
        };
        if !tokens.starts_with(&child.edge) {
            return false;
        }
        let remaining = tokens.get(child.edge.len()..).unwrap_or_default();
        child.has_exact_paired_endpoint(remaining)
    }

    fn touch_entry(
        &self,
        entry_id: u64,
        access: u64,
        accessed_at: Instant,
        paired_only: bool,
    ) -> bool {
        if let Some(cached) = self.cached.as_ref().filter(|cached| {
            cached.entry_id == entry_id && (!paired_only || cached.endpoint.is_paired())
        }) {
            cached.last_accessed.set(access);
            cached.last_accessed_at.set(accessed_at);
            return true;
        }

        self.children
            .values()
            .any(|child| child.touch_entry(entry_id, access, accessed_at, paired_only))
    }

    /// Remove paired endpoints last touched at or before `cutoff`.
    ///
    /// The endpoint owns both target and dSpark capabilities, so taking the
    /// single `CachedState` is the atomic eviction operation. Target-only
    /// endpoints and shared path blocks needed by surviving descendants remain.
    fn remove_expired_paired(&mut self, cutoff: Instant) -> usize {
        let mut removed = 0;
        if self.cached.as_ref().is_some_and(|cached| {
            cached.endpoint.is_paired() && cached.last_accessed_at.get() <= cutoff
        }) {
            self.cached = None;
            removed += 1;
        }
        for child in self.children.values_mut() {
            removed += child.remove_expired_paired(cutoff);
        }
        removed
    }

    fn accumulate_paired_stats(&self, stats: &mut PairedPrefixCacheStats) {
        if let Some(pair) = self
            .cached
            .as_ref()
            .and_then(|cached| cached.endpoint.dflash())
        {
            stats.entries = stats.entries.saturating_add(1);
            stats.target_bytes = stats.target_bytes.saturating_add(pair.target_bytes());
            stats.dflash_bytes = stats.dflash_bytes.saturating_add(pair.dflash_bytes());
        }

        for child in self.children.values() {
            child.accumulate_paired_stats(stats);
        }
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
            self.edge_blocks = EdgeData::merge(
                std::mem::replace(&mut self.edge_blocks, EdgeData::None),
                only_child.edge_blocks,
            );
            self.cached = only_child.cached;
            self.children = only_child.children;
        }
    }
}

impl EdgeData {
    const fn tokens(&self) -> usize {
        match self {
            Self::Paged(blocks) => blocks.tokens,
            Self::None => 0,
        }
    }

    /// Concatenate two consecutive edges' block payloads into one. Used when
    /// `prune` collapses a node into its sole child after eviction.
    fn merge(parent: Self, child: Self) -> Self {
        match (parent, child) {
            (Self::Paged(mut p), Self::Paged(c)) => {
                p.tokens += c.tokens;
                p.context = p.context.or(c.context);
                for (pl, cl) in p.layers.iter_mut().zip(c.layers) {
                    pl.append_from(cl);
                }
                Self::Paged(p)
            }
            // A paged edge followed by a non-paged one (or vice versa) only
            // happens around Cloned endpoints, which carry no path blocks; the
            // paged side (if any) is structurally irrelevant once merged.
            (Self::Paged(p), Self::None) => Self::Paged(p),
            (Self::None, other) => other,
        }
    }
}

impl CachedLayerData {
    /// Append another edge segment's blocks for the same layer onto `self`.
    fn append_from(&mut self, other: Self) {
        match (self, other) {
            (Self::Kv(a), Self::Kv(b)) => a.extend(b),
            (Self::TurboQuantKv(a), Self::TurboQuantKv(b)) => a.extend(b),
            // Empty/Gdn layers carry no per-block run; keep the existing one.
            // (Gdn snapshots are endpoint-level, not split across edges, and
            // only ever appear on a single terminal edge.)
            _ => {}
        }
    }

    /// Split this layer's block run at `n` blocks, returning `(head, tail)`.
    /// `Arc` clones are reference bumps -- no array data is copied.
    fn split_at_blocks(&self, n: usize) -> (Self, Self) {
        match self {
            Self::Kv(blocks) => {
                let head = blocks.iter().take(n).map(Arc::clone).collect();
                let tail = blocks.iter().skip(n).map(Arc::clone).collect();
                (Self::Kv(head), Self::Kv(tail))
            }
            Self::TurboQuantKv(blocks) => {
                let head = blocks.iter().take(n).map(Arc::clone).collect();
                let tail = blocks.iter().skip(n).map(Arc::clone).collect();
                (Self::TurboQuantKv(head), Self::TurboQuantKv(tail))
            }
            Self::Gdn(snap) => (Self::Gdn(snap.clone()), Self::Empty),
            Self::Empty => (Self::Empty, Self::Empty),
        }
    }

    /// Drop the leading `n` blocks, returning the remaining tail.
    fn drop_leading(&self, n: usize) -> Self {
        self.split_at_blocks(n).1
    }

    /// Keep only the last `n` blocks of this layer's run.
    fn take_last_blocks(&self, n: usize) -> Self {
        match self {
            Self::Kv(blocks) => {
                let start = blocks.len().saturating_sub(n);
                Self::Kv(blocks.iter().skip(start).map(Arc::clone).collect())
            }
            Self::TurboQuantKv(blocks) => {
                let start = blocks.len().saturating_sub(n);
                Self::TurboQuantKv(blocks.iter().skip(start).map(Arc::clone).collect())
            }
            Self::Gdn(snap) => Self::Gdn(snap.clone()),
            Self::Empty => Self::Empty,
        }
    }
}

/// Shared insert/lookup parameters threaded through the recursion.
struct Ctx {
    block_size: usize,
    context: Option<Arc<TurboQuantContext>>,
    entry_id: u64,
    last_accessed: u64,
    last_accessed_at: Instant,
}

#[derive(Debug, Clone, Copy, Default)]
struct InsertOutcome {
    added: bool,
    was_paired: bool,
    is_paired: bool,
}

/// Build an `EdgeData::Paged` from a per-layer block run spanning `tokens`,
/// or `EdgeData::None` when there are no blocks (e.g. the remainder of a
/// `Cloned` insert). `context` is the shared `TurboQuant` context (dense: `None`).
fn edge_blocks_from(
    blocks: Option<Vec<CachedLayerData>>,
    context: Option<Arc<TurboQuantContext>>,
) -> EdgeData {
    blocks.map_or(EdgeData::None, |layers| {
        let tokens = layer_run_tokens(&layers);
        EdgeData::Paged(EdgeBlocks {
            layers,
            tokens,
            context,
        })
    })
}

/// Number of tokens a per-layer block run covers (block count x block size,
/// inferred from a non-`Empty` layer's block dimension).
fn layer_run_tokens(layers: &[CachedLayerData]) -> usize {
    for layer in layers {
        match layer {
            CachedLayerData::Kv(blocks) => {
                if let Some(b) = blocks.first() {
                    let per_block =
                        usize::try_from(b.keys.shape().get(2).copied().unwrap_or(0)).unwrap_or(0);
                    return blocks.len() * per_block;
                }
            }
            CachedLayerData::TurboQuantKv(blocks) => {
                if let Some(b) = blocks.first() {
                    let per_block =
                        usize::try_from(b.key_norms.shape().get(1).copied().unwrap_or(0))
                            .unwrap_or(0);
                    return blocks.len() * per_block;
                }
            }
            CachedLayerData::Gdn(_) | CachedLayerData::Empty => {}
        }
    }
    0
}

/// Drop the leading `n_tokens` worth of blocks (`n_tokens / block_size` blocks)
/// from every layer of an incoming block run. Used to discard the incoming
/// duplicates of blocks that already live on the trie's shared edges.
fn drop_leading_blocks(
    blocks: Option<Vec<CachedLayerData>>,
    n_tokens: usize,
    block_size: usize,
) -> Option<Vec<CachedLayerData>> {
    let n_blocks = n_tokens / block_size;
    blocks.map(|layers| layers.iter().map(|l| l.drop_leading(n_blocks)).collect())
}

/// Split an existing edge's blocks at `n_tokens` (`n_tokens / block_size`
/// blocks), returning `(shared_head, leftover_tail)` as `EdgeData`s.
fn split_edge_blocks(edge: EdgeData, n_tokens: usize, block_size: usize) -> (EdgeData, EdgeData) {
    match edge {
        EdgeData::None => (EdgeData::None, EdgeData::None),
        EdgeData::Paged(blocks) => {
            let n_blocks = n_tokens / block_size;
            let head_tokens = n_blocks * block_size;
            let tail_tokens = blocks.tokens.saturating_sub(head_tokens);
            let mut head_layers = Vec::with_capacity(blocks.layers.len());
            let mut tail_layers = Vec::with_capacity(blocks.layers.len());
            for layer in &blocks.layers {
                let (h, t) = layer.split_at_blocks(n_blocks);
                head_layers.push(h);
                tail_layers.push(t);
            }
            (
                EdgeData::Paged(EdgeBlocks {
                    layers: head_layers,
                    tokens: head_tokens,
                    context: blocks.context.clone(),
                }),
                EdgeData::Paged(EdgeBlocks {
                    layers: tail_layers,
                    tokens: tail_tokens,
                    context: blocks.context,
                }),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// PagedPrefixCache impl
// ---------------------------------------------------------------------------

impl PagedPrefixCache {
    fn next_instance_id() -> CacheInstanceId {
        let id = NEXT_CACHE_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("paired cache instance ID exhausted");
        CacheInstanceId(id)
    }

    pub fn new(max_entries: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "PagedPrefixCache block_size must be > 0");
        Self {
            root: RadixNode::empty(),
            num_cached: 0,
            num_paired: 0,
            max_cached: max_entries,
            max_paired: MAX_PAIRED_RADIX_ENTRIES,
            block_size,
            instance_id: Self::next_instance_id(),
            epoch: PairedCacheEpoch(0),
            revision: CachePublicationRevision(0),
            next_entry_id: 1,
            access_clock: 1,
            paired_idle_ttl: None,
        }
    }

    /// Apply one idle TTL to paired target+dSpark radix endpoints.
    ///
    /// `None` disables expiry. The target-only radix remains governed by its
    /// existing entry cap; the TTL is intentionally scoped to the larger paired
    /// capability whose dSpark full-attention state grows with context.
    pub(crate) const fn set_paired_idle_ttl(&mut self, ttl: Option<Duration>) {
        self.paired_idle_ttl = ttl;
    }

    #[must_use]
    pub(crate) const fn paired_prepare_ticket(&self) -> PairedPrepareTicket {
        PairedPrepareTicket {
            instance_id: self.instance_id,
            block_size: self.block_size,
            epoch: self.epoch,
            revision: self.revision,
        }
    }

    /// Find the longest cached prefix that matches the beginning of `tokens`.
    ///
    /// Returns `None` if no prefix matches or if the match is shorter than one
    /// block. On hit, blocks along the matched path are gathered into a
    /// contiguous `AnyCache`.
    pub fn find_longest_prefix(&mut self, tokens: &[u32]) -> Option<PagedPrefixMatch> {
        self.evict_idle_paired();
        let (prefix_len, entry_id, result) = {
            let mut scratch: Vec<&EdgeBlocks> = Vec::new();
            let matched = self.root.find_deepest_match(
                tokens,
                0,
                self.block_size,
                self.block_size,
                LookupPolicy::TargetAny,
                &mut scratch,
            )?;
            (matched.prefix_len, matched.entry_id, materialize(&matched))
        };
        match result {
            Ok(cache) => {
                tracing::debug!(prefix_len, "Prefix cache hit");
                if let Some(entry_id) = entry_id {
                    let access = self.next_access();
                    let _ = self
                        .root
                        .touch_entry(entry_id, access, Instant::now(), false);
                }
                Some(PagedPrefixMatch { prefix_len, cache })
            }
            Err(e) => {
                tracing::warn!(error = %e, "Prefix cache materialize failed");
                None
            }
        }
    }

    /// Select the deepest exact stored endpoint that owns both target and
    /// dFlash state.
    ///
    /// This performs no MLX materialization or dFlash fork. The returned plan
    /// owns immutable handles for the target endpoint and the exact dFlash
    /// sidecar selected under the same lock. Selection does not refresh LRU.
    pub(crate) fn plan_longest_paired_prefix(
        &mut self,
        tokens: &[u32],
    ) -> Result<Option<PagedPairedLookupPlan>, PairedCacheError> {
        debug_assert!(
            !higgs_models::mlx_exec::held(),
            "paired prefix selection must happen before acquiring the MLX execution gate"
        );
        self.evict_idle_paired();
        let mut scratch: Vec<&EdgeBlocks> = Vec::new();
        let Some(matched) = self.root.find_deepest_match(
            tokens,
            0,
            self.block_size,
            self.block_size,
            LookupPolicy::TargetAndPairExact,
            &mut scratch,
        ) else {
            return Ok(None);
        };
        let MatchEndpoint::Stored(endpoint) = &matched.kind else {
            return Ok(None);
        };
        let Some(dflash) = endpoint.dflash() else {
            return Ok(None);
        };
        let prefix_len = matched.prefix_len;
        let prefix_tokens = tokens
            .get(..prefix_len)
            .ok_or(PairedCacheError::PrefixMismatch {
                stored_len: prefix_len,
                requested_len: tokens.len(),
            })?;
        let entry_id = matched
            .entry_id
            .ok_or_else(|| PairedCacheError::TargetMaterialization {
                details: "paired lookup selected an endpoint without stable identity".to_owned(),
            })?;
        let target = own_target_materialization_plan(&matched)?;
        let dflash = dflash.plan_fork(prefix_tokens)?;
        Ok(Some(PagedPairedLookupPlan {
            prefix_len,
            target,
            dflash,
            touch: PairedTouchToken {
                instance_id: self.instance_id,
                epoch: self.epoch,
                entry_id,
            },
        }))
    }

    /// Store a prefix and its cache state as paged blocks.
    ///
    /// For dense KV caches, the K/V arrays are sliced into block-sized views
    /// (lazy, nearly free) and inserted into the radix trie one block per edge
    /// segment. Where the new sequence shares a leading run of blocks with an
    /// existing entry, the shared blocks already live on the trie's edges and
    /// are reused -- the incoming duplicates are dropped, so storage is
    /// deduplicated. For `TurboQuant` caches with deferred quantization a full
    /// clone fallback is used. Only block-aligned tokens are stored in the trie.
    pub fn store(&mut self, prefix_tokens: &[u32], cache: &AnyCache) {
        if self.max_cached == 0 {
            return;
        }
        let Some(prepared) = self.prepare_store(prefix_tokens, cache) else {
            return;
        };
        let stored_len = prepared.total_tokens;
        if stored_len != prefix_tokens.len() {
            self.remove_exact_paired_endpoint(prefix_tokens);
        }
        let target = CachedEndpoint::TargetOnly(prepared.endpoint);
        self.publish_prepared(
            prefix_tokens,
            stored_len,
            prepared.blocks,
            prepared.context,
            target,
        );
    }

    /// Refresh target state loaded from the trusted disk cache unless the exact
    /// endpoint already owns a proven target/dFlash pair.
    ///
    /// This is intentionally narrower than [`Self::store`]: arbitrary live
    /// target replacement still demotes speculative continuity. Disk snapshots
    /// may serve the current target-only request, but they cannot replace one
    /// half of an existing pair. At an exact paired endpoint the complete
    /// original target representation, dFlash sidecar, identity, LRU age, and
    /// publication revision therefore remain untouched.
    pub(crate) fn store_disk_refresh_preserving_pair(
        &mut self,
        prefix_tokens: &[u32],
        cache: &AnyCache,
    ) {
        if self.max_cached == 0 {
            return;
        }
        let Some(prepared) = self.prepare_store(prefix_tokens, cache) else {
            return;
        };
        let stored_len = prepared.total_tokens;
        let Some(tokens_to_store) = prefix_tokens.get(..stored_len) else {
            return;
        };
        if self.root.has_exact_paired_endpoint(tokens_to_store) {
            return;
        }
        let target = CachedEndpoint::TargetOnly(prepared.endpoint);
        self.publish_prepared(
            prefix_tokens,
            stored_len,
            prepared.blocks,
            prepared.context,
            target,
        );
    }

    /// Prepare one exact target endpoint plus dFlash sidecar without borrowing
    /// the radix.
    ///
    /// Slicing/evaluation happens here under the process MLX gate. Because this
    /// is an associated function rather than a method, callers do not need (and
    /// should not hold) the prefix-cache mutex. The opaque result can only be
    /// published through [`Self::commit_prepared_pair`]. The checkpoint must
    /// already name an exactly representable target boundary; preparation never
    /// relabels it to a shorter paged floor.
    pub(crate) fn prepare_paired_prefix(
        ticket: PairedPrepareTicket,
        checkpoint: RadixPairCheckpoint<'_>,
    ) -> Result<PreparedPairedPrefix, PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "paired radix preparation requires the process MLX execution gate"
        );
        let checkpoint_len = checkpoint.tokens().len();
        let prepared = Self::prepare_pair_target(ticket, checkpoint.tokens(), checkpoint.target())?;
        if prepared.tokens.len() != checkpoint_len {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: prepared.tokens.len(),
                requested_len: checkpoint_len,
            });
        }
        let dflash = checkpoint.into_radix_snapshot(prepared.target_bytes)?;
        Ok(prepared.attach_dflash(dflash))
    }

    /// Legacy independently-labelled preparation retained only for unit
    /// fixtures. Production publication must consume [`RadixPairCheckpoint`].
    #[cfg(test)]
    pub(crate) fn prepare_paired_prefix_from_parts(
        ticket: PairedPrepareTicket,
        prefix_tokens: &[u32],
        cache: &AnyCache,
        dflash: DFlashSnapshot,
    ) -> Result<PreparedPairedPrefix, PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "paired radix preparation requires the process MLX execution gate"
        );
        let prepared = Self::prepare_pair_target(ticket, prefix_tokens, cache)?;
        let dflash = RadixDFlashSnapshot::new(dflash, &prepared.tokens, prepared.target_bytes)?;
        Ok(prepared.attach_dflash(dflash))
    }

    fn prepare_pair_target(
        ticket: PairedPrepareTicket,
        prefix_tokens: &[u32],
        cache: &AnyCache,
    ) -> Result<PreparedPairTarget, PairedCacheError> {
        let mut prepared =
            Self::prepare_store_with_block_size(ticket.block_size, prefix_tokens, cache)
                .ok_or_else(|| PairedCacheError::TargetMaterialization {
                    details: "target cache cannot be represented at this radix key".to_owned(),
                })?;
        validate_prepared_pair_boundary(&prepared)?;
        let stored_len = prepared.total_tokens;
        let tokens = prefix_tokens
            .get(..stored_len)
            .ok_or(PairedCacheError::PrefixMismatch {
                stored_len,
                requested_len: prefix_tokens.len(),
            })?;

        // A cloned endpoint is the existing target path for hybrid / fallback
        // caches. Freeze that one stored target before the live cache advances;
        // unlike the previous implementation, no second target copy is kept in
        // the dFlash sidecar.
        if matches!(prepared.endpoint, CachedData::Cloned(_)) {
            let frozen = cache.try_deep_clone().map_err(|error| {
                PairedCacheError::TargetMaterialization {
                    details: error.to_string(),
                }
            })?;
            let expected = i32::try_from(stored_len)
                .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: stored_len })?;
            frozen
                .validate_absolute_boundary(expected)
                .map_err(|error| PairedCacheError::TargetBoundary {
                    expected,
                    details: error.to_string(),
                })?;
            prepared.endpoint = CachedData::Cloned(frozen);
        }

        let target_bytes = prepared_target_bytes(&prepared);
        Ok(PreparedPairTarget {
            ticket,
            tokens: tokens.into(),
            blocks: prepared.blocks,
            context: prepared.context,
            target: prepared.endpoint,
            target_bytes,
        })
    }

    /// Publish a fully prepared pair using only ownership moves and trie/LRU
    /// mutation. No MLX graph construction, evaluation, or dFlash fork occurs.
    pub(crate) fn commit_prepared_pair(
        &mut self,
        prepared: PreparedPairedPrefix,
    ) -> Result<(), PairedCacheError> {
        if prepared.ticket.instance_id != self.instance_id {
            return Err(PairedCacheError::ForeignCacheInstance {
                prepared: prepared.ticket.instance_id.0,
                current: self.instance_id.0,
            });
        }
        if prepared.ticket.block_size != self.block_size {
            return Err(PairedCacheError::TargetMaterialization {
                details: format!(
                    "paired prefix was prepared for block size {} but cache uses {}",
                    prepared.ticket.block_size, self.block_size
                ),
            });
        }
        if prepared.ticket.epoch != self.epoch {
            return Err(PairedCacheError::StaleEpoch {
                prepared: prepared.ticket.epoch.0,
                current: self.epoch.0,
            });
        }
        if prepared.ticket.revision != self.revision {
            return Err(PairedCacheError::StaleRevision {
                prepared: prepared.ticket.revision.0,
                current: self.revision.0,
            });
        }
        if self.max_cached == 0 {
            return Ok(());
        }
        let PreparedPairedPrefix {
            ticket: _,
            tokens,
            blocks,
            context,
            endpoint,
        } = prepared;
        let stored_len = tokens.len();
        self.publish_prepared(&tokens, stored_len, blocks, context, endpoint);
        Ok(())
    }

    #[cfg(test)]
    fn store_paired(
        &mut self,
        prefix_tokens: &[u32],
        cache: &AnyCache,
        dflash: DFlashSnapshot,
    ) -> Result<(), PairedCacheError> {
        let ticket = self.paired_prepare_ticket();
        let _exec = higgs_models::mlx_exec::acquire();
        let prepared =
            Self::prepare_paired_prefix_from_parts(ticket, prefix_tokens, cache, dflash)?;
        self.commit_prepared_pair(prepared)
    }

    fn prepare_store(&self, prefix_tokens: &[u32], cache: &AnyCache) -> Option<PreparedStore> {
        Self::prepare_store_with_block_size(self.block_size, prefix_tokens, cache)
    }

    fn prepare_store_with_block_size(
        block_size: usize,
        prefix_tokens: &[u32],
        cache: &AnyCache,
    ) -> Option<PreparedStore> {
        // Dense KV may legitimately extend beyond a stripped key: paging slices
        // only the key's complete blocks. Whole-clone fallbacks cannot be
        // truncated and therefore require an exact cache/key boundary.
        let prepared = match slice_into_blocks(cache, block_size, prefix_tokens.len()) {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::warn!(error = %error, "Failed to page cache, using clone fallback");
                let offset = kv_offset(cache).and_then(|value| usize::try_from(value).ok());
                if offset.is_some_and(|value| value != prefix_tokens.len()) {
                    tracing::warn!(
                        cache_tokens = offset.unwrap_or_default(),
                        key_tokens = prefix_tokens.len(),
                        "Skipping whole-clone prefix cache store at a different boundary"
                    );
                    return None;
                }
                PreparedStore {
                    blocks: None,
                    context: None,
                    total_tokens: prefix_tokens.len(),
                    endpoint: CachedData::Cloned(cache.clone()),
                }
            }
        };
        if matches!(&prepared.endpoint, CachedData::Cloned(_)) {
            let offset = kv_offset(cache).and_then(|value| usize::try_from(value).ok());
            if offset.is_some_and(|value| value != prepared.total_tokens) {
                tracing::warn!(
                    cache_tokens = offset.unwrap_or_default(),
                    key_tokens = prepared.total_tokens,
                    "Skipping whole-clone prefix cache store at a different boundary"
                );
                return None;
            }
        }
        Some(prepared)
    }

    fn next_entry_id(&mut self) -> u64 {
        let entry_id = self.next_entry_id;
        self.next_entry_id = self
            .next_entry_id
            .checked_add(1)
            .expect("paired radix entry ID exhausted");
        entry_id
    }

    fn next_access(&mut self) -> u64 {
        let access = self.access_clock;
        self.access_clock = self
            .access_clock
            .checked_add(1)
            .expect("paired radix access clock exhausted");
        access
    }

    fn advance_revision(&mut self) {
        self.revision = CachePublicationRevision(
            self.revision
                .0
                .checked_add(1)
                .expect("paired radix publication revision exhausted"),
        );
    }

    fn publish_prepared(
        &mut self,
        prefix_tokens: &[u32],
        stored_len: usize,
        blocks: Option<Vec<CachedLayerData>>,
        context: Option<Arc<TurboQuantContext>>,
        endpoint: CachedEndpoint,
    ) {
        let Some(tokens_to_store) = prefix_tokens.get(..stored_len) else {
            return;
        };
        let entry_id = self.next_entry_id();
        let last_accessed = self.next_access();
        let ctx = Ctx {
            block_size: self.block_size,
            context,
            entry_id,
            last_accessed,
            last_accessed_at: Instant::now(),
        };
        let outcome = Self::insert(
            &mut self.root,
            tokens_to_store,
            0,
            blocks.clone(),
            blocks.as_ref(),
            endpoint,
            &ctx,
        );

        if outcome.added {
            self.num_cached += 1;
        }
        match (outcome.was_paired, outcome.is_paired) {
            (false, true) => self.num_paired += 1,
            (true, false) => {
                debug_assert!(self.num_paired > 0, "paired endpoint count underflow");
                self.num_paired = self.num_paired.saturating_sub(1);
            }
            (false, false) | (true, true) => {}
        }
        while self.num_paired > self.max_paired {
            self.evict_oldest_paired();
        }
        while self.num_cached > self.max_cached {
            self.evict_lru();
        }
        self.advance_revision();
    }

    /// Insert `tokens` (with optional per-layer block run `blocks` spanning all
    /// of `tokens`) marking the terminal node with `endpoint`.
    ///
    /// `blocks` covers exactly the tokens still being placed (`[pos, len)`).
    /// As the trie is descended, the block run is sliced so that each edge
    /// segment carries precisely the blocks for the tokens on that edge. When
    /// the descent reaches an already-stored edge whose tokens match, the
    /// existing edge blocks are reused and the incoming ones for that span are
    /// discarded -- this is where shared prefixes deduplicate.
    fn insert(
        node: &mut RadixNode,
        tokens: &[u32],
        pos: usize,
        blocks: Option<Vec<CachedLayerData>>,
        full_blocks: Option<&Vec<CachedLayerData>>,
        endpoint: CachedEndpoint,
        ctx: &Ctx,
    ) -> InsertOutcome {
        let block_size = ctx.block_size;
        if pos >= tokens.len() {
            let is_new = node.cached.is_none();
            // Overwrite: refresh this terminal edge's blocks from the full
            // incoming run so a re-store with changed KV replaces the stale
            // blocks. (For identical tokens the model produces identical KV, so
            // this is a no-op in production; it keeps re-stores correct.) The
            // terminal edge owns the LAST `edge.len()/block_size` blocks of the
            // sequence; slice those from the full run.
            if !is_new {
                if let Some(full) = full_blocks {
                    let edge_blocks = node.edge_blocks.tokens() / block_size;
                    let refreshed: Vec<CachedLayerData> = full
                        .iter()
                        .map(|l| l.take_last_blocks(edge_blocks))
                        .collect();
                    node.edge_blocks = edge_blocks_from(Some(refreshed), ctx.context.clone());
                }
            }
            let previous = node.cached.take();
            let was_paired = previous
                .as_ref()
                .is_some_and(|cached| cached.endpoint.is_paired());
            let is_paired = endpoint.is_paired();
            node.cached = Some(CachedState {
                endpoint,
                entry_id: ctx.entry_id,
                last_accessed: Cell::new(ctx.last_accessed),
                last_accessed_at: Cell::new(ctx.last_accessed_at),
            });
            return InsertOutcome {
                added: is_new,
                was_paired,
                is_paired,
            };
        }

        let Some(&next_token) = tokens.get(pos) else {
            return InsertOutcome::default();
        };

        if node.children.contains_key(&next_token) {
            let Some(child) = node.children.get(&next_token) else {
                return InsertOutcome::default();
            };

            let remaining = tokens.get(pos..).unwrap_or_default();
            let common = child
                .edge
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common == child.edge.len() {
                // Whole child edge matched: its blocks are reused as-is. Drop
                // the incoming blocks for this span (they are byte-identical KV)
                // and recurse with the remainder.
                let remainder = drop_leading_blocks(blocks, common, block_size);
                let Some(child_mut) = node.children.get_mut(&next_token) else {
                    return InsertOutcome::default();
                };
                return Self::insert(
                    child_mut,
                    tokens,
                    pos + common,
                    remainder,
                    full_blocks,
                    endpoint,
                    ctx,
                );
            }

            // Partial match -- split the child edge at `common`.
            let Some(mut old_child) = node.children.remove(&next_token) else {
                return InsertOutcome::default();
            };

            let common_edge = old_child.edge.get(..common).unwrap_or_default().to_vec();
            let leftover_edge = old_child.edge.get(common..).unwrap_or_default().to_vec();

            let Some(&leftover_key) = leftover_edge.first() else {
                return InsertOutcome::default();
            };
            old_child.edge = leftover_edge;

            // Split the existing edge's blocks at the same token boundary so the
            // shared leading blocks live on the new `split` parent (referenced by
            // both children's paths) and the rest stay on `old_child`.
            let (shared_blocks, leftover_blocks) =
                split_edge_blocks(old_child.edge_blocks, common, block_size);
            old_child.edge_blocks = leftover_blocks;

            let mut split = RadixNode {
                edge: common_edge,
                edge_blocks: shared_blocks,
                cached: None,
                children: HashMap::new(),
            };
            split.children.insert(leftover_key, old_child);

            // The incoming blocks for `[pos, pos+common)` are duplicates of the
            // shared blocks now on `split`; drop them and keep the remainder.
            let remainder = drop_leading_blocks(blocks, common, block_size);

            if pos + common >= tokens.len() {
                let is_paired = endpoint.is_paired();
                split.cached = Some(CachedState {
                    endpoint,
                    entry_id: ctx.entry_id,
                    last_accessed: Cell::new(ctx.last_accessed),
                    last_accessed_at: Cell::new(ctx.last_accessed_at),
                });
                node.children.insert(next_token, split);
                return InsertOutcome {
                    added: true,
                    was_paired: false,
                    is_paired,
                };
            }

            let new_edge = tokens.get(pos + common..).unwrap_or_default().to_vec();
            let Some(&new_key) = new_edge.first() else {
                node.children.insert(next_token, split);
                return InsertOutcome::default();
            };
            let is_paired = endpoint.is_paired();
            let new_leaf = RadixNode::leaf(
                new_edge,
                edge_blocks_from(remainder, ctx.context.clone()),
                endpoint,
                ctx.entry_id,
                ctx.last_accessed,
                ctx.last_accessed_at,
            );
            split.children.insert(new_key, new_leaf);

            node.children.insert(next_token, split);
            return InsertOutcome {
                added: true,
                was_paired: false,
                is_paired,
            };
        }

        // No matching child -- create a new leaf carrying all remaining blocks.
        let new_edge = tokens.get(pos..).unwrap_or_default().to_vec();
        let is_paired = endpoint.is_paired();
        let new_leaf = RadixNode::leaf(
            new_edge,
            edge_blocks_from(blocks, ctx.context.clone()),
            endpoint,
            ctx.entry_id,
            ctx.last_accessed,
            ctx.last_accessed_at,
        );
        node.children.insert(next_token, new_leaf);
        InsertOutcome {
            added: true,
            was_paired: false,
            is_paired,
        }
    }

    fn remove_exact_paired_endpoint(&mut self, prefix_tokens: &[u32]) {
        if self.root.remove_exact_paired_endpoint(prefix_tokens) {
            debug_assert!(self.num_cached > 0, "paired endpoint count drift");
            debug_assert!(self.num_paired > 0, "paired endpoint count drift");
            self.num_cached -= 1;
            self.num_paired -= 1;
            self.root.prune();
        }
    }

    fn evict_lru(&mut self) {
        if let Some((_, entry_id)) = self.root.oldest_cached() {
            if let Some(was_paired) = self.root.remove_cached_by_id(entry_id) {
                self.num_cached -= 1;
                if was_paired {
                    self.num_paired -= 1;
                }
                self.root.prune();
            }
        }
    }

    fn evict_oldest_paired(&mut self) {
        if let Some((_, entry_id)) = self.root.oldest_paired()
            && let Some(was_paired) = self.root.remove_cached_by_id(entry_id)
        {
            debug_assert!(was_paired, "paired LRU selected a target-only endpoint");
            self.num_cached -= 1;
            self.num_paired -= 1;
            self.root.prune();
        }
    }

    pub const fn len(&self) -> usize {
        self.num_cached
    }

    pub const fn is_empty(&self) -> bool {
        self.num_cached == 0
    }

    pub const fn paired_entry_count(&self) -> usize {
        self.num_paired
    }

    /// Refresh paired LRU after successful post-lock materialization.
    ///
    /// Epoch and entry identity make stale tokens harmless: clearing or
    /// replacing an endpoint invalidates every previously selected touch.
    pub(crate) fn touch_paired(&mut self, token: PairedTouchToken) -> bool {
        self.evict_idle_paired();
        if token.instance_id != self.instance_id || token.epoch != self.epoch {
            return false;
        }
        let access = self.access_clock;
        if !self
            .root
            .touch_entry(token.entry_id, access, Instant::now(), true)
        {
            return false;
        }
        self.access_clock = self
            .access_clock
            .checked_add(1)
            .expect("paired radix access clock exhausted");
        true
    }

    /// Evict every expired paired endpoint as one target+dSpark ownership unit.
    ///
    /// The publication revision advances once when anything is removed, so a
    /// pair prepared against the pre-expiry trie cannot later commit through a
    /// stale ticket.
    pub(crate) fn evict_idle_paired(&mut self) -> usize {
        self.evict_idle_paired_at(Instant::now())
    }

    fn evict_idle_paired_at(&mut self, now: Instant) -> usize {
        let Some(ttl) = self.paired_idle_ttl else {
            return 0;
        };
        let Some(cutoff) = now.checked_sub(ttl) else {
            return 0;
        };
        let removed = self.root.remove_expired_paired(cutoff);
        if removed == 0 {
            return 0;
        }
        debug_assert!(removed <= self.num_paired);
        debug_assert!(removed <= self.num_cached);
        self.num_paired = self.num_paired.saturating_sub(removed);
        self.num_cached = self.num_cached.saturating_sub(removed);
        self.root.prune();
        self.advance_revision();
        removed
    }

    pub fn paired_stats(&self) -> PairedPrefixCacheStats {
        let mut stats = PairedPrefixCacheStats::default();
        self.root.accumulate_paired_stats(&mut stats);
        debug_assert_eq!(
            stats.entries, self.num_paired,
            "paired radix accounting drifted from the trie"
        );
        stats
    }

    pub fn clear(&mut self) {
        self.epoch = PairedCacheEpoch(
            self.epoch
                .0
                .checked_add(1)
                .expect("paired cache epoch exhausted"),
        );
        self.root = RadixNode::empty();
        self.num_cached = 0;
        self.num_paired = 0;
    }

    /// Test-only: collect, per layer-0 dense block, its `Arc` pointer identity
    /// and `strong_count` across every edge in the trie. Distinct pointers ==
    /// distinct stored blocks (no duplication); a `strong_count > 1` means the
    /// block is physically shared by multiple prefixes' paths.
    #[cfg(test)]
    #[allow(clippy::as_conversions)]
    fn layer0_block_stats(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.root.collect_layer0_blocks(&mut out);
        out
    }
}

#[cfg(test)]
#[allow(clippy::as_conversions)]
impl RadixNode {
    /// Walk the trie, appending `(arc_ptr_as_usize, strong_count)` for the
    /// layer-0 dense KV blocks on every edge.
    fn collect_layer0_blocks(&self, out: &mut Vec<(usize, usize)>) {
        if let EdgeData::Paged(blocks) = &self.edge_blocks {
            if let Some(CachedLayerData::Kv(layer0)) = blocks.layers.first() {
                for b in layer0 {
                    out.push((Arc::as_ptr(b) as usize, Arc::strong_count(b)));
                }
            }
        }
        for child in self.children.values() {
            child.collect_layer0_blocks(out);
        }
    }
}

// ---------------------------------------------------------------------------
// Slice & materialize helpers
// ---------------------------------------------------------------------------

/// Check if any layer in the cache uses `TurboQuant`.
#[allow(dead_code)]
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

/// Outcome of preparing a cache for storage: the per-layer block run (if paged),
/// how many tokens it covers, and the endpoint metadata for the trie node.
struct PreparedStore {
    /// `None` for `Cloned` endpoints (no path blocks).
    blocks: Option<Vec<CachedLayerData>>,
    /// Shared `TurboQuant` context for paged-TQ blocks; `None` for dense/clone.
    context: Option<Arc<TurboQuantContext>>,
    total_tokens: usize,
    endpoint: CachedData,
}

/// Prepared target representation waiting for the exact checkpoint-owned
/// dFlash sidecar. This intermediate cannot be committed to the radix.
struct PreparedPairTarget {
    ticket: PairedPrepareTicket,
    tokens: Box<[u32]>,
    blocks: Option<Vec<CachedLayerData>>,
    context: Option<Arc<TurboQuantContext>>,
    target: CachedData,
    target_bytes: usize,
}

impl PreparedPairTarget {
    fn attach_dflash(self, dflash: RadixDFlashSnapshot) -> PreparedPairedPrefix {
        PreparedPairedPrefix {
            ticket: self.ticket,
            tokens: self.tokens,
            blocks: self.blocks,
            context: self.context,
            endpoint: CachedEndpoint::TargetAndDflash {
                target: self.target,
                dflash: Arc::new(dflash),
            },
        }
    }
}

/// Opaque, fully evaluated paired endpoint ready for a CPU-only radix commit.
pub(crate) struct PreparedPairedPrefix {
    ticket: PairedPrepareTicket,
    tokens: Box<[u32]>,
    blocks: Option<Vec<CachedLayerData>>,
    context: Option<Arc<TurboQuantContext>>,
    endpoint: CachedEndpoint,
}

fn validate_prepared_pair_boundary(prepared: &PreparedStore) -> Result<(), PairedCacheError> {
    let expected = i32::try_from(prepared.total_tokens).map_err(|_| {
        PairedCacheError::PrefixLengthOverflow {
            len: prepared.total_tokens,
        }
    })?;
    match &prepared.endpoint {
        CachedData::Cloned(cache) => cache.validate_absolute_boundary(expected).map_err(|error| {
            PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            }
        }),
        CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => {
            let layers =
                prepared
                    .blocks
                    .as_ref()
                    .ok_or_else(|| PairedCacheError::TargetBoundary {
                        expected,
                        details: "paged endpoint has no target block payload".to_owned(),
                    })?;
            if layers.is_empty() {
                return Err(PairedCacheError::TargetBoundary {
                    expected,
                    details: "paged endpoint has no target layers".to_owned(),
                });
            }
            for (index, layer) in layers.iter().enumerate() {
                let actual = match layer {
                    CachedLayerData::Kv(blocks) => blocks.iter().fold(0usize, |total, block| {
                        total.saturating_add(
                            block
                                .keys
                                .shape()
                                .get(2)
                                .copied()
                                .and_then(|value| usize::try_from(value).ok())
                                .unwrap_or(0),
                        )
                    }),
                    CachedLayerData::TurboQuantKv(blocks) => {
                        blocks.iter().fold(0usize, |total, block| {
                            total.saturating_add(
                                block
                                    .key_norms
                                    .shape()
                                    .get(1)
                                    .copied()
                                    .and_then(|value| usize::try_from(value).ok())
                                    .unwrap_or(0),
                            )
                        })
                    }
                    CachedLayerData::Gdn(_) | CachedLayerData::Empty => 0,
                };
                if actual != prepared.total_tokens {
                    return Err(PairedCacheError::TargetBoundary {
                        expected,
                        details: format!(
                            "paged target layer {index} covers {actual} tokens instead of {}",
                            prepared.total_tokens
                        ),
                    });
                }
            }
            Ok(())
        }
    }
}

fn prepared_target_bytes(prepared: &PreparedStore) -> usize {
    match &prepared.endpoint {
        CachedData::Cloned(cache) => cache.estimated_bytes(),
        CachedData::Paged { .. } | CachedData::TurboQuantPaged { .. } => prepared
            .blocks
            .as_deref()
            .map_or(0, estimated_cached_layers_bytes),
    }
}

fn estimated_cached_layers_bytes(layers: &[CachedLayerData]) -> usize {
    layers.iter().fold(0usize, |total, layer| {
        total.saturating_add(match layer {
            CachedLayerData::Kv(blocks) => blocks.iter().fold(0usize, |layer_total, block| {
                layer_total
                    .saturating_add(block.keys.nbytes())
                    .saturating_add(block.values.nbytes())
            }),
            CachedLayerData::TurboQuantKv(blocks) => {
                blocks.iter().fold(0usize, |layer_total, block| {
                    layer_total
                        .saturating_add(block.key_codes.nbytes())
                        .saturating_add(block.key_norms.nbytes())
                        .saturating_add(block.key_gammas.nbytes())
                        .saturating_add(block.value_codes.nbytes())
                        .saturating_add(block.value_norms.nbytes())
                })
            }
            CachedLayerData::Gdn(snapshot) => snapshot
                .conv_state
                .as_ref()
                .map_or(0, Array::nbytes)
                .saturating_add(snapshot.ssm_state.as_ref().map_or(0, Array::nbytes)),
            CachedLayerData::Empty => 0,
        })
    })
}

/// Slice a cache into block-aligned paged data.
fn slice_into_blocks(
    cache: &AnyCache,
    block_size: usize,
    max_tokens: usize,
) -> Result<PreparedStore, Exception> {
    // Hybrid caches (GDN+KV) can't be block-paged because GDN sequential state
    // doesn't align to block boundaries. The KV offset would mismatch the GDN
    // offset after materialization, producing corrupt attention. Use clone instead.
    let AnyCache::KV(kv_layers) = cache else {
        return Ok(PreparedStore {
            blocks: None,
            context: None,
            total_tokens: max_tokens,
            endpoint: CachedData::Cloned(cache.clone()),
        });
    };

    let offset = kv_offset(cache).unwrap_or(0);
    let offset_usize = usize::try_from(offset).unwrap_or(0);
    let num_blocks = offset_usize.min(max_tokens) / block_size;
    if num_blocks == 0 {
        return Err(Exception::custom("Cache too short for paging"));
    }
    let total_tokens = num_blocks * block_size;
    let block_size_i32 =
        i32::try_from(block_size).map_err(|_| Exception::custom("block_size overflow"))?;

    // Slice KV layers as TQ blocks when actually quantized, dense otherwise.
    let mut tq_context: Option<Arc<TurboQuantContext>> = None;
    let layers: Vec<CachedLayerData> = kv_layers
        .iter()
        .map(|layer_opt| {
            let Some(kv) = layer_opt.as_ref() else {
                return Ok(CachedLayerData::Empty);
            };
            if kv.is_quantized() {
                if tq_context.is_none() {
                    tq_context = kv.turbo_arrays().map(|(c, ..)| Arc::clone(c));
                }
                slice_tq_layer(kv, num_blocks, block_size_i32)
            } else {
                slice_kv_layer(Some(kv), num_blocks, block_size_i32)
            }
        })
        .collect::<Result<_, _>>()?;

    let endpoint = tq_context
        .as_ref()
        .map_or(CachedData::Paged { is_hybrid: false }, |context| {
            CachedData::TurboQuantPaged {
                context: Arc::clone(context),
                is_hybrid: false,
            }
        });

    Ok(PreparedStore {
        blocks: Some(layers),
        context: tq_context,
        total_tokens,
        endpoint,
    })
}

/// Slice a single `TurboQuant` KV layer into blocks along axis 1.
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
        blocks.push(TqBlock::new(
            slice_axis1(key_codes, start, end)?,
            slice_axis1(key_norms, start, end)?,
            slice_axis1(key_gammas, start, end)?,
            slice_axis1(value_codes, start, end)?,
            slice_axis1(value_norms, start, end)?,
        )?);
    }

    Ok(CachedLayerData::TurboQuantKv(blocks))
}

/// Slice a single KV layer into blocks.
fn slice_kv_layer(
    kv_opt: Option<&SteppingKeyValueCache>,
    num_blocks: usize,
    block_size: i32,
) -> Result<CachedLayerData, Exception> {
    let Some(kv) = kv_opt else {
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
        blocks.push(KvBlock::new(k, v)?);
    }

    Ok(CachedLayerData::Kv(blocks))
}

/// Flatten the per-edge block runs along a root -> node path (plus an optional
/// partially-matched final edge) into a single per-layer block run, in order.
///
/// Each `EdgeBlocks` has the same layer layout, so layer `l`'s full run is the
/// in-order concatenation of every edge's `layers[l]`.
fn flatten_path_layers(
    full_path: &[&EdgeBlocks],
    partial_tail: Option<&EdgeBlocks>,
) -> Vec<CachedLayerData> {
    let mut out: Vec<CachedLayerData> = Vec::new();
    let segments = full_path.iter().copied().chain(partial_tail);
    for edge in segments {
        if out.is_empty() {
            out.clone_from(&edge.layers);
        } else {
            for (acc, seg) in out.iter_mut().zip(edge.layers.iter()) {
                acc.append_from(seg.clone());
            }
        }
    }
    out
}

/// Freeze an owned target reconstruction plan while the endpoint and its
/// dFlash sidecar are selected under the same radix lock.
///
/// All operations here are CPU-only immutable handle clones. A later endpoint
/// refresh can replace trie storage without changing this already-selected
/// target/drafter pair.
fn own_target_materialization_plan(
    matched: &MatchResult<'_>,
) -> Result<TargetMaterializationPlan, PairedCacheError> {
    let MatchEndpoint::Stored(endpoint) = &matched.kind else {
        return Err(PairedCacheError::TargetMaterialization {
            details: "paired lookup selected a structural target endpoint".to_owned(),
        });
    };
    match endpoint.target() {
        CachedData::Cloned(cache) => Ok(TargetMaterializationPlan::Cloned(cache.clone())),
        CachedData::Paged { is_hybrid } => Ok(TargetMaterializationPlan::DensePaged {
            layers: flatten_path_layers(&matched.full_path, matched.partial_tail.as_ref()),
            is_hybrid: *is_hybrid,
        }),
        CachedData::TurboQuantPaged {
            context, is_hybrid, ..
        } => Ok(TargetMaterializationPlan::TurboQuantPaged {
            layers: flatten_path_layers(&matched.full_path, matched.partial_tail.as_ref()),
            context: Arc::clone(context),
            is_hybrid: *is_hybrid,
        }),
    }
}

/// Materialize a matched prefix into a contiguous `AnyCache`.
///
/// Blocks are gathered from every edge along the match's `full_path` (root ->
/// matched node) plus any `partial_tail` (leading whole blocks of a partially
/// matched final edge). Shared leading edges contribute their blocks exactly
/// once, so the reconstruction is byte-identical to the originally stored KV
/// for the matched span.
fn materialize(m: &MatchResult) -> Result<AnyCache, Exception> {
    match &m.kind {
        // deep_clone (not shallow clone): a Hybrid Cloned cache's KV layers
        // update IN PLACE, so a shallow clone shares those buffers with the
        // stored radix entry. The reuser's suffix prefill would then mutate the
        // stored entry, corrupting it for every subsequent reuse (cascading
        // wholesale divergence). deep_clone gives the reuser independent
        // buffers and leaves the stored entry frozen. Dense paged caches are
        // reconstructed from blocks below and are already independent.
        MatchEndpoint::Stored(endpoint) => match endpoint.target() {
            CachedData::Cloned(cache) => try_clone_target_for_materialization(cache),
            CachedData::Paged { is_hybrid, .. } => {
                let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
                if *is_hybrid {
                    materialize_hybrid(&layers)
                } else {
                    materialize_kv(&layers)
                }
            }
            CachedData::TurboQuantPaged {
                context, is_hybrid, ..
            } => {
                let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
                if *is_hybrid {
                    materialize_tq_hybrid(&layers, context)
                } else {
                    materialize_tq_kv(&layers, context)
                }
            }
        },
        MatchEndpoint::PartialPaged { context } => {
            let layers = flatten_path_layers(&m.full_path, m.partial_tail.as_ref());
            context.as_ref().map_or_else(
                || materialize_kv(&layers),
                |ctx| materialize_tq_kv(&layers, ctx),
            )
        }
    }
}

fn try_clone_target_for_materialization(cache: &AnyCache) -> Result<AnyCache, Exception> {
    #[cfg(test)]
    if FAIL_NEXT_CLONED_TARGET_MATERIALIZATION.with(|fail| fail.replace(false)) {
        return Err(Exception::custom(
            "injected cloned target materialization failure",
        ));
    }
    cache.try_deep_clone()
}

#[cfg(test)]
fn fail_next_cloned_target_materialization_for_test() {
    FAIL_NEXT_CLONED_TARGET_MATERIALIZATION.with(|fail| fail.set(true));
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
fn gather_blocks(blocks: &[Arc<KvBlock>]) -> Result<SteppingKeyValueCache, Exception> {
    let Some(first) = blocks.first() else {
        return Ok(SteppingKeyValueCache::new());
    };

    if blocks.len() == 1 {
        return SteppingKeyValueCache::from_arrays(first.keys.clone(), first.values.clone());
    }

    let key_arrays: Vec<Array> = blocks.iter().map(|b| b.keys.clone()).collect();
    let value_arrays: Vec<Array> = blocks.iter().map(|b| b.values.clone()).collect();
    let keys = concatenate_axis(&key_arrays, 2)?;
    let values = concatenate_axis(&value_arrays, 2)?;

    SteppingKeyValueCache::from_arrays(keys, values)
}

/// Gather TQ blocks into a single `SteppingKeyValueCache` with TQ storage.
fn gather_tq_blocks(
    blocks: &[Arc<TqBlock>],
    context: &Arc<TurboQuantContext>,
) -> Result<SteppingKeyValueCache, Exception> {
    if blocks.is_empty() {
        return Ok(SteppingKeyValueCache::new());
    }

    // Concatenate all block arrays along axis 1 (the sequence dimension).
    let concat1 = |arrays: Vec<Array>| -> Result<Array, Exception> {
        match arrays.len() {
            0 => Err(Exception::custom("empty TQ block array")),
            1 => arrays
                .into_iter()
                .next()
                .ok_or_else(|| Exception::custom("empty TQ block array")),
            _ => concatenate_axis(&arrays, 1),
        }
    };

    let key_codes = concat1(blocks.iter().map(|b| b.key_codes.clone()).collect())?;
    let key_norms = concat1(blocks.iter().map(|b| b.key_norms.clone()).collect())?;
    let key_gammas = concat1(blocks.iter().map(|b| b.key_gammas.clone()).collect())?;
    let value_codes = concat1(blocks.iter().map(|b| b.value_codes.clone()).collect())?;
    let value_norms = concat1(blocks.iter().map(|b| b.value_norms.clone()).collect())?;

    // Total tokens = sum of block sizes along axis 1.
    let total = key_norms.shape().get(1).copied().unwrap_or(0);

    SteppingKeyValueCache::from_turbo_arrays(
        Arc::clone(context),
        key_codes,
        key_norms,
        key_gammas,
        value_codes,
        value_norms,
        total,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::identity_op,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::disallowed_methods
)]
mod tests {
    use super::*;
    use crate::cache::paired::DflashTapFrontier;
    use higgs_models::cache::KeyValueCache;
    use higgs_models::dflash::{DFlashConfig, DFlashDrafter, DFlashSnapshot};

    /// Create a KV cache with `num_layers` layers, each containing `seq_len`
    /// tokens of shape `[1, 2, seq_len, 8]`.
    fn make_kv_cache(num_layers: usize, seq_len: i32) -> AnyCache {
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|_| {
                let keys = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                let values = Array::zeros::<f32>(&[1, 2, seq_len, 8]).unwrap();
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
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
                    Some(LayerCache::KV(
                        SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
                    ))
                }
            })
            .collect();
        AnyCache::Hybrid(layers)
    }

    fn make_mutable_hybrid_cache(seq_len: i32) -> AnyCache {
        let mut kv = SteppingKeyValueCache::new();
        let ones = Array::ones::<f32>(&[1, 2, seq_len, 8]).unwrap();
        let (keys, values) = kv.update_and_fetch(ones.clone(), ones).unwrap();
        mlx_rs::transforms::eval([&keys, &values]).unwrap();
        AnyCache::Hybrid(vec![
            Some(LayerCache::KV(kv)),
            Some(LayerCache::Arrays(ArraysCache {
                conv_state: Some(Array::zeros::<f32>(&[1, 4, 4]).unwrap()),
                ssm_state: Some(Array::zeros::<f32>(&[1, 16]).unwrap()),
                conv_pos: 0,
                offset: seq_len,
            })),
        ])
    }

    fn append_hybrid_tail(cache: &mut AnyCache, value: f32) {
        let layers = cache.as_hybrid_mut().unwrap();
        let Some(LayerCache::KV(kv)) = layers.first_mut().and_then(Option::as_mut) else {
            panic!("expected hybrid KV layer");
        };
        let scalar = Array::from_f32(value);
        let keys = Array::full::<f32>(&[1, 2, 1, 8], &scalar).unwrap();
        let values = Array::full::<f32>(&[1, 2, 1, 8], &scalar).unwrap();
        let (keys, values) = kv.update_and_fetch(keys, values).unwrap();
        mlx_rs::transforms::eval([&keys, &values]).unwrap();
        let Some(LayerCache::Arrays(arrays)) = layers.get_mut(1).and_then(Option::as_mut) else {
            panic!("expected hybrid recurrent layer");
        };
        arrays.offset += 1;
    }

    fn hybrid_kv_token_is(cache: &AnyCache, token: i32, expected: f32) -> bool {
        let AnyCache::Hybrid(layers) = cache else {
            panic!("expected hybrid cache");
        };
        let Some(LayerCache::KV(kv)) = layers.first().and_then(Option::as_ref) else {
            panic!("expected hybrid KV layer");
        };
        let token = slice_axis2(kv.keys().unwrap(), token, token + 1).unwrap();
        let scalar = Array::from_f32(expected);
        let expected = Array::full::<f32>(&[1, 2, 1, 8], &scalar).unwrap();
        let all = token.array_eq(&expected, None).unwrap().all(None).unwrap();
        all.eval().unwrap();
        all.item::<bool>()
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
                .map_or(0, KeyValueCache::offset),
            AnyCache::Hybrid(layers) => layers
                .iter()
                .find_map(|l| match l {
                    Some(LayerCache::KV(c)) => Some(KeyValueCache::offset(c)),
                    _ => None,
                })
                .unwrap_or(0),
        }
    }

    fn tiny_dflash_drafter() -> DFlashDrafter {
        let config: DFlashConfig = serde_json::from_str(
            r#"{
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 8,
                "vocab_size": 8,
                "dflash_config": {
                    "target_layer_ids": [0]
                }
            }"#,
        )
        .unwrap();
        DFlashDrafter::new(config).unwrap()
    }

    fn dflash_snapshot(boundary: i32) -> DFlashSnapshot {
        let mut drafter = tiny_dflash_drafter();
        let cache = drafter.make_cache();
        let taps = (boundary > 0)
            .then(|| Array::zeros::<f32>(&[1, boundary, 4]).unwrap())
            .into_iter()
            .collect::<Vec<_>>();
        let _exec = higgs_models::mlx_exec::acquire();
        drafter.seal_after_taps(cache, &taps, boundary).unwrap()
    }

    fn cold_live_pair_with_tokens(tokens: &[u32]) -> (LivePair, DFlashDrafter) {
        assert!(!tokens.is_empty());
        let _exec = higgs_models::mlx_exec::acquire();
        let drafter = tiny_dflash_drafter();
        let target = AnyCache::KV(vec![Some(SteppingKeyValueCache::new())]);
        let pair = LivePair::cold(target, drafter.make_cache(), 1).unwrap();
        let (pair, ()) = pair
            .advance_known(tokens, |exact, target, _dflash, _frontier| {
                let rows = i32::try_from(exact.len()).unwrap();
                let AnyCache::KV(layers) = target else {
                    panic!("test checkpoint target must be dense KV");
                };
                let kv = layers[0].as_mut().unwrap();
                let keys = Array::zeros::<f32>(&[1, 2, rows, 8]).unwrap();
                let values = Array::zeros::<f32>(&[1, 2, rows, 8]).unwrap();
                let (keys, values) = kv.update_and_fetch(keys, values).unwrap();
                mlx_rs::transforms::eval([&keys, &values]).unwrap();
                let taps = Array::zeros::<f32>(&[1, rows, 4]).unwrap();
                let frontier = DflashTapFrontier::new(0, rows, vec![taps], 1).unwrap();
                Ok::<_, String>((frontier, ()))
            })
            .unwrap();
        (pair, drafter)
    }

    fn prepare_proven_checkpoint(
        ticket: PairedPrepareTicket,
        tokens: &[u32],
    ) -> (LivePair, Result<PreparedPairedPrefix, PairedCacheError>) {
        let (pair, mut drafter) = cold_live_pair_with_tokens(tokens);
        let exec = higgs_models::mlx_exec::acquire();
        pair.checkpoint_for_radix(&mut drafter, &exec, |checkpoint| {
            PagedPrefixCache::prepare_paired_prefix(ticket, checkpoint)
        })
        .unwrap()
    }

    fn advance_proven_pair(pair: LivePair, suffix: &[u32]) -> LivePair {
        let _exec = higgs_models::mlx_exec::acquire();
        pair.advance_known(suffix, |exact, target, _dflash, frontier| {
            let rows = i32::try_from(exact.len()).unwrap();
            let AnyCache::KV(layers) = target else {
                panic!("test checkpoint target must be dense KV");
            };
            let kv = layers[0].as_mut().unwrap();
            let keys = Array::ones::<f32>(&[1, 2, rows, 8]).unwrap();
            let values = Array::ones::<f32>(&[1, 2, rows, 8]).unwrap();
            let (keys, values) = kv.update_and_fetch(keys, values).unwrap();
            mlx_rs::transforms::eval([&keys, &values]).unwrap();
            let taps = Array::ones::<f32>(&[1, rows, 4]).unwrap();
            let target_boundary = frontier.target_boundary().checked_add(rows).unwrap();
            let frontier =
                DflashTapFrontier::new(frontier.draft_boundary(), target_boundary, vec![taps], 1)
                    .unwrap();
            Ok::<_, String>((frontier, ()))
        })
        .unwrap()
        .0
    }

    fn find_paired(
        cache: &mut PagedPrefixCache,
        tokens: &[u32],
    ) -> Option<MaterializedPairedPrefix> {
        let plan = cache.plan_longest_paired_prefix(tokens).unwrap()?;
        let (matched, touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };
        assert!(
            cache.touch_paired(touch),
            "a successfully materialized live pair must refresh its retained endpoint"
        );
        Some(matched)
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
    fn dense_cache_longer_than_stripped_key_pages_to_key_floor() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let stripped_key: Vec<u32> = (0..50).collect();
        let full_cache = make_kv_cache(2, 64);

        cache.store(&stripped_key, &full_cache);

        let mut query = stripped_key;
        query.push(999);
        let matched = cache
            .find_longest_prefix(&query)
            .expect("dense cache should slice to the stripped key's block floor");
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
            AnyCache::Hybrid(_) => panic!("Expected KV cache"),
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
            AnyCache::KV(_) => panic!("Expected Hybrid cache"),
        }
    }

    /// Hybrid (GDN/SSM + KV) caches are deliberately NOT block-paged: GDN
    /// sequential state cannot be split at a block boundary without corrupting
    /// attention (the failure mlx-lm #980 reports as "hybrid + paging is
    /// impossible"). higgs sidesteps it by storing the whole cache as a clone.
    ///
    /// This test pins that decision two ways:
    ///   (1) `slice_into_blocks` on a hybrid returns the `Cloned` endpoint with
    ///       no blocks — the moment someone makes hybrid block-paged, this fails;
    ///   (2) the clone round-trips GDN state (`conv_state`/`ssm_state`) and KV
    ///       byte-identically over a non-block-aligned length (65 tokens), proving
    ///       reconstruction is exact and length-agnostic (a paged path would
    ///       realign to 64 and drop the GDN state).
    /// A future change that *correctly* enables hybrid paging must keep this green.
    #[test]
    fn test_hybrid_cache_is_cloned_not_paged_and_byte_identical() {
        // array_eq over the whole array, by value (respects strided slice views).
        let arrays_equal = |a: &Array, b: &Array| -> bool {
            a.array_eq(b, None)
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>()
        };

        // Distinct, non-zero content so byte-identity is meaningful, not vacuous.
        let conv = Array::from_slice(
            &(0..16).map(|x| x as f32 + 1.0).collect::<Vec<_>>(),
            &[1, 4, 4],
        );
        let ssm = Array::from_slice(
            &(0..16)
                .map(|x| (x as f32).mul_add(0.5, -3.0))
                .collect::<Vec<_>>(),
            &[1, 16],
        );
        let seq: i32 = 65; // intentionally NOT a multiple of DEFAULT_BLOCK_SIZE
        let n = (2 * seq * 8) as usize;
        let keys = Array::from_slice(
            &(0..n).map(|x| x as f32 * 0.01).collect::<Vec<_>>(),
            &[1, 2, seq, 8],
        );
        let values = Array::from_slice(
            &(0..n).map(|x| x as f32 * -0.02).collect::<Vec<_>>(),
            &[1, 2, seq, 8],
        );

        let hybrid = AnyCache::Hybrid(vec![
            Some(LayerCache::Arrays(ArraysCache {
                conv_state: Some(conv.clone()),
                ssm_state: Some(ssm.clone()),
                conv_pos: 3,
                offset: seq,
            })),
            Some(LayerCache::KV(
                SteppingKeyValueCache::from_arrays(keys.clone(), values.clone()).unwrap(),
            )),
        ]);

        // (1) Invariant: hybrid is stored as a CLONE, never sliced into blocks.
        let prepared = slice_into_blocks(&hybrid, DEFAULT_BLOCK_SIZE, seq as usize).unwrap();
        assert!(
            prepared.blocks.is_none(),
            "hybrid must NOT be sliced into blocks (GDN state can't split at a boundary)"
        );
        assert!(
            matches!(prepared.endpoint, CachedData::Cloned(_)),
            "hybrid endpoint must be Cloned — the deliberate mlx-lm #980 avoidance"
        );

        // (2) Byte-identity: store -> retrieve reconstructs GDN state + KV exactly,
        // for the full non-block-aligned length.
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let prefix: Vec<u32> = (0..seq as u32).collect();
        cache.store(&prefix, &hybrid);
        let mut query = prefix.clone();
        query.push(999);
        let matched = cache
            .find_longest_prefix(&query)
            .expect("hybrid clone must be retrievable");

        let AnyCache::Hybrid(layers) = &matched.cache else {
            panic!("expected hybrid cache");
        };
        let Some(LayerCache::Arrays(ac)) = layers[0].as_ref() else {
            panic!("layer 0 must be GDN");
        };
        assert!(
            arrays_equal(ac.conv_state.as_ref().unwrap(), &conv),
            "conv_state must round-trip byte-identically"
        );
        assert!(
            arrays_equal(ac.ssm_state.as_ref().unwrap(), &ssm),
            "ssm_state must round-trip byte-identically"
        );
        assert_eq!(ac.conv_pos, 3, "conv_pos must survive the clone");
        assert_eq!(ac.offset, seq, "offset must survive the clone");
        let Some(LayerCache::KV(kv)) = layers[1].as_ref() else {
            panic!("layer 1 must be KV");
        };
        assert!(
            arrays_equal(kv.keys().unwrap(), &keys),
            "KV keys must round-trip byte-identically"
        );
        assert!(
            arrays_equal(kv.values().unwrap(), &values),
            "KV values must round-trip byte-identically"
        );
    }

    #[test]
    fn hybrid_store_skips_clone_when_cache_offset_exceeds_key_len() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);
        let hybrid = make_hybrid_cache(8, 65);
        let stripped_key: Vec<u32> = (0..60).collect();

        cache.store(&stripped_key, &hybrid);

        let mut query = stripped_key;
        query.push(999);
        assert!(
            cache.find_longest_prefix(&query).is_none(),
            "Hybrid clones are whole-cache snapshots; storing offset>key_len would replay stale suffix state"
        );
        assert_eq!(cache.len(), 0);
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
        // Stored via clone fallback since too short for block paging.
        // Clone fallback still makes the prefix findable.
        let mut query: Vec<u32> = prefix;
        query.push(999);
        let matched = cache
            .find_longest_prefix(&query)
            .expect("clone fallback should be findable");
        assert_eq!(matched.prefix_len, 16);
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
        let mut kv = SteppingKeyValueCache::from_arrays(keys, values).unwrap();
        assert_eq!(KeyValueCache::offset(&kv), 32);

        // Simulate a decode step
        let new_k = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let new_v = Array::zeros::<f32>(&[1, 2, 1, 8]).unwrap();
        let (rk, rv) = kv.update_and_fetch(new_k, new_v).unwrap();
        assert_eq!(rk.shape(), &[1, 2, 33, 8]);
        assert_eq!(rv.shape(), &[1, 2, 33, 8]);
        assert_eq!(KeyValueCache::offset(&kv), 33);
    }

    // -- Radix-tree block sharing tests --------------------------------------

    /// KV cache whose K/V values are a deterministic function of absolute token
    /// position, so block content is position-distinct and reconstruction can be
    /// verified byte-for-byte. Element at (token `t`, head `h`, dim `d`) =
    /// `base + t*1000 + h*100 + d`.
    fn make_kv_cache_content(num_layers: usize, seq_len: i32, base: f32) -> AnyCache {
        let s = seq_len as usize;
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|layer| {
                let mut data = vec![0.0_f32; 1 * 2 * s * 8];
                for h in 0..2 {
                    for t in 0..s {
                        for d in 0..8 {
                            let idx = ((h * s) + t) * 8 + d;
                            data[idx] = base
                                + (layer as f32) * 1_000_000.0
                                + (t as f32) * 1000.0
                                + (h as f32) * 100.0
                                + d as f32;
                        }
                    }
                }
                let keys = Array::from_slice(&data, &[1, 2, seq_len, 8]);
                let values = Array::from_slice(&data, &[1, 2, seq_len, 8]);
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    /// Layer-0 keys array of a KV cache.
    fn cache_keys(cache: &AnyCache, layer: usize) -> Array {
        match cache {
            AnyCache::KV(layers) => layers[layer].as_ref().unwrap().keys().unwrap().clone(),
            AnyCache::Hybrid(_) => panic!("expected KV"),
        }
    }

    /// Assert the first `n` tokens (axis 2) of two `[1, H, *, 8]` key arrays are
    /// byte-identical. Uses MLX `array_eq` + `all` so strided slice views are
    /// compared by VALUE (raw `as_slice` would read the contiguous backing
    /// buffer and ignore strides).
    fn assert_keys_eq_first_n(got: &Array, expected: &Array, n: i32) {
        let g = slice_axis2(got, 0, n).unwrap();
        let e = slice_axis2(expected, 0, n).unwrap();
        let eq = g.array_eq(&e, None).unwrap();
        let all = eq.all(None).unwrap();
        assert!(
            all.item::<bool>(),
            "reconstructed keys differ from stored KV over first {n} tokens"
        );
    }

    /// (a) Two inserts sharing a leading prefix must PHYSICALLY share the
    /// overlapping blocks: one stored `Arc` per shared block (not 2x), with a
    /// strong_count reflecting the shared reference.
    #[test]
    fn test_shared_prefix_dedups_blocks() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Both sequences share the first 64 tokens (2 blocks), diverge after.
        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1064);
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(2000..2064);

        cache.store(&seq_a, &make_kv_cache(1, 128));
        cache.store(&seq_b, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);

        // Total distinct layer-0 blocks: 2 shared + 2 (a-only) + 2 (b-only) = 6.
        // Without sharing (storing each prefix's blocks independently) it would
        // be 2*4 = 8. Distinct-Arc count == 6 IS the dedup proof: the shared
        // leading blocks are stored once on the common parent edge, not copied
        // into each prefix's storage.
        let stats = cache.layer0_block_stats();
        assert_eq!(stats.len(), 6, "expected 6 distinct blocks, got {stats:?}");

        // Both prefixes still reconstruct correctly.
        let mut q_a = seq_a.clone();
        q_a.push(9);
        let mut q_b = seq_b.clone();
        q_b.push(9);
        assert_eq!(cache.find_longest_prefix(&q_a).unwrap().prefix_len, 128);
        assert_eq!(cache.find_longest_prefix(&q_b).unwrap().prefix_len, 128);
    }

    /// Stronger sharing proof: store a short prefix, then a longer one extending
    /// it. The short prefix's blocks are REUSED by the long prefix's path (same
    /// Arc), so storing the extension adds no duplicate of the shared blocks.
    #[test]
    fn test_extension_reuses_shared_block_arcs() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let short: Vec<u32> = (0..64).collect(); // 2 blocks
        cache.store(&short, &make_kv_cache(1, 64));
        let before = cache.layer0_block_stats();
        assert_eq!(before.len(), 2);

        // Extend: shares the first 64 tokens, adds 64 more (2 new blocks).
        let long: Vec<u32> = (0..128).collect(); // 4 blocks
        cache.store(&long, &make_kv_cache(1, 128));

        let after = cache.layer0_block_stats();
        // 4 distinct blocks total (2 shared reused + 2 new) -- not 2 + 4 = 6.
        assert_eq!(
            after.len(),
            4,
            "extension must reuse shared blocks: {after:?}"
        );

        // The two original block Arcs are still present by pointer identity.
        let before_ptrs: std::collections::HashSet<usize> =
            before.iter().map(|(p, _)| *p).collect();
        let after_ptrs: std::collections::HashSet<usize> = after.iter().map(|(p, _)| *p).collect();
        assert!(
            before_ptrs.is_subset(&after_ptrs),
            "original shared block Arcs must survive the extension"
        );
    }

    /// (b) `find_longest_prefix` returns the DEEPEST shared match for a query
    /// that overlaps the stored prefix only partially -- including divergence
    /// in the MIDDLE of a block, where only whole shared blocks may be reused.
    #[test]
    fn test_deepest_match_mid_block_divergence() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        // Store 96 tokens (3 blocks) of content-distinct KV.
        let stored: Vec<u32> = (0..96).collect();
        let kv = make_kv_cache_content(2, 96, 7.0);
        let expected_keys = cache_keys(&kv, 0);
        cache.store(&stored, &kv);

        // Query shares 40 tokens then diverges mid-block-2 (block boundary 32).
        // No endpoint is stored at depth 32 -- this exercises the RadixAttention
        // intra-edge block-boundary match.
        let mut query: Vec<u32> = (0..40).collect();
        query.extend(5000..5060);
        let result = cache.find_longest_prefix(&query).unwrap();

        // Deepest block-aligned match below 40 tokens is 32 (1 block).
        assert_eq!(result.prefix_len, 32);
        assert_eq!(kv_cache_offset(&result.cache), 32);

        // Reconstruction must be byte-identical to the first 32 tokens of the
        // originally stored KV.
        assert_keys_eq_first_n(&cache_keys(&result.cache, 0), &expected_keys, 32);
    }

    /// Byte-identical reconstruction across a SHARED block boundary: a query
    /// reusing a fully-shared 2-block prefix rebuilds the exact stored KV.
    #[test]
    fn test_shared_block_reconstruction_byte_identical() {
        let mut cache = PagedPrefixCache::new(10, DEFAULT_BLOCK_SIZE);

        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1032);
        let kv_a = make_kv_cache_content(1, 96, 11.0);
        let expected = cache_keys(&kv_a, 0);
        cache.store(&seq_a, &kv_a);

        // seq_b shares the first 64 tokens, then diverges. After the split the
        // first two blocks live on the shared parent edge.
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(3000..3032);
        cache.store(&seq_b, &make_kv_cache_content(1, 96, 22.0));

        // Query reusing the shared 64-token (2-block) prefix. The match lands on
        // the shared parent edge (no stored endpoint there) via an intra-edge
        // block-boundary match.
        let mut query: Vec<u32> = (0..64).collect();
        query.push(9);
        let result = cache.find_longest_prefix(&query).unwrap();
        assert_eq!(result.prefix_len, 64);

        // The shared blocks came from seq_a (stored first); reconstruction of
        // the first 64 tokens must byte-match seq_a's stored KV.
        assert_keys_eq_first_n(&cache_keys(&result.cache, 0), &expected, 64);
    }

    #[test]
    fn production_checkpoint_derives_identity_and_returns_one_proven_live_pair() {
        let mut cache = PagedPrefixCache::new(10, 1);
        let tokens = vec![7];
        let ticket = cache.paired_prepare_ticket();
        let (continued, prepared) = prepare_proven_checkpoint(ticket, &tokens);
        assert_eq!(continued.token_len(), 1);
        cache.commit_prepared_pair(prepared.unwrap()).unwrap();

        assert!(
            cache.plan_longest_paired_prefix(&[8]).unwrap().is_none(),
            "production publication exposes no external token label that could relabel an equal-length checkpoint"
        );

        let plan = cache
            .plan_longest_paired_prefix(&[7, 99])
            .unwrap()
            .expect("the checkpoint-owned key must select its exact endpoint");
        let (pair, touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize(1).unwrap().into_pair_and_touch()
        };
        assert_eq!(pair.token_len(), 1);
        assert!(cache.touch_paired(touch));
    }

    #[test]
    fn production_checkpoint_rejects_paged_floor_relabelling() {
        let cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..10).collect();
        let ticket = cache.paired_prepare_ticket();
        let (continued, prepared) = prepare_proven_checkpoint(ticket, &tokens);

        assert_eq!(
            continued.token_len(),
            10,
            "a rejected publication must not invalidate its independent live continuation"
        );
        let Err(error) = prepared else {
            panic!("an exact checkpoint must not be relabelled to its paged floor");
        };
        assert!(matches!(
            error,
            PairedCacheError::PrefixMismatch {
                stored_len: 8,
                requested_len: 10
            }
        ));
    }

    #[test]
    fn proven_radix_snapshot_forks_independent_live_pairs() {
        let mut cache = PagedPrefixCache::new(10, 1);
        let tokens = vec![7];
        let ticket = cache.paired_prepare_ticket();
        let (_continued, prepared) = prepare_proven_checkpoint(ticket, &tokens);
        cache.commit_prepared_pair(prepared.unwrap()).unwrap();

        let left_plan = cache.plan_longest_paired_prefix(&[7, 99]).unwrap().unwrap();
        let right_plan = cache
            .plan_longest_paired_prefix(&[7, 100])
            .unwrap()
            .unwrap();
        let (left, _left_touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            left_plan.materialize(1).unwrap().into_pair_and_touch()
        };
        let (right, _right_touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            right_plan.materialize(1).unwrap().into_pair_and_touch()
        };

        let left = advance_proven_pair(left, &[8]);
        assert_eq!(left.token_len(), 2);
        assert_eq!(
            right.token_len(),
            1,
            "advancing one proven fork must not mutate another fork or the retained snapshot"
        );
    }

    #[test]
    fn paired_store_rejects_snapshot_boundary_different_from_prepared_boundary() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..10).collect();
        let target = make_kv_cache(1, 10);

        let error = cache
            .store_paired(&tokens, &target, dflash_snapshot(9))
            .unwrap_err();

        assert!(matches!(
            error,
            crate::cache::paired::PairedCacheError::DFlashBoundary {
                expected: 8,
                actual: 9
            }
        ));
        assert!(cache.is_empty(), "failed paired publication must be atomic");
    }

    #[test]
    fn dense_paired_store_uses_the_existing_block_aligned_target_path() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        cache
            .store_paired(&tokens, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();

        let mut query = tokens;
        query.push(99);
        let matched = find_paired(&mut cache, &query).unwrap();
        assert_eq!(matched.prefix_len, 8);
        assert_eq!(kv_cache_offset(&matched.cache), 8);
        assert_eq!(matched.dflash_cache.position(), 8);

        let mut scratch = Vec::new();
        let selected = cache
            .root
            .find_deepest_match(
                &query,
                0,
                cache.block_size,
                cache.block_size,
                LookupPolicy::TargetAndPairExact,
                &mut scratch,
            )
            .expect("paired endpoint must remain selectable");
        let MatchEndpoint::Stored(endpoint) = selected.kind else {
            panic!("paired lookup must select a stored endpoint");
        };
        assert!(
            matches!(endpoint.target(), CachedData::Paged { .. }),
            "dense paired endpoints must reuse paged target storage instead of owning a second full target clone"
        );
    }

    #[test]
    fn failed_paired_preparation_leaves_populated_trie_and_stats_unchanged() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let retained_pair: Vec<u32> = (0..8).collect();
        let target_only: Vec<u32> = (100..108).collect();
        cache
            .store_paired(&retained_pair, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        cache.store(&target_only, &make_kv_cache(1, 8));
        let before_len = cache.len();
        let before_stats = cache.paired_stats();

        let rejected_tokens: Vec<u32> = (200..208).collect();
        let rejected_target = make_kv_cache(1, 8);
        let wrong_boundary = dflash_snapshot(7);
        let ticket = cache.paired_prepare_ticket();
        let result = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(
                ticket,
                &rejected_tokens,
                &rejected_target,
                wrong_boundary,
            )
        };
        let Err(error) = result else {
            panic!("mismatched dFlash boundary must reject preparation");
        };

        assert!(matches!(
            error,
            crate::cache::paired::PairedCacheError::DFlashBoundary {
                expected: 8,
                actual: 7
            }
        ));
        assert_eq!(cache.len(), before_len);
        assert_eq!(cache.paired_stats(), before_stats);
        let mut paired_query = retained_pair;
        paired_query.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&paired_query)
                .unwrap()
                .is_some(),
            "the existing pair must remain selectable"
        );
        assert!(
            cache.find_longest_prefix(&target_only).is_some(),
            "the ordinary target-only entry must remain selectable"
        );
        let mut rejected_query = rejected_tokens;
        rejected_query.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&rejected_query)
                .unwrap()
                .is_none(),
            "failed preparation must not publish a partial endpoint"
        );
    }

    #[test]
    fn paired_lookup_selects_deepest_qualifying_exact_endpoint() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let pair_short: Vec<u32> = (0..4).collect();
        let target_deeper: Vec<u32> = (0..8).collect();
        let pair_deepest: Vec<u32> = (0..12).collect();
        let target_deepest: Vec<u32> = (0..16).collect();

        cache
            .store_paired(&pair_short, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache.store(&target_deeper, &make_kv_cache(1, 8));
        cache
            .store_paired(&pair_deepest, &make_kv_cache(1, 12), dflash_snapshot(12))
            .unwrap();
        cache.store(&target_deepest, &make_kv_cache(1, 16));

        let query: Vec<u32> = (0..20).collect();
        let matched = find_paired(&mut cache, &query).expect("paired endpoint should match");
        assert_eq!(matched.prefix_len, 12);
        assert_eq!(kv_cache_offset(&matched.cache), 12);
        assert_eq!(matched.dflash_cache.position(), 12);
    }

    #[test]
    fn paired_lookup_selection_is_owned_and_mlx_free() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let stored: Vec<u32> = (0..8).collect();
        cache
            .store_paired(&stored, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        let mut query = stored;
        query.push(99);

        assert!(!higgs_models::mlx_exec::held());
        let plan = cache
            .plan_longest_paired_prefix(&query)
            .unwrap()
            .expect("paired endpoint should produce an owned plan");
        assert_eq!(plan.prefix_len(), 8);

        cache.clear();
        assert!(
            cache.is_empty(),
            "the selected plan must not borrow the trie"
        );

        let _exec = higgs_models::mlx_exec::acquire();
        let (matched, _touch) = plan.materialize_unproven_for_test().unwrap();
        assert_eq!(kv_cache_offset(&matched.cache), 8);
        assert_eq!(matched.dflash_cache.position(), 8);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "paired prefix selection must happen before")]
    fn paired_lookup_selection_rejects_the_mlx_execution_gate() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let stored: Vec<u32> = (0..8).collect();
        cache
            .store_paired(&stored, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        let mut query = stored;
        query.push(99);

        let _exec = higgs_models::mlx_exec::acquire();
        let _ = cache.plan_longest_paired_prefix(&query);
    }

    #[test]
    fn paired_prepare_and_commit_separate_mlx_from_trie_mutation() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let target = make_kv_cache(1, 8);
        let snapshot = dflash_snapshot(8);
        let ticket = cache.paired_prepare_ticket();

        let prepared = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(ticket, &tokens, &target, snapshot)
                .unwrap()
        };
        assert!(
            cache.is_empty(),
            "preparation must not borrow or mutate the radix"
        );
        assert!(!higgs_models::mlx_exec::held());

        cache.commit_prepared_pair(prepared).unwrap();
        assert_eq!(cache.len(), 1);
        let mut query = tokens;
        query.push(99);
        assert!(
            cache.plan_longest_paired_prefix(&query).unwrap().is_some(),
            "CPU-only commit must publish the complete pair"
        );
    }

    #[test]
    fn paired_prepare_ticket_uses_its_captured_block_layout_for_dense_boundaries() {
        let block_four = PagedPrefixCache::new(10, 4).paired_prepare_ticket();
        let block_six = PagedPrefixCache::new(10, 6).paired_prepare_ticket();

        assert_eq!(block_four.store_boundary(13, false), 12);
        assert_eq!(block_six.store_boundary(13, false), 12);
        assert_eq!(block_four.store_boundary(11, false), 8);
        assert_eq!(block_six.store_boundary(11, false), 6);
        assert_eq!(block_four.store_boundary(11, true), 11);
        assert_eq!(block_six.store_boundary(11, true), 11);
    }

    #[test]
    fn paired_prepare_ticket_is_branded_to_one_cache_instance() {
        let source = PagedPrefixCache::new(10, 4);
        let mut destination = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let ticket = source.paired_prepare_ticket();
        let target = make_kv_cache(1, 8);
        let snapshot = dflash_snapshot(8);
        let prepared = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(ticket, &tokens, &target, snapshot)
                .unwrap()
        };

        let error = destination.commit_prepared_pair(prepared).unwrap_err();
        assert!(matches!(
            error,
            crate::cache::paired::PairedCacheError::ForeignCacheInstance { .. }
        ));
        assert!(destination.is_empty());
    }

    #[test]
    fn target_refresh_rejects_a_prepared_pair_from_an_earlier_revision() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        cache
            .store_paired(
                &tokens,
                &make_kv_cache_content(1, 8, 11.0),
                dflash_snapshot(8),
            )
            .unwrap();

        let ticket = cache.paired_prepare_ticket();
        let prepared_target = make_kv_cache_content(1, 8, 17.0);
        let prepared_snapshot = dflash_snapshot(8);
        let prepared = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(
                ticket,
                &tokens,
                &prepared_target,
                prepared_snapshot,
            )
            .unwrap()
        };
        let refreshed = make_kv_cache_content(1, 8, 29.0);
        let refreshed_keys = cache_keys(&refreshed, 0);
        cache.store(&tokens, &refreshed);

        let error = cache.commit_prepared_pair(prepared).unwrap_err();
        assert!(matches!(
            error,
            crate::cache::paired::PairedCacheError::StaleRevision { .. }
        ));
        let mut query = tokens;
        query.push(99);
        assert!(
            cache.plan_longest_paired_prefix(&query).unwrap().is_none(),
            "an older prepared pair must not resurrect continuity after target refresh"
        );
        let matched = cache
            .find_longest_prefix(&query)
            .expect("the refreshed target-only endpoint must survive");
        assert_keys_eq_first_n(&cache_keys(&matched.cache, 0), &refreshed_keys, 8);
    }

    #[test]
    fn first_competing_prepared_pair_commit_wins_the_revision() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let first_tokens: Vec<u32> = (0..8).collect();
        let second_tokens: Vec<u32> = (100..108).collect();
        let ticket = cache.paired_prepare_ticket();
        let first_target = make_kv_cache(1, 8);
        let second_target = make_kv_cache(1, 8);
        let first_snapshot = dflash_snapshot(8);
        let second_snapshot = dflash_snapshot(8);
        let (first, second) = {
            let _exec = higgs_models::mlx_exec::acquire();
            (
                PagedPrefixCache::prepare_paired_prefix_from_parts(
                    ticket,
                    &first_tokens,
                    &first_target,
                    first_snapshot,
                )
                .unwrap(),
                PagedPrefixCache::prepare_paired_prefix_from_parts(
                    ticket,
                    &second_tokens,
                    &second_target,
                    second_snapshot,
                )
                .unwrap(),
            )
        };

        cache.commit_prepared_pair(first).unwrap();
        let error = cache.commit_prepared_pair(second).unwrap_err();
        assert!(matches!(
            error,
            crate::cache::paired::PairedCacheError::StaleRevision { .. }
        ));
        let mut first_query = first_tokens;
        first_query.push(999);
        let mut second_query = second_tokens;
        second_query.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&first_query)
                .unwrap()
                .is_some()
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&second_query)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn clear_rejects_stale_prepared_pair_and_stale_lookup_touch() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        let target = make_kv_cache(1, 8);
        let snapshot = dflash_snapshot(8);
        let ticket = cache.paired_prepare_ticket();
        let prepared = {
            let _exec = higgs_models::mlx_exec::acquire();
            PagedPrefixCache::prepare_paired_prefix_from_parts(ticket, &tokens, &target, snapshot)
                .unwrap()
        };

        cache.clear();
        let result = cache.commit_prepared_pair(prepared);
        assert!(matches!(
            result,
            Err(crate::cache::paired::PairedCacheError::StaleEpoch { .. })
        ));
        assert!(cache.is_empty());
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());

        cache
            .store_paired(&tokens, &target, dflash_snapshot(8))
            .unwrap();
        let mut query = tokens;
        query.push(99);
        let plan = cache.plan_longest_paired_prefix(&query).unwrap().unwrap();
        cache.clear();
        let (_matched, touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };
        assert!(
            !cache.touch_paired(touch),
            "a pre-clear lookup must not refresh or recreate cleared state"
        );
        assert!(cache.is_empty());
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
    }

    #[test]
    fn paired_touch_token_cannot_refresh_another_cache_instance() {
        let mut source = PagedPrefixCache::new(10, 4);
        let mut destination = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        source
            .store_paired(&tokens, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        destination
            .store_paired(&tokens, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        let mut query = tokens;
        query.push(99);
        let plan = source.plan_longest_paired_prefix(&query).unwrap().unwrap();
        let (_matched, foreign_touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };
        assert!(!destination.touch_paired(foreign_touch));

        let plan = source.plan_longest_paired_prefix(&query).unwrap().unwrap();
        let (_matched, local_touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };
        assert!(source.touch_paired(local_touch));
    }

    #[test]
    fn unrelated_publication_does_not_invalidate_a_materialized_pair_touch() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let paired_tokens: Vec<u32> = (0..8).collect();
        cache
            .store_paired(&paired_tokens, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        let mut query = paired_tokens;
        query.push(99);
        let plan = cache.plan_longest_paired_prefix(&query).unwrap().unwrap();
        let (_matched, touch) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test().unwrap()
        };

        let unrelated_tokens: Vec<u32> = (100..108).collect();
        cache.store(&unrelated_tokens, &make_kv_cache(1, 8));

        assert!(
            cache.touch_paired(touch),
            "publication revisions must not invalidate unrelated lookup touches"
        );
    }

    #[test]
    fn cloned_target_materialization_failure_returns_a_cache_miss() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        cache
            .store_paired(&tokens, &make_mutable_hybrid_cache(8), dflash_snapshot(8))
            .unwrap();
        let mut query = tokens;
        query.push(99);

        fail_next_cloned_target_materialization_for_test();
        assert!(
            cache.find_longest_prefix(&query).is_none(),
            "a fallible device clone must degrade to a cache miss instead of panicking"
        );
        assert!(
            cache.find_longest_prefix(&query).is_some(),
            "a failed materialization must leave the frozen endpoint intact"
        );
    }

    #[test]
    fn paired_lookup_never_uses_partial_edge_match() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let paired_prefix: Vec<u32> = (0..4).collect();
        let stored: Vec<u32> = (0..12).collect();
        cache
            .store_paired(&paired_prefix, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache.store(&stored, &make_kv_cache(1, 12));

        let mut query: Vec<u32> = (0..8).collect();
        query.extend(100..104);
        assert_eq!(
            cache.find_longest_prefix(&query).unwrap().prefix_len,
            8,
            "ordinary target lookup should retain partial block sharing"
        );
        let paired = find_paired(&mut cache, &query)
            .expect("shallower exact pair must survive a deeper partial target candidate");
        assert_eq!(paired.prefix_len, 4);
        assert_eq!(paired.dflash_cache.position(), 4);
    }

    #[test]
    fn paired_lookup_forks_independent_live_dflash_branches() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let stored: Vec<u32> = (0..4).collect();
        cache
            .store_paired(&stored, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        let mut query = stored;
        query.push(99);

        let mut first = find_paired(&mut cache, &query).unwrap();
        {
            let _exec = higgs_models::mlx_exec::acquire();
            let mut drafter = tiny_dflash_drafter();
            let tap = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
            drafter.prime_taps(&[tap], &mut first.dflash_cache).unwrap();
        }
        assert_eq!(first.dflash_cache.position(), 5);

        let second = find_paired(&mut cache, &query).unwrap();
        assert_eq!(
            second.dflash_cache.position(),
            4,
            "advancing one fork must not mutate the retained snapshot"
        );
    }

    #[test]
    fn concurrent_owned_forks_advance_without_cross_commit() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let stored: Vec<u32> = (0..4).collect();
        cache
            .store_paired(&stored, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        let mut query = stored;
        query.push(99);
        let left_plan = cache.plan_longest_paired_prefix(&query).unwrap().unwrap();
        let right_plan = cache.plan_longest_paired_prefix(&query).unwrap().unwrap();
        let start = Arc::new(std::sync::Barrier::new(3));

        let spawn_fork =
            |plan: PagedPairedLookupPlan, advance: i32, start: Arc<std::sync::Barrier>| {
                std::thread::spawn(move || {
                    start.wait();
                    let _exec = higgs_models::mlx_exec::acquire();
                    let (mut matched, _touch) = plan.materialize_unproven_for_test().unwrap();
                    let mut drafter = tiny_dflash_drafter();
                    let taps = Array::zeros::<f32>(&[1, advance, 4]).unwrap();
                    drafter
                        .prime_taps(&[taps], &mut matched.dflash_cache)
                        .unwrap();
                    (
                        matched.dflash_cache.position(),
                        kv_cache_offset(&matched.cache),
                    )
                })
            };
        let left = spawn_fork(left_plan, 1, Arc::clone(&start));
        let right = spawn_fork(right_plan, 2, Arc::clone(&start));
        start.wait();

        assert_eq!(left.join().unwrap(), (5, 4));
        assert_eq!(right.join().unwrap(), (6, 4));

        let retained = find_paired(&mut cache, &query).unwrap();
        assert_eq!(
            retained.dflash_cache.position(),
            4,
            "neither live branch may commit back into the retained snapshot"
        );
    }

    #[test]
    fn failed_materialization_does_not_refresh_paired_lru() {
        let mut cache = PagedPrefixCache::new(8, 4);
        let pair_a: Vec<u32> = (0..4).collect();
        let pair_b: Vec<u32> = (100..104).collect();
        let pair_c: Vec<u32> = (200..204).collect();
        cache
            .store_paired(&pair_a, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache
            .store_paired(&pair_b, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();

        let mut query_a = pair_a.clone();
        query_a.push(999);
        let plan = cache.plan_longest_paired_prefix(&query_a).unwrap().unwrap();
        plan.fail_materialization_for_test();
        let failed = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan.materialize_unproven_for_test()
        };
        assert!(failed.is_err());

        cache
            .store_paired(&pair_c, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        let mut query_b = pair_b;
        query_b.push(999);
        let mut query_c = pair_c;
        query_c.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&query_a)
                .unwrap()
                .is_none(),
            "failed materialization must leave A as the deterministic oldest pair"
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&query_b)
                .unwrap()
                .is_some()
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&query_c)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn live_hybrid_tail_mutation_cannot_change_frozen_paired_target() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..4).collect();
        let mut live = make_mutable_hybrid_cache(4);
        cache
            .store_paired(&tokens, &live, dflash_snapshot(4))
            .unwrap();

        {
            let _exec = higgs_models::mlx_exec::acquire();
            append_hybrid_tail(&mut live, 9.0);
        }
        assert!(
            hybrid_kv_token_is(&live, 4, 9.0),
            "live cache must expose the appended tail"
        );

        let mut query = tokens;
        query.push(99);
        let matched = find_paired(&mut cache, &query).unwrap();
        assert_eq!(kv_cache_offset(&matched.cache), 4);
        assert!(
            hybrid_kv_token_is(&matched.cache, 4, 0.0),
            "paired target must fork the frozen pre-mutation buffer"
        );
    }

    #[test]
    fn target_only_refresh_demotes_same_key_pair() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..8).collect();
        cache
            .store_paired(
                &tokens,
                &make_kv_cache_content(1, 8, 11.0),
                dflash_snapshot(8),
            )
            .unwrap();
        let refreshed = make_kv_cache_content(1, 8, 29.0);
        let refreshed_keys = cache_keys(&refreshed, 0);

        cache.store(&tokens, &refreshed);

        let mut query = tokens;
        query.push(99);
        assert!(
            find_paired(&mut cache, &query).is_none(),
            "target-only refresh must remove speculative continuity"
        );
        let matched = cache
            .find_longest_prefix(&query)
            .expect("ordinary refreshed target state must remain reusable");
        assert_keys_eq_first_n(&cache_keys(&matched.cache, 0), &refreshed_keys, 8);
    }

    #[test]
    fn non_block_aligned_target_refresh_replaces_the_same_floored_pair_boundary() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let tokens: Vec<u32> = (0..10).collect();
        cache
            .store_paired(&tokens, &make_kv_cache(1, 10), dflash_snapshot(8))
            .unwrap();

        cache.store(&tokens, &make_kv_cache(1, 10));

        let mut query = tokens;
        query.push(99);
        assert!(
            find_paired(&mut cache, &query).is_none(),
            "floored target refresh must remove the stale exact pair"
        );
        assert_eq!(
            cache.find_longest_prefix(&query).unwrap().prefix_len,
            8,
            "ordinary dense refresh must remain reusable at its floored boundary"
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
    }

    #[test]
    fn mid_block_split_then_target_refresh_demotes_pair() {
        let mut cache = PagedPrefixCache::new(10, 4);
        let seq_a: Vec<u32> = (0..12).collect();
        let mut seq_b: Vec<u32> = (0..6).collect();
        seq_b.extend(100..106);
        cache
            .store_paired(&seq_a, &make_kv_cache(1, 12), dflash_snapshot(12))
            .unwrap();
        cache.store(&seq_b, &make_kv_cache(1, 12));

        cache.store(&seq_a, &make_kv_cache(1, 12));

        let mut query = seq_a;
        query.push(99);
        assert!(
            find_paired(&mut cache, &query).is_none(),
            "same-key target-only refresh must demote even after a mid-edge split"
        );
        assert_eq!(
            cache.find_longest_prefix(&query).unwrap().prefix_len,
            12,
            "the refreshed ordinary target must remain available"
        );
    }

    #[test]
    fn lru_eviction_and_clear_drop_whole_paired_endpoints() {
        let mut cache = PagedPrefixCache::new(2, 4);
        let pair_a: Vec<u32> = (0..8).collect();
        let pair_b: Vec<u32> = (100..108).collect();
        let target_c: Vec<u32> = (200..208).collect();
        cache
            .store_paired(&pair_a, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        cache
            .store_paired(&pair_b, &make_kv_cache(1, 8), dflash_snapshot(8))
            .unwrap();
        let mut query_b = pair_b.clone();
        query_b.push(999);
        assert!(find_paired(&mut cache, &query_b).is_some());

        cache.store(&target_c, &make_kv_cache(1, 8));

        let mut query_a = pair_a;
        query_a.push(999);
        assert!(find_paired(&mut cache, &query_a).is_none());
        assert!(
            cache.find_longest_prefix(&query_a).is_none(),
            "LRU eviction must remove the paired target half with its dSpark sidecar"
        );
        assert!(find_paired(&mut cache, &query_b).is_some());
        let after_lru = cache.paired_stats();
        assert_eq!(cache.paired_entry_count(), 1);
        assert_eq!(after_lru.entries, 1);
        assert!(after_lru.target_bytes > 0);
        assert!(after_lru.dflash_bytes > 0);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
        assert!(find_paired(&mut cache, &query_b).is_none());
    }

    #[test]
    fn paired_radix_cap_evicts_oldest_pair_and_memory_plateaus() {
        let mut cache = PagedPrefixCache::new(8, 4);
        let pair_a: Vec<u32> = (0..4).collect();
        let pair_b: Vec<u32> = (100..104).collect();
        let pair_c: Vec<u32> = (200..204).collect();
        let target_only: Vec<u32> = (300..304).collect();
        cache.store(&target_only, &make_kv_cache(1, 4));
        cache
            .store_paired(&pair_a, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache
            .store_paired(&pair_b, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();

        let before = cache.paired_stats();
        assert_eq!(before.entries, MAX_PAIRED_RADIX_ENTRIES);
        assert_eq!(cache.paired_entry_count(), MAX_PAIRED_RADIX_ENTRIES);
        assert_eq!(cache.len(), MAX_PAIRED_RADIX_ENTRIES + 1);
        assert!(before.target_bytes > 0);
        assert!(before.dflash_bytes > 0);

        let mut query_a = pair_a.clone();
        query_a.push(999);
        assert!(
            find_paired(&mut cache, &query_a).is_some(),
            "successfully materialize and touch pair A so pair B becomes oldest"
        );
        cache
            .store_paired(&pair_c, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();

        let after = cache.paired_stats();
        assert_eq!(after.entries, MAX_PAIRED_RADIX_ENTRIES);
        assert_eq!(cache.paired_entry_count(), MAX_PAIRED_RADIX_ENTRIES);
        assert_eq!(cache.len(), MAX_PAIRED_RADIX_ENTRIES + 1);
        assert_eq!(after.target_bytes, before.target_bytes);
        assert_eq!(after.dflash_bytes, before.dflash_bytes);

        let mut query_b = pair_b;
        query_b.push(999);
        let mut query_c = pair_c;
        query_c.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&query_a)
                .unwrap()
                .is_some()
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&query_b)
                .unwrap()
                .is_none()
        );
        assert!(
            cache.find_longest_prefix(&query_b).is_none(),
            "the cap must evict the complete target+dSpark endpoint, not demote it"
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&query_c)
                .unwrap()
                .is_some()
        );
        assert!(
            cache.find_longest_prefix(&target_only).is_some(),
            "paired pressure must not evict an ordinary target-only endpoint"
        );
    }

    #[test]
    fn paired_radix_ttl_evicts_whole_endpoint_and_invalidates_prepared_publication() {
        let mut cache = PagedPrefixCache::new(8, 4);
        let paired: Vec<u32> = (0..4).collect();
        let target_only: Vec<u32> = (100..104).collect();
        let replacement: Vec<u32> = (200..204).collect();
        cache.store(&target_only, &make_kv_cache(1, 4));
        cache
            .store_paired(&paired, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache.set_paired_idle_ttl(Some(Duration::from_secs(999)));

        assert_eq!(cache.evict_idle_paired(), 0);
        assert_eq!(cache.paired_entry_count(), 1);

        let ticket = cache.paired_prepare_ticket();
        let replacement_target = make_kv_cache(1, 4);
        let replacement_dflash = dflash_snapshot(4);
        let _exec = higgs_models::mlx_exec::acquire();
        let prepared = PagedPrefixCache::prepare_paired_prefix_from_parts(
            ticket,
            &replacement,
            &replacement_target,
            replacement_dflash,
        )
        .unwrap();
        drop(_exec);

        cache.set_paired_idle_ttl(Some(Duration::ZERO));
        assert_eq!(
            cache.evict_idle_paired(),
            1,
            "expiry must take the one endpoint that owns both cache halves"
        );
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
        assert_eq!(
            cache.len(),
            1,
            "paired expiry must leave unrelated target-only endpoints intact"
        );
        assert!(cache.find_longest_prefix(&target_only).is_some());
        let mut paired_query = paired;
        paired_query.push(999);
        assert!(
            cache
                .plan_longest_paired_prefix(&paired_query)
                .unwrap()
                .is_none()
        );
        assert!(
            cache.find_longest_prefix(&paired_query).is_none(),
            "TTL expiry must remove the target half with the dSpark sidecar"
        );
        assert!(matches!(
            cache.commit_prepared_pair(prepared).unwrap_err(),
            PairedCacheError::StaleRevision { .. }
        ));
    }

    #[test]
    fn paired_ttl_prunes_exact_endpoint_without_damaging_shared_target_descendant() {
        let mut cache = PagedPrefixCache::new(8, 4);
        let paired: Vec<u32> = (0..4).collect();
        let target_descendant: Vec<u32> = (0..8).collect();
        let expected_target = make_kv_cache_content(1, 8, 23.0);
        let expected_keys = cache_keys(&expected_target, 0);
        cache
            .store_paired(
                &paired,
                &make_kv_cache_content(1, 4, 23.0),
                dflash_snapshot(4),
            )
            .unwrap();
        cache.store(&target_descendant, &expected_target);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.paired_entry_count(), 1);

        cache.set_paired_idle_ttl(Some(Duration::ZERO));
        assert_eq!(cache.evict_idle_paired(), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());

        let mut query = target_descendant;
        query.push(999);
        assert!(
            cache.plan_longest_paired_prefix(&query).unwrap().is_none(),
            "expiry must remove the exact paired capability"
        );
        let retained = cache
            .find_longest_prefix(&query)
            .expect("shared descendant target must survive paired endpoint pruning");
        assert_eq!(retained.prefix_len, 8);
        assert_keys_eq_first_n(&cache_keys(&retained.cache, 0), &expected_keys, 8);
    }

    #[test]
    fn successful_paired_touch_renews_ttl_without_reviving_an_idle_peer() {
        fn set_last_accessed_at(node: &RadixNode, entry_id: u64, accessed_at: Instant) -> bool {
            if let Some(cached) = node
                .cached
                .as_ref()
                .filter(|cached| cached.entry_id == entry_id)
            {
                cached.last_accessed_at.set(accessed_at);
                return true;
            }
            node.children
                .values()
                .any(|child| set_last_accessed_at(child, entry_id, accessed_at))
        }

        let mut cache = PagedPrefixCache::new(8, 4);
        let pair_a: Vec<u32> = (0..4).collect();
        let pair_b: Vec<u32> = (100..104).collect();
        cache
            .store_paired(&pair_a, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache
            .store_paired(&pair_b, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        cache.set_paired_idle_ttl(Some(Duration::from_secs(10)));

        let mut query_a = pair_a;
        query_a.push(999);
        let plan_a = cache.plan_longest_paired_prefix(&query_a).unwrap().unwrap();
        let entry_a = plan_a.touch.entry_id;
        drop(plan_a);

        let mut query_b = pair_b;
        query_b.push(999);
        let plan_b = cache.plan_longest_paired_prefix(&query_b).unwrap().unwrap();
        let (_matched_b, touch_b) = {
            let _exec = higgs_models::mlx_exec::acquire();
            plan_b.materialize_unproven_for_test().unwrap()
        };
        let entry_b = touch_b.entry_id;

        let baseline = Instant::now();
        let five_seconds_old = baseline.checked_sub(Duration::from_secs(5)).unwrap();
        assert!(set_last_accessed_at(&cache.root, entry_a, five_seconds_old));
        assert!(set_last_accessed_at(&cache.root, entry_b, five_seconds_old));
        assert!(
            cache.touch_paired(touch_b),
            "a successful materialized fork must renew the retained pair"
        );

        let six_seconds_later = baseline.checked_add(Duration::from_secs(6)).unwrap();
        assert_eq!(cache.evict_idle_paired_at(six_seconds_later), 1);
        assert!(
            cache
                .plan_longest_paired_prefix(&query_a)
                .unwrap()
                .is_none(),
            "the untouched peer must expire"
        );
        assert!(
            cache
                .plan_longest_paired_prefix(&query_b)
                .unwrap()
                .is_some(),
            "the successfully touched pair must retain both cache halves"
        );
        assert_eq!(cache.paired_entry_count(), 1);
        let stats = cache.paired_stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.target_bytes > 0);
        assert!(stats.dflash_bytes > 0);
    }

    #[test]
    fn paired_stats_clear_to_zero() {
        let mut cache = PagedPrefixCache::new(8, 4);
        let pair: Vec<u32> = (0..4).collect();
        cache
            .store_paired(&pair, &make_kv_cache(1, 4), dflash_snapshot(4))
            .unwrap();
        assert_eq!(cache.paired_entry_count(), 1);
        assert_ne!(cache.paired_stats(), PairedPrefixCacheStats::default());

        cache.clear();

        assert_eq!(cache.paired_entry_count(), 0);
        assert_eq!(cache.paired_stats(), PairedPrefixCacheStats::default());
    }

    /// Deterministic KV content as a PURE function of (token value, position,
    /// layer, head, dim). Because it depends on the token VALUE, two prefixes
    /// that share tokens share content (so block dedup is invisible to the
    /// check), while a wrong-block reuse or an offset shift yields different
    /// content — exactly the corruption a fuzzer must catch.
    fn token_val(token: u32, t: usize, layer: usize, h: usize, d: usize) -> f32 {
        (token as f32).mul_add(
            31.0,
            (t as f32) * 7.0 + (layer as f32) * 1_000_000.0 + (h as f32) * 100.0 + d as f32,
        )
    }

    fn make_kv_from_tokens(num_layers: usize, tokens: &[u32]) -> AnyCache {
        let s = tokens.len();
        let layers: Vec<Option<SteppingKeyValueCache>> = (0..num_layers)
            .map(|layer| {
                let mut data = vec![0.0_f32; 2 * s * 8];
                for h in 0..2 {
                    for (t, &tok) in tokens.iter().enumerate() {
                        for d in 0..8 {
                            data[((h * s) + t) * 8 + d] = token_val(tok, t, layer, h, d);
                        }
                    }
                }
                let keys = Array::from_slice(&data, &[1, 2, s as i32, 8]);
                let values = Array::from_slice(&data, &[1, 2, s as i32, 8]);
                Some(SteppingKeyValueCache::from_arrays(keys, values).unwrap())
            })
            .collect();
        AnyCache::KV(layers)
    }

    /// Tiny deterministic xorshift64 PRNG (seed must be non-zero) — keeps the
    /// fuzzer reproducible without a `rand` dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    /// Fuzz the radix trie: thousands of randomized store/query/clear ops over a
    /// small shared-prefix alphabet, asserting after every op that
    ///   1. every reconstruction is byte-identical to the content implied by the
    ///      query's tokens (catches wrong-block reuse AND offset shifts),
    ///   2. `prefix_len` is block-aligned and within the query,
    ///   3. the LRU entry cap holds,
    ///   4. `clear()` frees every block, and
    ///   5. stored blocks never explode past the live bound (leak guard).
    ///
    /// Pure-Rust trie logic over small synthetic MLX arrays — no model/GPU. loom
    /// is intentionally not used: it cannot model the MLX FFI buffers the blocks
    /// hold; the `Send + Sync` soundness invariant plus
    /// `radix_blocks_reconstruct_serialized_across_threads` cover the concurrency
    /// contract.
    #[test]
    fn fuzz_radix_random_ops_reconstruct_and_stay_bounded() {
        const BLOCK: usize = 4;
        const LAYERS: usize = 2;
        const MAX_ENTRIES: usize = 6;
        const ALPHABET: u64 = 5;
        const MAX_LEN: u64 = 20;
        let block_cap = MAX_ENTRIES * (MAX_LEN as usize / BLOCK);

        for seed in [0x1234_5678_u64, 0x9E37_79B9, 0xDEAD_BEEF, 0x0BAD_F00D] {
            let mut rng = Rng(seed);
            let mut cache = PagedPrefixCache::new(MAX_ENTRIES, BLOCK);
            let mut recent: Vec<Vec<u32>> = Vec::new();

            for _ in 0..700 {
                match rng.below(10) {
                    0 => {
                        cache.clear();
                        assert!(cache.is_empty());
                        assert!(
                            cache.layer0_block_stats().is_empty(),
                            "clear must free every block"
                        );
                        recent.clear();
                    }
                    1..=4 => {
                        let len = 1 + rng.below(MAX_LEN) as usize;
                        let toks: Vec<u32> = (0..len).map(|_| rng.below(ALPHABET) as u32).collect();
                        cache.store(&toks, &make_kv_from_tokens(LAYERS, &toks));
                        recent.push(toks);
                        if recent.len() > 32 {
                            recent.remove(0);
                        }
                    }
                    _ => {
                        let q: Vec<u32> = if !recent.is_empty() && rng.below(4) != 0 {
                            recent[rng.below(recent.len() as u64) as usize].clone()
                        } else {
                            let len = 1 + rng.below(MAX_LEN) as usize;
                            (0..len).map(|_| rng.below(ALPHABET) as u32).collect()
                        };
                        if let Some(m) = cache.find_longest_prefix(&q) {
                            // A reported match must be a real, non-empty prefix of
                            // the query. (Block-paged matches are block-aligned and
                            // >= BLOCK; a sub-block KV cache is stored as a Cloned
                            // endpoint and reused at its exact, shorter length —
                            // both are valid, but prefix_len must never be 0, the
                            // empty-match bug this fuzzer caught.)
                            assert!(
                                m.prefix_len >= 1 && m.prefix_len <= q.len(),
                                "prefix_len {} out of range for query len {} (seed {seed:#x})",
                                m.prefix_len,
                                q.len()
                            );
                            // The contract: prefix_len equals the reconstructed
                            // cache's actual token count.
                            assert_eq!(
                                kv_cache_offset(&m.cache) as usize,
                                m.prefix_len,
                                "offset != prefix_len (seed {seed:#x}, q {q:?}, prefix_len {})",
                                m.prefix_len
                            );
                            let expected = make_kv_from_tokens(LAYERS, &q[..m.prefix_len]);
                            let AnyCache::KV(rec) = &m.cache else {
                                panic!("expected KV reconstruction");
                            };
                            assert_eq!(
                                rec.len(),
                                LAYERS,
                                "reconstruction changed layer count (seed {seed:#x}, q {q:?})"
                            );
                            for (layer, rl) in rec.iter().enumerate() {
                                let rk = rl.as_ref().and_then(|c| c.keys());
                                assert!(
                                    rk.is_some(),
                                    "reconstructed layer {layer} missing keys: seed={seed:#x} q={q:?} prefix_len={} offset={}",
                                    m.prefix_len,
                                    kv_cache_offset(&m.cache),
                                );
                                assert_keys_eq_first_n(
                                    rk.unwrap(),
                                    &cache_keys(&expected, layer),
                                    m.prefix_len as i32,
                                );
                            }
                        }
                    }
                }

                assert!(cache.len() <= MAX_ENTRIES, "LRU entry cap exceeded");
                let stats = cache.layer0_block_stats();
                assert!(
                    stats.iter().all(|&(_, c)| c >= 1),
                    "every stored block must have strong_count >= 1"
                );
                assert!(stats.len() <= block_cap, "stored-block explosion (leak)");
            }
        }
    }

    /// Concurrency contract for the radix block cache. Blocks are `Arc`-shared
    /// and, in the server, reconstructed from DIFFERENT tokio blocking-pool
    /// threads across turns (each request runs in its own `spawn_blocking`). Two
    /// invariants must hold:
    /// 1. A block is genuinely `Send + Sync` — safe to hand to and read from
    ///    another thread — which requires it to hold only fully-EVALUATED MLX
    ///    buffers (enforced by `KvBlock::new`), never a pending lazy slice graph.
    /// 2. MLX eval must be serialized: MLX's Metal command buffer is process-
    ///    global and aborts (SIGABRT in `concatenate_gpu` → command encoder) on
    ///    concurrent eval. The engine serializes via the model `Mutex`; this test
    ///    mirrors that with a shared lock around reconstruction.
    ///
    /// It reconstructs from many threads (under the shared lock) and checks each
    /// thread's result is well-formed. Without evaluated blocks (invariant 1) the
    /// shared lazy graphs would race even under the lock; without the lock
    /// (invariant 2) the Metal command buffer aborts.
    #[test]
    fn radix_blocks_reconstruct_serialized_across_threads() {
        let cache = make_kv_cache_content(1, 128, 7.0);
        let AnyCache::KV(layers) = &cache else {
            panic!("expected KV cache");
        };
        let kv = layers[0].as_ref().expect("layer 0 present");
        let CachedLayerData::Kv(blocks) = slice_kv_layer(Some(kv), 4, 32).unwrap() else {
            panic!("expected Kv blocks");
        };
        let shared = Arc::new(blocks);
        let mlx_lock = Arc::new(std::sync::Mutex::new(()));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let s = Arc::clone(&shared);
                let lock = Arc::clone(&mlx_lock);
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        let guard = lock.lock().unwrap();
                        let c = gather_blocks(&s).unwrap();
                        let k = c.keys().expect("keys").clone();
                        let v = c.values().expect("values").clone();
                        mlx_rs::transforms::eval([&k, &v]).unwrap();
                        drop(guard);
                        assert_eq!(k.shape(), [1, 2, 128, 8].as_slice());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("reconstruction thread panicked");
        }
    }

    /// (c) Inserting then evicting frees only UNSHARED blocks while shared
    /// blocks stay alive as long as another prefix references them.
    #[test]
    fn test_eviction_frees_only_unshared_blocks() {
        let mut cache = PagedPrefixCache::new(2, DEFAULT_BLOCK_SIZE);

        // A and B share the first 64 tokens (2 blocks), diverge after.
        let mut seq_a: Vec<u32> = (0..64).collect();
        seq_a.extend(1000..1064);
        let mut seq_b: Vec<u32> = (0..64).collect();
        seq_b.extend(2000..2064);

        cache.store(&seq_a, &make_kv_cache(1, 128));
        cache.store(&seq_b, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);
        // 6 distinct blocks: 2 shared + 2 (a) + 2 (b).
        assert_eq!(cache.layer0_block_stats().len(), 6);

        // Touch A so it is the most-recently-used; B becomes the LRU victim
        // when C arrives.
        let mut q_a = seq_a.clone();
        q_a.push(9);
        assert!(cache.find_longest_prefix(&q_a).is_some());

        // Insert C (disjoint) -> evicts the LRU (B). B's UNSHARED blocks (the 2
        // after the shared prefix) are freed; the 2 SHARED blocks remain because
        // A still references them.
        let mut seq_c: Vec<u32> = (5000..5064).collect();
        seq_c.extend(6000..6064);
        cache.store(&seq_c, &make_kv_cache(1, 128));
        assert_eq!(cache.len(), 2);

        // A must still reconstruct fully (its shared blocks were NOT freed).
        let result_a = cache.find_longest_prefix(&q_a).unwrap();
        assert_eq!(result_a.prefix_len, 128);

        // Distinct blocks now: A keeps 4 (2 shared + 2 a-only), C has 4.
        // B's 2 unshared blocks are gone; the 2 formerly-shared blocks survive
        // (now referenced only by A). Total distinct = 8.
        let stats = cache.layer0_block_stats();
        assert_eq!(
            stats.len(),
            8,
            "B's unshared blocks should be freed, shared+A+C kept: {stats:?}"
        );

        // B is gone: a B-only query no longer reaches depth 128.
        let mut q_b = seq_b.clone();
        q_b.push(9);
        match cache.find_longest_prefix(&q_b) {
            None => {}
            Some(m) => assert!(
                m.prefix_len <= 64,
                "evicted B must not yield its full 128-token prefix, got {}",
                m.prefix_len
            ),
        }
    }
}
