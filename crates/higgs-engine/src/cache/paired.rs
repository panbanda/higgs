//! Correct-by-construction ownership for target/dSpark retained state.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use higgs_models::{
    AnyCache,
    dflash::{DFlashCache, DFlashDrafter, DFlashSnapshot},
    mlx_exec::MlxExecToken,
    turboquant::KvCacheConfig,
};
use mlx_rs::Array;

use super::disk_prefix_cache::hash_tokens;
use crate::decode::token_ledger::PairedLedgerProof;
use crate::error::EngineError;

/// Move-only target taps spanning one exact drafter-to-target boundary gap.
///
/// `draft_boundary` is already represented by the live drafter cache;
/// `target_boundary` is already represented by target KV. The arrays cover
/// exactly the intervening target rows and may be consumed only by the next
/// drafter round, a cache-only extension, or final sealing.
#[derive(Debug)]
pub(crate) struct DflashTapFrontier {
    draft_boundary: i32,
    target_boundary: i32,
    pub(crate) taps: Vec<Array>,
    expected_taps: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DflashSealDemotion {
    MissingOrUnsupportedTaps {
        rows: i32,
        collected: usize,
        required: usize,
    },
}

impl DflashTapFrontier {
    pub(crate) fn validate_parts(
        draft_boundary: i32,
        target_boundary: i32,
        taps: &[Array],
        expected_taps: usize,
    ) -> Result<(), EngineError> {
        if draft_boundary < 0 || target_boundary < draft_boundary {
            return Err(EngineError::Generation(format!(
                "invalid dSpark tap frontier: draft={draft_boundary} target={target_boundary}"
            )));
        }
        let rows = target_boundary
            .checked_sub(draft_boundary)
            .ok_or_else(|| EngineError::Generation("dSpark tap frontier overflow".to_owned()))?;
        if rows == 0 {
            if !taps.is_empty() {
                return Err(EngineError::Generation(format!(
                    "empty dSpark tap frontier carries {} arrays",
                    taps.len()
                )));
            }
        } else {
            if taps.len() != expected_taps {
                return Err(EngineError::Generation(format!(
                    "dSpark tap frontier has {} layers for {expected_taps} configured taps",
                    taps.len()
                )));
            }
            for (index, tap) in taps.iter().enumerate() {
                let shape = tap.shape();
                if shape.len() != 3 || shape[0] <= 0 || shape[1] != rows || shape[2] <= 0 {
                    return Err(EngineError::Generation(format!(
                        "dSpark tap frontier layer {index} must be [B, {rows}, H], got {shape:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn new(
        draft_boundary: i32,
        target_boundary: i32,
        taps: Vec<Array>,
        expected_taps: usize,
    ) -> Result<Self, EngineError> {
        Self::validate_parts(draft_boundary, target_boundary, &taps, expected_taps)?;
        Ok(Self {
            draft_boundary,
            target_boundary,
            taps,
            expected_taps,
        })
    }

    #[must_use]
    pub(crate) const fn draft_boundary(&self) -> i32 {
        self.draft_boundary
    }

    #[must_use]
    pub(crate) const fn target_boundary(&self) -> i32 {
        self.target_boundary
    }

    #[must_use]
    const fn expected_taps(&self) -> usize {
        self.expected_taps
    }

    pub(crate) fn rows(&self) -> Result<i32, EngineError> {
        self.target_boundary
            .checked_sub(self.draft_boundary)
            .ok_or_else(|| EngineError::Generation("dSpark tap frontier overflow".to_owned()))
    }

    pub(crate) fn validate_live_draft(&self, draft_cache: &DFlashCache) -> Result<(), EngineError> {
        let actual = draft_cache.position();
        if actual != self.draft_boundary {
            return Err(EngineError::Generation(format!(
                "dSpark tap frontier starts at {0}, but live drafter is at {actual}",
                self.draft_boundary
            )));
        }
        Ok(())
    }

    pub(crate) fn append(
        self,
        next_target_boundary: i32,
        next_taps: Vec<Array>,
    ) -> Result<Self, EngineError> {
        let next = Self::new(
            self.target_boundary,
            next_target_boundary,
            next_taps,
            self.expected_taps,
        )?;
        if next.taps.is_empty() {
            return Ok(self);
        }
        if self.taps.is_empty() {
            return Self::new(
                self.draft_boundary,
                next.target_boundary,
                next.taps,
                self.expected_taps,
            );
        }
        let taps = self
            .taps
            .into_iter()
            .zip(next.taps)
            .map(|(current, added)| {
                mlx_rs::ops::concatenate_axis(&[&current, &added], 1).map_err(EngineError::Mlx)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            self.draft_boundary,
            next.target_boundary,
            taps,
            self.expected_taps,
        )
    }
}

/// Unforgeable-within-this-module identity for one live target/dSpark branch.
///
/// A fresh epoch is minted by [`LivePair::cold`] or a proven
/// [`PairedCache::resume`]. The exact token ledger, both cache halves, and tap
/// frontier move with that epoch until sealing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PairBranchEpoch(u64);

static NEXT_PAIR_BRANCH_EPOCH: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
static FAIL_NEXT_RADIX_CHECKPOINT_FORK: AtomicBool = AtomicBool::new(false);

fn next_pair_branch_epoch_from(counter: &AtomicU64) -> Result<PairBranchEpoch, PairedCacheError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map(PairBranchEpoch)
        .map_err(|_| PairedCacheError::PairBranchEpochOverflow)
}

fn next_pair_branch_epoch() -> Result<PairBranchEpoch, PairedCacheError> {
    next_pair_branch_epoch_from(&NEXT_PAIR_BRANCH_EPOCH)
}

/// Move-only identity binding one completion ledger to one exact live pair
/// revision and starting token boundary.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PairLedgerKey {
    epoch: PairBranchEpoch,
    revision: u64,
    base_boundary: usize,
}

impl PairLedgerKey {
    #[must_use]
    pub(crate) const fn base_boundary(&self) -> usize {
        self.base_boundary
    }

    #[cfg(test)]
    pub(crate) const fn for_test(base_boundary: usize) -> Self {
        Self {
            epoch: PairBranchEpoch(0),
            revision: 0,
            base_boundary,
        }
    }
}

/// Private identity for the exact token boundary represented by both caches.
///
/// The hash is only a fast rejection hint. Exact token equality is the
/// authority, so even an equal-length FNV collision cannot claim continuity.
#[derive(Debug, PartialEq, Eq)]
struct PrefixStamp {
    /// `Some` only for the correct-by-construction [`LivePair`] path.
    ///
    /// `None` identifies the legacy `#[cfg(test)]` fixture constructor, which
    /// deliberately cannot claim shared live-branch provenance.
    branch_epoch: Option<PairBranchEpoch>,
    hash: u64,
    len: usize,
    tokens: Box<[u32]>,
}

impl PrefixStamp {
    /// Legacy test-fixture stamp. This proves exact lookup identity, but cannot
    /// prove that independently supplied cache halves came from it.
    #[cfg(test)]
    fn new(tokens: &[u32]) -> Self {
        Self::from_tokens(tokens.to_vec())
    }

    fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            branch_epoch: None,
            hash: hash_tokens(&tokens),
            len: tokens.len(),
            tokens: tokens.into_boxed_slice(),
        }
    }

    fn from_live_branch(branch_epoch: PairBranchEpoch, tokens: Vec<u32>) -> Self {
        Self {
            branch_epoch: Some(branch_epoch),
            hash: hash_tokens(&tokens),
            len: tokens.len(),
            tokens: tokens.into_boxed_slice(),
        }
    }

    fn matches(&self, tokens: &[u32]) -> bool {
        self.matches_hashed(tokens, hash_tokens(tokens))
    }

    fn matches_hashed(&self, tokens: &[u32], hash: u64) -> bool {
        self.len == tokens.len() && self.hash == hash && self.tokens.as_ref() == tokens
    }

    fn boundary(&self) -> Result<i32, PairedCacheError> {
        i32::try_from(self.len)
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: self.len })
    }
}

/// Construction failures for a target/dFlash retained pair.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PairedCacheError {
    #[error("live paired-cache branch epoch counter overflow")]
    PairBranchEpochOverflow,
    #[error("prefix length {len} exceeds the cache boundary range")]
    PrefixLengthOverflow { len: usize },
    #[error(
        "retained prefix does not match requested tokens (stored length {stored_len}, requested length {requested_len})"
    )]
    PrefixMismatch {
        stored_len: usize,
        requested_len: usize,
    },
    #[error("target cache does not represent absolute boundary {expected}: {details}")]
    TargetBoundary { expected: i32, details: String },
    #[error(
        "dFlash cache boundary {actual} does not match the retained prefix boundary {expected}"
    )]
    DFlashBoundary { expected: i32, actual: i32 },
    #[error(
        "dSpark tap frontier target boundary {actual} does not match the exact token boundary {expected}"
    )]
    FrontierTargetBoundary { expected: i32, actual: i32 },
    #[error("dSpark tap frontier is invalid: {details}")]
    Frontier { details: String },
    #[error("dSpark tap count {actual} does not match the live pair's expected count {expected}")]
    DFlashTapCount { expected: usize, actual: usize },
    #[error("retained target/dSpark pair has legacy unproven provenance")]
    UnprovenPair,
    #[error("retained target/dSpark pair metadata unexpectedly has another owner")]
    SharedPairMetadata,
    #[error("live paired-cache advance carries a target half from another branch")]
    ForeignTargetBranch,
    #[error("live paired-cache advance carries a dFlash half from another paired branch")]
    ForeignDFlashPairBranch,
    #[error("live paired-cache branch revision overflow")]
    BranchRevisionOverflow,
    #[error("failed to seal live dFlash branch: {details}")]
    DFlashSeal { details: String },
    #[error("live target/dSpark known advance failed: {details}")]
    Advance { details: String },
    #[error("live target/dSpark decode transition failed: {details}")]
    Decode { details: String },
    #[error("paired token-ledger proof belongs to another branch or revision")]
    ForeignLedgerProof,
    #[error("failed to fork retained dFlash state: {details}")]
    DFlashFork { details: String },
    #[error("failed to materialize retained target state: {details}")]
    TargetMaterialization { details: String },
    #[error(
        "prepared paired prefix belongs to cache instance {prepared}, current instance is {current}"
    )]
    ForeignCacheInstance { prepared: u64, current: u64 },
    #[error("prepared paired prefix belongs to cache epoch {prepared}, current epoch is {current}")]
    StaleEpoch { prepared: u64, current: u64 },
    #[error(
        "prepared paired prefix belongs to cache publication revision {prepared}, current revision is {current}"
    )]
    StaleRevision { prepared: u64, current: u64 },
}

/// Immutable identity and accounting shared by every ownership form.
///
/// Radix plans clone this `Arc` for lock-free exact-key selection while the
/// single target+dFlash payload remains protected by its mutex.
#[derive(Debug)]
struct PairMetadata {
    stamp: PrefixStamp,
    target_bytes: usize,
    dflash_bytes: usize,
}

/// The one validated target+dFlash ownership core.
///
/// Session retention owns this directly. Radix retention puts the same type
/// behind a mutex and shares only its immutable metadata with lookup plans.
#[derive(Debug)]
struct SealedPair {
    target: AnyCache,
    dflash: DFlashSnapshot,
    metadata: Arc<PairMetadata>,
}

impl SealedPair {
    #[cfg(test)]
    fn new(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        Self::from_stamp(target, dflash, PrefixStamp::new(tokens))
    }

    fn from_live_branch(
        target: AnyCache,
        dflash: DFlashSnapshot,
        branch_epoch: PairBranchEpoch,
        tokens: Vec<u32>,
    ) -> Result<Self, PairedCacheError> {
        Self::from_stamp(
            target,
            dflash,
            PrefixStamp::from_live_branch(branch_epoch, tokens),
        )
    }

    fn from_stamp(
        target: AnyCache,
        dflash: DFlashSnapshot,
        stamp: PrefixStamp,
    ) -> Result<Self, PairedCacheError> {
        let expected = stamp.boundary()?;
        Self::validate_boundaries(&target, dflash.position(), expected)?;
        let metadata = Arc::new(PairMetadata {
            target_bytes: target.estimated_bytes(),
            dflash_bytes: dflash.estimated_bytes(),
            stamp,
        });
        Ok(Self {
            target,
            dflash,
            metadata,
        })
    }

    fn validate_boundaries(
        target: &AnyCache,
        dflash_position: i32,
        expected: i32,
    ) -> Result<(), PairedCacheError> {
        target
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        if dflash_position != expected {
            return Err(PairedCacheError::DFlashBoundary {
                expected,
                actual: dflash_position,
            });
        }
        Ok(())
    }

    #[must_use]
    #[cfg(test)]
    fn prefix_len(&self) -> usize {
        self.metadata.stamp.len
    }

    #[must_use]
    #[cfg(test)]
    fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.metadata.stamp.matches(tokens)
    }

    #[must_use]
    fn estimated_bytes(&self) -> (usize, usize) {
        (self.metadata.target_bytes, self.metadata.dflash_bytes)
    }

    #[cfg(test)]
    fn into_live(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        if !self.matches_prefix(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: self.prefix_len(),
                requested_len: expected_tokens.len(),
            });
        }
        Ok(self.into_live_unchecked())
    }

    #[cfg(test)]
    fn into_live_unchecked(self) -> (AnyCache, DFlashCache) {
        let Self {
            target,
            dflash,
            metadata: _,
        } = self;
        (target, dflash.into_live())
    }

    #[must_use]
    fn demote(self) -> AnyCache {
        let Self {
            target,
            dflash,
            metadata: _,
        } = self;
        drop(dflash);
        target
    }
}

/// Move-owned live target/dSpark branch with an exact forwarded-token ledger.
///
/// The stable invariant allows the target to run ahead of dFlash while the
/// exact intervening target taps remain move-owned:
///
/// - target boundary = `tokens.len()`;
/// - dFlash position = frontier draft boundary;
/// - frontier target boundary = `tokens.len()`.
///
/// Session prefill/decode and exact-endpoint radix publication both use this
/// coordinator directly. Radix entries retain only an immutable dSpark
/// sidecar beside the existing deduplicated target path and fork back into a
/// fresh `LivePair`.
#[derive(Debug)]
struct LiveTargetHalf {
    epoch: PairBranchEpoch,
    cache: AnyCache,
}

#[derive(Debug)]
struct LiveDFlashHalf {
    epoch: PairBranchEpoch,
    cache: DFlashCache,
}

#[derive(Debug)]
struct LiveFrontierHalf {
    epoch: PairBranchEpoch,
    frontier: DflashTapFrontier,
}

#[derive(Debug)]
pub(crate) struct LivePair {
    epoch: PairBranchEpoch,
    revision: u64,
    target: LiveTargetHalf,
    dflash: LiveDFlashHalf,
    frontier: LiveFrontierHalf,
    tokens: Vec<u32>,
}

/// One-way handoff into the legacy stateless decode loop.
///
/// The authoritative token ledger and branch publication capability are
/// intentionally discarded. These parts may drive the current request, but
/// cannot be relabelled or sealed back into a retained pair.
#[derive(Debug)]
pub(crate) struct LivePairParts {
    pub(crate) target: AnyCache,
    pub(crate) dflash: DFlashCache,
    pub(crate) frontier: DflashTapFrontier,
}

/// Evaluated exact-boundary artifacts produced by the sole live-pair sealing
/// primitive.
///
/// Session publication consumes these into a `SealedPair`; radix publication
/// forks the dSpark snapshot and continues from the same evaluated target.
struct PreparedLiveSeal {
    epoch: PairBranchEpoch,
    target: AnyCache,
    snapshot: DFlashSnapshot,
    tokens: Vec<u32>,
    expected_taps: usize,
}

/// Move-only decode transaction for one exact live target/dSpark branch.
#[derive(Debug)]
pub(crate) struct LivePairDecodeLease {
    epoch: PairBranchEpoch,
    revision: u64,
    base_boundary: usize,
    target: LiveTargetHalf,
    dflash: LiveDFlashHalf,
    frontier: LiveFrontierHalf,
    tokens: Vec<u32>,
}

impl LivePair {
    pub(crate) fn cold(
        target: AnyCache,
        dflash: DFlashCache,
        expected_taps: usize,
    ) -> Result<Self, PairedCacheError> {
        SealedPair::validate_boundaries(&target, dflash.position(), 0)?;
        let epoch = next_pair_branch_epoch()?;
        let frontier = DflashTapFrontier::new(0, 0, Vec::new(), expected_taps)
            .map_err(Self::frontier_error)?;
        let pair = Self {
            epoch,
            revision: 0,
            target: LiveTargetHalf {
                epoch,
                cache: target,
            },
            dflash: LiveDFlashHalf {
                epoch,
                cache: dflash,
            },
            frontier: LiveFrontierHalf { epoch, frontier },
            tokens: Vec::new(),
        };
        pair.validate_stable()?;
        Ok(pair)
    }

    /// Construct one fresh live branch from an already aligned immutable
    /// boundary.
    ///
    /// The token ledger is authoritative: callers supply owned tokens, not an
    /// independent length label. Every retained/radix fork enters through this
    /// helper so target, drafter, frontier, and branch identity are branded
    /// together.
    fn from_clean_boundary(
        target: AnyCache,
        dflash: DFlashCache,
        tokens: Vec<u32>,
        expected_taps: usize,
    ) -> Result<Self, PairedCacheError> {
        let expected = Self::token_boundary(tokens.len())?;
        SealedPair::validate_boundaries(&target, dflash.position(), expected)?;
        let epoch = next_pair_branch_epoch()?;
        let frontier = DflashTapFrontier::new(expected, expected, Vec::new(), expected_taps)
            .map_err(Self::frontier_error)?;
        let pair = Self {
            epoch,
            revision: 0,
            target: LiveTargetHalf {
                epoch,
                cache: target,
            },
            dflash: LiveDFlashHalf {
                epoch,
                cache: dflash,
            },
            frontier: LiveFrontierHalf { epoch, frontier },
            tokens,
        };
        pair.validate_stable()?;
        Ok(pair)
    }

    /// Consume this pair through one known exact prefill suffix.
    ///
    /// The suffix is appended to the pair-owned ledger before `advance` runs,
    /// and the callback receives a slice backed by that owned ledger. On any
    /// callback or postcondition error both cache halves and the frontier drop;
    /// no value capable of sealing is returned.
    pub(crate) fn advance_known<R, E, F>(
        self,
        suffix: &[u32],
        advance: F,
    ) -> Result<(Self, R), PairedCacheError>
    where
        E: std::fmt::Display,
        F: FnOnce(
            &[u32],
            &mut AnyCache,
            &mut DFlashCache,
            DflashTapFrontier,
        ) -> Result<(DflashTapFrontier, R), E>,
    {
        self.validate_stable()?;
        let Self {
            epoch,
            revision,
            mut target,
            mut dflash,
            frontier,
            mut tokens,
        } = self;
        let base_boundary = tokens.len();
        let expected_taps = frontier.frontier.expected_taps();
        let new_len = base_boundary
            .checked_add(suffix.len())
            .ok_or(PairedCacheError::PrefixLengthOverflow { len: usize::MAX })?;
        i32::try_from(new_len)
            .map_err(|_| PairedCacheError::PrefixLengthOverflow { len: new_len })?;
        tokens.extend_from_slice(suffix);
        let exact_suffix = tokens
            .get(base_boundary..)
            .ok_or(PairedCacheError::PrefixLengthOverflow { len: new_len })?;
        let (next_frontier, result) = advance(
            exact_suffix,
            &mut target.cache,
            &mut dflash.cache,
            frontier.frontier,
        )
        .map_err(|error| PairedCacheError::Advance {
            details: error.to_string(),
        })?;
        if next_frontier.expected_taps() != expected_taps {
            return Err(PairedCacheError::DFlashTapCount {
                expected: expected_taps,
                actual: next_frontier.expected_taps(),
            });
        }
        let revision = revision
            .checked_add(1)
            .ok_or(PairedCacheError::BranchRevisionOverflow)?;
        let pair = Self {
            epoch,
            revision,
            target,
            dflash,
            frontier: LiveFrontierHalf {
                epoch,
                frontier: next_frontier,
            },
            tokens,
        };
        pair.validate_stable()?;
        Ok((pair, result))
    }

    /// Consume this pair through one exact target-plus-taps prefill suffix.
    ///
    /// Unlike [`Self::advance_known`], the caller cannot choose either frontier
    /// boundary or tap count. Prefill must start from an aligned pair; this
    /// coordinator derives the new target boundary from the pair-owned token
    /// ledger and the new draft boundary from the live dFlash cache after the
    /// callback has projected any completed chunks. The callback supplies only
    /// the target operation's result and final unconsumed taps.
    pub(crate) fn prefill_known<R, F>(
        self,
        suffix: &[u32],
        prefill: F,
    ) -> Result<(Self, R), PairedCacheError>
    where
        F: FnOnce(&[u32], &mut AnyCache, &mut DFlashCache) -> Result<(R, Vec<Array>), EngineError>,
    {
        if suffix.is_empty() {
            return Err(PairedCacheError::Advance {
                details: "known dSpark prefill requires a non-empty suffix".to_owned(),
            });
        }
        self.advance_known(suffix, |exact_suffix, target, draft, frontier| {
            let pending_rows = frontier.rows()?;
            if pending_rows != 0 {
                return Err(EngineError::Generation(format!(
                    "known dSpark prefill requires an aligned frontier, found {pending_rows} pending target rows"
                )));
            }
            let suffix_rows = i32::try_from(exact_suffix.len()).map_err(|_| {
                EngineError::Generation(format!(
                    "known dSpark prefill suffix length {} exceeds i32",
                    exact_suffix.len()
                ))
            })?;
            let target_boundary = frontier
                .target_boundary()
                .checked_add(suffix_rows)
                .ok_or_else(|| {
                    EngineError::Generation(
                        "known dSpark prefill target boundary overflow".to_owned(),
                    )
                })?;
            let expected_taps = frontier.expected_taps();
            let (result, taps) = prefill(exact_suffix, target, draft)?;
            let next_frontier =
                DflashTapFrontier::new(draft.position(), target_boundary, taps, expected_taps)?;
            Ok((next_frontier, result))
        })
    }

    /// Move this exact pair into a decode lease and independently move its
    /// opaque identity into a token ledger.
    pub(crate) fn begin_decode(
        self,
    ) -> Result<(LivePairDecodeLease, PairLedgerKey), PairedCacheError> {
        self.validate_stable()?;
        let key = PairLedgerKey {
            epoch: self.epoch,
            revision: self.revision,
            base_boundary: self.tokens.len(),
        };
        let lease = LivePairDecodeLease {
            epoch: self.epoch,
            revision: self.revision,
            base_boundary: self.tokens.len(),
            target: self.target,
            dflash: self.dflash,
            frontier: self.frontier,
            tokens: self.tokens,
        };
        Ok((lease, key))
    }

    #[must_use]
    pub(crate) fn token_len(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub(crate) const fn target_is_hybrid(&self) -> bool {
        matches!(&self.target.cache, AnyCache::Hybrid(_))
    }

    /// Consume a validated pair into non-publishable stateless decode state.
    pub(crate) fn into_stateless_parts(self) -> Result<LivePairParts, PairedCacheError> {
        self.validate_stable()?;
        let Self {
            target,
            dflash,
            frontier,
            tokens: _,
            ..
        } = self;
        Ok(LivePairParts {
            target: target.cache,
            dflash: dflash.cache,
            frontier: frontier.frontier,
        })
    }

    /// Best-effort target-only retention compression without exposing either
    /// cache half or an independently supplied token label.
    pub(crate) fn quantize_target_for_retention(
        &mut self,
        config: KvCacheConfig,
        _exec: &MlxExecToken,
    ) -> Result<usize, PairedCacheError> {
        self.validate_stable()?;
        let expected = Self::token_boundary(self.tokens.len())?;
        let layers = self
            .target
            .cache
            .quantize_for_retention(config)
            .map_err(|error| PairedCacheError::TargetMaterialization {
                details: error.to_string(),
            })?;
        self.target
            .cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        Ok(layers)
    }

    /// Produce the only publishable session outcome from this live branch.
    ///
    /// A deterministic tap-capability mismatch is resolved before dFlash
    /// sealing begins and may therefore retain the independently validated
    /// target half. Once [`Self::seal`] starts, any error consumes the whole
    /// branch and returns no publication.
    pub(crate) fn seal_for_session(
        self,
        drafter: &mut DFlashDrafter,
        exec: &MlxExecToken,
    ) -> Result<SessionDsparkPublication, PairedCacheError> {
        self.validate_stable()?;
        let required = drafter.config.num_taps();
        let configured = self.frontier.frontier.expected_taps();
        if configured != required {
            let rows = self
                .frontier
                .frontier
                .rows()
                .map_err(Self::frontier_error)?;
            let reason = DflashSealDemotion::MissingOrUnsupportedTaps {
                rows,
                collected: self.frontier.frontier.taps.len(),
                required,
            };
            return self.into_target_only_session_publication(reason, exec);
        }

        let pair = self.seal(drafter, exec)?;
        Ok(SessionDsparkPublication {
            state: RetainedState::Paired(pair),
            demotion: None,
        })
    }

    fn into_target_only_session_publication(
        self,
        reason: DflashSealDemotion,
        _exec: &MlxExecToken,
    ) -> Result<SessionDsparkPublication, PairedCacheError> {
        self.validate_stable()?;
        let Self {
            target,
            dflash,
            frontier,
            tokens,
            ..
        } = self;
        let expected = Self::token_boundary(tokens.len())?;
        target
            .cache
            .eval()
            .map_err(|error| PairedCacheError::TargetMaterialization {
                details: error.to_string(),
            })?;
        target
            .cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        drop(dflash);
        drop(frontier);
        let target = RetainedTarget::from_evaluated_tokens(target.cache, tokens)?;
        Ok(SessionDsparkPublication {
            state: RetainedState::TargetOnly(target),
            demotion: Some(reason),
        })
    }

    /// Consume this exact live branch into one publishable retained pair.
    ///
    /// Sealing is direct: the only owned dFlash cache and frontier taps are
    /// passed to the supplied drafter under an explicit MLX execution token.
    /// No caller can substitute a same-position snapshot or publication label.
    pub(crate) fn seal(
        self,
        drafter: &mut DFlashDrafter,
        exec: &MlxExecToken,
    ) -> Result<PairedCache, PairedCacheError> {
        let PreparedLiveSeal {
            epoch,
            target,
            snapshot,
            tokens,
            expected_taps: _,
        } = self.prepare_live_seal(drafter, exec)?;
        let sealed = SealedPair::from_live_branch(target, snapshot, epoch, tokens)?;
        Ok(PairedCache { sealed })
    }

    /// Seal one exact radix checkpoint and continue from an independent live
    /// fork.
    ///
    /// The preparation callback runs synchronously while its opaque checkpoint
    /// borrows this branch's exact target cache. Its result may fail without
    /// invalidating the independently forked continuation. Sealing or snapshot
    /// forking failures occur before the callback receives any publishable
    /// checkpoint and therefore fail closed.
    pub(crate) fn checkpoint_for_radix<R, E, F>(
        self,
        drafter: &mut DFlashDrafter,
        exec: &MlxExecToken,
        prepare: F,
    ) -> Result<(Self, Result<R, E>), PairedCacheError>
    where
        F: for<'a> FnOnce(RadixPairCheckpoint<'a>) -> Result<R, E>,
    {
        let PreparedLiveSeal {
            epoch,
            target,
            snapshot,
            tokens,
            expected_taps,
        } = self.prepare_live_seal(drafter, exec)?;

        #[cfg(test)]
        if FAIL_NEXT_RADIX_CHECKPOINT_FORK.swap(false, Ordering::SeqCst) {
            return Err(PairedCacheError::DFlashFork {
                details: "injected radix checkpoint fork failure".to_owned(),
            });
        }
        let live_dflash = snapshot
            .fork_live()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: error.to_string(),
            })?;
        let stamp = PrefixStamp::from_live_branch(epoch, tokens.clone());
        let continued = Self::from_clean_boundary(target, live_dflash, tokens, expected_taps)?;
        let checkpoint = RadixPairCheckpoint::new(&continued.target.cache, snapshot, stamp)?;
        let prepared = prepare(checkpoint);
        continued.validate_stable()?;
        Ok((continued, prepared))
    }

    /// Evaluate and seal one exact live branch through the only target+dSpark
    /// boundary-validation implementation.
    fn prepare_live_seal(
        self,
        drafter: &mut DFlashDrafter,
        _exec: &MlxExecToken,
    ) -> Result<PreparedLiveSeal, PairedCacheError> {
        self.validate_stable()?;
        let Self {
            epoch,
            revision: _,
            target,
            dflash,
            frontier,
            tokens,
        } = self;
        let configured_taps = drafter.config.num_taps();
        let expected_taps = frontier.frontier.expected_taps();
        if configured_taps != expected_taps {
            return Err(PairedCacheError::DFlashTapCount {
                expected: expected_taps,
                actual: configured_taps,
            });
        }
        let expected = Self::token_boundary(tokens.len())?;
        target
            .cache
            .eval()
            .map_err(|error| PairedCacheError::TargetMaterialization {
                details: error.to_string(),
            })?;
        target
            .cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        let snapshot = drafter
            .seal_after_taps(dflash.cache, &frontier.frontier.taps, expected)
            .map_err(|error| PairedCacheError::DFlashSeal {
                details: error.to_string(),
            })?;
        Ok(PreparedLiveSeal {
            epoch,
            target: target.cache,
            snapshot,
            tokens,
            expected_taps,
        })
    }

    fn validate_stable(&self) -> Result<(), PairedCacheError> {
        let expected = Self::token_boundary(self.tokens.len())?;
        Self::validate_parts(
            self.epoch,
            &self.target,
            &self.dflash,
            &self.frontier,
            expected,
        )
    }

    fn validate_parts(
        epoch: PairBranchEpoch,
        target: &LiveTargetHalf,
        dflash: &LiveDFlashHalf,
        frontier: &LiveFrontierHalf,
        expected_target_boundary: i32,
    ) -> Result<(), PairedCacheError> {
        if target.epoch != epoch {
            return Err(PairedCacheError::ForeignTargetBranch);
        }
        if dflash.epoch != epoch || frontier.epoch != epoch {
            return Err(PairedCacheError::ForeignDFlashPairBranch);
        }
        target
            .cache
            .validate_absolute_boundary(expected_target_boundary)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected: expected_target_boundary,
                details: error.to_string(),
            })?;
        if frontier.frontier.target_boundary() != expected_target_boundary {
            return Err(PairedCacheError::FrontierTargetBoundary {
                expected: expected_target_boundary,
                actual: frontier.frontier.target_boundary(),
            });
        }
        let expected_draft = frontier.frontier.draft_boundary();
        let actual_draft = dflash.cache.position();
        if actual_draft != expected_draft {
            return Err(PairedCacheError::DFlashBoundary {
                expected: expected_draft,
                actual: actual_draft,
            });
        }
        DflashTapFrontier::validate_parts(
            expected_draft,
            expected_target_boundary,
            &frontier.frontier.taps,
            frontier.frontier.expected_taps(),
        )
        .map_err(Self::frontier_error)
    }

    fn token_boundary(len: usize) -> Result<i32, PairedCacheError> {
        i32::try_from(len).map_err(|_| PairedCacheError::PrefixLengthOverflow { len })
    }

    fn frontier_error(error: EngineError) -> PairedCacheError {
        PairedCacheError::Frontier {
            details: error.to_string(),
        }
    }
}

impl LivePairDecodeLease {
    /// Run one target-authoritative decode transaction.
    ///
    /// The frontier is moved into the callback and a replacement must be
    /// returned. An error drops the target, dFlash cache, and frontier
    /// together, so no partial decode state can later seal.
    pub(crate) fn run<R, E, F>(self, decode: F) -> Result<(Self, R), PairedCacheError>
    where
        E: std::fmt::Display,
        F: FnOnce(
            &mut AnyCache,
            &mut DFlashCache,
            DflashTapFrontier,
        ) -> Result<(DflashTapFrontier, R), E>,
    {
        let Self {
            epoch,
            revision,
            base_boundary,
            mut target,
            mut dflash,
            frontier,
            tokens,
        } = self;
        let expected_taps = frontier.frontier.expected_taps();
        let (next_frontier, result) =
            decode(&mut target.cache, &mut dflash.cache, frontier.frontier).map_err(|error| {
                PairedCacheError::Decode {
                    details: error.to_string(),
                }
            })?;
        if next_frontier.expected_taps() != expected_taps {
            return Err(PairedCacheError::DFlashTapCount {
                expected: expected_taps,
                actual: next_frontier.expected_taps(),
            });
        }
        let lease = Self {
            epoch,
            revision,
            base_boundary,
            target,
            dflash,
            frontier: LiveFrontierHalf {
                epoch,
                frontier: next_frontier,
            },
            tokens,
        };
        lease.validate_structural()?;
        Ok((lease, result))
    }

    /// Reunite decode state with the exact forwarded completion proof.
    pub(crate) fn finish(mut self, proof: PairedLedgerProof) -> Result<LivePair, PairedCacheError> {
        let expected_key = PairLedgerKey {
            epoch: self.epoch,
            revision: self.revision,
            base_boundary: self.base_boundary,
        };
        let forwarded_suffix = proof
            .into_forwarded_suffix_for(&expected_key)
            .ok_or(PairedCacheError::ForeignLedgerProof)?;
        let final_boundary = self
            .base_boundary
            .checked_add(forwarded_suffix.len())
            .ok_or(PairedCacheError::PrefixLengthOverflow { len: usize::MAX })?;
        let expected = LivePair::token_boundary(final_boundary)?;
        self.target
            .cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        if self.frontier.frontier.target_boundary() != expected {
            return Err(PairedCacheError::FrontierTargetBoundary {
                expected,
                actual: self.frontier.frontier.target_boundary(),
            });
        }
        self.validate_structural()?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(PairedCacheError::BranchRevisionOverflow)?;
        self.tokens.extend_from_slice(&forwarded_suffix);
        let pair = LivePair {
            epoch: self.epoch,
            revision,
            target: self.target,
            dflash: self.dflash,
            frontier: self.frontier,
            tokens: self.tokens,
        };
        pair.validate_stable()?;
        Ok(pair)
    }

    fn validate_structural(&self) -> Result<(), PairedCacheError> {
        LivePair::validate_parts(
            self.epoch,
            &self.target,
            &self.dflash,
            &self.frontier,
            self.frontier.frontier.target_boundary(),
        )
    }
}

/// One immutable retained target/dFlash boundary.
///
/// Both halves are private and the type is not `Clone`, so callers cannot
/// publish or move one retained half independently of the other.
#[derive(Debug)]
pub(crate) struct PairedCache {
    sealed: SealedPair,
}

impl PairedCache {
    /// Validate and publish one exact shared target/dFlash token boundary.
    ///
    /// The caller must have evaluated `target` while holding the process MLX
    /// execution gate before transferring it here. `DFlashSnapshot` is already
    /// evaluated by `DFlashDrafter::seal_after_taps`.
    ///
    /// # Legacy test fixture
    ///
    /// This constructor accepts independently supplied halves and therefore
    /// proves boundary/key equality but not shared prefill provenance.
    /// Production code must use [`LivePair::seal`].
    #[cfg(test)]
    pub(crate) fn new(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        SealedPair::new(target, dflash, tokens).map(|sealed| Self { sealed })
    }

    /// Consume one proven retained pair into a fresh live branch.
    ///
    /// `expected_tokens` is only a comparison key. The live token ledger is
    /// moved out of the private retained stamp, so a caller cannot relabel the
    /// cache with an equal-length slice. Legacy test pairs created by
    /// [`Self::new`] remain intentionally unproven and cannot enter this
    /// correct-by-construction coordinator.
    pub(crate) fn resume(
        self,
        expected_tokens: &[u32],
        expected_taps: usize,
    ) -> Result<LivePair, PairedCacheError> {
        let SealedPair {
            target,
            dflash,
            metadata,
        } = self.sealed;
        if !metadata.stamp.matches(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: metadata.stamp.len,
                requested_len: expected_tokens.len(),
            });
        }
        if metadata.stamp.branch_epoch.is_none() {
            return Err(PairedCacheError::UnprovenPair);
        }
        let PairMetadata { stamp, .. } =
            Arc::try_unwrap(metadata).map_err(|_| PairedCacheError::SharedPairMetadata)?;
        let tokens = stamp.tokens.into_vec();
        let live_dflash = dflash.into_live();
        LivePair::from_clean_boundary(target, live_dflash, tokens, expected_taps)
    }

    #[must_use]
    #[cfg(test)]
    pub(crate) fn prefix_len(&self) -> usize {
        self.sealed.prefix_len()
    }

    /// Revalidate the lookup key before this pair is selected for reuse.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.sealed.matches_prefix(tokens)
    }

    #[must_use]
    pub(crate) fn estimated_bytes(&self) -> (usize, usize) {
        self.sealed.estimated_bytes()
    }

    /// Consume both immutable halves into one live target/dFlash branch.
    ///
    /// Rechecking the private stamp here prevents a stale session/radix lookup
    /// from turning a structurally valid pair into continuity for another key.
    #[cfg(test)]
    pub(crate) fn into_live(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), PairedCacheError> {
        self.sealed.into_live(expected_tokens)
    }

    /// Explicitly abandon speculative continuity and retain only target state.
    ///
    /// Consuming `self` makes demotion a whole-pair ownership transition; the
    /// drafter snapshot cannot remain accidentally associated with the target.
    #[must_use]
    pub(crate) fn demote(self) -> AnyCache {
        self.sealed.demote()
    }

    #[cfg(test)]
    fn into_live_unchecked(self) -> (AnyCache, DFlashCache) {
        self.sealed.into_live_unchecked()
    }
}

/// Synchronous, non-escapable proof for one radix publication attempt.
///
/// The target is borrowed from the exact continued [`LivePair`], while the
/// sealed drafter snapshot and authoritative prefix stamp came from that same
/// branch transition. Private fields prevent callers from substituting any of
/// the three pieces. The lifetime prevents the checkpoint itself from escaping
/// [`LivePair::checkpoint_for_radix`].
#[derive(Debug)]
pub(crate) struct RadixPairCheckpoint<'a> {
    target: &'a AnyCache,
    dflash: DFlashSnapshot,
    stamp: PrefixStamp,
}

impl<'a> RadixPairCheckpoint<'a> {
    fn new(
        target: &'a AnyCache,
        dflash: DFlashSnapshot,
        stamp: PrefixStamp,
    ) -> Result<Self, PairedCacheError> {
        if stamp.branch_epoch.is_none() {
            return Err(PairedCacheError::UnprovenPair);
        }
        let expected = stamp.boundary()?;
        SealedPair::validate_boundaries(target, dflash.position(), expected)?;
        Ok(Self {
            target,
            dflash,
            stamp,
        })
    }

    #[must_use]
    pub(crate) const fn target(&self) -> &AnyCache {
        self.target
    }

    #[must_use]
    pub(crate) fn tokens(&self) -> &[u32] {
        &self.stamp.tokens
    }

    /// Finish target byte accounting and consume this checkpoint into the only
    /// production radix sidecar constructor.
    pub(crate) fn into_radix_snapshot(
        self,
        target_bytes: usize,
    ) -> Result<RadixDFlashSnapshot, PairedCacheError> {
        RadixDFlashSnapshot::from_checkpoint(self, target_bytes)
    }
}

/// Immutable dFlash sidecar attached to one exact radix endpoint.
///
/// The target half remains represented by the radix endpoint's existing
/// paged/cloned storage. This type owns only the drafter snapshot and the
/// private metadata that binds it to that exact target endpoint.
#[derive(Debug)]
pub(crate) struct RadixDFlashSnapshot {
    // `DFlashSnapshot` is immutable after sealing but its MLX arrays are
    // `!Sync`. This mutex safely shares one frozen snapshot across owned lookup
    // plans and is never acquired while the radix mutex is held.
    dflash: Mutex<DFlashSnapshot>,
    metadata: Arc<PairMetadata>,
    #[cfg(test)]
    fail_next_fork: AtomicBool,
}

impl RadixDFlashSnapshot {
    fn from_checkpoint(
        checkpoint: RadixPairCheckpoint<'_>,
        target_bytes: usize,
    ) -> Result<Self, PairedCacheError> {
        let RadixPairCheckpoint {
            target,
            dflash,
            stamp,
        } = checkpoint;
        if stamp.branch_epoch.is_none() {
            return Err(PairedCacheError::UnprovenPair);
        }
        let expected = stamp.boundary()?;
        SealedPair::validate_boundaries(target, dflash.position(), expected)?;
        let metadata = Arc::new(PairMetadata {
            target_bytes,
            dflash_bytes: dflash.estimated_bytes(),
            stamp,
        });
        Ok(Self {
            dflash: Mutex::new(dflash),
            metadata,
            #[cfg(test)]
            fail_next_fork: AtomicBool::new(false),
        })
    }

    /// Legacy independently-labelled fixture constructor.
    ///
    /// Production radix publication must consume [`RadixPairCheckpoint`].
    #[cfg(test)]
    pub(crate) fn new(
        dflash: DFlashSnapshot,
        tokens: &[u32],
        target_bytes: usize,
    ) -> Result<Self, PairedCacheError> {
        let stamp = PrefixStamp::new(tokens);
        let expected = stamp.boundary()?;
        let actual = dflash.position();
        if actual != expected {
            return Err(PairedCacheError::DFlashBoundary { expected, actual });
        }
        let metadata = Arc::new(PairMetadata {
            target_bytes,
            dflash_bytes: dflash.estimated_bytes(),
            stamp,
        });
        Ok(Self {
            dflash: Mutex::new(dflash),
            metadata,
            #[cfg(test)]
            fail_next_fork: AtomicBool::new(false),
        })
    }

    #[must_use]
    pub(crate) fn prefix_len(&self) -> usize {
        self.metadata.stamp.len
    }

    #[must_use]
    pub(crate) fn matches_prefix(&self, tokens: &[u32]) -> bool {
        self.metadata.stamp.matches(tokens)
    }

    #[must_use]
    pub(crate) fn target_bytes(&self) -> usize {
        self.metadata.target_bytes
    }

    #[must_use]
    pub(crate) fn dflash_bytes(&self) -> usize {
        self.metadata.dflash_bytes
    }

    /// Select the exact frozen pair without doing any MLX work.
    ///
    /// The returned plan owns the snapshot through an `Arc`, so callers may
    /// release the radix/prefix mutex before forking both live halves.
    pub(crate) fn plan_fork(
        self: &Arc<Self>,
        expected_tokens: &[u32],
    ) -> Result<RadixDFlashForkPlan, PairedCacheError> {
        if !self.matches_prefix(expected_tokens) {
            return Err(PairedCacheError::PrefixMismatch {
                stored_len: self.prefix_len(),
                requested_len: expected_tokens.len(),
            });
        }
        Ok(RadixDFlashForkPlan {
            snapshot: Arc::clone(self),
        })
    }

    fn fork_live(&self) -> Result<DFlashCache, PairedCacheError> {
        debug_assert!(
            higgs_models::mlx_exec::held(),
            "radix dFlash fork requires the process MLX execution gate"
        );
        #[cfg(test)]
        if self.fail_next_fork.swap(false, Ordering::SeqCst) {
            return Err(PairedCacheError::DFlashFork {
                details: "injected dFlash fork failure".to_owned(),
            });
        }
        let snapshot = self
            .dflash
            .lock()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: format!("retained dFlash snapshot lock is poisoned: {error}"),
            })?;
        let live = snapshot
            .fork_live()
            .map_err(|error| PairedCacheError::DFlashFork {
                details: error.to_string(),
            })?;
        let expected = self.metadata.stamp.boundary()?;
        let actual = live.position();
        if actual != expected {
            return Err(PairedCacheError::DFlashBoundary { expected, actual });
        }
        Ok(live)
    }

    #[cfg(test)]
    fn fail_next_fork_for_test(&self) {
        self.fail_next_fork.store(true, Ordering::SeqCst);
    }
}

/// Owned, exact-identity plan for a post-prefix-lock dFlash fork.
#[derive(Debug)]
pub(crate) struct RadixDFlashForkPlan {
    snapshot: Arc<RadixDFlashSnapshot>,
}

impl RadixDFlashForkPlan {
    #[must_use]
    pub(crate) fn prefix_len(&self) -> usize {
        self.snapshot.prefix_len()
    }

    /// Fork both endpoint halves into one freshly branded live pair.
    ///
    /// The target was selected from the same radix endpoint as this plan. Its
    /// exact boundary is checked against the private proven stamp; no external
    /// token label participates in construction.
    pub(crate) fn materialize_pair(
        self,
        target: AnyCache,
        expected_taps: usize,
    ) -> Result<LivePair, PairedCacheError> {
        if self.snapshot.metadata.stamp.branch_epoch.is_none() {
            return Err(PairedCacheError::UnprovenPair);
        }
        let tokens = self.snapshot.metadata.stamp.tokens.to_vec();
        let dflash = self.snapshot.fork_live()?;
        LivePair::from_clean_boundary(target, dflash, tokens, expected_taps)
    }

    /// Legacy raw-drafter materialization for pre-migration unit fixtures.
    #[cfg(test)]
    pub(crate) fn materialize(self) -> Result<DFlashCache, PairedCacheError> {
        self.snapshot.fork_live()
    }

    #[cfg(test)]
    pub(crate) fn fail_materialization_for_test(&self) {
        self.snapshot.fail_next_fork_for_test();
    }
}

/// Session/radix retained ownership, with paired state represented atomically.
#[derive(Debug)]
pub(crate) enum RetainedState {
    TargetOnly(RetainedTarget),
    Paired(PairedCache),
}

/// One target-only retained cache inseparably keyed by its exact token prefix.
#[derive(Debug)]
pub(crate) struct RetainedTarget {
    cache: AnyCache,
    stamp: PrefixStamp,
}

impl RetainedTarget {
    fn from_evaluated_tokens(cache: AnyCache, tokens: Vec<u32>) -> Result<Self, PairedCacheError> {
        let stamp = PrefixStamp::from_tokens(tokens);
        let expected = stamp.boundary()?;
        cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        Ok(Self { cache, stamp })
    }

    fn evaluate(
        cache: AnyCache,
        tokens: Vec<u32>,
        _exec: &MlxExecToken,
    ) -> Result<Self, PairedCacheError> {
        let retained = Self::from_evaluated_tokens(cache, tokens)?;
        retained
            .cache
            .eval()
            .map_err(|error| PairedCacheError::TargetMaterialization {
                details: error.to_string(),
            })?;
        let expected = retained.stamp.boundary()?;
        retained
            .cache
            .validate_absolute_boundary(expected)
            .map_err(|error| PairedCacheError::TargetBoundary {
                expected,
                details: error.to_string(),
            })?;
        Ok(retained)
    }

    #[cfg(test)]
    fn unchecked_for_test(cache: AnyCache, tokens: Vec<u32>) -> Self {
        Self {
            cache,
            stamp: PrefixStamp::from_tokens(tokens),
        }
    }
}

/// Atomic session publication derived from one live pair-owned token ledger.
///
/// No constructor accepts a caller token vector, and no duplicate session key
/// exists outside the retained state. Paired publication moves the live ledger
/// into the private sealed stamp; target-only demotion moves it into the keyed
/// target wrapper.
#[derive(Debug)]
pub(crate) struct SessionDsparkPublication {
    state: RetainedState,
    demotion: Option<DflashSealDemotion>,
}

impl SessionDsparkPublication {
    #[must_use]
    pub(crate) fn demotion(&self) -> Option<&DflashSealDemotion> {
        self.demotion.as_ref()
    }

    #[must_use]
    pub(crate) fn into_state(self) -> RetainedState {
        self.state
    }
}

impl RetainedState {
    pub(crate) fn target_only(
        target: AnyCache,
        tokens: Vec<u32>,
        exec: &MlxExecToken,
    ) -> Result<Self, PairedCacheError> {
        RetainedTarget::evaluate(target, tokens, exec).map(Self::TargetOnly)
    }

    #[cfg(test)]
    pub(crate) fn target_only_unchecked_for_test(target: AnyCache, tokens: Vec<u32>) -> Self {
        Self::TargetOnly(RetainedTarget::unchecked_for_test(target, tokens))
    }

    #[cfg(test)]
    pub(crate) fn paired(
        target: AnyCache,
        dflash: DFlashSnapshot,
        tokens: &[u32],
    ) -> Result<Self, PairedCacheError> {
        PairedCache::new(target, dflash, tokens).map(Self::Paired)
    }

    /// Only target-only state may use the historical TurboQuant exemption from
    /// `max_session_tokens`. A paired dSpark snapshot remains uncompressed and
    /// grows with context, so exempting the target half would leave the whole
    /// retained pair unbounded.
    #[must_use]
    pub(crate) const fn allows_target_only_cap_exemption(&self) -> bool {
        matches!(self, Self::TargetOnly(_))
    }

    #[must_use]
    pub(crate) fn tokens(&self) -> &[u32] {
        match self {
            Self::TargetOnly(target) => &target.stamp.tokens,
            Self::Paired(pair) => &pair.sealed.metadata.stamp.tokens,
        }
    }

    #[must_use]
    pub(crate) fn paired_estimated_bytes(&self) -> Option<(usize, usize)> {
        match self {
            Self::TargetOnly(_) => None,
            Self::Paired(pair) => Some(pair.estimated_bytes()),
        }
    }

    /// Consume paired state only when it still matches the requested key.
    ///
    /// Target-only state and mismatched pairs are returned intact so the
    /// caller may explicitly demote or discard them.
    #[cfg(test)]
    pub(crate) fn into_paired(
        self,
        expected_tokens: &[u32],
    ) -> Result<(AnyCache, DFlashCache), Self> {
        match self {
            Self::TargetOnly(target) => Err(Self::TargetOnly(target)),
            Self::Paired(pair) if pair.matches_prefix(expected_tokens) => {
                Ok(pair.into_live_unchecked())
            }
            Self::Paired(pair) => Err(Self::Paired(pair)),
        }
    }

    /// Consume the retained state and return target-only continuity.
    #[must_use]
    pub(crate) fn demote(self) -> AnyCache {
        match self {
            Self::TargetOnly(target) => target.cache,
            Self::Paired(pair) => pair.demote(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::error::EngineError;
    use higgs_models::{
        AnyCache,
        cache::{KeyValueCache, SteppingKeyValueCache},
        dflash::{DFlashConfig, DFlashDrafter, DFlashSnapshot},
    };
    use mlx_rs::Array;

    use crate::decode::token_ledger::TokenLedger;

    use super::{
        DflashSealDemotion, DflashTapFrontier, LivePair, PairedCache, PairedCacheError,
        PrefixStamp, RetainedState,
    };

    fn target_cache(boundary: i32) -> AnyCache {
        let layer = if boundary == 0 {
            SteppingKeyValueCache::new()
        } else {
            let keys = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            let values = Array::zeros::<f32>(&[1, 1, boundary, 1]).unwrap();
            SteppingKeyValueCache::from_arrays(keys, values).unwrap()
        };
        let cache = AnyCache::KV(vec![Some(layer)]);
        let _exec = higgs_models::mlx_exec::acquire();
        cache.eval().unwrap();
        cache
    }

    fn test_drafter() -> DFlashDrafter {
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
        let mut drafter = test_drafter();
        let cache = drafter.make_cache();
        let taps = (boundary == 1)
            .then(|| Array::zeros::<f32>(&[1, 1, 4]).unwrap())
            .into_iter()
            .collect::<Vec<_>>();
        let _exec = higgs_models::mlx_exec::acquire();
        drafter.seal_after_taps(cache, &taps, boundary).unwrap()
    }

    fn advance_target_one(cache: &mut AnyCache, token: u32) {
        let AnyCache::KV(layers) = cache else {
            panic!("test target must use a KV cache");
        };
        let layer = layers
            .first_mut()
            .and_then(Option::as_mut)
            .expect("test target layer");
        let value = token as f32;
        let keys = Array::from_slice(&[value], &[1, 1, 1, 1]);
        let values = Array::from_slice(&[-value], &[1, 1, 1, 1]);
        layer.update_and_fetch(keys, values).unwrap();
    }

    fn target_ahead_live_pair(tokens: &[u32]) -> (LivePair, DFlashDrafter) {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let _exec = higgs_models::mlx_exec::acquire();
        let pair = pair
            .prefill_known(tokens, |exact, target, draft| {
                assert_eq!(exact, tokens);
                assert_eq!(draft.position(), 0);
                for &token in exact {
                    advance_target_one(target, token);
                }
                let rows = i32::try_from(exact.len()).unwrap();
                let taps = (rows > 0)
                    .then(|| Array::zeros::<f32>(&[1, rows, 4]).unwrap())
                    .into_iter()
                    .collect();
                Ok::<_, EngineError>(((), taps))
            })
            .unwrap()
            .0;
        pair.target.cache.eval().unwrap();
        (pair, drafter)
    }

    fn finish_one_forwarded_token(
        pair: LivePair,
        token: u32,
    ) -> Result<LivePair, PairedCacheError> {
        let _exec = higgs_models::mlx_exec::acquire();
        let (lease, key) = pair.begin_decode()?;
        let mut ledger = TokenLedger::new_paired(key);
        ledger.emit_pending(token).unwrap();
        let ticket = ledger.begin_cache_only_forward().unwrap();
        let (lease, ()) = lease.run(|target, _draft, frontier| {
            advance_target_one(target, token);
            let tap = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
            let next_boundary = frontier.target_boundary() + 1;
            let next = frontier.append(next_boundary, vec![tap])?;
            Ok::<_, EngineError>((next, ()))
        })?;
        ledger.complete_cache_only_forward(ticket).unwrap();
        lease.finish(ledger.into_paired_proof().unwrap())
    }

    #[test]
    fn live_pair_accepts_target_ahead_frontiers_across_tile_boundaries() {
        for boundary in [31_usize, 32, 33] {
            let tokens = vec![11; boundary];
            let (pair, _drafter) = target_ahead_live_pair(&tokens);

            assert_eq!(pair.tokens, tokens);
            assert_eq!(pair.dflash.cache.position(), 0);
            assert_eq!(pair.frontier.frontier.draft_boundary(), 0);
            assert_eq!(
                pair.frontier.frontier.target_boundary(),
                i32::try_from(boundary).unwrap()
            );
            pair.validate_stable().unwrap();
        }
    }

    #[test]
    fn known_advance_callback_receives_the_captured_exact_suffix() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let mut caller_suffix = vec![11, 12];
        let caller_ptr = caller_suffix.as_ptr();
        let _exec = higgs_models::mlx_exec::acquire();

        let (pair, captured_ptr) = pair
            .advance_known(&caller_suffix, |exact, target, _draft, frontier| {
                for &token in exact {
                    advance_target_one(target, token);
                }
                let taps = vec![Array::zeros::<f32>(&[1, 2, 4]).unwrap()];
                Ok::<_, EngineError>((frontier.append(2, taps)?, exact.as_ptr()))
            })
            .unwrap();
        caller_suffix[0] = 99;

        assert_ne!(captured_ptr, caller_ptr);
        assert_eq!(pair.tokens, [11, 12]);
    }

    #[test]
    fn known_prefill_derives_nonzero_boundaries_from_pair_owned_tokens() {
        let retained_tokens = vec![11];
        let (pair, mut drafter) = target_ahead_live_pair(&retained_tokens);
        let exec = higgs_models::mlx_exec::acquire();
        let sealed = pair.seal(&mut drafter, &exec).unwrap();
        let pair = sealed.resume(&retained_tokens, 1).unwrap();
        let mut caller_suffix = vec![12, 13];
        let caller_ptr = caller_suffix.as_ptr();

        let (pair, captured_ptr) = pair
            .prefill_known(&caller_suffix, |exact, target, draft| {
                assert_eq!(exact, [12, 13]);
                assert_eq!(
                    draft.position(),
                    1,
                    "the resumed drafter must begin at the retained boundary"
                );
                for &token in exact {
                    advance_target_one(target, token);
                }
                let taps = vec![Array::zeros::<f32>(&[1, 2, 4]).unwrap()];
                Ok::<_, EngineError>((exact.as_ptr(), taps))
            })
            .unwrap();
        caller_suffix[0] = 99;

        assert_ne!(
            captured_ptr, caller_ptr,
            "the callback must observe the pair-owned suffix"
        );
        assert_eq!(pair.tokens, [11, 12, 13]);
        assert_eq!(pair.dflash.cache.position(), 1);
        assert_eq!(pair.frontier.frontier.draft_boundary(), 1);
        assert_eq!(pair.frontier.frontier.target_boundary(), 3);
        assert_eq!(pair.frontier.frontier.rows().unwrap(), 2);
        pair.target.cache.validate_absolute_boundary(3).unwrap();
        pair.validate_stable().unwrap();
    }

    #[test]
    fn known_prefill_rejects_a_target_ahead_frontier_before_model_work() {
        let (pair, _drafter) = target_ahead_live_pair(&[11]);
        let callback_called = std::cell::Cell::new(false);

        let error = pair
            .prefill_known(&[12], |_exact, _target, _draft| {
                callback_called.set(true);
                Ok::<_, EngineError>(((), vec![Array::zeros::<f32>(&[1, 1, 4]).unwrap()]))
            })
            .unwrap_err();

        assert!(!callback_called.get());
        assert_eq!(
            error,
            PairedCacheError::Advance {
                details:
                    "Generation error: known dSpark prefill requires an aligned frontier, found 1 pending target rows"
                        .to_owned()
            }
        );
    }

    #[test]
    fn known_prefill_rejects_an_empty_suffix_before_model_work() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let callback_called = std::cell::Cell::new(false);

        let error = pair
            .prefill_known(&[], |_exact, _target, _draft| {
                callback_called.set(true);
                Ok::<_, EngineError>(((), Vec::new()))
            })
            .unwrap_err();

        assert!(!callback_called.get());
        assert_eq!(
            error,
            PairedCacheError::Advance {
                details: "known dSpark prefill requires a non-empty suffix".to_owned()
            }
        );
    }

    #[test]
    fn known_advance_error_cannot_return_a_publishable_pair() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let _exec = higgs_models::mlx_exec::acquire();

        let error = pair
            .advance_known(&[11], |_exact, target, _draft, _frontier| {
                advance_target_one(target, 11);
                Err::<(super::DflashTapFrontier, ()), _>("injected prefill failure")
            })
            .unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::Advance {
                details: "injected prefill failure".to_owned()
            }
        );
    }

    #[test]
    fn known_advance_cannot_change_the_expected_tap_count() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let _exec = higgs_models::mlx_exec::acquire();

        let error = pair
            .advance_known(&[11], |_exact, target, _draft, _frontier| {
                advance_target_one(target, 11);
                let taps = vec![
                    Array::zeros::<f32>(&[1, 1, 4]).unwrap(),
                    Array::zeros::<f32>(&[1, 1, 4]).unwrap(),
                ];
                let next = DflashTapFrontier::new(0, 1, taps, 2)?;
                Ok::<_, EngineError>((next, ()))
            })
            .unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::DFlashTapCount {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn decode_lease_cannot_change_the_expected_tap_count() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let (lease, _key) = pair.begin_decode().unwrap();

        let error = lease
            .run(|_target, _draft, _frontier| {
                let next = DflashTapFrontier::new(0, 0, vec![], 2)?;
                Ok::<_, EngineError>((next, ()))
            })
            .unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::DFlashTapCount {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn failed_cache_only_forward_returns_neither_lease_nor_paired_proof() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let (lease, key) = pair.begin_decode().unwrap();
        let mut ledger = TokenLedger::new_paired(key);
        ledger.emit_pending(11).unwrap();
        let _ticket = ledger.begin_cache_only_forward().unwrap();

        let error = lease
            .run(|target, _draft, _frontier| {
                advance_target_one(target, 11);
                Err::<(DflashTapFrontier, ()), _>("injected cache-only forward failure")
            })
            .unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::Decode {
                details: "injected cache-only forward failure".to_owned()
            }
        );
        assert!(
            ledger.into_paired_proof().is_err(),
            "the in-flight ledger must not mint a publication proof after the lease was consumed"
        );
    }

    #[test]
    fn visible_eos_is_excluded_before_live_pair_finish_and_session_seal() {
        let tokens = vec![11];
        let (pair, mut drafter) = target_ahead_live_pair(&tokens);
        let (lease, key) = pair.begin_decode().unwrap();
        let mut ledger = TokenLedger::new_paired(key);
        let eos = 2;
        ledger.emit_pending(eos).unwrap();
        ledger.exclude_pending_eos(&[eos]).unwrap();

        let pair = lease.finish(ledger.into_paired_proof().unwrap()).unwrap();

        assert_eq!(pair.tokens, tokens);
        pair.target.cache.validate_absolute_boundary(1).unwrap();
        assert_eq!(pair.frontier.frontier.target_boundary(), 1);
        let exec = higgs_models::mlx_exec::acquire();
        let publication = pair.seal_for_session(&mut drafter, &exec).unwrap();
        assert_eq!(
            publication.into_state().tokens(),
            [11],
            "the visible terminal token must never enter the retained prefix stamp"
        );
    }

    #[test]
    fn live_pair_direct_seal_consumes_frontier_without_a_publication_label() {
        // Keep this engine-level fixture below the dFlash 32-row projection
        // tile: dependency test builds intentionally expose placeholder model
        // weights. Tile-boundary frontier ownership is covered independently
        // above; model-level sealing tests exercise real projection weights.
        let tokens = vec![11];
        let (pair, mut drafter) = target_ahead_live_pair(&tokens);
        let exec = higgs_models::mlx_exec::acquire();

        let sealed = pair.seal(&mut drafter, &exec).unwrap();

        assert!(sealed.matches_prefix(&tokens));
        assert!(!sealed.matches_prefix(&[12]));
        assert!(
            sealed.sealed.metadata.stamp.branch_epoch.is_some(),
            "the correct-by-construction path must retain its live branch epoch"
        );
    }

    fn checkpoint_snapshot(
        pair: LivePair,
        drafter: &mut DFlashDrafter,
    ) -> (LivePair, std::sync::Arc<super::RadixDFlashSnapshot>) {
        let exec = higgs_models::mlx_exec::acquire();
        let (continued, prepared) = pair
            .checkpoint_for_radix(drafter, &exec, |checkpoint| {
                let target_bytes = checkpoint.target().estimated_bytes();
                Ok::<_, ()>(
                    checkpoint
                        .into_radix_snapshot(target_bytes)
                        .expect("valid proven checkpoint"),
                )
            })
            .unwrap();
        (
            continued,
            std::sync::Arc::new(prepared.expect("checkpoint preparation")),
        )
    }

    #[test]
    fn radix_checkpoint_owns_the_live_pairs_authoritative_tokens() {
        let mut caller_tokens = vec![11];
        let (pair, mut drafter) = target_ahead_live_pair(&caller_tokens);

        let (continued, snapshot) = checkpoint_snapshot(pair, &mut drafter);
        caller_tokens[0] = 99;

        assert_eq!(continued.tokens, [11]);
        assert!(snapshot.matches_prefix(&[11]));
        assert!(!snapshot.matches_prefix(&caller_tokens));
        assert!(
            snapshot.metadata.stamp.branch_epoch.is_some(),
            "production radix checkpoints must retain proven live provenance"
        );
    }

    #[test]
    fn radix_checkpoint_rejects_a_same_length_relabel() {
        let (pair, mut drafter) = target_ahead_live_pair(&[11]);
        let (_continued, snapshot) = checkpoint_snapshot(pair, &mut drafter);

        assert!(matches!(
            snapshot.plan_fork(&[12]).unwrap_err(),
            PairedCacheError::PrefixMismatch {
                stored_len: 1,
                requested_len: 1
            }
        ));
    }

    #[test]
    fn radix_snapshot_forks_independent_proven_live_pairs() {
        let (pair, mut drafter) = target_ahead_live_pair(&[11]);
        let (_continued, snapshot) = checkpoint_snapshot(pair, &mut drafter);
        let left_plan = snapshot.plan_fork(&[11]).unwrap();
        let right_plan = snapshot.plan_fork(&[11]).unwrap();
        let left_target = target_cache(1);
        let right_target = target_cache(1);
        let exec = higgs_models::mlx_exec::acquire();

        let left = left_plan.materialize_pair(left_target, 1).unwrap();
        let right = right_plan.materialize_pair(right_target, 1).unwrap();
        assert_ne!(left.epoch, right.epoch);
        drop(exec);
        let left = finish_one_forwarded_token(left, 12).unwrap();

        assert_eq!(left.tokens, [11, 12]);
        assert_eq!(right.tokens, [11]);
        right.validate_stable().unwrap();
        let third_target = target_cache(1);
        let _exec = higgs_models::mlx_exec::acquire();
        let third = snapshot
            .plan_fork(&[11])
            .unwrap()
            .materialize_pair(third_target, 1)
            .unwrap();
        assert_eq!(third.tokens, [11]);
    }

    #[test]
    fn radix_checkpoint_preparation_failure_preserves_the_continued_pair() {
        let (pair, mut drafter) = target_ahead_live_pair(&[11]);
        let exec = higgs_models::mlx_exec::acquire();

        let (continued, prepared) = pair
            .checkpoint_for_radix(&mut drafter, &exec, |_checkpoint| {
                Err::<(), _>("injected radix preparation failure")
            })
            .unwrap();

        assert_eq!(prepared, Err("injected radix preparation failure"));
        assert_eq!(continued.tokens, [11]);
        continued.validate_stable().unwrap();
        drop(exec);
        let continued = finish_one_forwarded_token(continued, 12).unwrap();
        assert_eq!(continued.tokens, [11, 12]);
    }

    #[test]
    fn radix_checkpoint_seal_and_fork_failures_expose_no_checkpoint() {
        let (seal_pair, mut broken_drafter) = target_ahead_live_pair(&[11]);
        broken_drafter.config.hidden_size = 5;
        let seal_called = std::cell::Cell::new(false);
        let exec = higgs_models::mlx_exec::acquire();

        let seal_error = seal_pair
            .checkpoint_for_radix(&mut broken_drafter, &exec, |_checkpoint| {
                seal_called.set(true);
                Ok::<(), ()>(())
            })
            .unwrap_err();
        assert!(matches!(seal_error, PairedCacheError::DFlashSeal { .. }));
        assert!(!seal_called.get());
        drop(exec);

        let (fork_pair, mut drafter) = target_ahead_live_pair(&[11]);
        let fork_called = std::cell::Cell::new(false);
        super::FAIL_NEXT_RADIX_CHECKPOINT_FORK.store(true, std::sync::atomic::Ordering::SeqCst);
        let exec = higgs_models::mlx_exec::acquire();
        let fork_error = fork_pair
            .checkpoint_for_radix(&mut drafter, &exec, |_checkpoint| {
                fork_called.set(true);
                Ok::<(), ()>(())
            })
            .unwrap_err();
        assert!(matches!(fork_error, PairedCacheError::DFlashFork { .. }));
        assert!(!fork_called.get());
    }

    #[test]
    fn legacy_radix_snapshot_cannot_materialize_a_proven_live_pair() {
        let snapshot = std::sync::Arc::new(
            super::RadixDFlashSnapshot::new(dflash_snapshot(1), &[11], 1).unwrap(),
        );
        let plan = snapshot.plan_fork(&[11]).unwrap();
        let target = target_cache(1);
        let _exec = higgs_models::mlx_exec::acquire();

        assert_eq!(
            plan.materialize_pair(target, 1).unwrap_err(),
            PairedCacheError::UnprovenPair
        );
    }

    #[test]
    fn session_publication_derives_tokens_only_from_the_live_pair() {
        let mut caller_tokens = vec![11];
        let (pair, mut drafter) = target_ahead_live_pair(&caller_tokens);
        let exec = higgs_models::mlx_exec::acquire();

        let publication = pair.seal_for_session(&mut drafter, &exec).unwrap();
        caller_tokens[0] = 99;
        let state = publication.into_state();
        let retained_tokens = state.tokens().to_vec();

        assert_eq!(retained_tokens, [11]);
        let RetainedState::Paired(pair) = state else {
            panic!("matching tap capability must publish a paired session");
        };
        assert!(pair.matches_prefix(&retained_tokens));
        assert!(!pair.matches_prefix(&caller_tokens));
    }

    #[test]
    fn deterministic_session_tap_mismatch_demotes_the_validated_target() {
        let (pair, _matching_drafter) = target_ahead_live_pair(&[11]);
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
                    "target_layer_ids": [0, 1]
                }
            }"#,
        )
        .unwrap();
        let mut incompatible = DFlashDrafter::new(config).unwrap();
        let exec = higgs_models::mlx_exec::acquire();

        let publication = pair.seal_for_session(&mut incompatible, &exec).unwrap();

        assert_eq!(
            publication.demotion(),
            Some(&DflashSealDemotion::MissingOrUnsupportedTaps {
                rows: 1,
                collected: 1,
                required: 2,
            })
        );
        let state = publication.into_state();
        assert_eq!(state.tokens(), [11]);
        let RetainedState::TargetOnly(target) = state else {
            panic!("deterministic capability loss must retain only the target");
        };
        target.cache.validate_absolute_boundary(1).unwrap();
    }

    #[test]
    fn effectful_session_seal_failure_returns_no_publication() {
        let (pair, mut drafter) = target_ahead_live_pair(&[11]);
        drafter.config.hidden_size = 5;
        let exec = higgs_models::mlx_exec::acquire();

        assert!(matches!(
            pair.seal_for_session(&mut drafter, &exec).unwrap_err(),
            PairedCacheError::DFlashSeal { .. }
        ));
    }

    #[test]
    fn proven_resume_moves_stored_tokens_and_mints_a_fresh_epoch() {
        let tokens = vec![11, 12];
        let (pair, mut drafter) = target_ahead_live_pair(&tokens);
        let sealed_epoch = pair.epoch;
        let exec = higgs_models::mlx_exec::acquire();
        let sealed = pair.seal(&mut drafter, &exec).unwrap();

        let resumed = sealed.resume(&tokens, 1).unwrap();

        assert_eq!(resumed.tokens, tokens);
        assert_eq!(
            resumed.dflash.cache.position(),
            i32::try_from(resumed.tokens.len()).unwrap(),
            "resumed dSpark state must start at the retained nonzero token boundary"
        );
        assert_ne!(resumed.epoch, sealed_epoch);
        resumed.validate_stable().unwrap();
    }

    #[test]
    fn proven_resume_rejects_a_same_length_different_prefix() {
        let tokens = vec![11, 12];
        let (pair, mut drafter) = target_ahead_live_pair(&tokens);
        let exec = higgs_models::mlx_exec::acquire();
        let sealed = pair.seal(&mut drafter, &exec).unwrap();

        assert_eq!(
            sealed.resume(&[11, 13], 1).unwrap_err(),
            PairedCacheError::PrefixMismatch {
                stored_len: 2,
                requested_len: 2
            }
        );
    }

    #[test]
    fn legacy_unproven_pair_cannot_resume_as_live_pair() {
        let pair = PairedCache::new(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert_eq!(
            pair.resume(&[], 1).unwrap_err(),
            PairedCacheError::UnprovenPair
        );
    }

    #[test]
    fn live_pair_rejects_a_same_length_target_half_from_another_branch() {
        let (mut left, mut left_drafter) = target_ahead_live_pair(&[11]);
        let (mut right, _right_drafter) = target_ahead_live_pair(&[12]);
        std::mem::swap(&mut left.target, &mut right.target);
        let exec = higgs_models::mlx_exec::acquire();

        let error = left.seal(&mut left_drafter, &exec).unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignTargetBranch);
    }

    #[test]
    fn live_pair_rejects_a_same_length_dflash_half_from_another_branch() {
        let (mut left, mut left_drafter) = target_ahead_live_pair(&[11]);
        let (mut right, _right_drafter) = target_ahead_live_pair(&[12]);
        std::mem::swap(&mut left.dflash, &mut right.dflash);
        let exec = higgs_models::mlx_exec::acquire();

        let error = left.seal(&mut left_drafter, &exec).unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignDFlashPairBranch);
    }

    #[test]
    fn live_pair_rejects_a_same_length_frontier_half_from_another_branch() {
        let (mut left, mut left_drafter) = target_ahead_live_pair(&[11]);
        let (mut right, _right_drafter) = target_ahead_live_pair(&[12]);
        std::mem::swap(&mut left.frontier, &mut right.frontier);
        let exec = higgs_models::mlx_exec::acquire();

        let error = left.seal(&mut left_drafter, &exec).unwrap_err();

        assert_eq!(error, PairedCacheError::ForeignDFlashPairBranch);
    }

    #[test]
    fn same_length_cross_pair_decode_proof_is_rejected() {
        let left_drafter = test_drafter();
        let right_drafter = test_drafter();
        let left = LivePair::cold(target_cache(0), left_drafter.make_cache(), 1).unwrap();
        let right = LivePair::cold(target_cache(0), right_drafter.make_cache(), 1).unwrap();
        let (_left_lease, left_key) = left.begin_decode().unwrap();
        let (right_lease, _right_key) = right.begin_decode().unwrap();
        let left_proof = TokenLedger::new_paired(left_key)
            .into_paired_proof()
            .unwrap();

        assert_eq!(
            right_lease.finish(left_proof).unwrap_err(),
            PairedCacheError::ForeignLedgerProof
        );
    }

    #[test]
    fn same_pair_decode_finish_accepts_the_exact_forwarded_suffix() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let pair = finish_one_forwarded_token(pair, 11).unwrap();

        assert_eq!(pair.tokens, [11]);
        assert_eq!(pair.revision, 1);
        pair.validate_stable().unwrap();
    }

    #[test]
    fn decode_finish_rejects_a_target_boundary_shorter_than_the_proof() {
        let drafter = test_drafter();
        let pair = LivePair::cold(target_cache(0), drafter.make_cache(), 1).unwrap();
        let (lease, key) = pair.begin_decode().unwrap();
        let mut ledger = TokenLedger::new_paired(key);
        ledger.emit_pending(11).unwrap();
        let ticket = ledger.begin_cache_only_forward().unwrap();
        ledger.complete_cache_only_forward(ticket).unwrap();

        assert!(matches!(
            lease
                .finish(ledger.into_paired_proof().unwrap())
                .unwrap_err(),
            PairedCacheError::TargetBoundary { expected: 1, .. }
        ));
    }

    #[test]
    fn paired_cache_accepts_one_exact_shared_boundary() {
        let pair = PairedCache::new(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert_eq!(pair.prefix_len(), 0);
        assert!(pair.matches_prefix(&[]));
        assert!(
            pair.sealed.metadata.stamp.branch_epoch.is_none(),
            "legacy compatibility construction must remain visibly unproven"
        );
    }

    #[test]
    fn prefix_length_must_fit_the_model_boundary_type() {
        let len = usize::try_from(i32::MAX).unwrap() + 1;
        let stamp = PrefixStamp {
            branch_epoch: None,
            hash: 0,
            len,
            tokens: Vec::new().into_boxed_slice(),
        };

        assert_eq!(
            stamp.boundary().unwrap_err(),
            PairedCacheError::PrefixLengthOverflow { len }
        );
    }

    #[test]
    fn live_pair_branch_counter_rejects_wraparound() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX);

        assert_eq!(
            super::next_pair_branch_epoch_from(&counter).unwrap_err(),
            PairedCacheError::PairBranchEpochOverflow
        );
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn paired_cache_rejects_target_boundary_mismatch() {
        let error = PairedCache::new(target_cache(0), dflash_snapshot(1), &[11]).unwrap_err();

        assert!(matches!(
            error,
            PairedCacheError::TargetBoundary { expected: 1, .. }
        ));
    }

    #[test]
    fn paired_cache_rejects_drafter_boundary_mismatch() {
        let error = PairedCache::new(target_cache(1), dflash_snapshot(0), &[11]).unwrap_err();

        assert_eq!(
            error,
            PairedCacheError::DFlashBoundary {
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn same_length_different_prefix_does_not_match() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        assert!(!pair.matches_prefix(&[12]));
    }

    #[test]
    fn exact_prefix_identity_rejects_a_simulated_hash_collision() {
        let stamp = PrefixStamp::new(&[11, 22]);

        assert!(
            !stamp.matches_hashed(&[11, 23], stamp.hash),
            "equal hash and length must not substitute for exact token equality"
        );
    }

    #[test]
    fn consuming_live_reuse_rechecks_the_exact_prefix() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let error = pair.into_live(&[12]).unwrap_err();
        assert_eq!(
            error,
            PairedCacheError::PrefixMismatch {
                stored_len: 1,
                requested_len: 1
            }
        );
    }

    #[test]
    fn consuming_live_reuse_moves_both_caches_together() {
        let pair = PairedCache::new(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let (target, dflash) = pair.into_live(&[11]).unwrap();
        target.validate_absolute_boundary(1).unwrap();
        assert_eq!(dflash.position(), 1);
    }

    #[test]
    fn retained_pair_can_only_be_demoted_by_consuming_the_whole_pair() {
        let retained = RetainedState::paired(target_cache(0), dflash_snapshot(0), &[]).unwrap();

        assert!(matches!(retained, RetainedState::Paired(_)));
        let target = retained.demote();
        target.validate_absolute_boundary(0).unwrap();
    }

    #[test]
    fn target_only_state_demotes_without_special_cases() {
        let target = target_cache(0);
        let exec = higgs_models::mlx_exec::acquire();
        let retained = RetainedState::target_only(target, vec![], &exec).unwrap();

        retained.demote().validate_absolute_boundary(0).unwrap();
    }

    #[test]
    fn retained_reuse_returns_nonmatching_state_intact() {
        let retained = RetainedState::paired(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let retained = retained.into_paired(&[12]).unwrap_err();
        assert!(matches!(retained, RetainedState::Paired(_)));
        retained.demote().validate_absolute_boundary(1).unwrap();
    }

    #[test]
    fn retained_reuse_moves_a_matching_pair_together() {
        let retained = RetainedState::paired(target_cache(1), dflash_snapshot(1), &[11]).unwrap();

        let (target, dflash) = retained.into_paired(&[11]).unwrap();
        target.validate_absolute_boundary(1).unwrap();
        assert_eq!(dflash.position(), 1);
    }
}
