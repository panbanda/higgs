//! Token-boundary accounting for cache-resident generation.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::paired::PairLedgerKey;

static NEXT_LEDGER_ID: AtomicU64 = AtomicU64::new(1);

fn try_next_ledger_id(counter: &AtomicU64) -> Option<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .ok()
}

fn next_ledger_id() -> u64 {
    let Some(ledger_id) = try_next_ledger_id(&NEXT_LEDGER_ID) else {
        // Wrapping could let a stale move-only ticket alias a future ledger.
        // Exhausting u64 identities is unrecoverable, so fail closed.
        std::process::abort();
    };
    ledger_id
}

/// The action required before this ledger can label a retained cache boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetentionAction {
    /// Every emitted token represented by the retention key has been forwarded.
    Ready { boundary: usize },
    /// Forward this visible non-EOS token once, without sampling another token.
    CacheOnlyForward { token: u32 },
    /// The trailing token is EOS: keep it in the response but exclude it from
    /// the retained key because it was never forwarded and is not conversational
    /// content.
    ExcludeEos { token: u32 },
    /// A cache-only forward has begun but has not yet succeeded.
    ForwardInFlight { token: u32 },
}

/// Token-ledger transition failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum LedgerError {
    #[error("token {token} is emitted but not forwarded")]
    PendingToken { token: u32 },
    #[error("cache-only forward for token {token} has not completed")]
    ForwardInFlight { token: u32 },
    #[error("no emitted pending token is available")]
    NoPendingToken,
    #[error("pending token {token} is not EOS")]
    PendingTokenIsNotEos { token: u32 },
    #[error("EOS token {token} was excluded; the ledger is terminal")]
    TerminalTokenExcluded { token: u32 },
    #[error("cache-only forward ticket belongs to another ledger or transition")]
    ForeignForwardTicket,
    #[error("a speculative round must produce one pending successor")]
    EmptySpeculativeRound,
    #[error("token boundary overflow")]
    BoundaryOverflow,
    #[error("cache-only forward ticket counter overflow")]
    ForwardTicketOverflow,
    #[error("token ledger is not bound to a live target/dSpark pair")]
    MissingPairBinding,
    #[error("pair-bound ledgers require a cache-only or speculative forward ticket")]
    UnverifiedPairedForwardClaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailState {
    Aligned,
    Pending { token: u32 },
    Forwarding { token: u32, ticket_id: u64 },
    ExcludedEos { token: u32 },
}

/// Opaque proof that a specific pending token entered a cache-only forward.
///
/// The ticket is intentionally neither `Clone` nor `Copy`: a successful model
/// forward consumes it exactly once when the ledger commits the token.
#[derive(Debug)]
struct TicketIdentity {
    ledger_id: u64,
    ticket_id: u64,
    token: u32,
}

#[derive(Debug)]
pub(crate) struct ForwardTicket(TicketIdentity);

/// Opaque proof that one exact pending token anchors a speculative round.
///
/// The drafter/target round takes ownership of this ticket and derives its
/// anchor token from it. The successful round then returns the same ticket to
/// the ledger transition, so a different same-length token sequence cannot be
/// published under the original cache boundary.
#[derive(Debug)]
pub(crate) struct SpeculativeTicket(TicketIdentity);

impl SpeculativeTicket {
    #[must_use]
    pub(crate) const fn token(&self) -> u32 {
        self.0.token
    }
}

/// Move-only proof that one pair-bound ledger reached an exact forwarded
/// completion suffix.
///
/// The proof owns both the opaque pair identity and the ledger's forwarded
/// prefix. A pending or in-flight token can never enter this value.
#[derive(Debug)]
pub(crate) struct PairedLedgerProof {
    key: PairLedgerKey,
    forwarded_suffix: Box<[u32]>,
}

impl PairedLedgerProof {
    /// Consume this proof only when it belongs to the expected pair identity.
    ///
    /// The owned key is never returned, even on mismatch, so it cannot be
    /// recycled into a second pair-bound token ledger.
    pub(crate) fn into_forwarded_suffix_for(self, expected: &PairLedgerKey) -> Option<Box<[u32]>> {
        (self.key == *expected).then_some(self.forwarded_suffix)
    }
}

/// Completion-token accounting for a cache-resident turn.
///
/// `emitted` is the response-visible sequence. `forwarded_len` is always a
/// prefix of it and is the only suffix allowed into a retention key. At most
/// one trailing emitted token may be pending; while it is pending or being
/// forwarded, [`Self::retainable_tokens`] fails closed.
#[derive(Debug)]
pub(crate) struct TokenLedger {
    base_boundary: usize,
    emitted: Vec<u32>,
    forwarded_len: usize,
    tail: TailState,
    ledger_id: u64,
    next_ticket_id: u64,
    pair_key: Option<PairLedgerKey>,
}

impl TokenLedger {
    #[must_use]
    pub(crate) fn new(base_boundary: usize) -> Self {
        Self {
            base_boundary,
            emitted: Vec::new(),
            forwarded_len: 0,
            tail: TailState::Aligned,
            ledger_id: next_ledger_id(),
            next_ticket_id: 1,
            pair_key: None,
        }
    }

    /// Start completion accounting for one exact move-owned live pair.
    ///
    /// The base boundary is derived from the opaque key; callers cannot supply
    /// an independent same-length label.
    #[must_use]
    pub(crate) fn new_paired(key: PairLedgerKey) -> Self {
        Self {
            base_boundary: key.base_boundary(),
            emitted: Vec::new(),
            forwarded_len: 0,
            tail: TailState::Aligned,
            ledger_id: next_ledger_id(),
            next_ticket_id: 1,
            pair_key: Some(key),
        }
    }

    /// Record a visible token whose target-cache transition already succeeded.
    pub(crate) fn record_forwarded(&mut self, token: u32) -> Result<(), LedgerError> {
        self.extend_forwarded([token])
    }

    /// Record a visible run already committed by a speculative verification
    /// transaction. This is atomic with respect to boundary overflow.
    pub(crate) fn extend_forwarded<I>(&mut self, tokens: I) -> Result<(), LedgerError>
    where
        I: IntoIterator<Item = u32>,
    {
        if self.pair_key.is_some() {
            return Err(LedgerError::UnverifiedPairedForwardClaim);
        }
        self.ensure_aligned_for_append()?;
        let incoming: Vec<u32> = tokens.into_iter().collect();
        let new_forwarded = self
            .forwarded_len
            .checked_add(incoming.len())
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.base_boundary
            .checked_add(new_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.emitted.extend(incoming);
        self.forwarded_len = new_forwarded;
        Ok(())
    }

    /// Emit one sampled token without claiming that the target cache contains
    /// it. No other token may be appended until this tail is forwarded or
    /// excluded as EOS.
    pub(crate) fn emit_pending(&mut self, token: u32) -> Result<(), LedgerError> {
        self.ensure_aligned_for_append()?;
        self.emitted.push(token);
        self.tail = TailState::Pending { token };
        Ok(())
    }

    /// Decide how to align the cache before publication.
    #[must_use]
    pub(crate) fn retention_action(&self, eos_token_ids: &[u32]) -> RetentionAction {
        match self.tail {
            TailState::Aligned | TailState::ExcludedEos { .. } => RetentionAction::Ready {
                boundary: self
                    .base_boundary
                    .checked_add(self.forwarded_len)
                    .unwrap_or(usize::MAX),
            },
            TailState::Pending { token } if eos_token_ids.contains(&token) => {
                RetentionAction::ExcludeEos { token }
            }
            TailState::Pending { token } => RetentionAction::CacheOnlyForward { token },
            TailState::Forwarding { token, .. } => RetentionAction::ForwardInFlight { token },
        }
    }

    /// Start the cache-only forward required for a visible non-EOS tail.
    pub(crate) fn begin_cache_only_forward(&mut self) -> Result<ForwardTicket, LedgerError> {
        self.begin_forward().map(ForwardTicket)
    }

    /// Bind the current visible pending token to one speculative target round.
    pub(crate) fn begin_speculative_round(&mut self) -> Result<SpeculativeTicket, LedgerError> {
        self.begin_forward().map(SpeculativeTicket)
    }

    fn begin_forward(&mut self) -> Result<TicketIdentity, LedgerError> {
        let TailState::Pending { token } = self.tail else {
            return Err(self.tail_error());
        };
        let ticket_id = self.next_ticket_id;
        self.next_ticket_id = self
            .next_ticket_id
            .checked_add(1)
            .ok_or(LedgerError::ForwardTicketOverflow)?;
        self.tail = TailState::Forwarding { token, ticket_id };
        Ok(TicketIdentity {
            ledger_id: self.ledger_id,
            ticket_id,
            token,
        })
    }

    /// Commit a pending token only after its cache-only target forward succeeds.
    pub(crate) fn complete_cache_only_forward(
        &mut self,
        ticket: ForwardTicket,
    ) -> Result<(), LedgerError> {
        let ticket = ticket.0;
        let TailState::Forwarding { token, ticket_id } = self.tail else {
            return Err(self.tail_error());
        };
        if ticket.ledger_id != self.ledger_id
            || ticket.ticket_id != ticket_id
            || ticket.token != token
        {
            return Err(LedgerError::ForeignForwardTicket);
        }
        let new_forwarded = self
            .forwarded_len
            .checked_add(1)
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.base_boundary
            .checked_add(new_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;
        if self.emitted.get(self.forwarded_len).copied() != Some(token) {
            return Err(LedgerError::ForeignForwardTicket);
        }
        self.forwarded_len = new_forwarded;
        self.tail = TailState::Aligned;
        Ok(())
    }

    /// Commit one target-authoritative speculative round as a single ledger
    /// transition.
    ///
    /// The in-flight ticket proves that the previously pending anchor entered
    /// the target cache. `successors[..len - 1]` were subsequently used as
    /// verified target inputs and are therefore cache-resident too; the final
    /// successor remains response-visible but unforwarded. All validation and
    /// boundary arithmetic happen before mutation, so callers can never expose
    /// a partially-accounted round.
    pub(crate) fn complete_speculative_round(
        &mut self,
        ticket: SpeculativeTicket,
        successors: &[u32],
    ) -> Result<(), LedgerError> {
        let ticket = ticket.0;
        let Some((&pending, forwarded_successors)) = successors.split_last() else {
            return Err(LedgerError::EmptySpeculativeRound);
        };
        let TailState::Forwarding { token, ticket_id } = self.tail else {
            return Err(self.tail_error());
        };
        if ticket.ledger_id != self.ledger_id
            || ticket.ticket_id != ticket_id
            || ticket.token != token
            || self.emitted.get(self.forwarded_len).copied() != Some(token)
        {
            return Err(LedgerError::ForeignForwardTicket);
        }
        let newly_forwarded = 1usize
            .checked_add(forwarded_successors.len())
            .ok_or(LedgerError::BoundaryOverflow)?;
        let new_forwarded = self
            .forwarded_len
            .checked_add(newly_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;
        self.base_boundary
            .checked_add(new_forwarded)
            .ok_or(LedgerError::BoundaryOverflow)?;

        self.emitted.extend_from_slice(successors);
        self.forwarded_len = new_forwarded;
        self.tail = TailState::Pending { token: pending };
        Ok(())
    }

    /// Exclude an emitted-but-unforwarded EOS from the retention key while
    /// keeping it in the response-visible sequence.
    pub(crate) fn exclude_pending_eos(
        &mut self,
        eos_token_ids: &[u32],
    ) -> Result<u32, LedgerError> {
        let TailState::Pending { token } = self.tail else {
            return Err(self.tail_error());
        };
        if !eos_token_ids.contains(&token) {
            return Err(LedgerError::PendingTokenIsNotEos { token });
        }
        self.tail = TailState::ExcludedEos { token };
        Ok(token)
    }

    /// Response-visible completion tokens, including an excluded terminal EOS.
    #[must_use]
    pub(crate) fn emitted_tokens(&self) -> &[u32] {
        &self.emitted
    }

    /// The one emitted tail that is not yet known to be cache-resident.
    #[must_use]
    pub(crate) const fn pending_token(&self) -> Option<u32> {
        match self.tail {
            TailState::Pending { token } | TailState::Forwarding { token, .. } => Some(token),
            TailState::Aligned | TailState::ExcludedEos { .. } => None,
        }
    }

    /// Tokens safe to append to the prompt when constructing a retention key.
    ///
    /// This method is the publication gate: it never returns the emitted vector
    /// while a token is pending or an attempted forward is still in flight.
    pub(crate) fn retainable_tokens(&self) -> Result<&[u32], LedgerError> {
        match self.tail {
            TailState::Pending { token } => Err(LedgerError::PendingToken { token }),
            TailState::Forwarding { token, .. } => Err(LedgerError::ForwardInFlight { token }),
            TailState::Aligned | TailState::ExcludedEos { .. } => self
                .emitted
                .get(..self.forwarded_len)
                .ok_or(LedgerError::BoundaryOverflow),
        }
    }

    /// Consume this ledger into a proof for its originating live pair.
    ///
    /// `emitted` is moved rather than copied. Truncating it to
    /// `forwarded_len` excludes a terminal EOS and is only reachable after
    /// pending/in-flight states have failed closed.
    pub(crate) fn into_paired_proof(mut self) -> Result<PairedLedgerProof, LedgerError> {
        match self.tail {
            TailState::Pending { token } => return Err(LedgerError::PendingToken { token }),
            TailState::Forwarding { token, .. } => {
                return Err(LedgerError::ForwardInFlight { token });
            }
            TailState::Aligned | TailState::ExcludedEos { .. } => {}
        }
        if self.forwarded_len > self.emitted.len() {
            return Err(LedgerError::BoundaryOverflow);
        }
        let key = self
            .pair_key
            .take()
            .ok_or(LedgerError::MissingPairBinding)?;
        self.emitted.truncate(self.forwarded_len);
        Ok(PairedLedgerProof {
            key,
            forwarded_suffix: self.emitted.into_boxed_slice(),
        })
    }

    fn ensure_aligned_for_append(&self) -> Result<(), LedgerError> {
        match self.tail {
            TailState::Aligned => Ok(()),
            TailState::Pending { token } => Err(LedgerError::PendingToken { token }),
            TailState::Forwarding { token, .. } => Err(LedgerError::ForwardInFlight { token }),
            TailState::ExcludedEos { token } => Err(LedgerError::TerminalTokenExcluded { token }),
        }
    }

    const fn tail_error(&self) -> LedgerError {
        match self.tail {
            TailState::Aligned => LedgerError::NoPendingToken,
            TailState::Pending { token } => LedgerError::PendingToken { token },
            TailState::Forwarding { token, .. } => LedgerError::ForwardInFlight { token },
            TailState::ExcludedEos { token } => LedgerError::TerminalTokenExcluded { token },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::cache::paired::PairLedgerKey;

    use super::{LedgerError, RetentionAction, TokenLedger};

    #[test]
    fn pending_non_eos_requires_cache_only_forward_before_retention() {
        let mut ledger = TokenLedger::new(41);
        ledger.emit_pending(7).unwrap();

        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::CacheOnlyForward { token: 7 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 7 }
        );

        let ticket = ledger.begin_cache_only_forward().unwrap();
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::ForwardInFlight { token: 7 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::ForwardInFlight { token: 7 }
        );

        ledger.complete_cache_only_forward(ticket).unwrap();
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::Ready { boundary: 42 }
        );
        assert_eq!(ledger.retainable_tokens().unwrap(), &[7]);
        assert_eq!(ledger.emitted_tokens(), &[7]);
    }

    #[test]
    fn emitted_eos_is_visible_but_excluded_from_the_cache_key() {
        let mut ledger = TokenLedger::new(12);
        ledger.record_forwarded(3).unwrap();
        ledger.emit_pending(99).unwrap();

        assert_eq!(
            ledger.retention_action(&[99, 100]),
            RetentionAction::ExcludeEos { token: 99 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 99 }
        );

        assert_eq!(ledger.exclude_pending_eos(&[99, 100]).unwrap(), 99);
        assert_eq!(ledger.emitted_tokens(), &[3, 99]);
        assert_eq!(ledger.retainable_tokens().unwrap(), &[3]);
        assert_eq!(
            ledger.retention_action(&[99, 100]),
            RetentionAction::Ready { boundary: 13 }
        );
    }

    #[test]
    fn no_key_can_include_a_pending_or_in_flight_token() {
        let mut ledger = TokenLedger::new(5);
        ledger.extend_forwarded([10, 11]).unwrap();
        ledger.emit_pending(12).unwrap();

        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 12 }
        );
        let ticket = ledger.begin_cache_only_forward().unwrap();
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::ForwardInFlight { token: 12 }
        );
        ledger.complete_cache_only_forward(ticket).unwrap();
        assert_eq!(ledger.retainable_tokens().unwrap(), &[10, 11, 12]);
    }

    #[test]
    fn only_one_trailing_pending_token_is_permitted() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(1).unwrap();
        assert_eq!(
            ledger.emit_pending(2).unwrap_err(),
            LedgerError::PendingToken { token: 1 }
        );
        assert_eq!(
            ledger.record_forwarded(2).unwrap_err(),
            LedgerError::PendingToken { token: 1 }
        );
    }

    #[test]
    fn eos_exclusion_rejects_a_non_eos_pending_token() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(4).unwrap();
        assert_eq!(
            ledger.exclude_pending_eos(&[99]).unwrap_err(),
            LedgerError::PendingTokenIsNotEos { token: 4 }
        );
        assert_eq!(ledger.pending_token(), Some(4));
    }

    #[test]
    fn forward_tickets_are_bound_to_their_originating_ledger() {
        let mut left = TokenLedger::new(0);
        let mut right = TokenLedger::new(0);
        left.emit_pending(8).unwrap();
        right.emit_pending(8).unwrap();

        let left_ticket = left.begin_cache_only_forward().unwrap();
        let right_ticket = right.begin_cache_only_forward().unwrap();
        assert_eq!(
            right.complete_cache_only_forward(left_ticket).unwrap_err(),
            LedgerError::ForeignForwardTicket
        );
        right.complete_cache_only_forward(right_ticket).unwrap();
        assert_eq!(right.retainable_tokens().unwrap(), &[8]);
    }

    #[test]
    fn excluded_eos_is_terminal_for_the_ledger() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(99).unwrap();
        ledger.exclude_pending_eos(&[99]).unwrap();

        assert_eq!(
            ledger.emit_pending(1).unwrap_err(),
            LedgerError::TerminalTokenExcluded { token: 99 }
        );
        assert_eq!(
            ledger.record_forwarded(1).unwrap_err(),
            LedgerError::TerminalTokenExcluded { token: 99 }
        );
    }

    #[test]
    fn boundary_overflow_is_reported_without_exposing_a_key() {
        let mut ledger = TokenLedger::new(usize::MAX);
        assert_eq!(
            ledger.record_forwarded(1).unwrap_err(),
            LedgerError::BoundaryOverflow
        );
        assert!(ledger.emitted_tokens().is_empty());
    }

    #[test]
    fn speculative_round_atomically_forwards_anchor_and_leaves_one_pending_successor() {
        let mut ledger = TokenLedger::new(40);
        ledger.emit_pending(11).unwrap();
        let ticket = ledger.begin_speculative_round().unwrap();

        ledger
            .complete_speculative_round(ticket, &[12, 13, 14])
            .unwrap();

        assert_eq!(ledger.emitted_tokens(), &[11, 12, 13, 14]);
        assert_eq!(ledger.pending_token(), Some(14));
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::CacheOnlyForward { token: 14 }
        );
        assert_eq!(
            ledger.retainable_tokens().unwrap_err(),
            LedgerError::PendingToken { token: 14 }
        );
    }

    #[test]
    fn speculative_round_rejects_an_empty_successor_set_without_publishing_a_boundary() {
        let mut ledger = TokenLedger::new(40);
        ledger.emit_pending(11).unwrap();
        let ticket = ledger.begin_speculative_round().unwrap();

        assert_eq!(
            ledger.complete_speculative_round(ticket, &[]).unwrap_err(),
            LedgerError::EmptySpeculativeRound
        );
        assert_eq!(
            ledger.retention_action(&[99]),
            RetentionAction::ForwardInFlight { token: 11 }
        );
    }

    #[test]
    fn speculative_ticket_is_bound_to_the_exact_pending_anchor() {
        let mut ledger = TokenLedger::new(0);
        ledger.emit_pending(17).unwrap();

        let ticket = ledger.begin_speculative_round().unwrap();

        assert_eq!(ticket.token(), 17);
        ledger.complete_speculative_round(ticket, &[18]).unwrap();
        assert_eq!(ledger.emitted_tokens(), &[17, 18]);
        assert_eq!(ledger.pending_token(), Some(18));
    }

    #[test]
    fn ordinary_ledger_cannot_mint_a_paired_proof() {
        assert_eq!(
            TokenLedger::new(0).into_paired_proof().unwrap_err(),
            LedgerError::MissingPairBinding
        );
    }

    #[test]
    fn paired_proof_rejects_a_pending_tail() {
        let mut ledger = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        ledger.emit_pending(7).unwrap();

        assert_eq!(
            ledger.into_paired_proof().unwrap_err(),
            LedgerError::PendingToken { token: 7 }
        );
    }

    #[test]
    fn paired_proof_rejects_an_in_flight_tail() {
        let mut ledger = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        ledger.emit_pending(7).unwrap();
        let _ticket = ledger.begin_cache_only_forward().unwrap();

        assert_eq!(
            ledger.into_paired_proof().unwrap_err(),
            LedgerError::ForwardInFlight { token: 7 }
        );
    }

    #[test]
    fn cache_only_token_enters_paired_proof_only_after_ticket_completion() {
        let mut pending = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        pending.emit_pending(7).unwrap();
        assert_eq!(
            pending.into_paired_proof().unwrap_err(),
            LedgerError::PendingToken { token: 7 }
        );

        let mut forwarded = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        forwarded.emit_pending(7).unwrap();
        let ticket = forwarded.begin_cache_only_forward().unwrap();
        forwarded.complete_cache_only_forward(ticket).unwrap();
        let proof = forwarded.into_paired_proof().unwrap();
        let suffix = proof
            .into_forwarded_suffix_for(&PairLedgerKey::for_test(10))
            .unwrap();

        assert_eq!(suffix.as_ref(), [7]);
    }

    #[test]
    fn paired_proof_excludes_a_terminal_eos() {
        let mut ledger = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        ledger.emit_pending(3).unwrap();
        let ticket = ledger.begin_cache_only_forward().unwrap();
        ledger.complete_cache_only_forward(ticket).unwrap();
        ledger.emit_pending(99).unwrap();
        ledger.exclude_pending_eos(&[99]).unwrap();

        let proof = ledger.into_paired_proof().unwrap();
        let suffix = proof
            .into_forwarded_suffix_for(&PairLedgerKey::for_test(10))
            .unwrap();

        assert_eq!(suffix.as_ref(), [3]);
    }

    #[test]
    fn paired_proof_contains_only_the_forwarded_speculative_prefix() {
        let mut ledger = TokenLedger::new_paired(PairLedgerKey::for_test(10));
        ledger.emit_pending(11).unwrap();
        let ticket = ledger.begin_speculative_round().unwrap();
        ledger
            .complete_speculative_round(ticket, &[12, 13, 99])
            .unwrap();
        ledger.exclude_pending_eos(&[99]).unwrap();

        let proof = ledger.into_paired_proof().unwrap();
        let suffix = proof
            .into_forwarded_suffix_for(&PairLedgerKey::for_test(10))
            .unwrap();

        assert_eq!(suffix.as_ref(), [11, 12, 13]);
    }

    #[test]
    fn paired_proof_mismatch_consumes_identity_without_returning_a_key() {
        let proof = TokenLedger::new_paired(PairLedgerKey::for_test(10))
            .into_paired_proof()
            .unwrap();

        let suffix = proof.into_forwarded_suffix_for(&PairLedgerKey::for_test(11));

        assert!(suffix.is_none());
    }

    #[test]
    fn paired_ledger_rejects_unverified_direct_forward_claims() {
        let mut ledger = TokenLedger::new_paired(PairLedgerKey::for_test(10));

        assert_eq!(
            ledger.record_forwarded(7).unwrap_err(),
            LedgerError::UnverifiedPairedForwardClaim
        );
        assert_eq!(
            ledger.extend_forwarded([7, 8]).unwrap_err(),
            LedgerError::UnverifiedPairedForwardClaim
        );
        assert!(ledger.emitted_tokens().is_empty());
    }

    #[test]
    fn ledger_identity_counter_rejects_wraparound() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX);

        assert!(super::try_next_ledger_id(&counter).is_none());
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), u64::MAX);
    }
}
