//! Scheduling decisions of the engine worker, kept free of MLX state so they
//! can be tested without a model.
//!
//! Every request carries its own capabilities (constraint, logprobs, whether
//! it may decode speculatively); the model carries the engine-wide ones
//! (batched decode, concurrency limit). Each turn of the worker loop asks
//! [`plan_round`] how to advance the active requests.

/// What the worker needs to know about one active request to schedule it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct RequestShape {
    /// The request decodes speculatively right now.
    pub speculating: bool,
    /// The request could decode speculatively, but is on the normal path.
    pub could_speculate: bool,
    /// A grammar or schema constrains its tokens, which must be read back
    /// every step.
    pub constrained: bool,
    /// The request wants per-token log-probabilities.
    pub logprobs: bool,
}

impl RequestShape {
    /// Whether the request can share a batched forward pass with others.
    pub const fn batchable(&self) -> bool {
        !self.constrained && !self.logprobs
    }
}

/// How the worker advances the active requests for one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundPlan {
    /// The lone request runs one speculative cycle (several tokens).
    Speculative,
    /// The lone request returns to speculative decode, then runs a cycle.
    ResumeSpeculation,
    /// Every request takes one token from a single batched forward pass.
    Batched,
    /// The lone request runs one step ahead of the token it reports, keeping
    /// the GPU busy while the CPU builds the next graph.
    Lookahead,
    /// Each request takes one token from its own forward pass.
    Sequential,
}

/// Engine-wide facts a round depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundFacts {
    /// The architecture and KV cache can decode several requests in one pass.
    pub batched_decode: bool,
}

/// Speculative decode drafts and rolls back one sequence, so it only runs when
/// nothing else is active.
pub fn speculation_must_yield(shapes: &[RequestShape]) -> bool {
    shapes.len() > 1 && shapes.iter().any(|shape| shape.speculating)
}

/// How to advance `shapes` this turn. Requests that are speculating must have
/// been handed over first when [`speculation_must_yield`] says so.
pub fn plan_round(facts: RoundFacts, shapes: &[RequestShape]) -> RoundPlan {
    if let [only] = shapes {
        if only.speculating {
            return RoundPlan::Speculative;
        }
        if only.could_speculate {
            return RoundPlan::ResumeSpeculation;
        }
    }
    if facts.batched_decode && shapes.iter().all(RequestShape::batchable) {
        return RoundPlan::Batched;
    }
    if shapes.len() == 1 && shapes.iter().all(RequestShape::batchable) {
        return RoundPlan::Lookahead;
    }
    RoundPlan::Sequential
}

/// How many requests may be decoding at once: unlimited, or one when the
/// model cannot batch its decode.
pub const fn concurrency_limit(batched_decode: bool, configured: Option<usize>) -> usize {
    match (batched_decode, configured) {
        (false, _) => 1,
        (true, Some(limit)) => {
            if limit == 0 {
                1
            } else {
                limit
            }
        }
        (true, None) => usize::MAX,
    }
}

/// How many more requests may start prefilling, given those already decoding
/// or prefilling.
pub const fn free_slots(limit: usize, busy: usize) -> usize {
    limit.saturating_sub(busy)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BATCHING: RoundFacts = RoundFacts {
        batched_decode: true,
    };
    const SEQUENTIAL: RoundFacts = RoundFacts {
        batched_decode: false,
    };

    const PLAIN: RequestShape = RequestShape {
        speculating: false,
        could_speculate: false,
        constrained: false,
        logprobs: false,
    };
    const SPECULATING: RequestShape = RequestShape {
        speculating: true,
        could_speculate: true,
        ..PLAIN
    };
    const CONSTRAINED: RequestShape = RequestShape {
        constrained: true,
        ..PLAIN
    };
    const LOGPROBS: RequestShape = RequestShape {
        logprobs: true,
        ..PLAIN
    };

    #[test]
    fn plain_requests_share_a_batched_pass() {
        assert_eq!(
            plan_round(BATCHING, &[PLAIN, PLAIN, PLAIN]),
            RoundPlan::Batched
        );
        assert_eq!(plan_round(BATCHING, &[PLAIN]), RoundPlan::Batched);
    }

    #[test]
    fn one_constrained_or_logprob_request_sends_everyone_sequential() {
        assert_eq!(
            plan_round(BATCHING, &[PLAIN, CONSTRAINED]),
            RoundPlan::Sequential
        );
        assert_eq!(
            plan_round(BATCHING, &[LOGPROBS, PLAIN]),
            RoundPlan::Sequential
        );
    }

    #[test]
    fn a_lone_constrained_or_logprob_request_is_sequential() {
        assert_eq!(plan_round(BATCHING, &[CONSTRAINED]), RoundPlan::Sequential);
        assert_eq!(plan_round(BATCHING, &[LOGPROBS]), RoundPlan::Sequential);
    }

    #[test]
    fn a_model_that_cannot_batch_runs_its_lone_request_ahead() {
        assert_eq!(plan_round(SEQUENTIAL, &[PLAIN]), RoundPlan::Lookahead);
        assert_eq!(
            plan_round(SEQUENTIAL, &[CONSTRAINED]),
            RoundPlan::Sequential
        );
    }

    #[test]
    fn a_model_that_cannot_batch_steps_several_requests_one_by_one() {
        assert_eq!(
            plan_round(SEQUENTIAL, &[PLAIN, PLAIN]),
            RoundPlan::Sequential
        );
    }

    #[test]
    fn a_lone_speculating_request_runs_cycles() {
        assert_eq!(plan_round(BATCHING, &[SPECULATING]), RoundPlan::Speculative);
        assert_eq!(
            plan_round(SEQUENTIAL, &[SPECULATING]),
            RoundPlan::Speculative
        );
    }

    #[test]
    fn a_lone_request_that_could_speculate_resumes() {
        let handed_over = RequestShape {
            could_speculate: true,
            ..PLAIN
        };
        assert_eq!(
            plan_round(BATCHING, &[handed_over]),
            RoundPlan::ResumeSpeculation
        );
    }

    #[test]
    fn speculation_yields_when_anything_else_is_active() {
        assert!(speculation_must_yield(&[SPECULATING, PLAIN]));
        assert!(speculation_must_yield(&[PLAIN, SPECULATING]));
        assert!(!speculation_must_yield(&[SPECULATING]));
        assert!(!speculation_must_yield(&[PLAIN, PLAIN]));
    }

    #[test]
    fn a_request_that_could_speculate_batches_when_not_alone() {
        let handed_over = RequestShape {
            could_speculate: true,
            ..PLAIN
        };
        assert_eq!(
            plan_round(BATCHING, &[handed_over, PLAIN]),
            RoundPlan::Batched
        );
    }

    #[test]
    fn models_that_cannot_batch_serve_one_request_at_a_time() {
        assert_eq!(concurrency_limit(false, None), 1);
        assert_eq!(concurrency_limit(false, Some(8)), 1);
    }

    #[test]
    fn batching_models_are_unlimited_unless_configured() {
        assert_eq!(concurrency_limit(true, None), usize::MAX);
        assert_eq!(concurrency_limit(true, Some(4)), 4);
        assert_eq!(concurrency_limit(true, Some(0)), 1);
    }

    #[test]
    fn free_slots_never_underflow() {
        assert_eq!(free_slots(1, 0), 1);
        assert_eq!(free_slots(1, 1), 0);
        assert_eq!(free_slots(1, 3), 0);
        assert_eq!(free_slots(usize::MAX, 5), usize::MAX - 5);
    }
}
