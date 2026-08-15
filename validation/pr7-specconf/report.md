# Validation report: pr7-specconf (confidence-gated speculative drafts) — NEGATIVE RESULT

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p7-confidence-drafts
- model: trevon/Qwen3.6-27B-mtp (8-bit, dedicated MTP head)

## Pre-registered acceptance criteria
1. Decode t/s >= current MTP on both suites (code + prose).
2. >= 5% better on at least one suite.
3. Greedy-output equality invariants hold.

## Method
bench_speculative (existing harness; it launches the server per trial), trials baseline + mtp_adaptive,
3 measured repeats each, 256 max tokens, code prompt (Rust LRU cache) and prose prompt (essay).
A/B = HIGGS_MTP_CONFIDENCE_MIN=0 (gate off; identical behavior to main) vs 0.10 (default-on).

## Results (median tok/s, mtp_adaptive)
| Suite | Gate off | Gate 0.10 | Delta | Gate 0.50 (sensitivity) |
| --- | ---: | ---: | ---: | ---: |
| code | 24.34 | 24.44 | +0.4% | 24.53 (+0.8%) |
| prose | 19.47 | 19.52 | +0.25% | - |
Baselines (MTP off) agree across conditions: 15.42-15.56 tok/s.

Criterion 1 PASS (no regression on either suite). Criterion 2 FAIL (max observed +0.8%, needed +5%).
Criterion 3 PASS (all mtp invariants/tests green; gating only shortens draft proposals, never changes
verification semantics).

## Interpretation
On a model with a real MTP head and adaptive depth already enabled, wasted drafts are too rare for
per-token confidence gating to matter: adaptive depth already shrinks depth after rejects, and the
batched verifier makes an occasional wasted draft cheap. The mechanism is sound and costs nothing
measurable, but its benefit at realistic settings is < 1%.

## Verdict: FAIL (below pre-registered effect size) — do not merge
Recorded as a negative result in docs/ds4-analysis-2026-08.md. Might be revisited if a future draft
head has materially lower acceptance rates (e.g. cross-model drafting).
