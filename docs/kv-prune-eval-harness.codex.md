# Codex task: KV-prune accuracy-sweep eval harness

Self-contained spec. You do **not** need the conversation that produced it.

## Background (why this exists)

`higgs` is adding TIM/TIMRUN-style **KV-cache pruning** to run long-horizon
reasoning on a stock (un-finetuned) Qwen3.6 MoE. The mechanism is already built
and proven in `crates/higgs-models/src/cache.rs`:

```rust
// On SteppingKeyValueCache (the dense decode cache):
pub fn prune_span(&mut self, a: i32, b: i32, rope: RopeShift) -> Result<(), Exception>
// RopeShift { base: f32, dims: i32, scale: f32, traditional: bool }
```

`prune_span(a, b, rope)` drops the half-open token span `[a, b)` from the cache,
compacts the survivors, and re-rotates the surviving suffix by `R(-(b-a))` so
positions stay dense. Proven bit-equivalent (f32 tol) to never inserting those
tokens — see test `prune_span_equiv_never_inserted`. Dense path only (errors on
TurboQuant).

The open question this harness answers: **how aggressively can we prune a stock
Qwen3.6 MoE's KV before reasoning accuracy degrades?** The paper's
quality-preserving regime is ~50–60% pruned. If our knee is near there, the
training-free thesis holds.

## What you own (the separable part)

A **grading + metrics + problem-set** module. You do NOT touch the model decode
loop or `prune_span` — those are wired on the higgs side against a clean
interface you define. Build a new crate-local module (suggest
`crates/higgs-bench/src/prune_eval.rs`) exposing:

```rust
/// One reasoning problem with a checkable final answer.
pub struct Problem { pub id: String, pub prompt: String, pub answer: String }

/// Curated set: ~50 items, mix of GSM8K-style arithmetic word problems and
/// MATH-style short-answer. Hard-code them (no network at run time). Answers
/// are exact strings (canonical numeric form, e.g. "42", "-3/4").
pub fn problem_set() -> Vec<Problem>;

/// Extract the model's final answer from free-form output and exact-match it
/// against `expected`. Handle: trailing "#### N" (GSM8K), "\boxed{...}" (MATH),
/// "The answer is X.", and bare-last-number fallback. Normalize whitespace,
/// commas in numbers, and trivial fraction forms.
pub fn grade(model_output: &str, expected: &str) -> bool;

/// One row of the sweep result.
pub struct SweepRow {
    pub prune_pct: u32,      // target prune rate this row was run at
    pub accuracy: f32,       // fraction graded correct
    pub mean_tok_per_s: f32,
    pub peak_resident_kv: u32, // max tokens resident across the run
    pub n: u32,
}

/// Render rows as a fixed-width table: prune% | acc | tok/s | peakKV | n,
/// plus a one-line summary naming the knee (highest prune_pct whose accuracy is
/// within `tol` of the prune=0 row).
pub fn render_table(rows: &[SweepRow], tol: f32) -> String;
```

Write a `grade` unit test proving the general logic (each extraction format +
one false case), not per-problem assertions. Keep the problem set small and
high quality.

## The interface boundary (higgs side wires this)

The higgs-side runner (built separately) will, per problem and per target
prune rate, drive the existing decode loop in
`crates/higgs-engine/src/simple.rs` (`generate_inner`) with a **prune policy**:

- Keep the first `S = 4` tokens always (attention sinks).
- Keep the most recent `W` tokens always.
- When resident length exceeds the budget implied by the target prune rate,
  call `prune_span(S, S + k, rope)` to evict the oldest `k` non-sink tokens,
  looping `for c in caches.iter_mut().flatten() { c.prune_span(..) }` across all
  layers. `rope = RopeShift { base: rope_theta, dims: head_dim, scale: 1.0,
  traditional: false }` (read from model config).

It collects `(model_output, tok_per_s, peak_resident_kv)` per problem and calls
your `grade` + `render_table`. **Token-age pruning only here — no Thread-2
schema yet** (that's a later phase); this isolates the mechanism's accuracy
curve.

## Deliverable

`cargo test -p higgs-bench prune_eval` passes (your grader tests). Module
compiles clean (`cargo clippy -p higgs-bench`, nursery lints) and `cargo fmt`.
Do not run the 35B sweep yourself — the higgs side does that once your grader +
table land.

## Sweep matrix (for the final run, FYI)

Target prune rates `{0, 25, 40, 55, 70}` %, N≈50 problems, model
`qwen3_moe` / `qwen3_5_moe` (Qwen3.6 35B-A3B). Headline = highest prune% whose
accuracy stays within noise of the 0% row, plus tok/s and peak-KV at that point.

## Measured results (preliminary — 2026-06-24)

These runs were done on a real model (9B / 35B Qwen3.x) in the session that
built this harness. They are **small-N and exploratory**, not a final sweep.
Recorded here so the findings aren't lost in chat history.

### 1. Age-based KV pruning — NEGATIVE

Sweep (n=6, age-based policy, `sink=4 min_window=64`):

```
prune% | acc  | peakKV | tok/s | n
  0%   | 100% |   310  | 21.3  | 6
 50%   |   0% |   155  |  8.2  | 6
knee: accuracy holds (within 5%) up to ~0% prune
```

At 50% prune, accuracy collapsed to 0% on running-state reasoning, and
throughput *dropped* (21→8 tok/s). The paper's ~50–60% quality-preserving
knee did **not** reproduce for the naive token-age policy: it evicts exactly
the state these tasks carry. The `prune_span` mechanism is bit-exact (proven);
the *policy* is the problem.

### 2. Self-maintenance (checkpoint) vs prune vs full — PROMISING

Running-counter task, K signed ops, gold = final value:

```
        | K=18      | K=36           | peak KV
full    | OK (59)   | WRONG (65/68)  | 515 / 833
prune50%| WRONG(201)| WRONG (57)     | 435 / 588
checkpt | OK (59)   | OK (68)        | 239 / 241   <-- self-state summary
```

Checkpoint/self-maintenance stayed correct where BOTH full-context and 50%
pruning failed, at ~¼–½ the peak KV. Best signal of the three. Still small-N.

### 3. Working-memory "self-notes vs one-shot" — INCONCLUSIVE

Drift curve (N=5 tasks/K, full context, no pruning):

```
  K | one-shot | self-notes | max_resident_KV
 24 |   4/5    |    5/5     |       1490
 42 |   2/5    |    3/5     |       2275
 60 |   5/5    |    ...     |       ...
```

Rate run (N=8): one-shot 7/8, self-notes 8/8 (notes won the single case
one-shot drifted), NOTES emitted 8/8, comparable tokens.

Verdict from the session: **"N=5 is too noisy — per-task difficulty swamps K …
self-notes neither reliably helps nor hurts at this difficulty. Promising, not
proven."** One-shot accuracy was non-monotonic (80→40→100→60%), so the small
deltas are within noise. At the tested difficulty the model often solves
one-shot, so notes have nothing to fix.

### What a real proof still needs

- Harder/longer chains (K≈70–100) where one-shot fails 30–50%, so there is
  drift to claw back — otherwise no discrimination.
- Much larger N (≥200, or multi-seed) for error bars; N=5–8 is a coin flip.
- Turn the print-and-eyeball harness into asserted thresholds (baseline sanity
  + declared knee), so a regression can't pass silently.
- For pruning specifically: the negative result suggests the lever is
  *structural* (protect-facts / self-maintenance), not token-age. Focus there.
