# Validation report: pr4-toolgreedy (structural greedy sampling) — NEGATIVE RESULT

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p4-structural-greedy
- model: mlx-community/Qwen3-1.7B-4bit

## Pre-registered acceptance criteria
1. Malformed tool-call rate strictly decreases at temp 0.8/1.0.
2. Throughput regression <= 2%.
3. Payload diversity preserved (no degenerate repetition).

## Method
100-150 tool-call prompts per condition against a live server (two tools, varied cities/products),
scoring each response: well-formed tool_calls with valid JSON + required args, vs malformed.
Round 1 (max_tokens 256) was invalidated by a measurement confound: Qwen3 thinking mode consumed the
budget and truncated calls mid-JSON (nearly all "malformed" cases had exactly 256 completion tokens).
Round 2 removed it (max_tokens 1024, temp 1.0, N=150/side).

## Results (round 2)
| Side | Well-formed | Malformed | Rate | Throughput tok/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline (main) | 148/150 | 2 | 1.33% | 301.9 |
| Candidate (structural greedy) | 147/150 | 3 | 2.00% | 302.1 |

Criterion 1 FAIL: no decrease (difference is within noise; nominally worse).
Criterion 2 PASS: throughput unchanged (+0.06%).
Criterion 3 PASS: argument payloads vary across prompts on both sides.

## Failure taxonomy (the decisive finding)
Every residual malformed case on BOTH sides is the same mode: syntactically VALID JSON emitted
without the <tool_call> wrapper, so the parser does not lift it into tool_calls. Zero cases of
corrupted JSON syntax (the failure mode structural greedy targets) were observed in 300 samples.
Wrapper omission happens in the "outside" region where request sampling is intentionally preserved —
by design out of this mechanism's reach. The dominant failure in round 1 (45-51% "no tool call") is
tool-avoidance, also out of reach.

## Verdict: FAIL — do not merge
The mechanism is implemented and unit-tested (region classifier handles nested JSON/escapes), but on
Qwen3-class instruct models the structural-syntax malformation it prevents effectively does not occur
(~0% base rate at temp 1.0). Adding a per-step classifier to the hot decode loop is not justified by
zero measured benefit. Negative result to be recorded in docs/ds4-analysis-2026-08.md. A plausible
rework (out of scope here): address wrapper omission instead, e.g. constrained emission of the wrapper
once arguments-JSON begins.
