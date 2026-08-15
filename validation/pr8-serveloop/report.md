# Validation report: pr8-serveloop (mixed-prefill yield quantum)

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p8-serving-loop
- model: mlx-community/Qwen3-1.7B-4bit, batch mode (batch = true)

## Pre-registered acceptance criteria
1. Client A's inter-token p95 during client B's long prefill reduced >= 2x.
2. Exactness: outputs identical with the knob on vs off (greedy).
3. Single-client throughput unregressed.

## Method
Two-client harness: A streams a 700-token greedy generation; after A's 30th token, B submits a
~6000-token prompt (8-token completion). A's inter-token gaps classified before/during/after B's
request window. Sync = prefill_yield_tokens unset (current behavior); Yield = prefill_yield_tokens 512.

## Results

### Criterion 1 — PASS (13.4x)
| Phase | Sync p50/p95/max (ms) | Yield-512 p50/p95/max (ms) |
| --- | --- | --- |
| before B | 3.6 / 3.8 / 3.8 | 3.7 / 3.8 / 3.8 |
| during B's prefill | 3.9 / 2640.8 / 2640.8 | 138.9 / 196.9 / 203.2 |
| after B | 3.9 / 4.1 / 11.2 | 3.8 / 4.0 / 6.4 |
Sync mode freezes A for the full prefill (one 2.64 s gap). With the 512-token quantum the worst gap
is 203 ms. p95 reduction: 2640.8 -> 196.9 ms = 13.4x (criterion >= 2x).
B's own request completed slightly faster with yielding (2.46 s vs 2.71 s wall).

### Criterion 2 — PASS
Greedy outputs byte-identical between modes for a short prompt (200 tokens) and a ~7k-token prompt.

### Criterion 3 — PASS
Steady-state inter-token gaps identical (p50 3.6-3.9 ms both modes, before and after phases).

### Config surface (repo policy)
prefill_yield_tokens shipped through ModelConfig -> doctor (0/None disables; rejects 1-127; warns
128-511) -> batch engine; README + higgs init template updated.

Verdict: PASS
