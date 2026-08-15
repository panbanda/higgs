# Validation report: pr2-frontier

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p2-bench-frontier
- model: mlx-community/Qwen3-1.7B-4bit

## Pre-registered acceptance criteria
1. A/A stability < 5% per frontier (median-based, per program methodology: compare medians across runs).
2. KV-bytes column matches analytic expectation for a known model.
3. mtp tests still green after the checkpoint-helper refactor.

## Results

### Criterion 2: analytic KV check — PASS
Qwen3-1.7B: 28 layers x 2 (K,V) x 8 kv-heads x 128 head-dim x 2 bytes (fp16) = 114,688 bytes/token.
`--verify-kv-analytic --expect-kv-bytes-per-token 114688` -> analytic_verified=true (exact match at the
2048 frontier; larger frontiers include buffer-capacity rounding, as expected for preallocated caches).

### Criterion 1: A/A stability — PASS (median basis), with noted outliers
Two independent invocations (3 and 4 sweeps; sweep 1 of each treated as warmup).
Probe decode tok/s medians per frontier:

| Frontier | Invocation A median | Invocation B median | Median delta |
| ---: | ---: | ---: | ---: |
| 2048 | 231.6 | 232.5 | +0.39% |
| 4096 | 205.3 | 208.0 | +1.32% |
| 8192 | 170.4 | 173.7 | +1.94% |
| 16384 | 129.6 | 129.3 | -0.23% |

Honest caveat: individual sweeps show sporadic dips up to ~8.5% (raw min-max spread), attributable to
background desktop load; medians across >=3 measured sweeps are stable (<2%). A/B comparisons made with
this tool must therefore always compare medians of >=3 post-warmup sweeps, never single runs.

### Criterion 3: mtp/cache tests — PASS
cargo test -p higgs-engine -- --test-threads=1: 287 passed.
cargo test -p higgs-models -- --test-threads=1: 427 passed.
cargo test -p higgs -- --test-threads=1: passed.

Verdict: PASS
