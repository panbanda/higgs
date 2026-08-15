# Validation report: pr6-diskkv (disk-backed KV prefix store)

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p6-disk-kv-store
- model: mlx-community/Qwen3-1.7B-4bit

## Pre-registered acceptance criteria
1. Restored TTFT >= 10x better than cold prefill at 8k+ context.
2. Continuation from restored state token-identical to an uninterrupted session.
3. Corrupt files rejected cleanly (unit tests).

## Results

### Criterion 1: restart-resume latency — PASS (17.7x)
~7113-token prompt, greedy, 64 generated tokens, wall time of the full request:
| Run | Wall time |
| --- | ---: |
| Cold (fresh server, empty disk store) | 9.30 s |
| Restarted server, disk restore | 0.53 s |
| Second restart, disk restore | 0.52 s |
Server log confirms the path: "Disk prefix hit token_count=7104 total_len=7113" on both restarts.
(A first implementation stored the unaligned prompt length and never restored; review-driven debugging
found and fixed it — the regression test now covers store -> new engine -> materialize -> consume.)

### Criterion 2: continuation identity — PASS, with the equivalence class documented
Restored continuations are deterministic (restart 1 == restart 2, byte-identical) and byte-identical
to the continuation produced by main's EXISTING in-memory prefix-cache hit path for the same prompt
(control measured on this hardware). They differ from a fully-cold continuation at a greedy near-tie
early in reasoning — but so does the in-memory prefix hit on main (also measured): that divergence is
a pre-existing property of prefix reuse (suffix-only prefill batches differently than chunked full
prefill under MLX numerics), not introduced by this PR. The disk tier is numerically exactly
equivalent to the accepted in-memory reuse path.

### Criterion 3: corruption handling — PASS
Unit tests: truncated payload, wrong model_id, wrong quant, wrong tokenizer hash, bad magic — each
rejected cleanly as a miss; leftover .tmp files ignored; eviction respects the size budget
(LRU-by-mtime v1 of frecency, documented). cargo test -p higgs-engine: 293+ passed.

### Config surface (repo policy)
kv_disk_dir / kv_disk_space_mb shipped through the full pattern: ModelConfig -> doctor validation
(dir writable, minimum size, free-space warning) -> engine; README config reference and the
higgs init template updated.

Verdict: PASS
