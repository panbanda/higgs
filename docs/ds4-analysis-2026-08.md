# ds4 (DwarfStar) analysis: what transfers to Higgs

**Date:** 2026-08-13
**Subject:** Deep analysis of [antirez/ds4](https://github.com/antirez/ds4) and a
prioritized proposal for techniques Higgs should adopt to get faster and/or more
accurate.

## 1. What ds4 is, and why the comparison is useful

ds4 ("DwarfStar") is a self-contained C inference engine specialized for a small
set of MoE models (DeepSeek V4 Flash/PRO, GLM 5.2) on Metal, CUDA, and ROCm. It
is deliberately narrow: model loading, prompt rendering, tool calls, KV state,
the HTTP server, and a coding agent are built and tested together against a
handful of GGUF layouts.

Higgs is a different animal: a Rust engine over Apple MLX (safetensors, Metal
only, `unsafe_code = "deny"`, JIT Metal kernel strings where MLX has gaps). So
ds4's CPU SIMD kernels, GGUF/IQ2 formats, and CUDA/ROCm backends do not port.
What *does* port are the systems ideas — and ds4 has several that are directly
relevant because both projects serve the same workload shape: long-context,
multi-turn, tool-calling agent sessions on Apple Silicon, running MoE models
(Higgs: Qwen3-Next, Qwen3-MoE, DeepSeek-V2, Gemma-4 MoE).

The rest of this document is a technique-by-technique applicability review,
followed by a prioritized proposal.

## 2. Technique inventory and applicability

| # | ds4 technique | Higgs today | Verdict |
|---|---|---|---|
| 1 | MLA compressed-latent KV cache (+ FP8 KV quant) | Caches full decompressed per-head K/V (`deepseek_v2.rs:444-486`) | **Adopt — biggest speed/memory win** |
| 2 | Asymmetric routed-expert quantization + imatrix calibration | Uniform pre-quantized MLX checkpoints | **Adopt — biggest accuracy win** |
| 3 | Greedy sampling for tool-call *syntax*, request sampling for payloads | Sampling params applied uniformly | **Adopt — tool-call reliability** |
| 4 | On-disk KV store (byte-prefix keyed, frecency eviction, survives restarts) | In-memory prefix caches only | **Adopt — agent TTFT** |
| 5 | Exact tool-call replay (`tool id → exact sampled bytes`) | Re-renders tool calls from client JSON | **Adopt — protects KV prefix hits** |
| 6 | Confidence-gated speculative drafts (`--dspark-confidence`) | Adaptive draft *depth* only (`mtp.rs:112`) | Adopt — incremental decode gain |
| 7 | Official-continuation logprob scoring + deterministic regression gates | No accuracy-regression harness | **Adopt — protects the hand-written kernels** |
| 8 | Frontier benchmarking (incremental prefill + KV snapshot/restore) | Whole-run averages in `higgs-bench` | Adopt — cheap |
| 9 | Micro-batched serving: mixed prefill/decode quantum, exact ordered fallback | One-prefill-per-iteration; batching limited to 4 dense archs | Partial adopt |
| 10 | SSD streaming of routed experts + profiled hot-expert preload | Models must be resident | Future — capacity mode |
| 11 | `--power N` GPU duty-cycle throttling | None | Optional — cheap, good on MacBooks |
| 12 | Pipeline/tensor parallelism across machines (TCP/RDMA) | Single machine | Out of scope near-term |
| 13 | CPU SIMD quant kernels, GGUF/IQ2_XXS, CUDA/ROCm, DSML specifics | — | Does not transfer |

## 3. Proposal

### Tier 1 — do these

#### P1. MLA latent KV cache with weight absorption (speed + memory)

ds4 caches DeepSeek's *compressed* KV latent and quantizes those rows to FP8
(`dsv4_fp8_kv_quantize_row_inplace_cpu`, ds4.c). Higgs's `DeepSeekV2Attention`
instead decompresses through `kv_b_proj` and caches full per-head K
(`nope+rope`) and V every step (`crates/higgs-models/src/deepseek_v2.rs:444-486`).

For DeepSeek-V2-Lite (16 heads, nope 128, rope 64, v 128) that is
`16 × (192 + 128) = 5120` floats/token of KV versus the latent's
`kv_lora_rank + rope = 512 + 64 = 576` — a **8.9× KV memory reduction**, and
proportionally less memory traffic in decode attention, which is what bounds
long-context decode throughput on Apple Silicon.

The standard absorption trick (from the DeepSeek-V2 paper, used by every
serious MLA implementation including ds4):

- Cache per token: `kv_latent` (post-`kv_a_layernorm`, `kv_lora_rank` dims) and
  the RoPE'd `k_pe` (`qk_rope_head_dim` dims, shared across heads).
- At decode, fold the K-nope half of `kv_b_proj` into the query
  (`q_nope' = q_nope @ W_kb_knope`), so scores are computed directly against the
  latent — attention becomes MQA over a 576-dim shared key.
- Fold the V half of `kv_b_proj` into the output projection
  (`o' = (attn @ latent) @ W_kb_v @ W_o`).

Implementation sketch: a new `MlaKeyValueCache` variant in
`crates/higgs-models/src/cache.rs` storing `[B, 1, seq, kv_lora_rank + rope]`,
a decode path in `deepseek_v2.rs` using the absorbed matmuls (prefill can keep
the current decompressed path, which is compute-friendly for long sequences —
this prefill/decode split is exactly what ds4 does). On top, TurboQuant
(`crates/higgs-models/src/turboquant.rs`) can quantize the latent rows the same
way ds4 FP8-quantizes its compressed KV — the latent is just another
`[H=1, seq, D]` stream, and 576 dims per token at 3–4 bits makes 100k-token
contexts cheap.

Payoff: large decode speedup at long context + ~9× KV memory for MLA models.
Risk: moderate — needs careful nope/rope bookkeeping; validated by P7's
regression gates.

#### P2. Asymmetric MoE quantization recipe + imatrix calibration (accuracy)

ds4's headline accuracy result: 2-bit models that "behave well, work under
coding agents, call tools in a reliable way" — achieved not by a better
low-bit codec but by **spending bits asymmetrically**. Only the routed experts
(the overwhelming majority of parameters) are quantized to ~2 bits; attention,
shared experts, router, embeddings, and output head stay at Q8/F16. Quant
error is steered by an **imatrix** (importance matrix) collected on the *real
inference graph* over a calibration corpus of rendered chat prompts, tool-call
transcripts, and reasoning traces (`gguf-tools/imatrix/README.md`) — for
gate/up the squared FFN input activations, for down the squared route-weighted
SwiGLU rows.

Higgs consumes pre-quantized MLX checkpoints with a single global
`{group_size, bits}` (`transformer.rs:58-62`). MoE models pay for that
uniformity twice: attention/router precision is wasted budget at 8 bits and
badly missed at 4.

Proposal:

1. **Loader**: support per-tensor quantization overrides (MLX checkpoints
   produced by `mlx_lm.convert --quant-predicate` already encode per-layer
   `{bits, group_size}` in `config.json`; make sure `QLinear`/`SwitchMlpWeights`
   construction honors them rather than assuming the global config).
2. **Calibration collector**: a `higgs quantize --calibrate` mode that runs the
   dataset through the normal engine graph recording squared activations at the
   routed-expert boundaries (the hook points already exist in
   `SwitchMlpWeights::forward_*`, `qwen3_next.rs:2048-2330`), emitting an
   imatrix file consumable by the quantizer.
3. **Recipe + published checkpoints**: routed experts 2–3 bit (imatrix-guided),
   shared experts / attention / router / embeddings / lm_head at 6–8 bit, for
   Qwen3-Next-80B and Qwen3-MoE. This is what makes 80B-class MoE genuinely
   good on 64–128 GB Macs instead of merely runnable.

Payoff: strictly better accuracy at a fixed memory budget (or bigger models at
the same budget). Risk: low — offline tooling, no hot-path changes.

#### P3. Structural greedy sampling for tool calls (accuracy/reliability)

ds4 forces `temperature=0` while the model is emitting tool-call *structure*
(tags, parameter headers, JSON punctuation, closing markers) and restores the
request's sampling settings inside argument payloads (README, "Tool call
handling"). Deterministic syntax keeps calls parseable; sampled payloads avoid
the repetitive-text failure mode of greedy long file bodies.

Higgs already streams tokens through a tool-call parser state machine
(`crates/higgs-engine/src/tool_parser.rs`); the sampler
(`higgs-models/src/lib.rs:947`) just never hears about it. Wire the parser
state into the per-step sampling decision: when the parser is inside structural
syntax, sample greedily; inside string/payload regions, use request params.
This measurably reduces malformed tool calls at high temperatures — an
accuracy win agent users feel immediately.

Payoff: fewer failed/retried tool calls. Risk: low; the state machine already
exists.

#### P4. Disk-backed KV store (agent TTFT)

ds4 persists KV checkpoints to disk (`ds4_kvstore.{c,h}`): files keyed by a
SHA of the **rendered byte prefix** (not token IDs — so a stateless client
resending a longer version of the same prompt hits the cache), a frecency
eviction score with a 6-hour hit half-life, quantized payloads, and
`min_tokens` / `continued_interval` / boundary-alignment knobs. Agent sessions
resume after a server restart with zero prefill.

Higgs has two in-memory prefix caches (`prompt_cache.rs`,
`paged_prefix_cache.rs`) that die with the process. But the expensive
ingredient already exists: `PagedPrefixCache` stores TurboQuant-compressed
blocks (`CachedLayerData`), so a disk tier is mostly (de)serialization plus
ds4's proven index/eviction design:

- Key: hash of the rendered prompt byte prefix, exactly like ds4 (Higgs
  already strips the generation-prompt suffix for its in-memory key,
  `simple.rs:701-711` — same normalization applies).
- Payload: TurboQuant block codes + norms/gammas (3-bit keys make a 32k-token
  DeepSeek/Qwen KV state small enough that SSD load is far cheaper than
  re-prefill), plus GDN state snapshots for hybrid models.
- Eviction: ds4's frecency score (hits decayed by half-life, scaled by size)
  is simple and battle-tested; adopt it as-is.
- Config: `kv_disk_dir` + `kv_disk_space_mb` in higgs config, validated in
  `crates/higgs/src/doctor.rs` per project policy.

Payoff: near-zero TTFT for returning agent sessions; prefix reuse across
restarts and across evicted sessions. Risk: low-moderate; format versioning
must include model ID + quant config (ds4 stores both in its header).

### Tier 2 — strong candidates after Tier 1

#### P5. Exact tool-call replay

The subtle problem ds4 solves: stateless API clients send back *normalized
JSON* tool calls, not the exact bytes the model sampled. If the server
re-renders them even slightly differently, the byte prefix no longer matches
the KV checkpoint and the next turn silently re-prefills the whole
conversation. ds4 keeps a bounded map `tool id → exact sampled tool-call
block` (radix-tree backed, persisted inside KV cache files) and renders
replayed tool calls from those exact bytes; canonicalization is only the
fallback.

Higgs generates tool-call IDs and re-renders history through minijinja chat
templates, so it has this failure mode today — it just manifests as
mysteriously cold prefix caches in tool-heavy sessions. Fix: record the exact
generated text span per tool-call ID at parse time (`tool_parser.rs` sees it),
consult the map during prompt rendering, and count "prefix broken by re-render"
in cache stats so the win is observable. This multiplies the value of P4.

#### P6. Confidence-gated speculative drafts

Higgs's MTP path adapts draft *depth* between blocks
(`AdaptiveDraftDepth`, `mtp.rs:112`). ds4 additionally prunes each proposed
suffix by draft-model confidence per token (default threshold 0.6 on Metal),
so verification cost is only paid where acceptance is likely. Add per-token
confidence gating to `mtp_cycle` — the draft logits are already materialized;
this is a cheap filter that raises effective acceptance rate, particularly on
non-code text. Also worth copying: ds4's honest framing that verifier-kept
state can diverge in FP reduction order from one-token decode — Higgs's
`--quality`-equivalent should document the same.

#### P7. Quality regression harness (protects everything else)

ds4 treats accuracy as a regression-tested property:

- `--logprob-vectors`: compare local token bytes + top-logprob slices against
  reference continuation vectors → catches tokenizer, template, attention,
  kernel, and quant regressions in one cheap check.
- A deterministic greedy gate: 4 fixed questions must produce *exactly* N
  generated tokens and fixed answers (`ds4-eval --questions 4 --temp 0`) —
  any inference drift shows up as a token-count diff.
- `score_official`: how much probability a quantized model assigns to
  reference continuations, compared before/after any quantization change.

Higgs has strict lint/coverage CI but no accuracy regression net, while
carrying five+ hand-written Metal kernels (TurboQuant scores/values/pack,
QGEMV, GatedDelta, Bonsai bits=1) whose failure mode is *silent quality
drift*, not crashes. Proposal: a `higgs-eval` fixture directory of reference
continuations (generated once from unquantized checkpoints of the small
supported models), a logprob-comparison test behind a model-download gate, and
a deterministic greedy token-count gate runnable in the existing
`cargo test -p higgs -- --test-threads=1` flow for CI machines that have a
small model available. Low effort; makes P1/P2/P6 safe to land.

#### P8. Frontier benchmarking in `higgs-bench`

ds4-bench reports *instantaneous* prefill and generation throughput at context
frontiers (2048, 4096, …): incremental prefill per interval, KV snapshot →
128-token greedy probe → restore → continue. One run yields the full
throughput-vs-context curve as CSV. `higgs-bench` reports whole-run averages,
which hide exactly the long-context decay that P1 targets. Higgs already has
the required primitive (`trim_by` rollback / cache checkpointing from the
speculative path). Cheap to add, and it is the instrument that proves P1's win.

#### P9. Serving-loop refinements

Two ds4-server ideas for `batch_engine.rs`:

- **Mixed-prefill quantum**: while any decode is active, prefill yields every N
  tokens (default 128) rather than completing a whole chunk — bounds decoder
  stall (`--mixed-prefill-quantum`). Higgs does at most one prefill per
  iteration but the chunk (512+) can still stall decoders; make the yield
  quantum explicit and configurable.
- **Exact ordered fallback as a contract**: ds4 documents that when a native
  batched kernel is unavailable, sessions run in fixed order producing results
  identical to separate evaluation — concurrency without batching. Higgs
  already pipelines per-request decode for non-batchable archs; adopting the
  "exactness contract" framing (and testing it) lets batching expand to
  MoE/MLA/hybrid archs safely, fallback-first, native kernels later.

### Tier 3 — future / optional

- **P10. SSD-streamed routed experts.** ds4's capacity mode: non-routed weights
  resident, routed experts in an in-memory cache backed by the file, plus a
  *profiled hot-expert preload list* (`ds4_streaming_hotlist.inc` — static
  `{layer, expert}` pairs sorted by measured hit rate). This is what lets
  128 GB machines run 300 GB-class models. For Higgs this is a large project
  (MLX lazy loading + explicit expert cache + routing-stats profiler), but it
  is the only item on this list that changes *which models Macs can run at
  all*. The routing-stats profiler is a sensible first step — it is also the
  data source for smarter expert-quantization budgets in P2.
- **P11. `--power N`.** Duty-cycle throttling by inserting sleeps between
  layers (prefill) / tokens (decode). Trivial to implement in the decode loop,
  no output change, and genuinely valuable on the fanless/laptop hardware
  Higgs targets.
- **P12. Multi-machine parallelism.** ds4's pipeline parallelism (layer slices
  over TCP) and two-Mac tensor parallelism over Thunderbolt RDMA are impressive
  but represent a different product scope; revisit only if capacity demand
  (P10) proves out first.

### Explicitly not transferable

- CPU SIMD quant/dot kernels and the Q8_0/IQ2_XXS/Q2_K codecs — Higgs is
  MLX/Metal with `unsafe_code = "deny"`; equivalent work goes through MLX ops
  or JIT Metal kernels (the existing `turboquant.rs` / `qwen3_next.rs` /
  `metal_kernel.rs` pattern). The *bit-budget recipe* transfers (P2); the
  codec code does not.
- GGUF loading — Higgs is safetensors/MLX-checkpoint native; adopting GGUF
  would fork the model-loading story for little gain.
- CUDA/ROCm backends, DSML-format specifics, directional steering.

## 4. Suggested sequencing

1. **P7 first** (regression harness) — it is the safety net for everything else.
2. **P1** (MLA latent cache) with P8 (frontier bench) to measure it.
3. **P2** (asymmetric quant + imatrix) — offline tooling, parallelizable with P1.
4. **P3 + P5** (tool-call greedy syntax + exact replay) — small, agent-facing.
5. **P4** (disk KV store), then **P6/P9**, then evaluate Tier 3.

Per project policy, each landed item that adds config surface must update
`doctor.rs` validation, `README.md`, and the `higgs init` template in
`daemon.rs`.

## 5. Results: claimed vs measured (2026-08-15, Apple M4 Max 128 GB, macOS 26.5.1)

Every "Adopt" item (P1-P9) was implemented and validated on hardware against pre-registered
acceptance criteria. Negative results are recorded as first-class outcomes. Per-PR evidence is
committed under validation/ on each branch.

| Item | PR | Claimed | Measured | Verdict |
| --- | --- | --- | --- | --- |
| P7 quality regression harness | #252 | deterministic gates; perturbation trips | 3 consecutive checks byte-identical; perturbation probe exits non-zero; baselines recorded for Qwen3-1.7B-4bit + DeepSeek-V2-Lite-4bit | SHIPPED |
| P8 frontier benchmarking | #253 | stable per-frontier measurement | A/A median deltas 0.2-1.9% per frontier; measured KV bytes/token exactly matches analytic (114,688 B/tok for Qwen3-1.7B) | SHIPPED |
| P1 MLA latent KV cache | #254 | ~9x KV; decode wins at long ctx | 8.9x KV bytes/token (276,480 -> 31,104); decode 1.36x @8k, 1.71x @16k, 2.19x @32k; dense 3x32k sweep OOM-killed while MLA completed; 11/12 prompts token-exact vs decompressed baseline (drift 0.01-0.06 logprob, one near-tie flip) | PASS (opt-in, default off) |
| P4 disk-backed KV store | #257 | restart-resume TTFT >= 10x at 8k+ | 17.7x (9.30 s -> 0.53 s, ~7100-token prompt); restored output byte-identical to the in-memory prefix-hit path; corruption suite green | PASS |
| P9 serving-loop yield quantum | #259 | decode stall during long prefill cut >= 2x | inter-token p95 during a 6k prefill: 2640.8 ms -> 196.9 ms (13.4x); greedy outputs byte-identical; steady-state unchanged | PASS |
| P2 asymmetric MoE quantization | #260 | per-tensor loading + asymmetric >= uniform quality | Loader half PASS: real mixed checkpoint crashes main, loads + generates on the branch; width validation turns silent garbage into load errors. Calibration half shipped (scripts/quantize: imatrix collector + layer-granularity recipe solver + convert wrapper). Quality claim MEASURED NEGATIVE at the ~4.5 eff-bpw budget: asymmetric (8.68 GB) median max-dlogprob 4.18 vs uniform-4bit (8.84 GB) 3.41 vs bf16 reference; uniform allocation wins at this budget (ds4's regime is far lower budgets). Scoring also surfaced+fixed two real loader bugs (default-equal overrides; fused switch_mlp expert paths) | LOADER + TOOLING MERGED; quality claim NEGATIVE at 4.5 bpw |
| P3 structural greedy sampling | #255 (draft) | malformed tool calls decrease at temp | malformed-syntax base rate ~0% on Qwen3-class at temp 1.0 (2/150 vs 3/150; all residual failures are wrapper omission, out of mechanism's reach); throughput unchanged | NEGATIVE - do not merge |
| P5 exact tool-call replay | #256 (draft) | tool-turn prefix breaks ~0; TTFT improves | mechanism works (replay hits recorded, byte-identical splice, decoy-guarded) but TTFT delta ~0: thinking-token divergence precedes tool calls; 64-token block quantization hides sub-block extensions; typical calls span <= 1 block | NEGATIVE - do not merge |
| P6 confidence-gated drafts | #258 (draft) | fewer wasted drafts -> throughput | +0.25% to +0.8% (pre-registered floor: +5% on one suite); adaptive depth + batched verify already make wasted drafts cheap | NEGATIVE - do not merge |

**Merge status (2026-08-15):** PRs #251, #252, #253, #254, #257, #259, #260, and this
document's PR are merged. The three negative-result drafts, #255, #256, and #258, were
closed unmerged per the go/no-go protocol. During the merge train, pre-merge review (codex
sol, medium) caught and fixed: quality-gate/frontier suite integration gaps with the merged
runner, disk-store hardening (payload checksum, weight-bound identity, longest-prefix
lookup, bounded allocation), an MLA-vs-disk-store interaction (MLA prefixes cleanly rejected
from disk persistence), and real-checkpoint quantization compatibility (boolean predicate
maps, router-gate overrides).

Methodology notes carried forward: all A/B comparisons are medians of >= 3 post-warmup runs
(single-run spreads up to ~8% from desktop background load); token-identity claims are greedy-only;
two apparent effects during validation turned out to be measurement confounds and were re-run
(thinking-mode max_tokens truncation in P3; a stale server process answering on a reused port in the
harness). Review-cycle catches before any merge: a per-head weight-layout scramble in the P1
absorption (parity delta 210), a disk store that never restored (unaligned key length) in P4, and a
request-time crash on mixed checkpoints for plain-transformer models in P2.
