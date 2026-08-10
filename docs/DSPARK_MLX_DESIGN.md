# dSpark on MLX: a correct-by-construction design

## Goal and current status

Run Prism dSpark against Bonsai-27B on Apple Silicon without changing the
target model's autoregressive output semantics. Proposal quality and target
verification throughput are separate requirements:

- the drafter proposes up to four tokens;
- the target remains the sole authority for every emitted token;
- speculative execution must not change target state, sampling, or output;
- dSpark projection tiling must be stable for identical target taps, without
  assuming the target itself is invariant to outer prefill shape;
- optimized kernels must pass the same state and numerical gates as the
  reference implementation.

[The Bonsai-27B paper](https://www.alphaxiv.org/abs/2607.bonsai-27b) reports its
dSpark throughput gains on H100 CUDA and identifies net-positive Apple Silicon
verification as open work. Reproducing the same quality and multiplier on
Metal therefore requires backend work; it cannot be inferred from the
published CUDA result.

The architecture and synthetic exactness gates described below are in place.
The real Bonsai-27B checkpoint also loads and completes a target forward in an
unsandboxed Metal run. The earlier uncaught MLX exception was caused by the
restricted test sandbox exposing no Metal devices, not by the checkpoint or
target loader.

The last controlled powered result before the TG-LUT work was a normalized
release run with the full frozen Q4 head, greedy/no-thinking sampling, block
size and draft cap four, a fresh 26-token one-shot Fibonacci prefill, an
eight-token warmup, and 128 generated tokens. Its ABBA aggregate was 23.31
tok/s for AR decode and 22.55 tok/s for speculative decode, or 0.967x. It
reported `tau = 4.536`, 99/110 exact draft matches (90%), a 201.2 ms speculative
round, byte-exact output, and only 0.27% AR endpoint drift. Power remained
connected and charging throughout. Cap three was also near parity but worse
than cap four, so it is not the default.

The experimental row4 TG-LUT integration subsequently produced 19.51 tok/s AR
decode and 29.01 tok/s speculative decode in the same ABBA harness, or 1.487x,
with wall throughput increasing from 15.72 to 21.63 tok/s (1.376x). The best
speculative sample was 29.38 tok/s and the average speculative round was
156.35 ms. It retained `tau = 4.536`, the 90% exact-draft-match rate,
byte-exact output, and 0.55% AR endpoint drift. This was deliberately allowed
to run on a discharging battery, so it is a diagnostic rather than a promotable
powered result. Scaling it by the earlier same-machine powered/battery AR ratio
estimates roughly 34.7 tok/s, but that number remains an estimate until the
powered rerun. These are narrow measurements from one prompt and two samples,
not a general Apple speedup claim.

The row4 integration has since moved from a duplicate experimental side layout
to the authoritative dense-MLP parameter representation. Synthetic exactness
tests and a real Bonsai-27B load/forward smoke test pass. A one-launch gate/up
kernel is also implemented behind a separate additional flag. A subsequent
OFF--ON--OFF battery diagnostic measured 29.31 tok/s speculative decode and
21.95 tok/s wall throughput with fusion off, versus 27.29 and 20.82 tok/s with
fusion on; AR decode was effectively unchanged at 20.42 versus 20.33 tok/s.
The off result therefore reached 1.435x decode and 1.325x wall speedup, and its
best speculative sample reached 29.47 tok/s. Acceptance and byte parity were
unchanged. This validates the primary-layout path under real decoding but does
not support enabling gate/up fusion. The run ended at 61% on a discharging
battery with no external source, so a powered primary-row4 result is still
required for promotion.

The later full-Q4 ABBA battery gate measured 19.95 tok/s AR decode and 29.14
tok/s dSpark decode (1.461x), with 90% matched-draft acceptance. Its end-to-end
wall throughput was 16.14 versus 21.80 tok/s (1.35x); the lower wall figure
includes prefill and request orchestration and is not a dSpark decode
regression. A separate paired-radix run measured 16.8 tok/s AR versus 25.59
tok/s dSpark decode (1.52x) at 87.8% acceptance. These remain machine-local
battery measurements pending the final stable-power rerun.

## Pinned sidecar contract

A dSpark sidecar is not a generic drafter that may be paired with any
shape-compatible target. Its configuration pins three quality-critical facts.

### Tap semantics

`tap_semantics` must be `post_layer_residual_v1`. Prism's authoritative GGUF
runtime captures `cur` after the indexed target layer has completed, so every
configured layer id means the post-layer residual stream, not that layer's
input or an intermediate normalized value. Higgs rejects dSpark configs that
omit or change this semantic, and validates tap ids before model execution.

### Exact target binding

The converter records a `higgs-target-artifact-v1` manifest containing the
normalized relative path, byte size, and full SHA-256 digest of the target
`config.json` and every selected base safetensors file. Loading recomputes that
exact selected set and fails closed for missing, extra, renamed, resized, or
content-modified entries. Optional MTP sidecars are deliberately outside this
base-target identity.

This prevents a sidecar trained for one target artifact from silently loading
against another checkpoint that happens to have the same dimensions.

### Reference proposal head

The reference/default quality profile keeps dSpark's frozen Q4 output head.
That is the proposal distribution trained and published by Prism.
`--reuse-target-head` is an experimental compact profile: it removes the Q4
copy and proposes through the paired target's Q1 head. Target verification
still protects final generation semantics, but proposals are numerically
different and acceptance may be lower. Acceptance comparisons must therefore
identify which head profile was used.

## Canonical target transition

Define one target transition, used by ordinary decode and by the verifier:

```text
step(token, committed_state, sampling_state)
  -> (target_token, post_layer_taps, next_committed_state, next_sampling_state)
```

The following are part of the transition and cannot be approximated:

- packed projection reduction order;
- convolution history order and activation;
- recurrent-state arithmetic;
- RoPE positions and attention reduction order;
- KV append position;
- penalty history and random-number consumption;
- post-layer tap positions exposed to the drafter.

The default dSpark verifier applies the canonical `S=1` transition to the
anchor and matching draft tokens, stopping at the first rejection or bonus
token. It advances only positions that commit, so no approximate rollback is
needed. `HIGGS_DFLASH_VERIFY_MODE=block` enables the experimental batched tape
verifier for profiling; unsupported domains fail closed to canonical `S=1`.
The production decode loop caps its final proposal/verification round to the
remaining output-token budget, avoiding unused tail positions while preserving
the trained four-position path for full rounds.

The tape verifier selects its short-row schedule explicitly: one-position
RoPE, one K/V append and one-query attention per causal prefix, canonical
convolution addition, and the shared recurrent tape kernel. Dense and active
TurboQuant caches use the same one-query primitive as AR. Sampling, penalties,
forced-thinking behavior, stop conditions, and RNG state remain part of the
transaction rather than host-side approximations.

Canonical `S=1` is the correctness oracle, not the final fast path. Serializing
a target/sample barrier for each accepted position cannot by itself deliver a
speculative target speedup.

## Typed, transactional drafter state

`DFlashCache` owns three distinct pieces of state:

```text
DFlashCache {
    layers: projected per-layer K/V,
    pending_taps: raw post-layer tap rows not yet forming a full tile,
    position: absolute number of target rows ingested,
}
```

`position` is explicit and absolute. It is never inferred from retained KV
length: sliding-attention layers may evict old rows while the RoPE position
continues increasing. The projected position is derived only as
`position - pending_rows`.

Cache updates are staged. Priming clones the layer state, validates all tap
shapes, builds and materializes the candidate projections, and commits layer
state, raw carry, and absolute position together. A `DFlashForwardTransaction`
similarly owns lazy proposal output plus candidate layer state. Commit checks
that its base position still matches the live cache, preventing stale or
partially evaluated work from becoming visible. Dropping a failed transaction
leaves the committed cache unchanged.

## Paired cache lifecycle

Reusable dSpark state is one capability, never two independently discoverable
caches:

```text
Cold -> LivePair -> SealedPair -> session move / radix fork -> LivePair
                    |
                    +-> deterministic capability loss -> TargetOnly
                    +-> effectful seal/fork failure     -> publish nothing
```

`LivePair` move-owns the target state, dSpark state, exact token ledger, tap
frontier, and branch epoch. A decode lease prevents either cache half from
being published until target advancement, tap advancement, and the token
ledger have all committed the same boundary. `SealedPair` validates every
target KV layer, every recurrent-cache absolute offset, the dSpark position,
and the exact token identity. The dSpark fixed-tile remainder is preserved; it
is valid internal state and is not projected merely to make a cache entry.

All known target-plus-taps prefill enters through the same `LivePair`
coordinator, whether the branch is cold, session-resumed, or radix-forked. It
derives both boundaries from pair-owned state and accepts only the final
unconsumed tap rows from model execution. Canonical stateless and session
decode likewise share one transaction driver for anchor-ticket creation,
history derivation, and ledger commit; the session backend adds lease
validation before that commit, while streaming/termination policy remains an
outer delivery concern.

Retained sessions move the whole pair out of the session map and mint a fresh
live branch epoch. The radix cache stores an immutable dSpark sidecar only at
an exact endpoint in the existing target trie; each hit materializes the
deduplicated target path and independently forks the endpoint's dSpark
snapshot. Paired radix lookup is memory-only and never combines a target from
disk with a separately selected sidecar.

Session and radix pairs report target and dSpark bytes independently. Session
TTL/count/token limits and radix TTL/LRU/cap eviction remove the complete
ownership unit. The initial paired-radix cap is two entries.

The first release intentionally excludes disk-persisted dSpark snapshots,
partial/interior drafter radix hits, batch/paged-engine integration,
sampled-history transactions, and cached logits for a full prompt hit. A full
paired hit without a generation suffix therefore fails closed to fresh
prefill.

### Fixed 32-row context tiling

Q4 projection results can depend on the row schedule selected by MLX. Therefore
the dSpark numerical contract uses fixed 32-row raw-tap tiles, independent of
the target prefill's outer memory chunks:

1. concatenate the configured post-layer taps for each target row;
2. prepend the cache's raw carry;
3. project and append every complete 32-row tile;
4. retain fewer than 32 raw rows in `pending_taps`;
5. on proposal forward, flush complete tiles and the one final fixed remainder
   before running the noise queries.

For identical raw taps, their submission chunking cannot change this Q4
projection schedule. This does not make target outer-prefill chunking
numerically invariant. The real Bonsai target exhibits sequence-shape drift,
so changing its forward shape can change target taps and therefore proposals
even though the downstream drafter tiling is fixed. Target prefill shares the
ordinary target backbone traversal and evaluation barriers; only the final
hidden row is vocabulary-projected, while all requested post-layer taps are
materialized before the corresponding drafter-cache commit.

## Transactional block verifier

The optimized target verifier should expose a transaction instead of mutating
the committed target cache directly:

```text
begin_verify(anchor_and_drafts, &committed_state) -> VerifyTxn
VerifyTxn.targets()                              -> target rows
VerifyTxn.commit(prefix_len)                     -> committed state + taps
```

`prefix_len` is always in `1..=5`. A transaction contains:

- full-attention K/V written beyond the committed offset, with the live offset
  unchanged until commit;
- the initial GDN state plus canonical per-position innovations and QKV input;
- candidate convolution history;
- per-position post-layer taps;
- target logits/samples and their sequential history/RNG transaction.

Commit selects exactly one prefix. KV commit advances by that prefix. GDN
commit replays the prefix through the same recurrence source used by forward,
or selects a captured prefix state. Convolution rebuild uses the same
chronological lag order as `S=1`. Rejection never uses an inverse operation or
an approximate restore.

## Shared numerical primitives

Repeated `S=1` and an optimized `S=N` path must be bit-equal at every layer.
Both shapes therefore share the numerical primitives that define the model:

1. **Convolution:** current tap first, then newest-to-oldest history, with the
   same rounded addition order and `silu_direct` implementation.
2. **GDN recurrence:** plain forward, tape capture, and accepted-prefix replay
   are generated from the same Metal recurrence source. Recurrent state and
   replay tape have an explicit f32 boundary, including stateless
   initialization; bf16 input must never silently create bf16 recurrent state.
3. **Full attention:** each short-block query uses the same one-query SDPA or
   TurboQuant code-domain reduction as AR. K/V is appended one row at a time so
   activation boundaries and packed-cache contents remain identical.
4. **Q1 projections:** each verifier position preserves the existing SIMD
   reduction order. No verifier optimization may reduce across positions.
5. **Target traversal:** normal and tap-producing forwards use one backbone
   loop, including the same layer materialization barriers.

## Two different Q1 problems

The narrow verifier and large prefill need different kernels and must not be
conflated.

### Exact narrow verification

The current packed Q1 multi-row path runs the proven decode QMV independently
for each verifier row in grid Z. It has exact parity with concatenated repeated
QMV across the tested verifier shapes, dtypes, bias forms, leading dimensions,
and non-row-contiguous inputs. A real-shape M sweep corrected the initial model
that five rows caused five full packed-weight reads: for BF16 N=17,408 and
K=5,120, M=5 measured 2.92x the M=1 time after specialization. Cache and GPU
parallelism already reuse a substantial fraction of the packed weights even
though the source expresses independent rows.

An opt-in aligned-N specialization removes output-row bounds checks when N
fills the complete threadgroup tile. It is bit-exact against the guarded
kernel, retains the guarded fallback for unaligned N, and improved the same
M=5 microbenchmark by 1.088x. A full-model ABBA run passed byte-exact output,
acceptance, and thermal-endpoint gates, but it ran at 5% battery with no charger
power and used a 128-token warmup rather than the baseline's default eight.
Its absolute throughput is therefore invalid for promotion; the benchmark must
be repeated under real external power with the original warmup policy. The
first powered warmup-eight rerun was also rejected: AR throughput ramped from
19.90 to 21.73 tok/s and exceeded the 3% endpoint-drift gate while the battery
recovered from 7% to 10%. Its manual aggregate is diagnostic only. The final
rerun began at 21% on a negotiated 68 W adapter, retained the original
warmup-eight policy, passed byte-exact output and the 3% stability gate, and
produced the controlled result above. This establishes whole-model safety and
current performance; it is not a same-window aligned-on/off attribution.

The ignored full-model harness now checks macOS power state before model load
and after ABBA. It requires AC power, confirms that an internal battery sees an
external source, and defaults to a 20% battery floor. Set
`HIGGS_TEST_MIN_BATTERY_PERCENT` to change the floor, or explicitly set
`HIGGS_TEST_REQUIRE_AC=0` for a diagnostic battery run whose throughput will
not be comparable to the powered baseline.

The M5 row-cohort prototype was bit-exact on its gates but measured only
0.96--1.00x the shipped grid-Z path on the tested representative shapes, so it
was dropped. Padding five rows into stock Steel QMM changed bf16 results and
lost throughput, so it is not a verifier solution.

A custom BM8/BN32 floating `simdgroup_matrix` prototype also failed its
performance gate. After fixing redundant packed-word loads, the best BF16/BK32
variant took 1.867 ms versus 1.418 ms for the interleaved pre-specialization QMV
baseline on N=17,408, K=5,120, M=5. FP16 and FP32 fragments were no faster;
increasing BK to 128 regressed to 4.058 ms. CPU affine-oracle, row-independence,
and cross-fragment output checks passed, so the result rejects this kernel
shape rather than merely a broken implementation. The prototype remains in an
isolated worktree and is not part of this design.

Prism's CUDA path obtains a qualitatively different advantage: it quantizes
the five activation rows to Q8_1, reuses a Q1 block across all five right-hand
sides, and uses packed integer DP4A operations. The M4/Xcode Metal toolchain has
no corresponding integer dot or integer matrix contraction. Emulating that
topology is possible, but Q8 activation quantization would also change the
canonical BF16 target reduction and therefore cannot replace the exact path by
default. It is a secondary research backend, not a direct CUDA port.

### Large prefill

For more than the narrow packed-row limit, the current path dequantizes large
Q1 matrices before regular MLX matmul. The proper large-prefill path is a true
packed affine 1-bit QMM based on the Prism/Steel BlockMMA design, adapted to the
MLX version and Apple GPU generation Higgs actually ships. This work should
have separate prompt/logit parity and performance gates; its faster blocked
reduction is not automatically valid for the exact narrow verifier.

## What the WebGPU demo establishes

The public 27B browser demo is not a dSpark implementation. It loads one
`Bonsai-27B-Q1_0.gguf` target and performs autoregressive decode; its pipeline
depth controls asynchronous target-step submission and readback rather than
speculative proposal depth. There is no drafter artifact, hidden-state tap,
acceptance loop, or rollback path, and no public WebGPU dSpark benchmark.

Its numerical contract also differs from this backend. The browser GGUF uses
Q1_0 sign blocks and is roughly 3.8 GB, whereas the MLX artifact is about 5.13
GB and stores affine one-bit groups plus the bundled vision tower. WebGPU's
small-M kernels use FP16 activation lookup tables or FP16 subgroup-matrix
operands, FP32 accumulation, split-K reduction, and M=5 padding. They do not
preserve the current BF16 QMV reduction tree.

The reusable idea is the four-bit lookup identity

```text
s * sum(sign_i * x_i) = 2s * sum(bit_i * x_i) - s * sum(x_i)
```

not the browser shader verbatim. Adopting it requires a frozen alternate Q1
reduction plan selected before prefill and used by both AR and block verify for
the entire request. It can be internally correct by construction, but it will
not be byte-identical to the legacy MLX target and must pass model-quality and
long-state qualification before becoming a supported backend.

An isolated Metal prototype separated two implementations of that idea. A
global-materialized LUT was slower even when construction was excluded: its M5
end-to-end ratios were 0.776x and 0.528x on the two dominant shapes. The
faithful threadgroup-local schedule was materially different and passed its
hard kernel gate. Against interleaved aligned QMV, the selected scalar-contract
variant measured:

| M | N | K | aligned QMV | TG LUT | ratio |
|---:|---:|---:|---:|---:|---:|
| 5 | 17,408 | 5,120 | 946.0 us | 639.5 us | 1.479x |
| 5 | 5,120 | 17,408 | 894.1 us | 583.5 us | 1.532x |
| 1 | 17,408 | 5,120 | 356.5 us | 337.7 us | 1.056x |
| 1 | 5,120 | 17,408 | 364.5 us | 381.4 us | 0.956x |

The generic M1--M4 and M6--M8 plans use a 256-thread/N256/K128 tile, an FP16
threadgroup LUT, independent scalar FP32 accumulators, and two barriers per K
group. The native M5 specialization uses a 5 KiB LUT and five accumulators
while sharing each packed weight/scale read across all five verifier rows.
Distinct stacked rows match their separate M1 evaluations bit-for-bit under
the new plan.

The opt-in full-model trial (`HIGGS_BONSAI_TG_LUT4=1`) produced the 29.01 tok/s
battery result above with workgroup size 256. A 160-thread trial reached 29.59
tok/s initially but fell to 27.56 tok/s and was rejected when AR endpoint drift
reached 7.24%, above the 3% stability gate.

The supported integration no longer retains both canonical and row4-packed
weights. At load time, each eligible projection moves its packed arrays into
the authoritative `weight` and `scales` parameters and records only row4
dimensions as layout metadata. Forward reconstructs a validated borrowed row4
view from those current parameters, avoiding stale shadow handles if module
parameters change. M=1 through M=8 execute the exact packed schedule (with the
native M5 specialization), and wider inputs dequantize directly from row4
before MLX matmul instead of first rebuilding canonical packing.

The exact affine check promotes 191 of Bonsai-27B's 192 dense-MLP projections.
`model.layers.31.mlp.down_proj` has one FP16 half-subnormal bias mismatch and
therefore remains in the canonical general-affine path, retaining its required
1.328125 MiB bias. That fallback retains only its canonical weight; it does not
create a duplicate row4 copy. Synthetic row-count, layout, dequantization, and
fused-projection exactness tests pass, as does a real-checkpoint load/forward
smoke test.

This physical parameter layout is currently an inference representation.
Generic checkpoint export would mislabel the row4 buffers as source-canonical
parameters, so exporting a promoted model requires explicit demotion and is
unsupported as-is.

The paired gate/up microbenchmark and full-model battery diagnostic reject the
current fusion schedule as a throughput optimization:

| Rows | Separate median | Fused median | Median speedup |
|---:|---:|---:|---:|
| 1 | 447.96 us | 446.83 us | 1.0025x |
| 5 | 752.08 us | 762.42 us | 0.9864x |

Each microbenchmark result contains 31 samples. In the stable end-to-end
comparison, fusion-off versus fusion-on was 29.31 versus 27.29 tok/s for
speculative decode and 21.95 versus 20.82 tok/s for wall throughput, while AR
decode differed by only 0.44%. The leading fusion-off control was visibly cold,
however. Latency-space interpolation between the two off controls estimates
that fusion improved AR decode by 2.23% while reducing speculative decode by
2.40%; the speculative speedup ratio fell from an interpolated 1.406x to
1.342x. The fused kernel saves one launch and one activation-LUT construction,
but it does not reduce gate/up weight traffic and keeps both projections'
accumulators live. The measured M=5 regression is consistent with added
register pressure or reduced scheduling freedom. That mechanism is an
inference, not a profiler attribution, and the exact end-to-end penalty remains
order- and battery-confounded. The fused path therefore remains opt-in and
disabled by default.

## Remaining performance hypotheses

At the measured accepted length, 30 tok/s requires a 151.2 ms round, only about
5.2 ms or 3.4% below the 156.35 ms battery diagnostic. Reaching 40 tok/s still
requires roughly 113.4 ms, about 42.9 ms or 27.5% below that round time. The
remaining exact experiments are ranked as follows.

1. **Rejected current gate/up fusion schedule.** With the base row4 path enabled,
   `HIGGS_BONSAI_TG_LUT4_FUSED_MLP=1` now runs symmetric row4 gate and up
   projections in one launch for M=1 through M=5. It shares the activation LUT
   while preserving an independent FP32 accumulator and the existing output-
   rounding boundary for each projection; SiLU/multiply remains a separate
   authoritative operation. Synthetic fused-versus-separate outputs are exact.
   The paired microbenchmark was flat at M=1 by median and 1.36% slower at M=5.
   The adjacent stable battery passes observed 7.40% lower speculative decode
   and 5.43% lower wall throughput with fusion, while sandwich interpolation
   estimates a smaller 2.40% speculative-decode regression. The direction is
   consistent; the magnitude is not yet causal. It remains available only for
   explicit A/B work. Changing the tile or accumulator strategy would require
   a new exactness and powered-performance promotion cycle.
2. **Dense prefix attention.** Append all five post-RoPE K/V rows once, then
   evaluate each query against its exact chronological prefix. Grid rows must
   never reduce together, committed cache length advances only by the accepted
   prefix, and TurboQuant continues to fail closed to the one-row schedule.
   The synchronized full-attention branch was only a 17.2 ms upper bound,
   including projections, so the recoverable time is materially smaller.
3. **Fused dSpark Markov reductions.** Preserve the four true sequential token
   dependencies, but fuse gather, packed-Q4 projection, base-logit addition,
   and exact lowest-index argmax for each position. This can avoid materialized
   vocabulary-width intermediates; it cannot remove the dependency between
   positions. The entire drafter plus host component was previously estimated
   near 25 ms, so this fusion alone cannot close the round gap.

GDN scheduling is not a fourth large opportunity. The fused convolution runs
all five chronological steps in one kernel and the recurrence runs its time
loop inside one dispatch. From the 1.70x convolution microbenchmark and the
2.02 ms full-round saving, current convolution is inferred at only about
2.89 ms per round. The synchronized 64.4 ms GDN-branch trace includes its large
Q1 projections and must not be attributed to recurrence launches.

## Required gates

Before block verification becomes the default, tests must cover:

- whole-model Q1 logits, post-layer taps, full-attention KV, f32 GDN state,
  convolution history, and partial commit for repeated `S=1` versus blocks of
  lengths 2 through 5;
- forced first mismatches at draft positions 0 through 3;
- consecutive partial commits and full acceptance;
- greedy and sampled requests, including penalties and RNG consumption;
- code, math, structured, and high-entropy prose prompts;
- cold/warm caches and multiple context and prefill-chunk lengths;
- exact generated bytes and stop/length boundaries;
- both head profiles, with acceptance reported separately;
- the exact bound target artifact, not merely a synthetic shape-compatible
  model.

Synthetic whole-model gates prove transition mechanics but do not replace
real-checkpoint gates. Real-checkpoint reports must separate proposal, target
verification, target head/argmax, host barriers, commit, prefill, and
end-to-end timings, alongside warmed AR decode in the same thermal window. The
small full-Q4 Fibonacci baseline above is a parity result; 30 tok/s remains a
target rather than a measured result.
