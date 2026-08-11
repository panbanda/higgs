# Bonsai-Q1

Higgs supports MLX affine 1-bit checkpoints with `quantization.bits = 1` and
`quantization.group_size = 128` on the pinned `oxideai/mlx-rs` revision.
Upstream MLX does not ship the required affine 1-bit decode kernels, so Higgs
provides runtime JIT Metal kernels for packed matvec and dequantization.

Two layouts are supported:

- Qwen3-shaped Bonsai checkpoints use the dedicated packed engine in
  `crates/higgs-models/src/bonsai_q1.rs`.
- Qwen3.5 hybrid checkpoints, including Bonsai-27B, use the existing
  `qwen3_next` architecture with affine 1-bit operations dispatched to the
  same Higgs Metal kernels.

Single-token decode and narrow multi-token forwards remain packed. For
canonical Qwen3.5 weights, the established packed path covers up to eight
flattened rows by default, including speculative verifier batches. It applies
the QMV reduction independently in grid Z and is bit-exact with concatenated
repeated QMV across the tested verifier domains. Set
`HIGGS_BONSAI_QMM_MAX_ROWS=0` to disable it, or raise the cap to at most 64 for
explicit A/B experiments.

With `HIGGS_BONSAI_TG_LUT4=1`, eligible symmetric Qwen3.5 dense-MLP
projections are instead promoted at load time to the threadgroup-local LUT's
row4 physical layout. The row4 arrays become the authoritative `weight` and
`scales` parameters; layout state stores only dimensions, and each forward
constructs a shape- and dtype-checked borrowed view of the current parameters.
There is no retained canonical copy. This path preserves the exact scalar
contract for M=1 through M=8. M=5 has a native schedule, while the other row
counts use the generic schedule.

Wider prefill currently dequantizes each selected matrix to the input dtype
before regular MLX matmul. Promoted projections dequantize directly from row4;
they do not reconstruct a canonical packed array first. This is a separate
performance problem from exact narrow verification. The intended
large-prefill optimization is a packed affine 1-bit QMM adapted from
Prism/Steel BlockMMA, with its own target-prompt parity gates. A blocked QMM
reduction must not silently replace the exact QMV order used by speculative
verification.

For Qwen3.5 Q1 checkpoints, the loader validates every affine scale/bias pair.
When a tensor is exactly symmetric (`bias = -scale / 2`), Higgs releases its
bias array and derives the bias in the Metal kernel. Any non-symmetric tensor
keeps the general affine path. Set `HIGGS_BONSAI_SYMMETRIC_Q1=0` to retain all
bias tensors for A/B debugging.

On the current Bonsai-27B artifact, 191 of 192 dense-MLP projections pass that
exact promotion check. `model.layers.31.mlp.down_proj` has one FP16
half-subnormal bias mismatch, so it deliberately stays in the canonical affine
path and retains its required 1.328125 MiB bias. It has one canonical weight
array, not a canonical/row4 duplicate pair. All eligible transforms for a layer
complete before any parameter is installed, so a failed transform leaves the
original layer parameters intact.

The promoted in-memory representation is currently inference-only. Generic
checkpoint export would write the physical row4 arrays under parameter names
whose source format expects canonical packing. Export therefore requires an
explicit demotion step and is unsupported as-is.

Qwen3.5 checkpoints packaged as multimodal models currently load only their
text backbone. Higgs does not expose the vision tower, so image input remains
unsupported for these checkpoints.

## Bonsai-27B dSpark

Higgs converts Prism's public
[`Bonsai-27B-dspark-Q4_1.gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
into an MLX sidecar for the DFlash recurrent-tape implementation.

The reference/default quality profile preserves dSpark's full frozen Q4 output
head. Reproduce it with the exact paired target directory:

```bash
python scripts/convert_dspark_gguf.py \
  Bonsai-27B-dspark-Q4_1.gguf Bonsai-27B-dspark-mlx \
  --target-dir ~/.cache/lm-studio/models/prism-ml/Bonsai-27B-mlx-1bit
```

The converter losslessly repacks GGUF Q4_1 blocks into MLX affine Q4/group-32.
It omits the duplicate token embedding because dSpark is tied to the paired
target and uses that target embedding for the draft block. The generated
configuration pins `tap_semantics = post_layer_residual_v1`: Prism's
authoritative source captures post-layer `cur`, so Higgs rejects missing or
different tap semantics.

Conversion also creates a `higgs-target-artifact-v1` binding over the target
`config.json` and every selected base safetensors file. Each entry contains its
normalized relative path, size, and full SHA-256 digest. The paired-target
loader recomputes this manifest and fails closed on any mismatch. Optional MTP
sidecars are excluded from the base-target identity.

### Experimental compact profile

`--reuse-target-head` omits dSpark's frozen Q4 output copy and uses the paired
Bonsai Q1 head for proposals:

```bash
python scripts/convert_dspark_gguf.py \
  Bonsai-27B-dspark-Q4_1.gguf Bonsai-27B-dspark-mlx-compact \
  --target-dir ~/.cache/lm-studio/models/prism-ml/Bonsai-27B-mlx-1bit \
  --reuse-target-head
```

This is an experimental storage optimization, not the reference quality
profile. The Q1 target head and frozen Q4 dSpark head are not numerically
identical, so the compact profile can lower proposal acceptance even though
target verification preserves final generation semantics. The published
[`peppi314/Bonsai-27B-dSpark-MLX-4bit`](https://huggingface.co/peppi314/Bonsai-27B-dSpark-MLX-4bit)
is this compact target-head conversion; benchmark it as such rather than as the
full Prism-head profile.

Load the converted directory as `draft_model`, or set `HIGGS_DFLASH_PATH`.
dSpark always runs its trained four-position non-causal trunk; generic adaptive
block sizing and early-exit verification remain disabled. It defaults to the
canonical `S=1` target verifier. `HIGGS_DFLASH_VERIFY_MODE=block` opts into the
experimental batched verifier, with unsupported model, row, or sampling
domains failing closed to canonical `S=1`. The generation loop is tail-aware:
the final round caps proposal and verification work to the remaining output
budget instead of always paying for four draft positions.

No-thinking, greedy decoding is the reference benchmark mode for this
checkpoint. Record which output-head profile is active whenever reporting
acceptance. `HIGGS_DSPARK_DRAFT_CAP=1..4` may cap vocabulary-head and verify
positions for an explicit experiment, but the trained/default block is four.
`HIGGS_DSPARK_TARGET_HEAD=1` can make a full conversion use the target Q1 head;
a sidecar created with `--reuse-target-head` always uses it.

## Cache and numerical contract

The dSpark cache has typed projected layer state, raw pending taps, and an
explicit absolute target position. Position is never inferred from retained KV
length, which can shrink under sliding attention. Cache mutations are staged,
materialized, and committed together; stale forward transactions cannot commit
against an advanced live cache.

Raw concatenated post-layer taps use a fixed 32-row projection schedule.
Incomplete tiles carry across outer target-prefill chunks, and proposal forward
flushes the final fixed remainder. Given identical raw taps, changing how they
are submitted to the drafter cannot select a different Q4 projection schedule.
This is not a target-prefill chunk-invariance claim: the real Bonsai target has
measurable sequence-shape drift, so changing the target's outer forward shape
can change the taps before they reach the drafter.

The target tape path uses one-position RoPE, one-row K/V append, one-query dense
or TurboQuant attention, and the canonical convolution addition order. Plain
GDN forward, tape capture, and prefix replay share one recurrence source, with
an explicit f32 recurrent-state and replay-tape boundary. Normal and
tap-producing target forwards also share one backbone traversal and its layer
materialization barriers.

## Performance status

The real Bonsai-27B checkpoint loads and completes a target forward when run
outside the restricted test sandbox. The earlier uncaught MLX exception came
from the sandbox exposing no Metal devices; it was an environmental device-
enumeration failure, not a checkpoint-loader failure.

A normalized powered release-build Fibonacci baseline before the TG-LUT work
used the full frozen Q4 proposal head, greedy/no-thinking decoding, block size
and draft cap four, a fresh 26-token one-shot prefill, an eight-token warmup,
and 128 generated tokens. Across two samples of that single prompt, it reported
`tau = 4.536`, with 99 exact draft matches out of 110 comparisons (90%). The
ABBA aggregate was 23.31 tok/s for AR decode and 22.55 tok/s for speculative
decode, or 0.967x, with a 201.2 ms speculative round. Output was byte-exact, AR
endpoint drift was 0.27%, and external power remained connected and charging.
A cap-three trial was also near parity but performed worse than cap four, so
four remains the default.

A fresh AC-gated row4/TG-LUT release benchmark was run on 2026-07-20 with the full frozen Q4 dSpark head, greedy/no-thinking decoding, block size and draft cap four, warmup eight, and the same 128-token paired ABBA workload. It passed the preflight and postflight AC checks, the 3% AR endpoint-drift gate, byte-exact speculative-vs-AR output, and the release acceptance floor. The aggregate result was 19.55 tok/s AR decode and 28.67 tok/s dSpark decode, or 1.466x. Wall throughput was 15.85 tok/s AR and 21.29 tok/s dSpark, or 1.343x. Acceptance stayed at `tau = 4.536` with 90.00% exact draft matches, and AR endpoint drift was 0.85% decode / 0.56% wall. This supersedes the earlier powered pre-TG-LUT result as the promotable 1-bit row4/TG-LUT baseline.

The experimental row4 TG-LUT path later measured 19.51 tok/s AR decode and
29.01 tok/s speculative decode (1.487x) in the ABBA harness, while wall
throughput rose from 15.72 to 21.63 tok/s (1.376x). Its best speculative sample
was 29.38 tok/s and its average speculative round was 156.35 ms. Acceptance and
byte-exact parity were unchanged, and AR endpoint drift was 0.55%. Because this
run was on a discharging battery, it is diagnostic rather than a promotable
powered result. The earlier powered/battery AR ratio projects roughly 34.7
tok/s, but only a powered rerun can confirm that estimate. These are narrow
measurements, not evidence of 40 tok/s or a result that generalizes beyond this
prompt and machine.

After replacing the duplicate side layout with authoritative row4 parameters,
an OFF--ON--OFF battery diagnostic compared the optional fused gate/up kernel
with the separate projection kernels. The stable fused-on pass measured 20.33
tok/s AR decode, 27.29 tok/s speculative decode, 16.42/20.82 tok/s AR/speculative
wall throughput, and 1548.1/1494.2 ms mean AR/speculative prefill. The final
stable fused-off pass measured 20.42 tok/s AR decode, 29.31 tok/s speculative
decode, 16.56/21.95 tok/s wall throughput, and 1507.1/1498.0 ms mean prefill.
Its best speculative sample was 29.47 tok/s. Thus the final off pass delivered
1.435x decode and 1.325x wall speedups, while the on pass delivered 1.342x and
1.268x. Both retained `tau = 4.536`, the 90% exact-draft-match rate, and
byte-exact output. The first cold off pass was slower and more variable, so it
is retained only as the leading half of the sandwich rather than used for the
absolute endpoint. Interpolating the two off controls in latency space estimates
that fusion improved AR decode by 2.23% but reduced speculative decode by 2.40%;
the adjacent on-to-final-off comparison observed a larger 7.40% speculative
difference whose magnitude remains order-confounded. This remained a
discharging-battery diagnostic, ending at 61% with no external source connected. It is retained as historical evidence because its absolute dSpark decode rate was slightly higher than the later AC gate, but the AC-gated 28.67 tok/s result is the publishable row4/TG-LUT claim.

[The Bonsai-27B paper](https://www.alphaxiv.org/abs/2607.bonsai-27b) reports its
dSpark throughput gains on H100 CUDA. It explicitly leaves net-positive Apple
Silicon verification as open work; the paper does not establish a Metal
speedup. The earlier exact M5 row-cohort prototype was dropped after measuring
only 0.96--1.00x the shipped grid-Z verifier path on the tested shapes. A custom
BM8 matrix prototype was likewise slower than QMV. The later threadgroup-local
row4 TG-LUT is a different kernel: its native M5 schedule shares each packed
weight/scale read across five FP32 accumulators and produced the full-model
diagnostic above. The replacement-layout integration now stores row4 as the
authoritative parameters for 191 projections, eliminating the former roughly
2.230 GiB of canonical twins for those promoted matrices. The exceptional
projection remains canonical only. The path is exact on synthetic M=1 through
M=8 gates, and the real checkpoint passes the load/forward smoke gate.

On the enabled row4 path, `HIGGS_BONSAI_TG_LUT4_FUSED_MLP=1` additionally
enables a one-launch gate/up kernel for M=1 through M=5. It shares the
activation LUT but retains separate FP32 accumulators and output-rounding
boundaries, so the existing SiLU/multiply remains authoritative. Synthetic
fused-versus-separate exact tests pass. A 31-sample paired microbenchmark was
flat at M=1 (1.0025x by median) and 0.9864x at M=5. Together with the battery
ABBA result above, this provides no basis to enable fusion by default; it
remains an explicit experiment. A powered confirmation is still pending. See
`docs/DSPARK_MLX_DESIGN.md` for the complete state contract, rejected paths,
and promotion gates.

## Default runtime policy

As of the July 20 AC-gated row4 release benchmark, Bonsai-27B 1-bit no longer requires the benchmark flag bundle for the proven path. When a Qwen3.5 affine 1-bit group-128 target is loaded, Higgs enables primary row4/TG-LUT promotion by default and keeps the rejected fused gate/up schedule off by default. A paired dSpark sidecar defaults to block verification when the model-domain validator accepts the request shape.

Escape hatches remain explicit:

```bash
HIGGS_BONSAI_TG_LUT4=0              # disable Q1 row4 promotion
HIGGS_DFLASH_VERIFY_MODE=canonical  # force S=1 dSpark verification
HIGGS_BONSAI_TG_LUT4_FUSED_MLP=1    # experimental; not default
```

The default dSpark schedule keeps Prism's full frozen Q4 proposal head (`HIGGS_DSPARK_TARGET_HEAD=0` behavior) and draft cap four. This is the AC-backed path that produced 19.55 tok/s AR decode, 28.67 tok/s dSpark decode, 1.466x decode speedup, 15.85 versus 21.29 tok/s wall throughput, and 90.00% exact draft matches.
