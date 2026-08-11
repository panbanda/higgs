# Ternary dSpark verifier optimization notes

Date: 2026-07-20

This note records the evidence trail for the Ternary-Bonsai-27B dSpark verifier work. The goal was to close the gap between the ternary dSpark speedup and the earlier Bonsai Q1 row4/TG-LUT result.

## Baseline context

Target model:

```text
/Users/peppi/.cache/lm-studio/models/prism-ml/Ternary-Bonsai-27B-mlx-2bit
```

Drafter sidecar:

```text
/Users/peppi/models/ternary-bonsai-27b-dspark-mlx
```

Reference dSpark flags:

```bash
HIGGS_DFLASH_VERIFY_MODE=block \
HIGGS_DFLASH_GATE=0 \
HIGGS_DSPARK_DRAFT_CAP=4 \
HIGGS_DSPARK_TARGET_HEAD=0
```

Earlier long Fibonacci reference:

```text
client decode: 13.87 tok/s
server decode: ~12.0 tok/s
accept_len: 4.23
spec_rounds: 30
```

The AR Fibonacci reference was about `13.51 tok/s`, so the old ternary speculative path was only a small win.

## What did not explain the gap

MLX version was not the blocker. Newer MLX/mlx-c experiments either failed API compatibility or regressed ternary decode.

Gate/up fusion was not enough:

```text
gate+up only: 1.061x
full MLP with fused gate/up: 1.026x
```

Tensor rank was not the blocker. `[1,5,K]` was comparable to `[5,K]` for MLX Q2 matmuls.

Native dSpark verifier batching is not safe as a default. It can help regular prompts, but on prose it dropped acceptance to `2.89`, increased rounds to `44`, and regressed throughput to about `8.28 tok/s`.

Zero-elision is not promising for this checkpoint. Real tensor scans showed zero-code density around `28-30%`, with no sampled groups near the `50-60%` threshold where sparse masks become plausible.

## Checkpoint structure

The actual ternary checkpoint has strict ternary affine structure:

```text
codes: q in {0, 1, 2}; code 3 unused
bias / scale: exactly -1.0
weight: scale * (q - 1)
```

This made a strict ternary kernel viable. It can drop the bias buffer and avoid generic affine Q2 math.

## Head argmax result

A verifier-only Q2 `lm_head -> argmax` candidate kernel avoids materializing full `[5, vocab]` logits.

Isolated kill gate:

```text
MLX qmm + argmax: 15686.5 us
candidate kernel: 11300.2 us
speedup: 1.388x
argmax parity: matched
```

The first runtime impact was real but small:

```text
head argmax only: 14.03 tok/s
```

The follow-up moved the final candidate reduction from CPU to a tiny GPU kernel returning `[1,5]` `uint32` ids directly. Isolated head timing was neutral because the projection dominates:

```text
MLX qmm + argmax:       15724.8 us
candidate + CPU reduce: 11284.3 us
candidate + GPU reduce: 11285.1 us
GPU vs CPU:             1.000x
argmax parity:          matched
```

End-to-end runtime did improve because the verifier no longer syncs/copies the candidate arrays to CPU:

```text
row2 MLP + CPU head reduce:         17.49 tok/s
row2 MLP + GPU head reduce:         18.55 tok/s
row2 MLP + GPU head + split-K down: 19.29 tok/s
```

The next head probe specialized the candidate kernel to the checkpoint's strict ternary affine structure (`bias = -scale`, `weight = scale * (q - 1)`). The generic synthetic benchmark initially failed parity with arbitrary affine biases, then matched once the benchmark used strict ternary biases:

```text
MLX qmm + argmax:            15494.7 us
affine candidate + GPU reduce: 11218.7 us
ternary candidate + GPU reduce: 8369.6 us
ternary vs affine candidate: 1.340x
ternary vs MLX qmm+argmax:  1.851x
argmax parity:              matched under strict ternary affine
```

Runtime with strict ternary head:

```text
row2 MLP + GPU head + split-K down + ternary head: 19.68 tok/s
```

Server telemetry stayed behavior-identical:

```text
accept_len: 4.23
spec_rounds: 30
server decode after warmup before split-K: ~16.0 tok/s
server decode after warmup with split-K:   ~16.5-16.8 tok/s
server decode with ternary head:           ~17.0 tok/s
```

The head path is available behind:

```bash
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

## Winning row2 MLP path

The breakthrough was a strict ternary row2 M=5 kernel for verifier MLP projections:

```text
bonsai_q2_row2_m5_ternary_direct
```

It uses row2-transposed Q2 weights and strict ternary math:

```text
output = scale * sum((q - 1) * x)
```

Projection microbench:

```text
gate/up M=5:
  MLX stock:    1394.2 us
  ternary row2:  935.8 us
  speedup:      1.49x

down M=5:
  MLX stock:    1366.2 us
  ternary row2: 1308.7 us
  speedup:      1.04x
```

Split-K was tested for `down_proj`, where `K=17408` and the direct row2 kernel exposes less row parallelism:

```text
gate/up M=5:
  MLX stock:     1356.9 us
  ternary row2:   929.0 us
  split-K2:       941.1 us
  split-K4:       935.0 us

down M=5:
  MLX stock:     1366.0 us
  ternary row2:  1295.6 us
  split-K2:      1164.8 us
  split-K4:      1144.8 us
```

Conclusion: split-K hurts or is neutral for gate/up, but wins for down. Runtime dispatch now uses split-K4 only for the Bonsai ternary `down_proj` shape (`N=5120`, `K=17408`) under the row2 MLP flag.

Full MLP microbench:

```text
MLX stock full MLP:    3557.4 us
ternary row2 full MLP: 2557.2 us
speedup:              1.391x
```

Hybrid row2 plus stock MLX down was tested after the row2 win:

```text
MLX stock full MLP:                 3518.8 us
ternary row2 full MLP:              2542.8 us
row2 gate/up + MLX stock down:      2642.2 us
row2 full MLP speedup:              1.384x
hybrid row2/MLX-down speedup:       1.332x
```

Conclusion: keeping `down_proj` on stock MLX does not recover throughput. The all-row2 MLP remains faster, so the remaining gap is not explained by a bad row2 down-projection dispatch alone.

Fusing row2 gate/up into a single Metal launch was also tested:

```text
MLX stock full MLP:                 3500.9 us
ternary row2 full MLP:              2525.7 us
fused row2 gate/up + row2 down:     2594.8 us
row2 full MLP speedup:              1.386x
fused row2 gate/up speedup:         1.349x
```

Conclusion: one fewer launch is not enough to win here. The fused gate/up kernel increases per-thread register/weight work enough that it loses to two independent row2 projection launches.

Power-of-two M-scaling was tested with the legal dSpark cap:

```bash
HIGGS_DSPARK_DRAFT_CAP=3
```

This gives verifier `M=4` (`anchor + 3 draft`) because the published dSpark artifact has `block_size=4`; attempts to set `HIGGS_DSPARK_DRAFT_CAP=7` were clamped back to `draft_cap=4`.

Long Fibonacci, 128 tokens:

```text
client decode mean: 13.68 tok/s
accept_len:         3.63
spec_rounds:        35
server decode:      ~12.0-12.4 tok/s
```

Conclusion: the feasible power-of-two verifier tile loses. The lower draft cap increases rounds and drops acceptance enough that any M=4 scaling benefit is erased. Testing M=8 would require a dSpark artifact or verifier schedule that can actually propose seven draft positions.

Runtime result with:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1 \
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Long Fibonacci, 128 tokens:

```text
previous long Fibonacci: 13.87 tok/s
row2 MLP + head argmax:  19.68 tok/s
```

Server telemetry:

```text
accept_len: 4.23
spec_rounds: 30
server decode before: ~12.0-12.1 tok/s
server decode after:  ~17.0 tok/s
```

This is about `1.46x` versus the AR Fibonacci baseline, still short of the `1.5x` target but now within roughly another three percent.

## Runtime controls

The row2 MLP path is opt-in:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1
```

The head argmax path is opt-in:

```bash
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Native dSpark verifier remains opt-in because it is prompt-sensitive:

```bash
HIGGS_DSPARK_NATIVE_VERIFY=1
```

## Next targets

The remaining gap is likely in `down_proj`, `lm_head`, and verifier overhead outside the MLP.

Recommended next moves:

1. Probe split-K variants for the strict ternary head candidate kernel, since the head has `N=248320`, `K=5120`, and still dominates isolated head time.
2. Explore radix-3 prepack only after proving it beats direct ternary on gate/up, down, or head in isolation.
3. Look for verifier scheduling overhead outside target forward now that arithmetic-only wins are smaller.
4. Test M=8 only if a dSpark artifact or schedule can produce seven draft positions; the current artifact clamps at four.
5. Keep native verifier scheduling as an explicit probe/flag until exactness and acceptance are understood across prose.
## Default runtime policy

The ternary path now defaults the winning exact verifier setup for affine 2-bit Qwen3Next/Bonsai targets:

```bash
HIGGS_DFLASH_VERIFY_MODE=block
HIGGS_DFLASH_GATE=0
HIGGS_DSPARK_DRAFT_CAP=4
HIGGS_DSPARK_TARGET_HEAD=0
```

Additional ternary defaults:

```bash
HIGGS_DSPARK_Q2_ROW2_MLP=1
HIGGS_DSPARK_Q2_HEAD_ARGMAX=1
```

Set either variable to `0` to force the older MLX stock path for that component.

Current best apples-to-apples Fibonacci result on AC power:

```text
AR decode:            13.51 tok/s
exact dSpark decode:  19.68 tok/s
speedup:              1.46x
accept_len:           4.23
spec_rounds:          30
```

The exact verifier path is the upstreamable default. Top-K/proposal probes remain local because the best measured top-K variant did not beat the exact path.
