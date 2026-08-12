# Bonsai-Q1

Higgs supports MLX affine 1-bit checkpoints with `quantization.bits = 1` and
`quantization.group_size = 128` on the pinned upstream `oxideai/mlx-rs`
revision. Upstream MLX does not ship the required 1-bit affine kernels, so Higgs
provides runtime JIT Metal kernels for packed matvec and dequantization.

Two layouts are supported:

- Qwen3-shaped Bonsai checkpoints use the dedicated packed engine in
  `crates/higgs-models/src/bonsai_q1.rs`.
- Qwen3.5 hybrid checkpoints, including Bonsai-27B, use the existing
  `qwen3_next` architecture with its affine 1-bit operations dispatched to the
  same Higgs Metal kernels.

Single-token decode and narrow multi-token forwards stay packed. For Qwen3.5,
the packed Metal path covers up to 8 flattened rows by default, including the
small verifier batches used by speculative decoding. Wider prefill inputs
dequantize the selected matrix to the input dtype before using regular MLX
matmul. Set `HIGGS_BONSAI_QMM_MAX_ROWS=0` to disable the narrow packed path, or
raise it up to 64 for A/B testing.

For Qwen3.5 Q1 checkpoints, the loader validates every affine scale/bias pair.
When a tensor is exactly symmetric (`bias = -scale / 2`), Higgs releases its
bias array and derives the bias in the Metal kernel. Any non-symmetric tensor
keeps the general affine path. Set `HIGGS_BONSAI_SYMMETRIC_Q1=0` to retain all
bias tensors for A/B debugging.

Qwen3.5 checkpoints packaged as multimodal models currently load the text
backbone only. Their vision tower is not exposed by Higgs, so image input remains
unsupported for those checkpoints.
