# mlx-rs 0.25.3 Capabilities

## Available Primitives

| Feature | Available | Module | Notes |
|---------|-----------|--------|-------|
| Conv1d | Yes | `nn::Conv1d` | Builder pattern, NLC weight format `[out, kernel, in]` |
| RmsNorm | Yes | `nn::RmsNorm`, `fast::rms_norm` | Fast path + module wrapper |
| Softmax | Yes | `ops::softmax_axis` | Supports `precise` flag |
| Softplus | Yes | `nn::softplus` | |
| Sigmoid | Yes | `nn::sigmoid` | |
| SiLU | Yes | `nn::silu` | |
| RoPE | Yes | `nn::Rope` | Builder with base, scale, traditional, offset |
| Categorical | Yes | `random::categorical` | |
| Quantize | Yes | `ops::quantize` | Returns `(w, scales, biases)` |
| Dequantize | Yes | `ops::dequantize` | |
| Quantized Matmul | Yes | `ops::quantized_matmul` | transpose, group_size, bits params |
| Argpartition | Yes | `ops::argpartition_axis` | For MoE top-k selection |
| Take | Yes | `Array::take_axis` | Index into array along axis |
| Take Along Axis | Yes | `Array::take_along_axis` | Gather by index array |
| Matmul | Yes | `Array::matmul` / `ops::matmul` | Supports batched |
| Expand Dims | Yes | `Array::expand_dims` | |
| Squeeze | Yes | `Array::squeeze_axes` | |
| Repeat | Yes | `Array::repeat_axis` | |
| Cumsum | Yes | `Array::cumsum` | reverse, inclusive options |
| SDPA | Yes | `fast::scaled_dot_product_attention` | Causal mask support |
| `gather_qmm` | Yes | `qwen3_next::gather_qmm` (FFI wrapper) | Direct call to `mlx_sys::mlx_gather_qmm`; used for fused MoE expert dispatch |
| Compile | Yes | `transforms::compile::compile` | Fuses element-wise ops into fewer GPU kernels |

## NOT Available

| Feature | Impact | Workaround |
|---------|--------|------------|
| `gather_mm` | MoE expert routing (non-quantized) | Dequantize + regular matmul per expert batch |
| Custom Metal kernels | GatedDeltaNet acceleration | Use `compile` to fuse element-wise ops; sequential fallback for recurrence |
| SwitchLinear | MoE layer type | Manual implementation with stacked weights + `gather_qmm` |

## Quantized Weight Loading

Pre-quantized MLX models (from HuggingFace) store weights with flat names:
- `layer.weight` (packed uint32)
- `layer.scales` (float16)
- `layer.biases` (float16, note plural)

But mlx-rs `QuantizedLinear` uses nested parameter names:
- `layer.inner.weight`
- `layer.scales`
- `layer.biases`

The `remap_quantized_key()` function in `higgs-models/src/lib.rs` handles this
by retrying `.weight` -> `.inner.weight` and `.bias` (singular) -> `.inner.bias`.
Note that `.biases` (plural) is a quantization parameter and is not remapped.

## ModuleParameters Derive

The `#[derive(ModuleParameters)]` macro generates parameter tree from field names.
Fields marked `#[param]` become parameters. `#[quantizable]` marks fields for
`nn::quantize()` transformation.

Parameter paths are dot-separated: `model.layers.0.self_attn.q_proj.inner.weight`.
These must match safetensors key names (after remapping) for weight loading to work.

## Conv1d Weight Format

MLX Conv1d expects weight shape `[out_channels, kernel_size, in_channels/groups]`.
Python MLX's `sanitize()` calls `moveaxis(2, 1)` on conv1d weights to convert
from PyTorch format `[out, in/groups, kernel]` to MLX format.
Pre-converted models already have weights in the correct format.
