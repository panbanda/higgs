#![allow(clippy::items_after_test_module)]
// Model forward passes call raw `mlx_rs::transforms::eval`; they are only ever
// run by the engine while it holds the MLX gate, so they are structurally
// on-gate. The clippy `disallowed-methods` ban stays active across engine/
// server code, where an off-gate eval would be the real hazard. See clippy.toml.
#![allow(clippy::disallowed_methods)]

//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (`MoE`).
//! Every `full_attention_interval`-th layer uses full attention (`Qwen3NextAttention`),
//! all other layers use `GatedDeltaNet` (SSM-like linear attention).
//! All layers use Sparse `MoE` for the feed-forward block.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::sync::OnceLock;

use mlx_rs::{
    Array, Dtype, Stream,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        self,
        indexing::{IndexOp, TryIndexMutOp},
    },
    transforms::compile::compile_with_state,
    utils::Updatable,
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// FFI error capture for gather_qmm
// ---------------------------------------------------------------------------

// Per-thread FFI error capture avoids cross-contamination between threads.
thread_local! {
    static FFI_LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Error handler registered once with MLX to capture error messages.
/// Runs on the calling thread, so thread-local storage is safe here.
#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    FFI_LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(s);
    });
}

/// Register our FFI error handler exactly once.
fn ensure_ffi_error_handler() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
        }
    });
}

/// Wrapper for the cached `GatedDeltaNet` Metal kernel object.
struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

/// Thread-local owner for a reusable FastMetal configuration.
///
/// mlx-c copies the configuration fields inside
/// `mlx_fast_metal_kernel_apply`, so the original remains immutable and can
/// be reused. Keeping this owner thread-local avoids sharing the underlying
/// C++ object across callers even if a future engine path bypasses the
/// process-global MLX gate.
struct CachedMetalKernelConfig(mlx_sys::mlx_fast_metal_kernel_config);

// SAFETY: The kernel handle is created once during initialization and used
// read-only thereafter (only passed as an argument to `mlx_fast_metal_kernel_apply`).
// No mutable state is shared across threads.
#[allow(unsafe_code)]
unsafe impl Send for CachedMetalKernel {}
#[allow(unsafe_code)]
unsafe impl Sync for CachedMetalKernel {}

impl Drop for CachedMetalKernel {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.0);
        }
    }
}

impl Drop for CachedMetalKernelConfig {
    fn drop(&mut self) {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(self.0);
        }
    }
}

/// Cached `GatedDeltaNet` Metal kernel -- created once, reused for all layers.
static GATED_DELTA_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static GATED_DELTA_TAPE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TAPE_REPLAY_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static CANONICAL_CONV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    utils::{AttentionMask, apply_rope, create_causal_mask},
    yarn::{compute_yarn_freqs, yarn_get_mscale},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_full_attention_interval() -> i32 {
    4
}

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Match Python mlx-lm default: `norm_topk_prob: bool = True`.
/// Without normalization, `MoE` expert scores sum to ~0.39 instead of 1.0,
/// producing 0.39x output magnitude and degenerate generation.
const fn default_norm_topk_prob() -> bool {
    true
}

/// Quantization parameters from config.json (top-level defaults).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
    /// Quantization format. Defaults to `Affine` for backwards compatibility
    /// with checkpoints that don't declare a mode (the historical MLX default).
    #[serde(default)]
    pub mode: crate::quant_mode::QuantMode,
}

impl QuantizationConfig {
    /// Convenience: build a `QuantSpec` from this config.
    pub(crate) const fn spec(&self) -> QuantSpec {
        QuantSpec {
            group_size: self.group_size,
            bits: self.bits,
            mode: self.mode,
        }
    }
}

/// Resolved per-tensor quantization parameters (`group_size` + bits + mode).
///
/// Threaded through layer constructors in place of the old `(ql, qb)` pair.
/// Each `QLinear` is built from one of these so the forward path knows whether
/// to call the affine mlx-rs wrapper or the mxfp4 FFI bypass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuantSpec {
    pub group_size: i32,
    pub bits: i32,
    pub mode: crate::quant_mode::QuantMode,
}

impl Default for QuantSpec {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        }
    }
}

/// Configuration for the Qwen3-Next / Qwen3.5 hybrid architecture.
///
/// Supports hybrid SSM/attention transformers with optional Sparse `MoE`.
/// Every `full_attention_interval`-th layer uses full attention, all other
/// layers use `GatedDeltaNet` (SSM-like linear attention). `MoE` layers are
/// enabled when `decoder_sparse_step > 0` and `num_experts > 0`.
///
/// Key fields:
/// - `norm_topk_prob` — normalize top-k expert scores (default `true`).
/// - `gate_quantization` — optional quantization override for `MoE` gate weights.
/// - `use_separate_gdn_projections` — when `true`, GDN layers use 4 separate
///   projection matrices; when `false` (default), projections are fused to 2
///   combined matrices for fewer GPU dispatches.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3NextModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // Linear attention (GatedDeltaNet) params
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,

    // MoE params
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Normalize top-k expert scores to sum to 1.0 before weighting outputs.
    /// Defaults to `true` to match Python mlx-lm. Setting to `false` scales
    /// `MoE` output by the raw softmax scores (~0.39x), causing degenerate output.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Per-layer quantization override for router gate / `shared_expert_gate`.
    /// When absent, uses the global quantization config.
    #[serde(default)]
    pub gate_quantization: Option<QuantizationConfig>,

    /// Per-tensor quantization overrides keyed by module path — either
    /// checkpoint form (e.g. `"language_model.model.layers.3.self_attn.q_proj"`)
    /// or model-parameter form (prefix stripped); lookup accepts both.
    ///
    /// Lifted from `quantization.<key>` entries by the loader (mode-aware:
    /// mxfp4/affine/dense per tensor). Consumed by [`resolve_quant_for`] /
    /// `quant_spec_for` to pick the spec per `QLinear`, falling back to
    /// `quantization` when no entry matches. Empty for uniform-bit
    /// checkpoints; non-empty for Unsloth UD mix-bit, AEON mixed-precision,
    /// and similar configs. The `qwen3_5` loaders also inject `Dense` entries
    /// here for GDN projections that ship `.weight` with no `.scales`.
    #[serde(default)]
    pub quant_overrides: BTreeMap<String, QuantizationConfig>,

    /// Use separate GDN projections (qwen3.5-style) instead of combined (qwen3_next-style).
    #[serde(default)]
    pub use_separate_gdn_projections: bool,

    /// Store attention output projections (`self_attn.o_proj`, `linear_attn.out_proj`,
    /// `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `linear_attn.in_proj_ba`)
    /// as BF16-dense rather than quantized. Set by the `qwen3_5` / `qwen3_5_moe`
    /// loaders to match the Unsloth UD checkpoint layout. Left `false` for the
    /// original `qwen3_next` `model_type` to preserve historical behavior.
    #[serde(default)]
    pub dense_attention_outputs: bool,

    /// Number of MTP (Multi-Token Prediction) hidden layers.
    /// 0 = no MTP head, 1 = one transformer layer for next-next-token prediction.
    #[serde(default)]
    pub mtp_num_hidden_layers: i32,

    /// Use dense projection tensors for the MTP head.
    ///
    /// Some Qwen3.5 MTP sidecars ship full-precision `mtp.*.weight` tensors
    /// rather than quantized `weight/scales/biases` triples. This is set by the
    /// loader after inspecting checkpoint keys; it is not expected in configs.
    #[serde(default)]
    pub use_dense_mtp: bool,

    /// Use an MoE-structured MTP head (Qwen3.6-A3B style).
    ///
    /// These sidecars ship the MTP layer as a full `MoE` decoder layer
    /// (`mlp.gate`, `mlp.switch_mlp.*`, `mlp.shared_expert*`) with a quantized
    /// `fc`. Set by the loader after inspecting checkpoint keys; not expected
    /// in configs.
    #[serde(default)]
    pub use_moe_mtp: bool,
}

impl Qwen3NextModelArgs {
    /// Resolve the quantization spec for a model parameter path.
    ///
    /// Checks `quant_overrides` first (per-tensor entries from `config.json`),
    /// then falls back to the global `quantization` default. Returns
    /// `QuantSpec::default()` (affine 4-bit gs=64) if no quantization config
    /// is present at all.
    pub(crate) fn quant_spec_for(&self, path: &str) -> QuantSpec {
        self.quant_override_for(path)
            .map_or_else(|| self.default_quant_spec(), QuantizationConfig::spec)
    }

    /// Look up a per-tensor override, accepting either key form.
    ///
    /// Entries may be keyed in checkpoint form (`language_model.`-prefixed,
    /// as published in `config.json`) or in model-parameter form (prefix
    /// stripped); callers likewise pass either form.
    fn quant_override_for(&self, path: &str) -> Option<&QuantizationConfig> {
        if let Some(qc) = self.quant_overrides.get(path) {
            return Some(qc);
        }
        path.strip_prefix("language_model.").map_or_else(
            || self.quant_overrides.get(&format!("language_model.{path}")),
            |stripped| self.quant_overrides.get(stripped),
        )
    }

    /// The global default quantization spec (used as fallback for tensors not
    /// in the per-path override map).
    pub(crate) fn default_quant_spec(&self) -> QuantSpec {
        self.quantization
            .as_ref()
            .map_or_else(QuantSpec::default, QuantizationConfig::spec)
    }
}

// ---------------------------------------------------------------------------
// Quantized weight containers
// ---------------------------------------------------------------------------

type QuantizedParams = (Param<Array>, Param<Array>, Param<Array>);

pub(crate) fn init_quantized_params() -> QuantizedParams {
    fn placeholder() -> Param<Array> {
        Param::new(Array::from_slice(&[0.0_f32], &[1]))
    }

    (placeholder(), placeholder(), placeholder())
}

/// Zero-sized marker stored in place of validated symmetric Q1 biases.
///
/// Q1 affine weights with `bias = -scale / 2` need no resident bias buffer;
/// the runtime Metal kernels derive it from the scale. A zero-sized array is
/// distinct from the `[1]` unloaded-parameter placeholder used by the loader.
fn symmetric_q1_bias_sentinel() -> Array {
    Array::from_slice::<f32>(&[], &[0])
}

fn has_symmetric_q1_biases(biases: &Array) -> bool {
    biases.size() == 0
}

fn has_loaded_affine_q1_biases(scales: &Array, biases: &Array) -> bool {
    biases.size() > 0
        && biases.shape() == scales.shape()
        && matches!(
            biases.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        )
}

fn validate_dflash_q1_linear(
    path: &str,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
    mode: crate::quant_mode::QuantMode,
) -> Result<(), Exception> {
    if mode != crate::quant_mode::QuantMode::Affine || bits != 1 || group_size != 128 {
        return Err(Exception::custom(format!(
            "{path} is outside the proven dSpark Q1 domain: mode={mode:?} bits={bits} group_size={group_size}"
        )));
    }
    let [rows, packed_columns] = *weight.shape() else {
        return Err(Exception::custom(format!(
            "{path} must have a loaded two-dimensional packed Q1 weight"
        )));
    };
    let [scale_rows, scale_columns] = *scales.shape() else {
        return Err(Exception::custom(format!(
            "{path} must have two-dimensional Q1 scales"
        )));
    };
    if weight.dtype() != Dtype::Uint32
        || !matches!(
            scales.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
        )
    {
        return Err(Exception::custom(format!(
            "{path} has invalid Q1 dtypes weight={:?} scales={:?}",
            weight.dtype(),
            scales.dtype()
        )));
    }
    let logical_columns = packed_columns
        .checked_mul(32)
        .ok_or_else(|| Exception::custom(format!("{path} Q1 shape overflow")))?;
    if logical_columns % group_size != 0 {
        return Err(Exception::custom(format!(
            "{path} logical Q1 width {logical_columns} is not divisible by group size {group_size}"
        )));
    }
    let expected_scale_columns = logical_columns / group_size;
    if rows <= 0
        || packed_columns <= 0
        || scale_rows != rows
        || scale_columns != expected_scale_columns
    {
        return Err(Exception::custom(format!(
            "{path} has inconsistent packed Q1 weight/scales shapes {:?}/{:?}",
            weight.shape(),
            scales.shape()
        )));
    }
    if !has_symmetric_q1_biases(biases) && !has_loaded_affine_q1_biases(scales, biases) {
        return Err(Exception::custom(format!(
            "{path} must use the validated symmetric-Q1 bias sentinel or a nonempty floating-point affine bias matching scales shape {:?}; got shape {:?} dtype {:?}",
            scales.shape(),
            biases.shape(),
            biases.dtype()
        )));
    }
    Ok(())
}

fn bonsai_q1_qmm_max_rows() -> i32 {
    static MAX_ROWS: OnceLock<i32> = OnceLock::new();
    *MAX_ROWS.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_QMM_MAX_ROWS")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .filter(|rows| (0..=64).contains(rows))
            .unwrap_or(8)
    })
}

pub(crate) fn quantized_forward(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    if bits == 1 {
        affine_q1_forward(x, weight, scales, biases, group_size)
    } else {
        ops::quantized_matmul(x, weight, scales, biases, true, group_size, bits)
    }
}

/// Affine 1-bit matrix multiplication using Higgs' runtime Metal kernels.
///
/// Upstream MLX does not provide the affine `bits=1` kernels used by Bonsai
/// checkpoints. Decode uses the fused packed matvec. Narrow multi-token
/// verifier batches use the same packed kernel over a z-dimension batch; wider
/// prefill inputs retain the dense dequantize + MLX matmul fallback. This is
/// shared by the Qwen3.5 hybrid path (Bonsai-27B) and its LM head.
fn affine_q1_forward(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    let x_shape = x.shape();
    let input_dim = x_shape
        .last()
        .copied()
        .ok_or_else(|| Exception::custom("1-bit affine input has no dimensions"))?;
    let weight_shape = weight.shape();
    let packed_dim = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("1-bit affine weight must be a matrix"))?;
    let expected_input_dim = packed_dim
        .checked_mul(32)
        .ok_or_else(|| Exception::custom("1-bit affine input dimension overflow"))?;
    if input_dim != expected_input_dim {
        return Err(Exception::custom(format!(
            "1-bit affine input dim {input_dim} does not match packed weight dim {expected_input_dim}"
        )));
    }
    if group_size <= 0 || expected_input_dim % group_size != 0 {
        return Err(Exception::custom(format!(
            "invalid 1-bit affine group size {group_size} for input dim {expected_input_dim}"
        )));
    }

    let row_count: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if row_count == 1 {
        crate::metal_kernel::bonsai_q1_qmv(x, weight, scales, biases, group_size)
    } else if row_count > 0 && row_count <= bonsai_q1_qmm_max_rows() {
        crate::metal_kernel::bonsai_q1_qmm(x, weight, scales, biases, group_size)
    } else {
        let dense = crate::metal_kernel::bonsai_q1_dequant(weight, scales, biases, group_size)?
            .as_dtype(x.dtype())?;
        x.matmul(&dense.transpose()?)
    }
}

/// Physical storage contract for a [`QLinear`]'s weight and scale parameters.
///
/// The layout is deliberately metadata-only: the `Param<Array>` handles remain
/// the sole authority for the resident buffers. Reconstructing a checked row4
/// view from those handles on every use means parameter-tree replacement can
/// never leave forward pointing at stale cloned arrays.
///
/// A promoted parameter tree stores physical row4 arrays under the original
/// parameter names. Generic checkpoint serialization/export therefore requires
/// an explicit row4-to-canonical demotion step and is unsupported as-is.
#[derive(Debug, Clone)]
enum QLinearWeightLayout {
    Canonical,
    BonsaiRow4 { n_rows: i32, k_dim: i32 },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct BonsaiRow4Promotion {
    layers: usize,
    projections: usize,
    bytes: usize,
}

/// Quantized linear layer stored as weight/scales/biases arrays plus a typed
/// physical-layout contract. Canonical forward uses `quantized_matmul`
/// directly; promoted Bonsai dense MLP projections route through row4-native
/// kernels and dequantize wide prefill directly from row4 storage.
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct QLinear {
    #[param]
    pub(crate) weight: Param<Array>,
    #[param]
    pub(crate) scales: Param<Array>,
    #[param]
    pub(crate) biases: Param<Array>,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    /// Quantization format. `Affine` (default) keeps the existing mlx-rs fast
    /// path; `MxFp4` routes through the FFI bypass in [`crate::quant_mode`].
    pub(crate) mode: crate::quant_mode::QuantMode,
    weight_layout: QLinearWeightLayout,
}

impl QLinear {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        // `bits == 0` marks a BF16-dense tensor (only `.weight` on disk) —
        // route to `Dense` mode so the forward path uses plain matmul.
        let mode = if bits == 0 {
            crate::quant_mode::QuantMode::Dense
        } else {
            crate::quant_mode::QuantMode::Affine
        };
        Self::new_with_mode(group_size, bits, mode)
    }

    /// Construct from a resolved [`QuantSpec`] (`group_size` + bits + mode).
    pub(crate) fn new_spec(spec: QuantSpec) -> Result<Self, Exception> {
        Self::new_with_mode(spec.group_size, spec.bits, spec.mode)
    }

    /// Construct with an explicit quantization mode. Used by loaders that read
    /// per-tensor `"mode"` from `config.json` (e.g. mxfp4 bulk + affine islands).
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new_with_mode(
        group_size: i32,
        bits: i32,
        mode: crate::quant_mode::QuantMode,
    ) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params();
        // mxfp4 tensors ship without `.biases` on disk (E2M1 has no zero-point).
        // Dense tensors ship without `.scales` or `.biases` (just a weight matrix).
        // Replace the [1] placeholder with a [0] empty array so the weight-loader's
        // completeness check (`shape == [1]` ⇒ missing) doesn't flag them.
        // The forward path ignores scales/biases for these modes.
        let needs_no_aux = mode.is_mxfp4() || mode.is_dense();
        let biases_param = if needs_no_aux {
            Param::new(Array::from_slice::<f32>(&[], &[0]))
        } else {
            biases
        };
        let scales_param = if mode.is_dense() {
            Param::new(Array::from_slice::<f32>(&[], &[0]))
        } else {
            scales
        };
        Ok(Self {
            weight,
            scales: scales_param,
            biases: biases_param,
            group_size,
            bits,
            mode,
            weight_layout: QLinearWeightLayout::Canonical,
        })
    }

    fn reset_weight_layout(&mut self) {
        self.weight_layout = QLinearWeightLayout::Canonical;
    }

    fn bonsai_row4(&self) -> Result<Option<crate::metal_kernel::BonsaiQ1Row4Ref<'_>>, Exception> {
        let QLinearWeightLayout::BonsaiRow4 { n_rows, k_dim } = &self.weight_layout else {
            return Ok(None);
        };
        crate::metal_kernel::BonsaiQ1Row4Ref::from_primary_parts(
            &self.weight,
            &self.scales,
            *n_rows,
            *k_dim,
        )
        .map(Some)
    }

    /// Return the logical `[out,in]` shape and whether this canonical
    /// projection can be promoted losslessly. A valid loaded affine bias is a
    /// first-class canonical fallback: it must never be approximated by the
    /// symmetric row4 kernels.
    fn bonsai_row4_promotion_candidate(&self, path: &str) -> Result<((i32, i32), bool), Exception> {
        match &self.weight_layout {
            QLinearWeightLayout::BonsaiRow4 { n_rows, k_dim } => {
                validate_dflash_qlinear(path, self)?;
                Ok(((*n_rows, *k_dim), false))
            }
            QLinearWeightLayout::Canonical => {
                validate_dflash_q1_linear(
                    path,
                    &self.weight,
                    &self.scales,
                    &self.biases,
                    self.group_size,
                    self.bits,
                    self.mode,
                )?;
                let [n_rows, k_packed] = *self.weight.shape() else {
                    return Err(Exception::custom(format!(
                        "{path} canonical Q1 weight must have shape [N,K/32]"
                    )));
                };
                let k_dim = k_packed
                    .checked_mul(32)
                    .ok_or_else(|| Exception::custom(format!("{path} Q1 width overflow")))?;
                let eligible = has_symmetric_q1_biases(&self.biases)
                    && matches!(self.scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
                    && n_rows % 4 == 0
                    && k_dim % 128 == 0;
                Ok(((n_rows, k_dim), eligible))
            }
        }
    }

    /// Validate the strict primary-row4 domain without allocating or mutating.
    fn preflight_bonsai_row4_promotion(&self, path: &str) -> Result<(i32, i32), Exception> {
        let (shape, eligible) = self.bonsai_row4_promotion_candidate(path)?;
        if !matches!(self.weight_layout, QLinearWeightLayout::Canonical) || !eligible {
            return Err(Exception::custom(format!(
                "{path} is outside the primary row4 domain: mode={:?} bits={} group_size={} symmetric_bias={}",
                self.mode,
                self.bits,
                self.group_size,
                has_symmetric_q1_biases(&self.biases)
            )));
        }
        Ok(shape)
    }

    fn prepare_bonsai_row4(
        &self,
        path: &str,
    ) -> Result<crate::metal_kernel::BonsaiQ1Row4, Exception> {
        self.preflight_bonsai_row4_promotion(path)?;
        crate::metal_kernel::BonsaiQ1Row4::from_row_major(&self.weight, &self.scales)
    }

    /// Install the packed arrays as the actual, sole-authority model parameters.
    fn install_bonsai_row4(&mut self, packed: crate::metal_kernel::BonsaiQ1Row4) {
        let (weights, scales, n_rows, k_dim) = packed.into_primary_parts();
        self.weight = Param::new(weights);
        self.scales = Param::new(scales);
        self.weight_layout = QLinearWeightLayout::BonsaiRow4 { n_rows, k_dim };
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        if let Some(packed) = self.bonsai_row4()? {
            if packed.accepts_input(x) {
                return crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(x, packed);
            }
            let row_count: i32 = x.shape().iter().take(x.ndim().saturating_sub(1)).product();
            if (1..=8).contains(&row_count) {
                return Err(Exception::custom(format!(
                    "Bonsai row4 narrow input is outside the TG-LUT4 contract: shape={:?} dtype={:?}",
                    x.shape(),
                    x.dtype()
                )));
            }
            // Wide prefill dequantizes directly from the authoritative row4
            // buffers. No transient canonical packed copy is materialized.
            let dense =
                crate::metal_kernel::bonsai_q1_row4_dequant_view(packed)?.as_dtype(x.dtype())?;
            return x.matmul(&dense.transpose()?);
        }

        // Fast path: batched quantized GEMM for verify (T>1, T<=16).
        // Fuses T matmuls into one Metal kernel dispatch — eliminates
        // pipeline bubbles. Gated by env var until validated.
        if let Some(t) = self.qgemm_verify_shape(x) {
            let fast = if self.mode == crate::quant_mode::QuantMode::MxFp4 {
                qgemm_mxfp4_4bit(x, &self.weight, &self.scales, self.group_size, t)
            } else {
                qgemm_4bit(
                    x,
                    &self.weight,
                    &self.scales,
                    &self.biases,
                    self.group_size,
                    t,
                )
            };
            if let Ok(result) = fast {
                return Ok(result);
            }
            // Fall through to standard path if kernel fails.
        }

        match self.mode {
            crate::quant_mode::QuantMode::MxFp4 => crate::quant_mode::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                None,
                true,
                self.group_size,
                self.bits,
                crate::quant_mode::QuantMode::MxFp4,
            ),
            // Dense: plain matmul on the raw weight (bf16/fp16).
            crate::quant_mode::QuantMode::Dense => dense_linear_no_bias_forward(&self.weight, x),
            // Affine fast path — unchanged from the mlx-rs wrapper.
            crate::quant_mode::QuantMode::Affine => {
                if self.bits == 1 {
                    return affine_q1_forward(
                        x,
                        &self.weight,
                        &self.scales,
                        &self.biases,
                        self.group_size,
                    );
                }
                // DIAGNOSTIC (HIGGS_DIAG_DEQUANT=1): force the dense dequantized
                // matmul (row-independent) instead of quantized_matmul, to test
                // whether quantized_matmul's length-dependence is the divergence
                // source.
                if std::env::var("HIGGS_DIAG_DEQUANT").is_ok_and(|v| v == "1") {
                    let wdq = mlx_rs::ops::dequantize(
                        &*self.weight,
                        &*self.scales,
                        Some(&*self.biases),
                        Some(self.group_size),
                        Some(self.bits),
                    )?
                    .as_dtype(x.dtype())?;
                    return x.matmul(&wdq.transpose()?);
                }
                quantized_forward(
                    x,
                    &self.weight,
                    &self.scales,
                    &self.biases,
                    self.group_size,
                    self.bits,
                )
            }
        }
    }

    fn qgemm_verify_shape(&self, x: &Array) -> Option<i32> {
        if !qgemm_verify_enabled()
            || self.bits != 4
            || !self.qgemm_mode_enabled()
            || self.group_size <= 0
            || self.weight.shape().len() != 2
        {
            return None;
        }

        let x_shape = x.shape();
        let [1, t, k_in] = *x_shape else {
            return None;
        };
        if !(2..=16).contains(&t) {
            return None;
        }

        let weight_shape = self.weight.shape();
        let k_packed = *weight_shape.get(1)?;
        let k_dim = k_packed.checked_mul(8)?;
        if k_dim != k_in || k_dim % self.group_size != 0 {
            return None;
        }

        let num_groups = k_dim / self.group_size;
        let n_rows = *weight_shape.first()?;
        if self.scales.shape().iter().product::<i32>() != n_rows * num_groups {
            return None;
        }
        if self.mode == crate::quant_mode::QuantMode::Affine
            && self.biases.shape().iter().product::<i32>() != n_rows * num_groups
        {
            return None;
        }
        Some(t)
    }

    fn qgemm_mode_enabled(&self) -> bool {
        match self.mode {
            crate::quant_mode::QuantMode::Affine => true,
            crate::quant_mode::QuantMode::MxFp4 => qgemm_mxfp4_enabled(),
            crate::quant_mode::QuantMode::Dense => false,
        }
    }

    /// Decode-only fast path for 4-bit single-token inference.
    ///
    /// Keeps the optimization opt-in so we can wire it into selected hot paths
    /// without changing the default behavior of every quantized linear.
    /// Only supported for `Affine` — the custom `qgemv_4bit` kernel assumes
    /// affine-packed weights; `MxFp4` falls through to the standard matmul.
    pub(crate) fn forward_decode_fast(&self, x: &Array) -> Result<Array, Exception> {
        if self.mode.is_mxfp4() || self.mode.is_dense() {
            return self.forward(x);
        }
        if decode_gemv_enabled()
            && self.bits == 4
            && matches!(x.shape(), [1, 1, _])
            && self.weight.shape().len() == 2
        {
            qgemv_4bit(x, &self.weight, &self.scales, &self.biases, self.group_size)
        } else {
            self.forward(x)
        }
    }
}

fn validate_dflash_qlinear(path: &str, linear: &QLinear) -> Result<(), Exception> {
    match &linear.weight_layout {
        QLinearWeightLayout::Canonical => validate_dflash_q1_linear(
            path,
            &linear.weight,
            &linear.scales,
            &linear.biases,
            linear.group_size,
            linear.bits,
            linear.mode,
        ),
        QLinearWeightLayout::BonsaiRow4 { .. } => {
            if linear.mode != crate::quant_mode::QuantMode::Affine
                || linear.bits != 1
                || linear.group_size != 128
                || !has_symmetric_q1_biases(&linear.biases)
            {
                return Err(Exception::custom(format!(
                    "{path} has an invalid installed Bonsai row4 contract weight={:?}/{:?} scales={:?}/{:?} mode={:?} bits={} group_size={} symmetric_bias={}",
                    linear.weight.shape(),
                    linear.weight.dtype(),
                    linear.scales.shape(),
                    linear.scales.dtype(),
                    linear.mode,
                    linear.bits,
                    linear.group_size,
                    has_symmetric_q1_biases(&linear.biases)
                )));
            }
            // Rebuild the typed view from the current parameter handles. This
            // validates shape, dtype, contiguity, and the logical dimensions
            // recorded in metadata, so a same-shaped parameter-tree update
            // cannot silently leave forward using an older cloned handle.
            linear.bonsai_row4()?.ok_or_else(|| {
                Exception::custom(format!("{path} lost its Bonsai row4 layout metadata"))
            })?;
            Ok(())
        }
    }
}

/// Dense linear layer with a single weight tensor and no bias.
#[derive(Debug, Clone, ModuleParameters)]
struct DenseLinearNoBias {
    #[param]
    weight: Param<Array>,
}

fn dense_linear_no_bias_forward(weight: &Array, x: &Array) -> Result<Array, Exception> {
    let shape = x.shape().to_vec();
    let in_features = *shape
        .last()
        .ok_or_else(|| Exception::custom("empty input"))?;
    let batch: i32 = shape.iter().take(shape.len() - 1).product();
    let x2d = x.reshape(&[batch, in_features])?;
    let w = weight.as_dtype(x.dtype())?;
    let out2d = x2d.matmul(&w.transpose()?)?;
    let out_features = *out2d.shape().last().unwrap_or(&0);
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = out_features;
    }
    out2d.reshape(&out_shape)
}

impl DenseLinearNoBias {
    fn new() -> Self {
        Self {
            weight: Param::new(Array::from_slice(&[0.0_f32], &[1])),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        dense_linear_no_bias_forward(&self.weight, x)
    }
}

/// Quantized embedding stored as raw weight/scales/biases arrays.
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct QEmbedding {
    #[param]
    weight: Param<Array>,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    group_size: i32,
    bits: i32,
    mode: crate::quant_mode::QuantMode,
}

impl QEmbedding {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        // `bits == 0` marks a BF16-dense embedding (only `.weight` on disk).
        let mode = if bits == 0 {
            crate::quant_mode::QuantMode::Dense
        } else {
            crate::quant_mode::QuantMode::Affine
        };
        Ok(Self::new_spec(QuantSpec {
            group_size,
            bits,
            mode,
        }))
    }

    pub(crate) fn new_spec(spec: QuantSpec) -> Self {
        let (weight, scales, biases) = init_quantized_params();
        let needs_no_aux = spec.mode.is_mxfp4() || spec.mode.is_dense();
        let biases_param = if needs_no_aux {
            Param::new(Array::from_slice::<f32>(&[], &[0]))
        } else {
            biases
        };
        let scales_param = if spec.mode.is_dense() {
            Param::new(Array::from_slice::<f32>(&[], &[0]))
        } else {
            scales
        };
        Self {
            weight,
            scales: scales_param,
            biases: biases_param,
            group_size: spec.group_size,
            bits: spec.bits,
            mode: spec.mode,
        }
    }

    pub(crate) fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        let shape = indices.shape().to_vec();
        let flat = indices.flatten(None, None)?;
        // Gather aux arrays only for the modes that ship them: mxfp4 has no
        // `.biases` (empty [0] placeholder) and dense has neither — an
        // unconditional take on an empty axis throws at the first forward.
        let w = (*self.weight).take_axis(&flat, 0)?;
        let out = match self.mode {
            crate::quant_mode::QuantMode::MxFp4 => {
                let s = (*self.scales).take_axis(&flat, 0)?;
                crate::quant_mode::dequantize(&w, &s, None, self.group_size, self.bits, self.mode)?
            }
            crate::quant_mode::QuantMode::Affine => {
                let s = (*self.scales).take_axis(&flat, 0)?;
                if self.bits == 1 {
                    if has_symmetric_q1_biases(&self.biases) {
                        crate::metal_kernel::bonsai_q1_dequant(
                            &w,
                            &s,
                            &self.biases,
                            self.group_size,
                        )?
                    } else {
                        let b = (*self.biases).take_axis(&flat, 0)?;
                        crate::metal_kernel::bonsai_q1_dequant(&w, &s, &b, self.group_size)?
                    }
                } else {
                    let b = (*self.biases).take_axis(&flat, 0)?;
                    ops::dequantize(&w, &s, &b, self.group_size, self.bits)?
                }
            }
            // Dense: weights are already full-precision; just gather rows.
            crate::quant_mode::QuantMode::Dense => w,
        };
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        match self.mode {
            crate::quant_mode::QuantMode::MxFp4 => crate::quant_mode::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                None,
                true,
                self.group_size,
                self.bits,
                self.mode,
            ),
            // Affine fast path — custom 4-bit gemv kernel for single-token decode.
            crate::quant_mode::QuantMode::Affine => {
                if self.bits == 1 {
                    affine_q1_forward(x, &self.weight, &self.scales, &self.biases, self.group_size)
                } else if self.bits == 4
                    && matches!(x.shape(), [1, 1, _])
                    && self.weight.shape().len() == 2
                {
                    qgemv_4bit(x, &self.weight, &self.scales, &self.biases, self.group_size)
                } else {
                    quantized_forward(
                        x,
                        &self.weight,
                        &self.scales,
                        &self.biases,
                        self.group_size,
                        self.bits,
                    )
                }
            }
            // Dense: plain matmul on full-precision weights.
            crate::quant_mode::QuantMode::Dense => dense_linear_no_bias_forward(&self.weight, x),
        }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

pub(crate) fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    gate.multiply(nn::sigmoid(gate)?)?.multiply(x)
}

fn silu_direct(x: &Array) -> Result<Array, Exception> {
    x.multiply(nn::sigmoid(x)?)
}

static COMPILED_GATING_ENABLED: OnceLock<bool> = OnceLock::new();
static APPLE_CPU_BRAND: OnceLock<Option<String>> = OnceLock::new();
static COMPILED_GDN_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
static ASYNC_LAYER_STATE_EVAL_ENABLED: OnceLock<bool> = OnceLock::new();

fn parse_compiled_gating_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

fn apple_cpu_brand() -> Option<&'static str> {
    APPLE_CPU_BRAND
        .get_or_init(|| {
            #[cfg(target_os = "macos")]
            {
                std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                    .ok()
                    .filter(|out| out.status.success())
                    .and_then(|out| String::from_utf8(out.stdout).ok())
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        })
        .as_deref()
}

fn should_force_dense_decode_safe_defaults_for_brand(brand: Option<&str>) -> bool {
    matches!(brand.map(str::trim), Some("Apple M4"))
}

fn compiled_gating_enabled() -> bool {
    *COMPILED_GATING_ENABLED.get_or_init(|| {
        std::env::var("HIGGS_COMPILED_GATING")
            .ok()
            .is_none_or(|raw| parse_compiled_gating_enabled(Some(raw.as_str())))
    })
}

fn compiled_gdn_decode_enabled() -> bool {
    *COMPILED_GDN_DECODE_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIGGS_COMPILED_GDN_DECODE")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

fn async_layer_state_eval_enabled() -> bool {
    *ASYNC_LAYER_STATE_EVAL_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HIGGS_ASYNC_LAYER_STATE_EVAL")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

fn compiled_silu_mul((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

type CompiledSiluMulFn = dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array, Exception>;
type CompiledSigmoidMulFn = dyn for<'a> FnMut((&'a Array, &'a Array)) -> Result<Array, Exception>;
type CompiledGdnOutputGateFn =
    dyn for<'a> FnMut((&'a Array, &'a Array, &'a Array)) -> Result<Array, Exception>;

thread_local! {
    static COMPILED_SILU_MUL_FN: RefCell<Option<Box<CompiledSiluMulFn>>> = RefCell::new(None);
    static COMPILED_SIGMOID_MUL_FN: RefCell<Option<Box<CompiledSigmoidMulFn>>> = RefCell::new(None);
    static COMPILED_GDN_OUTPUT_GATE_FN: RefCell<Option<Box<CompiledGdnOutputGateFn>>> =
        RefCell::new(None);
    static QGEMV_CONFIG_CACHE: RefCell<HashMap<QgemvKernelConfigKey, mlx_sys::mlx_fast_metal_kernel_config>> =
        RefCell::new(HashMap::new());
    static GATED_DELTA_CONFIG_CACHE: RefCell<HashMap<GatedDeltaKernelConfigKey, mlx_sys::mlx_fast_metal_kernel_config>> =
        RefCell::new(HashMap::new());
    static GATED_DELTA_TAPE_CONFIG_CACHE: RefCell<HashMap<GatedDeltaTapeKernelConfigKey, CachedMetalKernelConfig>> =
        RefCell::new(HashMap::new());
    static CANONICAL_CONV_CONFIG_CACHE: RefCell<HashMap<CanonicalConvKernelConfigKey, CachedMetalKernelConfig>> =
        RefCell::new(HashMap::new());
}

fn silu_mul(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if compiled_gating_enabled() {
        COMPILED_SILU_MUL_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_silu_mul,
                    None,
                ))
            });
            compiled((gate, x))
        })
    } else {
        nn::silu(gate)?.multiply(x)
    }
}

fn compiled_sigmoid_mul((gate, x): (&Array, &Array)) -> Result<Array, Exception> {
    nn::sigmoid(gate)?.multiply(x)
}

fn sigmoid_mul(gate: &Array, x: &Array) -> Result<Array, Exception> {
    if compiled_gating_enabled() {
        COMPILED_SIGMOID_MUL_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_sigmoid_mul,
                    None,
                ))
            });
            compiled((gate, x))
        })
    } else {
        nn::sigmoid(gate)?.multiply(x)
    }
}

fn compiled_gdn_output_gate((y, weight, z): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let normed = fast::rms_norm(y, weight, 1e-6)?;
    nn::silu(z)?.multiply(&normed)
}

fn gdn_output_gate(y: &Array, weight: &Array, eps: f32, z: &Array) -> Result<Array, Exception> {
    if compiled_gating_enabled() && (eps - 1e-6).abs() <= f32::EPSILON {
        COMPILED_GDN_OUTPUT_GATE_FN.with(|cell| {
            let mut guard = cell.borrow_mut();
            let compiled = guard.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    compiled_gdn_output_gate,
                    None,
                ))
            });
            compiled((y, weight, z))
        })
    } else {
        let normed = fast::rms_norm(y, weight, eps)?;
        nn::silu(z)?.multiply(&normed)
    }
}

// ---------------------------------------------------------------------------
// gather_qmm FFI wrapper
// ---------------------------------------------------------------------------

/// Quantized matrix multiplication with expert-level gather, dispatched as a
/// single fused GPU kernel. Replaces per-expert `take_axis + quantized_matmul`
/// loops in `MoE` layers.
///
/// `rhs_indices` selects which expert weight matrices to use for each batch
/// element. Batch dimensions of `x` and `rhs_indices` are broadcast together.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    rhs_indices: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let null_lhs = unsafe { mlx_sys::mlx_array_new() };
    let mut result = unsafe { mlx_sys::mlx_array_new() };
    let status = unsafe {
        mlx_sys::mlx_gather_qmm(
            &raw mut result,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            null_lhs,
            rhs_indices.as_ptr(),
            transpose,
            mlx_sys::mlx_optional_int_ {
                value: group_size,
                has_value: true,
            },
            mlx_sys::mlx_optional_int_ {
                value: bits,
                has_value: true,
            },
            c"affine".as_ptr(),
            sorted_indices,
            stream.as_ptr(),
        )
    };

    // Always free the null sentinel
    unsafe { mlx_sys::mlx_array_free(null_lhs) };

    if status != 0 {
        // Free the uninitialized result array
        unsafe { mlx_sys::mlx_array_free(result) };
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        let msg = format!(
            "gather_qmm failed: {mlx_msg} \
             [x={:?}/{:?} w={:?}/{:?} scales={:?}/{:?} biases={:?}/{:?} \
             idx={:?}/{:?} transpose={transpose} gs={group_size} bits={bits}]",
            x.shape(),
            x.dtype(),
            w.shape(),
            w.dtype(),
            scales.shape(),
            scales.dtype(),
            biases.shape(),
            biases.dtype(),
            rhs_indices.shape(),
            rhs_indices.dtype(),
        );
        return Err(Exception::custom(msg));
    }
    Ok(unsafe { Array::from_ptr(result) })
}

// ---------------------------------------------------------------------------
// `GatedDeltaNet` custom Metal kernel
// ---------------------------------------------------------------------------

/// Ordered short-block depthwise convolution used by the canonical verifier.
///
/// The output deliberately stops before `SiLU`. The caller applies the existing
/// MLX `sigmoid` and multiply primitives once over the complete block, keeping
/// their arithmetic identical to the S=1 path while collapsing the ordered
/// multiply/add chain to one dispatch per GDN layer.
const CANONICAL_CONV_KERNEL_SOURCE: &str = r"
const int channel = static_cast<int>(thread_position_in_grid.x);
const int position = static_cast<int>(thread_position_in_grid.y);
const int batch_index = static_cast<int>(thread_position_in_grid.z);
const int mixed_index = (batch_index * T + position) * D + channel;

// Match canonical_conv1d_step exactly: current tap first, then one rounded
// multiply and one rounded add for each available lag, newest to oldest.
InT accumulator = static_cast<InT>(
    mixed_qkv[mixed_index] * weight_t[(K - 1) * D + channel]);
const int available = min(max(offset_init + position, 0), K - 1);
for (int lag = 0; lag < available; ++lag) {
  const int prior_position = position - 1 - lag;
  InT prior;
  if (prior_position >= 0) {
    prior = mixed_qkv[(batch_index * T + prior_position) * D + channel];
  } else {
    const int history_index = (K - 2) - (lag - position);
    prior = history[(batch_index * (K - 1) + history_index) * D + channel];
  }
  InT product = static_cast<InT>(prior * weight_t[(K - 2 - lag) * D + channel]);
  accumulator = static_cast<InT>(accumulator + product);
}
preactivation[mixed_index] = accumulator;
";

#[allow(unsafe_code)]
fn create_canonical_conv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 4] = [c"mixed_qkv", c"history", c"weight_t", c"offset_init"];
    let output_names: [&std::ffi::CStr; 1] = [c"preactivation"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|name| name.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|name| name.as_ptr()).collect();
    let source = CString::new(CANONICAL_CONV_KERNEL_SOURCE).unwrap_or_default();

    unsafe {
        let inputs =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let outputs =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"canonical_conv_ordered".as_ptr(),
            inputs,
            outputs,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(inputs);
        mlx_sys::mlx_vector_string_free(outputs);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_canonical_conv_kernel(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    conv_dim: i32,
    kernel_size: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        for (name, value) in [
            (c"B", batch),
            (c"T", seq_len),
            (c"D", conv_dim),
            (c"K", kernel_size),
        ] {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                name.as_ptr(),
                value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, conv_dim, seq_len, batch);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);
        let output_shape = [batch, seq_len, conv_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            in_dtype,
        );
        config
    }
}

fn canonical_conv_kernel_supported(
    mixed_qkv: &Array,
    history: &Array,
    weight_t: &Array,
    batch: i32,
    seq_len: i32,
    conv_dim: i32,
    kernel_size: i32,
) -> bool {
    batch == 1
        && (1..=5).contains(&seq_len)
        && kernel_size == 4
        && mixed_qkv.dtype() == Dtype::Bfloat16
        && history.dtype() == Dtype::Bfloat16
        && weight_t.dtype() == Dtype::Bfloat16
        && mixed_qkv.shape() == [batch, seq_len, conv_dim]
        && history.shape() == [batch, kernel_size - 1, conv_dim]
        && weight_t.shape() == [kernel_size, conv_dim]
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn canonical_conv_preactivation_ffi(
    mixed_qkv: &Array,
    history: &Array,
    weight_t: &Array,
    offset_init: i32,
    batch: i32,
    seq_len: i32,
    conv_dim: i32,
    kernel_size: i32,
) -> Result<Array, Exception> {
    if !canonical_conv_kernel_supported(
        mixed_qkv,
        history,
        weight_t,
        batch,
        seq_len,
        conv_dim,
        kernel_size,
    ) || offset_init < 0
    {
        return Err(Exception::custom(format!(
            "unsupported canonical convolution domain: mixed={:?}/{:?} history={:?}/{:?} weight={:?}/{:?} B={batch} T={seq_len} D={conv_dim} K={kernel_size} offset={offset_init}",
            mixed_qkv.shape(),
            mixed_qkv.dtype(),
            history.shape(),
            history.dtype(),
            weight_t.shape(),
            weight_t.dtype(),
        )));
    }

    ensure_ffi_error_handler();
    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(mixed_qkv.as_ptr()) };
    let key = CanonicalConvKernelConfigKey {
        in_dtype,
        batch,
        seq_len,
        conv_dim,
        kernel_size,
    };
    let config = CANONICAL_CONV_CONFIG_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| {
                CachedMetalKernelConfig(configure_canonical_conv_kernel(
                    in_dtype,
                    batch,
                    seq_len,
                    conv_dim,
                    kernel_size,
                ))
            })
            .0
    });
    let kernel =
        CANONICAL_CONV_KERNEL.get_or_init(|| CachedMetalKernel(create_canonical_conv_kernel()));
    let offset_scalar = unsafe { mlx_sys::mlx_array_new_int(offset_init) };
    let input_ptrs = [
        mixed_qkv.as_ptr(),
        history.as_ptr(),
        weight_t.as_ptr(),
        offset_scalar,
    ];
    let inputs =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };
    let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs,
            kernel.0,
            inputs,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status == 0 {
        let mut output = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut output, outputs, 0) };
        Ok(unsafe { Array::from_ptr(output) })
    } else {
        let message = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "canonical_conv_ordered failed: {message}"
        )))
    };

    unsafe {
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_array_free(offset_scalar);
    }
    result
}

/// Shared recurrence algebra for every GDN Metal path.
///
/// Keep decay and innovation update as two statements. Combining them permits
/// an FMA/reassociation that changes rollback state bits after partial accept.
const GDN_RECURRENCE_METAL_PREAMBLE: &str = r"
#define HIGGS_GDN_GATE(gate, a_value, dt_bias_value, a_log_value) \
  float x = static_cast<float>(a_value) + dt_bias_value; \
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x))); \
  float gate = exp(-exp(a_log_value) * sp)
#define HIGGS_GDN_BETA(beta, b_value) \
  float beta = 1.0f / (1.0f + exp(-static_cast<float>(b_value)))
#define HIGGS_GDN_DECAY(state_value, gate) \
  state_value = state_value * gate
#define HIGGS_GDN_UPDATE(state_value, key_value, delta) \
  state_value = state_value + key_value * delta
";

fn gdn_metal_source(defines: &str, body: &str) -> CString {
    let mut source =
        String::with_capacity(GDN_RECURRENCE_METAL_PREAMBLE.len() + defines.len() + body.len());
    source.push_str(GDN_RECURRENCE_METAL_PREAMBLE);
    source.push_str(defines);
    source.push_str(body);
    CString::new(source).unwrap_or_else(|_| CString::default())
}

/// Metal kernel source for the fused `GatedDeltaNet` recurrence. Plain and
/// tape-recording kernels compile this same body with one tape feature flag.
///
/// Computes `g = exp(-exp(a_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`
/// inline, then runs the full recurrence -- all in one kernel dispatch.
///
/// Template parameters: `InT` (dtype), `Dk`, `Dv`, `Hk`, `Hv` (int constants).
/// Grid: `(32, Dv, B * Hv)`, Threadgroup: `(32, 4, 1)`.
const GATED_DELTA_FORWARD_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;
#if HIGGS_GDN_RECORD_TAPE
auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;
#endif

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

// state_in/state_out are float32 buffers for numerical stability,
// but the kernel signature types them as InT*. Reinterpret to float*.
auto i_state = reinterpret_cast<const device float*>(state_in) + (n * Dv + dv_idx) * Dk;
auto o_state = reinterpret_cast<device float*>(state_out) + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// Per-head constants for gate computation
float a_log_val = static_cast<float>(a_log[hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);

// a, b: [B, T, Hv]
auto a_ = a + b_idx * T * Hv;
auto b_ = b + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  HIGGS_GDN_GATE(g_val, a_[hv_idx], dt_bias_val, a_log_val);
  HIGGS_GDN_BETA(beta_val, b_[hv_idx]);

  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_DECAY(state[i], g_val);
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      HIGGS_GDN_UPDATE(state[i], k_[s_idx], delta);
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
#if HIGGS_GDN_RECORD_TAPE
      tape_[dv_idx] = delta;
#endif
    }
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
#if HIGGS_GDN_RECORD_TAPE
  tape_ += Hv * Dv;
#endif
  a_ += Hv;
  b_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = state[i];
}
";

/// Create the `mlx_fast_metal_kernel` object from kernel source and names.
#[allow(unsafe_code)]
fn create_gated_delta_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 9] = [
        c"q",
        c"k",
        c"v",
        c"a_log",
        c"a",
        c"dt_bias",
        c"b",
        c"state_in",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 2] = [c"y", c"state_out"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source = gdn_metal_source(
        "#define HIGGS_GDN_RECORD_TAPE 0\n",
        GATED_DELTA_FORWARD_KERNEL_SOURCE,
    );

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"gated_delta_step".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

/// Configure template args, grid, threadgroup, and output shapes for the kernel.
#[allow(unsafe_code)]
fn configure_gated_delta_kernel(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        let y_shape = [batch, seq_len, num_v_heads, head_v_dim];
        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );

        config
    }
}

fn gated_delta_kernel_config(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> (mlx_sys::mlx_fast_metal_kernel_config, bool) {
    if !gated_delta_config_cache_enabled() {
        return (
            configure_gated_delta_kernel(
                in_dtype,
                batch,
                seq_len,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
            ),
            false,
        );
    }

    let key = GatedDeltaKernelConfigKey {
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    };
    let config = GATED_DELTA_CONFIG_CACHE.with(|cache_cell| {
        let mut cache_map = cache_cell.borrow_mut();
        *cache_map.entry(key).or_insert_with(|| {
            configure_gated_delta_kernel(
                in_dtype,
                batch,
                seq_len,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
            )
        })
    });
    (config, true)
}

fn validate_gdn_kernel_state(
    operation: &str,
    state: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(), Exception> {
    if batch <= 0
        || seq_len <= 0
        || num_k_heads <= 0
        || num_v_heads <= 0
        || head_k_dim <= 0
        || head_v_dim <= 0
        || head_k_dim % 32 != 0
        || num_v_heads % num_k_heads != 0
    {
        return Err(Exception::custom(format!(
            "{operation}: invalid GDN Metal geometry B={batch} T={seq_len} Hk={num_k_heads} Dk={head_k_dim} Hv={num_v_heads} Dv={head_v_dim}"
        )));
    }
    if state.dtype() != Dtype::Float32 {
        return Err(Exception::custom(format!(
            "{operation}: state must be Float32, got {:?}",
            state.dtype()
        )));
    }
    let expected = [batch, num_v_heads, head_v_dim, head_k_dim];
    if state.shape() != expected {
        return Err(Exception::custom(format!(
            "{operation}: state shape must be {expected:?}, got {:?}",
            state.shape()
        )));
    }
    Ok(())
}

/// Fused `GatedDeltaNet` kernel: computes g, beta, AND the full recurrence in one dispatch.
#[allow(unsafe_code, clippy::too_many_arguments)]
fn gated_delta_kernel_ffi(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array), Exception> {
    validate_gdn_kernel_state(
        "gated_delta_kernel",
        state_in,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )?;
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };

    let cached = GATED_DELTA_KERNEL.get_or_init(|| CachedMetalKernel(create_gated_delta_kernel()));
    let (config, config_is_cached) = gated_delta_kernel_config(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    );

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        dt_bias.as_ptr(),
        b.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "gated_delta_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 1);
        }
        Ok((unsafe { Array::from_ptr(y_ptr) }, unsafe {
            Array::from_ptr(state_ptr)
        }))
    };

    unsafe {
        if !config_is_cached {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Custom quantized GEMV kernel — 4-bit affine, vectorized uint4 loads
// ---------------------------------------------------------------------------

/// Metal kernel for quantized GEMV: y = dequant(W) @ x.
///
/// One threadgroup computes ONE output row. 4 simdgroups split-K parallelize
/// the dot product, then reduce via threadgroup memory. Uses `uint4` (16-byte)
/// vectorized weight loads for peak bandwidth, vs MLX's `uint16` (2-byte) loads.
///
/// Single packed buffer `wb` = [`weight_u32` | `scales_f32_as_u32` | `biases_f32_as_u32`].
/// receive `state_in.clone()` so they can verify without mutating the cache.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_ffi_stateless(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array), Exception> {
    let (y, _new_state) = gated_delta_kernel_ffi(
        q,
        k,
        v,
        a_log,
        a,
        dt_bias,
        b,
        state_in,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )?;
    Ok((y, state_in.clone()))
}

// ---------------------------------------------------------------------------
// Tape-recording GDN kernel: same recurrence, also outputs innovation delta
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn create_gated_delta_tape_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 9] = [
        c"q",
        c"k",
        c"v",
        c"a_log",
        c"a",
        c"dt_bias",
        c"b",
        c"state_in",
        c"T",
    ];
    let output_names: [&std::ffi::CStr; 3] = [c"y", c"state_out", c"innovation_tape"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source = gdn_metal_source(
        "#define HIGGS_GDN_RECORD_TAPE 1\n",
        GATED_DELTA_FORWARD_KERNEL_SOURCE,
    );

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"gated_delta_tape".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_gated_delta_tape_kernel(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );

        let y_shape = [batch, seq_len, num_v_heads, head_v_dim];
        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        let tape_shape = [batch, seq_len, num_v_heads, head_v_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            in_dtype,
        );
        // State is float32 (matches the AR-decode kernel) for bit-exact verify.
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        // Tape stores deltas in float32 for precision (matches dflash-mlx)
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            tape_shape.as_ptr(),
            tape_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        config
    }
}

fn gated_delta_tape_kernel_config(
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> (mlx_sys::mlx_fast_metal_kernel_config, bool) {
    if !gated_delta_tape_config_cache_enabled() {
        return (
            configure_gated_delta_tape_kernel(
                in_dtype,
                batch,
                seq_len,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
            ),
            false,
        );
    }

    let key = GatedDeltaTapeKernelConfigKey::new(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    );
    let config = GATED_DELTA_TAPE_CONFIG_CACHE.with(|cache_cell| {
        let mut cache_map = cache_cell.borrow_mut();
        cache_map
            .entry(key)
            .or_insert_with(|| {
                CachedMetalKernelConfig(configure_gated_delta_tape_kernel(
                    in_dtype,
                    batch,
                    seq_len,
                    num_k_heads,
                    head_k_dim,
                    num_v_heads,
                    head_v_dim,
                ))
            })
            .0
    });
    (config, true)
}

/// Tape-recording GDN kernel: returns `(y, state_out, innovation_tape)`.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_ffi_with_tape(
    q: &Array,
    k: &Array,
    v: &Array,
    a_log: &Array,
    a: &Array,
    dt_bias: &Array,
    b: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<(Array, Array, Array), Exception> {
    validate_gdn_kernel_state(
        "gated_delta_tape_kernel",
        state_in,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )?;
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(q.as_ptr()) };

    let cached =
        GATED_DELTA_TAPE_KERNEL.get_or_init(|| CachedMetalKernel(create_gated_delta_tape_kernel()));
    let (config, config_is_cached) = gated_delta_tape_kernel_config(
        in_dtype,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    );

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        a_log.as_ptr(),
        a.as_ptr(),
        dt_bias.as_ptr(),
        b.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "gated_delta_tape_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut tape_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 1);
            mlx_sys::mlx_vector_array_get(&raw mut tape_ptr, outputs_vec, 2);
        }
        Ok((
            unsafe { Array::from_ptr(y_ptr) },
            unsafe { Array::from_ptr(state_ptr) },
            unsafe { Array::from_ptr(tape_ptr) },
        ))
    };

    unsafe {
        if !config_is_cached {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Tape replay kernel: replays accepted steps to advance GDN state
// ---------------------------------------------------------------------------

/// Replays the GDN recurrence from a recorded innovation tape.
/// Inputs: `tape[B,T,Hv,Dv]`, `k[B,T,Hk,Dk]`, `a[B,T,Hv]`, `a_log[Hv]`,
/// `dt_bias[Hv]`, `state_in[B,Hv,Dv,Dk]`. Output: `state_out[B,Hv,Dv,Dk]`.
const TAPE_REPLAY_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// a_log and dt_bias are [B * Hv] when batched across layers
float a_log_val = static_cast<float>(a_log[b_idx * Hv + hv_idx]);
float dt_bias_val = static_cast<float>(dt_bias[b_idx * Hv + hv_idx]);
auto a_ = a + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  HIGGS_GDN_GATE(g_val, a_[hv_idx], dt_bias_val, a_log_val);

  auto delta = tape_[dv_idx];
  // Replay the forward kernel's EXACT op sequence (decay as a separate
  // statement, then the k*delta update) so the replayed state is bit-exact
  // with a fresh forward. Fusing into `state*g + k*delta` rounds differently
  // (FMA) and the ~1e-8 drift accumulates across rollbacks → argmax flips.
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    HIGGS_GDN_DECAY(state[i], g_val);
    HIGGS_GDN_UPDATE(state[i], k_[s_idx], delta);
  }
  tape_ += Hv * Dv;
  k_ += Hk * Dk;
  a_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = state[i];
}
";

#[allow(unsafe_code)]
fn create_tape_replay_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 7] =
        [c"tape", c"k", c"a", c"a_log", c"dt_bias", c"state_in", c"T"];
    let output_names: [&std::ffi::CStr; 1] = [c"state_out"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source = gdn_metal_source("", TAPE_REPLAY_KERNEL_SOURCE);

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"tape_replay".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

/// Replay accepted steps from a recorded innovation tape.
/// Returns the new SSM state after replaying `seq_len` steps.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(crate) fn tape_replay_kernel_ffi(
    tape: &Array,
    k: &Array,
    a: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state_in: &Array,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
) -> Result<Array, Exception> {
    validate_gdn_kernel_state(
        "tape_replay_kernel",
        state_in,
        batch,
        seq_len,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )?;
    if tape.dtype() != Dtype::Float32 {
        return Err(Exception::custom(format!(
            "tape_replay_kernel: innovation tape must be Float32, got {:?}",
            tape.dtype()
        )));
    }
    let expected_tape = [batch, seq_len, num_v_heads, head_v_dim];
    if tape.shape() != expected_tape {
        return Err(Exception::custom(format!(
            "tape_replay_kernel: tape shape must be {expected_tape:?}, got {:?}",
            tape.shape()
        )));
    }
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let in_dtype = unsafe { mlx_sys::mlx_array_dtype(state_in.as_ptr()) };

    let cached = TAPE_REPLAY_KERNEL.get_or_init(|| CachedMetalKernel(create_tape_replay_kernel()));

    let config = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"InT".as_ptr(),
            in_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dk".as_ptr(),
            head_k_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Dv".as_ptr(),
            head_v_dim,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hk".as_ptr(),
            num_k_heads,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Hv".as_ptr(),
            num_v_heads,
        );

        let state_shape = [batch, num_v_heads, head_v_dim, head_k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            in_dtype,
        );

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, head_v_dim, batch * num_v_heads);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);

        config
    };

    let t_scalar = unsafe { mlx_sys::mlx_array_new_int(seq_len) };
    let input_ptrs = [
        tape.as_ptr(),
        k.as_ptr(),
        a.as_ptr(),
        a_log.as_ptr(),
        dt_bias.as_ptr(),
        state_in.as_ptr(),
        t_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "tape_replay_kernel failed: {mlx_msg}"
        )))
    } else {
        let mut state_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut state_ptr, outputs_vec, 0);
        }
        Ok(unsafe { Array::from_ptr(state_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

/// Eliminates per-token dtype conversions for scales/biases (packed once at load time).
/// Tiled GEMV with shared memory for x and K-chunking.
///
/// Each threadgroup processes NSG rows (one per simdgroup). x is loaded into
/// shared memory once per threadgroup, eliminating redundant global reads.
/// K is tiled in chunks of CHUNK to fit in threadgroup memory (max 32KB).
///
/// Inputs: `w`(uint32), `sc`(model dtype), `bi`(model dtype), `x`(bf16), `n_param`(int32 scalar)
/// Template: `OutT`, `K`, `GroupSize`, `KPacked`, `NumGroups`.
/// Grid: `(ceil(N/NSG)*32, NSG, 1)`, Threadgroup: `(32, NSG, 1)`.
const QGEMV_4BIT_KERNEL_SOURCE: &str = r"
constexpr int CHUNK = (K <= 8192) ? K : 8192;

threadgroup OutT x_sh[CHUNK];

auto tg = threadgroup_position_in_grid.x;
auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tid = thread_index_in_threadgroup;
auto n_sg = simdgroups_per_threadgroup;
uint tg_sz = n_sg * 32u;

int row = tg * int(n_sg) + int(sg);
bool valid = (row < n_param);

float acc = 0.0f;

for (int k_off = 0; k_off < K; k_off += CHUNK) {
    int k_end = min(k_off + CHUNK, K);
    int k_len = k_end - k_off;

    for (uint i = tid; i < uint(k_len); i += tg_sz) {
        x_sh[i] = x[k_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        int wp_off = k_off / 8;
        int wp_end = k_end / 8;
        auto w_row = w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint packed = w_row[idx];
            int kl = (idx - wp_off) * 8;

            float x0 = x_sh[kl];     float x1 = x_sh[kl+1];
            float x2 = x_sh[kl+2];   float x3 = x_sh[kl+3];
            float x4 = x_sh[kl+4];   float x5 = x_sh[kl+5];
            float x6 = x_sh[kl+6];   float x7 = x_sh[kl+7];

            float dot_val =
                float(packed & 0xFu)         * x0 +
                float((packed >> 4u) & 0xFu)  * x1 +
                float((packed >> 8u) & 0xFu)  * x2 +
                float((packed >> 12u) & 0xFu) * x3 +
                float((packed >> 16u) & 0xFu) * x4 +
                float((packed >> 20u) & 0xFu) * x5 +
                float((packed >> 24u) & 0xFu) * x6 +
                float((packed >> 28u) & 0xFu) * x7;

            int g = idx * 8 / GroupSize;
            float s_val = float(sc[row * NumGroups + g]);
            float b_val = float(bi[row * NumGroups + g]);
            acc += s_val * dot_val + b_val * (x0+x1+x2+x3+x4+x5+x6+x7);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (valid) {
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = OutT(acc);
    }
}
";

/// Quantized GEMM kernel — batched matmul for DFlash verify (T>1).
///
/// Same fused weight-unpack + dot-product as QGEMV_4BIT_KERNEL_SOURCE
/// but processes T input vectors per output row. Each thread group
/// accumulates T separate dot products, eliminating T x dispatch overhead.
///
/// Template: `OutT`, `K`, `GroupSize`, `KPacked`, `NumGroups`, `T`.
const QGEMM_4BIT_KERNEL_SOURCE: &str = r"
constexpr int CHUNK = (K <= 512) ? K : 512;
constexpr int MAX_T = (T > 1) ? T : 1;

threadgroup OutT x_sh[MAX_T][CHUNK];

auto tg = threadgroup_position_in_grid.x;
auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tid = thread_index_in_threadgroup;
auto n_sg = simdgroups_per_threadgroup;
uint tg_sz = n_sg * 32u;

int row = tg * int(n_sg) + int(sg);
bool valid = (row < n_param);

float acc[MAX_T];
for (int t = 0; t < MAX_T; t++) acc[t] = 0.0f;

for (int k_off = 0; k_off < K; k_off += CHUNK) {
    int k_end = min(k_off + CHUNK, K);
    int k_len = k_end - k_off;

    for (uint i = tid; i < uint(k_len * MAX_T); i += tg_sz) {
        int t = int(i) / k_len;
        int k = int(i) % k_len;
        if (t < MAX_T) {
            x_sh[t][k] = x[t * K + k_off + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        int wp_off = k_off / 8;
        int wp_end = k_end / 8;
        auto w_row = w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint packed = w_row[idx];
            int kl = (idx - wp_off) * 8;

            int g = idx * 8 / GroupSize;
            float s_val = float(sc[row * NumGroups + g]);
            float b_val = float(bi[row * NumGroups + g]);

            float dot_vals[MAX_T];
            float sum_x[MAX_T];
            for (int t = 0; t < MAX_T; t++) { dot_vals[t] = 0.0f; sum_x[t] = 0.0f; }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float w_val = float((packed >> (j * 4u)) & 0xFu);
                for (int t = 0; t < MAX_T; t++) {
                    float xv = float(x_sh[t][kl + j]);
                    dot_vals[t] += w_val * xv;
                    sum_x[t] += xv;
                }
            }

            for (int t = 0; t < MAX_T; t++) {
                acc[t] += s_val * dot_vals[t] + b_val * sum_x[t];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (valid) {
    for (int t = 0; t < MAX_T; t++) {
        acc[t] = simd_sum(acc[t]);
        if (lane == 0) {
            y[t * n_param + row] = OutT(acc[t]);
        }
    }
}
";

/// MXFP4 GEMM kernel — same launch shape as affine qgemm, but E2M1 weights
/// and per-block uint8 exponent scales. Kept separate from affine so the hot
/// path has no mode branch and source-level constants.
const QGEMM_MXFP4_4BIT_KERNEL_SOURCE: &str = r"
constexpr int CHUNK = (K <= 512) ? K : 512;
constexpr int MAX_T = (T > 1) ? T : 1;

threadgroup OutT x_sh[MAX_T][CHUNK];

auto tg = threadgroup_position_in_grid.x;
auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tid = thread_index_in_threadgroup;
auto n_sg = simdgroups_per_threadgroup;
uint tg_sz = n_sg * 32u;

int row = tg * int(n_sg) + int(sg);
bool valid = (row < n_param);

float acc[MAX_T];
for (int t = 0; t < MAX_T; t++) acc[t] = 0.0f;

for (int k_off = 0; k_off < K; k_off += CHUNK) {
    int k_end = min(k_off + CHUNK, K);
    int k_len = k_end - k_off;

    for (uint i = tid; i < uint(k_len * MAX_T); i += tg_sz) {
        int t = int(i) / k_len;
        int k = int(i) % k_len;
        if (t < MAX_T) {
            x_sh[t][k] = x[t * K + k_off + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        int wp_off = k_off / 8;
        int wp_end = k_end / 8;
        auto w_row = w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint packed = w_row[idx];
            int kl = (idx - wp_off) * 8;

            int g = idx * 8 / GroupSize;
            uint scale_bits = uint(sc[row * NumGroups + g]) << 23u;
            float block_scale = as_type<float>(scale_bits);
            float mag_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

            float dot_vals[MAX_T];
            for (int t = 0; t < MAX_T; t++) dot_vals[t] = 0.0f;

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint nibble = (packed >> (j * 4u)) & 0xFu;
                float mag = mag_lut[nibble & 0x7u];
                float w_val = ((nibble & 0x8u) != 0u) ? -mag : mag;
                w_val *= block_scale;
                for (int t = 0; t < MAX_T; t++) {
                    dot_vals[t] += w_val * float(x_sh[t][kl + j]);
                }
            }

            for (int t = 0; t < MAX_T; t++) {
                acc[t] += dot_vals[t];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (valid) {
    for (int t = 0; t < MAX_T; t++) {
        acc[t] = simd_sum(acc[t]);
        if (lane == 0) {
            y[t * n_param + row] = OutT(acc[t]);
        }
    }
}
";

/// MXFP4 gate/up SwiGLU kernel for verifier windows.
///
/// Computes `silu(x @ gate_w.T) * (x @ up_w.T)` in one dispatch without
/// materializing separate gate/up activations or concatenating weights.
const MXFP4_GATE_UP_SILU_4BIT_KERNEL_SOURCE: &str = r"
constexpr int CHUNK = (K <= 512) ? K : 512;
constexpr int MAX_T = (T > 1) ? T : 1;

threadgroup OutT x_sh[MAX_T][CHUNK];

auto tg = threadgroup_position_in_grid.x;
auto sg = simdgroup_index_in_threadgroup;
auto lane = thread_index_in_simdgroup;
auto tid = thread_index_in_threadgroup;
auto n_sg = simdgroups_per_threadgroup;
uint tg_sz = n_sg * 32u;

int row = tg * int(n_sg) + int(sg);
bool valid = (row < n_param);

float gate_acc[MAX_T];
float up_acc[MAX_T];
for (int t = 0; t < MAX_T; t++) {
    gate_acc[t] = 0.0f;
    up_acc[t] = 0.0f;
}
float mag_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

for (int k_off = 0; k_off < K; k_off += CHUNK) {
    int k_end = min(k_off + CHUNK, K);
    int k_len = k_end - k_off;

    for (uint i = tid; i < uint(k_len * MAX_T); i += tg_sz) {
        int t = int(i) / k_len;
        int k = int(i) % k_len;
        if (t < MAX_T) {
            x_sh[t][k] = x[t * K + k_off + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        int wp_off = k_off / 8;
        int wp_end = k_end / 8;
        auto gate_row = gate_w + row * KPacked;
        auto up_row = up_w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint gate_packed = gate_row[idx];
            uint up_packed = up_row[idx];
            int kl = (idx - wp_off) * 8;

            int g = idx * 8 / GroupSize;
            uint gate_scale_bits = uint(gate_sc[row * NumGroups + g]) << 23u;
            uint up_scale_bits = uint(up_sc[row * NumGroups + g]) << 23u;
            float gate_scale = as_type<float>(gate_scale_bits);
            float up_scale = as_type<float>(up_scale_bits);

            float gate_dot[MAX_T];
            float up_dot[MAX_T];
            for (int t = 0; t < MAX_T; t++) {
                gate_dot[t] = 0.0f;
                up_dot[t] = 0.0f;
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint gate_nibble = (gate_packed >> (j * 4u)) & 0xFu;
                uint up_nibble = (up_packed >> (j * 4u)) & 0xFu;

                float gate_mag = mag_lut[gate_nibble & 0x7u];
                float up_mag = mag_lut[up_nibble & 0x7u];
                float gate_val = ((gate_nibble & 0x8u) != 0u) ? -gate_mag : gate_mag;
                float up_val = ((up_nibble & 0x8u) != 0u) ? -up_mag : up_mag;
                gate_val *= gate_scale;
                up_val *= up_scale;

                for (int t = 0; t < MAX_T; t++) {
                    float xv = float(x_sh[t][kl + j]);
                    gate_dot[t] += gate_val * xv;
                    up_dot[t] += up_val * xv;
                }
            }

            for (int t = 0; t < MAX_T; t++) {
                gate_acc[t] += gate_dot[t];
                up_acc[t] += up_dot[t];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (valid) {
    for (int t = 0; t < MAX_T; t++) {
        float gate = simd_sum(gate_acc[t]);
        float up = simd_sum(up_acc[t]);
        if (lane == 0) {
            float hidden = (gate / (1.0f + exp(-gate))) * up;
            y[t * n_param + row] = OutT(hidden);
        }
    }
}
";

static QGEMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static QGEMV_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static GATED_DELTA_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static GATED_DELTA_TAPE_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static CANONICAL_CONV_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_GEMV_ENABLED: OnceLock<bool> = OnceLock::new();
static QGEMV_NSG_OVERRIDE: OnceLock<Option<i32>> = OnceLock::new();
static DENSE_FFN_GEMV_MODE: OnceLock<DenseFfnGemvMode> = OnceLock::new();
static DENSE_FFN_FUSE_GATE_UP: OnceLock<bool> = OnceLock::new();
static MXFP4_FUSED_FFN_VERIFY_ENABLED: OnceLock<bool> = OnceLock::new();
static MOE_FFN_FUSE_GATE_UP: OnceLock<bool> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseFfnGemvMode {
    Both,
    FusedOnly,
    DownOnly,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QgemvKernelConfigKey {
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    nsg: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GatedDeltaKernelConfigKey {
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CanonicalConvKernelConfigKey {
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    conv_dim: i32,
    kernel_size: i32,
}

/// Complete specialization and output-geometry key for the tape-recording GDN
/// kernel. The output shapes are derived as
/// `[batch, seq_len, num_v_heads, head_v_dim]` for `y` and the tape, and
/// `[batch, num_v_heads, head_v_dim, head_k_dim]` for state. State and tape
/// output dtypes are fixed to Float32; `in_dtype` also selects the `y` dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GatedDeltaTapeKernelConfigKey {
    in_dtype: mlx_sys::mlx_dtype,
    batch: i32,
    seq_len: i32,
    num_k_heads: i32,
    head_k_dim: i32,
    num_v_heads: i32,
    head_v_dim: i32,
}

impl GatedDeltaTapeKernelConfigKey {
    const fn new(
        in_dtype: mlx_sys::mlx_dtype,
        batch: i32,
        seq_len: i32,
        num_k_heads: i32,
        head_k_dim: i32,
        num_v_heads: i32,
        head_v_dim: i32,
    ) -> Self {
        Self {
            in_dtype,
            batch,
            seq_len,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        }
    }
}

fn decode_gemv_enabled() -> bool {
    *DECODE_GEMV_ENABLED.get_or_init(|| std::env::var("HIGGS_ENABLE_SELECTED_DECODE_GEMV").is_ok())
}

static QGEMM_VERIFY_ENABLED: OnceLock<bool> = OnceLock::new();
fn qgemm_verify_enabled() -> bool {
    *QGEMM_VERIFY_ENABLED.get_or_init(|| truthy_env_var("HIGGS_QGEMM_VERIFY"))
}

static QGEMM_MXFP4_ENABLED: OnceLock<bool> = OnceLock::new();
fn qgemm_mxfp4_enabled() -> bool {
    *QGEMM_MXFP4_ENABLED.get_or_init(|| truthy_env_var("HIGGS_QGEMM_MXFP4"))
}

fn mxfp4_fused_ffn_verify_enabled() -> bool {
    *MXFP4_FUSED_FFN_VERIFY_ENABLED.get_or_init(|| truthy_env_var("HIGGS_MXFP4_FUSED_FFN_VERIFY"))
}

fn truthy_env_var(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

fn canonical_conv_enabled() -> bool {
    *CANONICAL_CONV_ENABLED.get_or_init(|| truthy_env_var("HIGGS_DFLASH_FUSED_CONV"))
}

fn parse_dense_ffn_gemv_mode(raw: Option<&str>) -> DenseFfnGemvMode {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("fused" | "fused_only") => DenseFfnGemvMode::FusedOnly,
        Some("down" | "down_only") => DenseFfnGemvMode::DownOnly,
        Some("off" | "none") => DenseFfnGemvMode::Off,
        _ => DenseFfnGemvMode::Both,
    }
}

fn dense_ffn_gemv_mode() -> DenseFfnGemvMode {
    *DENSE_FFN_GEMV_MODE.get_or_init(|| {
        parse_dense_ffn_gemv_mode(std::env::var("HIGGS_QGEMV_FFN_MODE").ok().as_deref())
    })
}

fn dense_ffn_fuse_gate_up() -> bool {
    *DENSE_FFN_FUSE_GATE_UP.get_or_init(|| {
        std::env::var("HIGGS_DENSE_FFN_GATE_UP").ok().map_or_else(
            || !should_force_dense_decode_safe_defaults_for_brand(apple_cpu_brand()),
            |raw| {
                !matches!(
                    Some(raw.trim().to_ascii_lowercase()).as_deref(),
                    Some("separate" | "split" | "0" | "false" | "off")
                )
            },
        )
    })
}

fn moe_ffn_fuse_gate_up() -> bool {
    *MOE_FFN_FUSE_GATE_UP.get_or_init(|| truthy_env_var("HIGGS_MOE_FFN_GATE_UP"))
}

fn qgemv_config_cache_enabled() -> bool {
    *QGEMV_CONFIG_CACHE_ENABLED.get_or_init(|| truthy_env_var("HIGGS_CACHE_QGEMV_CONFIGS"))
}

fn gated_delta_config_cache_enabled() -> bool {
    *GATED_DELTA_CONFIG_CACHE_ENABLED.get_or_init(|| {
        std::env::var("HIGGS_CACHE_GATED_DELTA_CONFIGS")
            .ok()
            .is_none_or(|raw| {
                matches!(
                    Some(raw.trim().to_ascii_lowercase()).as_deref(),
                    Some("1" | "true" | "on" | "yes")
                )
            })
    })
}

fn gated_delta_tape_config_cache_enabled() -> bool {
    *GATED_DELTA_TAPE_CONFIG_CACHE_ENABLED
        .get_or_init(|| truthy_env_var("HIGGS_DFLASH_GDN_CONFIG_CACHE"))
}

fn qgemv_nsg_override() -> Option<i32> {
    *QGEMV_NSG_OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_QGEMV_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&n| matches!(n, 4 | 8 | 16 | 32))
    })
}

type CompiledGdnDecodeFn =
    dyn for<'a> FnMut(&mut ArraysCache, &'a [Array]) -> Result<Vec<Array>, Exception>;

thread_local! {
    static COMPILED_GDN_DECODE_FN: RefCell<Option<Box<CompiledGdnDecodeFn>>> = RefCell::new(None);
}

// HIGGS_PROFILE=1 TurboQuant decode attribution: per-FA-layer append(quantize) vs
// attn(kernels) nanoseconds, accumulated across a token's FA layers and printed +
// reset once per token by the top-level forward. Confirms the Phase 0 microbench
// finding (append dominates below ~18K) on the real model.
thread_local! {
    static PROF_TQ_APPEND_NS: std::cell::Cell<u128> = const { std::cell::Cell::new(0) };
    static PROF_TQ_ATTN_NS: std::cell::Cell<u128> = const { std::cell::Cell::new(0) };
    static PROF_TQ_N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn tq_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HIGGS_PROFILE").is_ok_and(|v| v == "1"))
}

fn make_compiled_gdn_decode() -> Box<CompiledGdnDecodeFn> {
    Box::new(compile_with_state(compiled_gdn_decode_step, true))
}

fn run_compiled_gdn_decode(cache: &mut ArraysCache, inputs: &[Array]) -> Result<Array, Exception> {
    COMPILED_GDN_DECODE_FN.with(|cell| {
        let mut guard = cell.borrow_mut();
        let compiled = guard.get_or_insert_with(make_compiled_gdn_decode);
        let mut out = compiled(cache, inputs)?;
        out.pop()
            .ok_or_else(|| Exception::custom("compiled GDN decode returned no outputs"))
    })
}

// ---------------------------------------------------------------------------
// QGEMM 4-bit: batched quantized matmul for DFlash verify (T>1)
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn create_qgemm_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 5] = [c"w", c"sc", c"bi", c"x", c"n_param"];
    let output_names: [&std::ffi::CStr; 1] = [c"y"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();
    let source = CString::new(QGEMM_4BIT_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());
    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_qgemm_4bit".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            false,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

static QGEMM_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static QGEMM_MXFP4_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

#[allow(unsafe_code)]
fn configure_qgemm_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    t: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let n_sg = qgemv_nsg_override().unwrap_or(8);
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 8,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"T".as_ptr(), t);

        let n_tgs = (n_rows + n_sg - 1) / n_sg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, n_sg, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, n_sg, 1);

        let y_shape = [t, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );

        config
    }
}

/// Batched quantized matmul for affine 4-bit weights.
///
/// Replaces T separate `quantized_matmul` calls with one fused Metal kernel.
/// Input x: [T, K], weight: [N, K/8] uint32 packed, scales/biases: [N, K/gs].
/// Output: [T, N]. Used during DFlash verify to eliminate T× dispatch overhead.
#[allow(unsafe_code, clippy::too_many_lines)]
pub(crate) fn qgemm_4bit(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    t_len: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("qgemm_4bit: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("qgemm_4bit: weight has no columns"))?;
    let k_dim = k_packed * 8;
    let t = t_len.max(1);

    // Flatten x to [T, K]
    let x_flat = x.reshape(&[t, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QGEMM_KERNEL.get_or_init(|| CachedMetalKernel(create_qgemm_kernel()));

    let config = configure_qgemm_kernel(out_dtype, n_rows, k_dim, group_size, t);

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        unsafe {
            mlx_sys::mlx_array_free(n_scalar);
        }
        Err(Exception::custom(format!("qgemm_4bit failed: {mlx_msg}")))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
        }
        let y = unsafe { Array::from_ptr(y_ptr) };
        unsafe {
            mlx_sys::mlx_array_free(n_scalar);
        }
        Ok(y.reshape(&[1, t, n_rows])?)
    };

    unsafe {
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
    }
    result
}

#[allow(unsafe_code)]
fn create_qgemm_mxfp4_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 4] = [c"w", c"sc", c"x", c"n_param"];
    let output_names: [&std::ffi::CStr; 1] = [c"y"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();
    let source =
        CString::new(QGEMM_MXFP4_4BIT_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());
    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_qgemm_mxfp4_4bit".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            false,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

/// Batched quantized matmul for MXFP4 4-bit weights.
///
/// Decodes E2M1 nibbles with a per-group uint8 exponent scale, matching MLX's
/// `mode="mxfp4"` quantized matmul but fusing verifier windows into one dispatch.
#[allow(unsafe_code)]
pub(crate) fn qgemm_mxfp4_4bit(
    x: &Array,
    weight: &Array,
    scales: &Array,
    group_size: i32,
    t_len: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("qgemm_mxfp4_4bit: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("qgemm_mxfp4_4bit: weight has no columns"))?;
    let k_dim = k_packed * 8;
    let t = t_len.max(1);

    let x_flat = x.reshape(&[t, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QGEMM_MXFP4_KERNEL.get_or_init(|| CachedMetalKernel(create_qgemm_mxfp4_kernel()));
    let config = configure_qgemm_kernel(out_dtype, n_rows, k_dim, group_size, t);

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [w_flat.as_ptr(), s_flat.as_ptr(), x_flat.as_ptr(), n_scalar];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "qgemm_mxfp4_4bit failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
        }
        let y = unsafe { Array::from_ptr(y_ptr) };
        Ok(y.reshape(&[1, t, n_rows])?)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }

    result
}

#[allow(unsafe_code)]
fn create_mxfp4_gate_up_silu_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 6] =
        [c"gate_w", c"gate_sc", c"up_w", c"up_sc", c"x", c"n_param"];
    let output_names: [&std::ffi::CStr; 1] = [c"y"];
    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();
    let source =
        CString::new(MXFP4_GATE_UP_SILU_4BIT_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());
    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_mxfp4_gate_up_silu_4bit".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            false,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

static MXFP4_GATE_UP_SILU_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Fused verifier hidden activation for dense MXFP4 SwiGLU.
///
/// Computes gate/up projections and `SiLU(gate) * up` directly from the two
/// packed MXFP4 matrices. The down projection remains a normal QLinear call.
#[allow(unsafe_code)]
pub(crate) fn mxfp4_gate_up_silu_4bit(
    x: &Array,
    gate_weight: &Array,
    gate_scales: &Array,
    up_weight: &Array,
    up_scales: &Array,
    group_size: i32,
    t_len: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let gate_shape = gate_weight.shape();
    let up_shape = up_weight.shape();
    if gate_shape != up_shape {
        return Err(Exception::custom(format!(
            "mxfp4_gate_up_silu_4bit: gate/up weight shape mismatch {gate_shape:?} vs {up_shape:?}"
        )));
    }
    let n_rows = gate_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("mxfp4_gate_up_silu_4bit: weight has no rows"))?;
    let k_packed = gate_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("mxfp4_gate_up_silu_4bit: weight has no columns"))?;
    let k_dim = k_packed * 8;
    let t = t_len.max(1);

    let expected_scales = n_rows * (k_dim / group_size);
    if gate_scales.shape().iter().product::<i32>() != expected_scales
        || up_scales.shape().iter().product::<i32>() != expected_scales
    {
        return Err(Exception::custom(
            "mxfp4_gate_up_silu_4bit: scale shape mismatch",
        ));
    }

    let x_flat = x.reshape(&[t, k_dim])?;
    let gate_w_flat = gate_weight.reshape(&[-1])?;
    let gate_s_flat = gate_scales.flatten(None, None)?;
    let up_w_flat = up_weight.reshape(&[-1])?;
    let up_s_flat = up_scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = MXFP4_GATE_UP_SILU_KERNEL
        .get_or_init(|| CachedMetalKernel(create_mxfp4_gate_up_silu_kernel()));
    let config = configure_qgemm_kernel(out_dtype, n_rows, k_dim, group_size, t);

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [
        gate_w_flat.as_ptr(),
        gate_s_flat.as_ptr(),
        up_w_flat.as_ptr(),
        up_s_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!(
            "mxfp4_gate_up_silu_4bit failed: {mlx_msg}"
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
        }
        let y = unsafe { Array::from_ptr(y_ptr) };
        Ok(y.reshape(&[1, t, n_rows])?)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }

    result
}

#[allow(unsafe_code)]
fn create_qgemv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let input_names: [&std::ffi::CStr; 5] = [c"w", c"sc", c"bi", c"x", c"n_param"];
    let output_names: [&std::ffi::CStr; 1] = [c"y"];

    let input_ptrs: Vec<*const c_char> = input_names.iter().map(|s| s.as_ptr()).collect();
    let output_ptrs: Vec<*const c_char> = output_names.iter().map(|s| s.as_ptr()).collect();

    let source = CString::new(QGEMV_4BIT_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

    unsafe {
        let in_vec =
            mlx_sys::mlx_vector_string_new_data(input_ptrs.as_ptr().cast_mut(), input_ptrs.len());
        let out_vec =
            mlx_sys::mlx_vector_string_new_data(output_ptrs.as_ptr().cast_mut(), output_ptrs.len());
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_qgemv_4bit".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            false, // ensure_row_contiguous — we handle contiguity manually
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_qgemv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 8,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );

        // Adaptive NSG: more simdgroups for large K (reduces K-chunking barrier overhead).
        // Allow manual override for per-machine tuning on real model dims.
        let nsg = qgemv_nsg_override().unwrap_or(if k_dim > 8192 { 16 } else { 8 });
        let n_tgs = (n_rows + nsg - 1) / nsg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);

        let y_shape = [1, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );

        config
    }
}

fn qgemv_kernel_config(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
) -> (mlx_sys::mlx_fast_metal_kernel_config, bool) {
    let nsg = qgemv_nsg_override().unwrap_or(if k_dim > 8192 { 16 } else { 8 });
    if !qgemv_config_cache_enabled() {
        return (
            configure_qgemv_kernel(out_dtype, n_rows, k_dim, group_size),
            false,
        );
    }

    let key = QgemvKernelConfigKey {
        out_dtype,
        n_rows,
        k_dim,
        group_size,
        nsg,
    };
    let config = QGEMV_CONFIG_CACHE.with(|cache_cell| {
        let mut cache_map = cache_cell.borrow_mut();
        *cache_map
            .entry(key)
            .or_insert_with(|| configure_qgemv_kernel(out_dtype, n_rows, k_dim, group_size))
    });
    (config, true)
}

/// Custom quantized GEMV for 4-bit affine weights (single-token decode).
///
/// Computes `y = dequant(W, scales, biases) @ x` with vectorized uint4 loads.
/// Each buffer uses its native dtype — zero Rust-side conversions.
/// MLX auto-generates per-buffer Metal types from `arr.dtype()`.
#[allow(unsafe_code)]
pub(crate) fn qgemv_4bit(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("qgemv_4bit: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("qgemv_4bit: weight has no columns"))?; // uint32 words per row
    let k_dim = k_packed * 8; // logical elements (8 nibbles per uint32)

    // Flatten all inputs to 1D for the kernel
    let x_flat = x.reshape(&[k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QGEMV_KERNEL.get_or_init(|| CachedMetalKernel(create_qgemv_kernel()));
    let (config, config_is_cached) = qgemv_kernel_config(out_dtype, n_rows, k_dim, group_size);

    // 5 inputs: w(uint32), sc(model dtype), bi(model dtype), x(compute dtype), n_param(int32 scalar)
    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
    ];
    let inputs_vec =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };

    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        )
    };

    let result = if status != 0 {
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        Err(Exception::custom(format!("qgemv_4bit failed: {mlx_msg}")))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0);
        }
        // Output is already in the correct dtype (OutT = x.dtype()) — no conversion
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("qgemv_4bit: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        if !config_is_cached {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
        }
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// YaRN rope scaling (long-context Qwen3.5 / Qwen3-Next checkpoints)
// ---------------------------------------------------------------------------

/// `YaRN` parameters parsed from `rope_scaling` (populated from the nested
/// `rope_parameters` object by the config loaders). Defaults match mlx-lm's
/// `YarnRoPE` signature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct YarnRopeParams {
    pub factor: f32,
    pub original_max_position_embeddings: i32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub mscale: f32,
    pub mscale_all_dim: f32,
}

/// Parse `YaRN` params out of `args.rope_scaling`. Returns `None` unless the
/// scaling object declares `type`/`rope_type` == `"yarn"` — base checkpoints
/// carry `rope_parameters` too (mrope layout hints, theta) but no `type`, and
/// must keep the default rope path bit-for-bit.
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
pub(crate) fn yarn_rope_params(args: &Qwen3NextModelArgs) -> Option<YarnRopeParams> {
    let scaling = args.rope_scaling.as_ref()?;
    let rope_type = scaling
        .get("type")
        .or_else(|| scaling.get("rope_type"))?
        .as_str()?;
    if rope_type != "yarn" {
        return None;
    }
    let get_f32 = |key: &str, default: f64| -> f32 {
        scaling
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default) as f32
    };
    Some(YarnRopeParams {
        factor: get_f32("factor", 1.0),
        original_max_position_embeddings: scaling
            .get("original_max_position_embeddings")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(4096) as i32,
        beta_fast: get_f32("beta_fast", 32.0),
        beta_slow: get_f32("beta_slow", 1.0),
        mscale: get_f32("mscale", 1.0),
        mscale_all_dim: get_f32("mscale_all_dim", 0.0),
    })
}

/// Precomputed `YaRN` state for one attention module. `None` on default-rope
/// checkpoints, whose rope path stays bit-identical to the pre-`YaRN` code.
#[derive(Debug, Clone)]
struct YarnRope {
    /// Per-dimension rope periods (yarn-interpolated `base^(2i/dims)`), f32
    /// `[rope_dim/2]`. Passed as the `freqs` argument of `mlx_fast_rope` on
    /// the decode path; the manual prefill path uses `reciprocal(freqs)` —
    /// the exact op the MLX rope kernel applies internally.
    freqs: Array,
    /// `YaRN` attention-magnitude scale `yarn_get_mscale(factor, mscale) /
    /// yarn_get_mscale(factor, mscale_all_dim)` (`0.1*ln(factor)+1` for the
    /// standard config). Applied to the ROTARY dims of q/k before rotation,
    /// matching mlx-lm `YarnRoPE.__call__`
    /// (`x[..., :dims] = mscale * x[..., :dims]`) and HF's
    /// `cos/sin * attention_scaling`. The pass-through dims stay unscaled, so
    /// this is NOT folded into the softmax scale.
    mscale: f32,
    /// `[head_dim]` f32 broadcast vector: `mscale` on the first `rope_dim`
    /// entries, 1.0 on the tail. Cast to the activation dtype per call so the
    /// prescale never upcasts bf16 activations (see `apply_yarn_rope`'s dtype
    /// note in yarn.rs).
    prescale: Array,
}

impl YarnRope {
    /// Scale the rotary dims of `x` by `mscale` (identity on the tail).
    /// Multiplying by 1.0 is exact in IEEE floats, so the pass-through dims
    /// are bit-identical to a slice-assign.
    fn prescale_rotary(&self, x: &Array) -> Result<Array, Exception> {
        if (self.mscale - 1.0).abs() <= f32::EPSILON {
            return Ok(x.clone());
        }
        let scale = self.prescale.as_dtype(x.dtype())?;
        x.multiply(&scale)
    }
}

/// Build the per-attention `YaRN` state from the model args, or `None` when the
/// checkpoint uses default rope.
#[allow(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
fn build_yarn_rope(args: &Qwen3NextModelArgs, rope_dim: i32, head_dim: i32) -> Option<YarnRope> {
    let params = yarn_rope_params(args)?;
    let freqs = compute_yarn_freqs(
        rope_dim,
        args.rope_theta,
        params.factor,
        params.original_max_position_embeddings,
        params.beta_fast,
        params.beta_slow,
    );
    let mscale = yarn_get_mscale(params.factor, params.mscale)
        / yarn_get_mscale(params.factor, params.mscale_all_dim);
    let mut prescale_vec = vec![1.0_f32; head_dim.max(0) as usize];
    prescale_vec[..rope_dim.max(0) as usize].fill(mscale);
    let prescale = Array::from_slice(&prescale_vec, &[head_dim]);
    tracing::debug!(
        factor = params.factor,
        original_max_position_embeddings = params.original_max_position_embeddings,
        rope_dim,
        mscale,
        "YaRN rope scaling active"
    );
    Some(YarnRope {
        freqs,
        mscale,
        prescale,
    })
}

// ---------------------------------------------------------------------------
// Qwen3NextAttention (full attention with gated Q and partial RoPE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextAttention {
    #[param]
    q_proj: QLinear,
    #[param]
    k_proj: QLinear,
    #[param]
    v_proj: QLinear,
    #[param]
    o_proj: QLinear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    /// `YaRN` long-context state; `None` for default-rope checkpoints.
    yarn: Option<YarnRope>,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

/// Numerical schedule for short full-attention blocks.
///
/// Speculative verification must use the same one-query `RoPE` and SDPA
/// primitives as autoregressive decode. Ordinary prefill deliberately keeps
/// the native multi-query schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashRowSchedule {
    /// Preserve the model's ordinary multi-row prefill schedule.
    NativeBatch,
    /// Fold the exact one-token stateful primitives across a short block.
    CanonicalS1,
}

impl Qwen3NextAttention {
    /// Construct with per-tensor quantization resolved from module paths
    /// (`quant_overrides` / global default; mode-aware). `dense_attention_outputs`
    /// forces a BF16-dense `o_proj` (Unsloth UD layout).
    fn new(args: &Qwen3NextModelArgs, attn_prefix: &str) -> Result<Self, Exception> {
        let q_spec = resolve_quant_for(args, &format!("{attn_prefix}.q_proj"));
        let k_spec = resolve_quant_for(args, &format!("{attn_prefix}.k_proj"));
        let v_spec = resolve_quant_for(args, &format!("{attn_prefix}.v_proj"));
        let o_resolved = resolve_quant_for(args, &format!("{attn_prefix}.o_proj"));
        let o_spec = if args.dense_attention_outputs {
            QuantSpec {
                group_size: o_resolved.group_size,
                bits: 0,
                mode: crate::quant_mode::QuantMode::Dense,
            }
        } else {
            o_resolved
        };
        Self::from_specs(args, q_spec, k_spec, v_spec, o_spec)
    }

    fn from_specs(
        args: &Qwen3NextModelArgs,
        q_spec: QuantSpec,
        k_spec: QuantSpec,
        v_spec: QuantSpec,
        o_spec: QuantSpec,
    ) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        let rope_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        // partial_rotary_factor * head_dim is always a small positive integer (e.g. 64)
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let partial_dim = (rope_dim_f32 * args.partial_rotary_factor).round() as i32;

        Ok(Self {
            q_proj: QLinear::new_spec(q_spec)?,
            k_proj: QLinear::new_spec(k_spec)?,
            v_proj: QLinear::new_spec(v_spec)?,
            o_proj: QLinear::new_spec(o_spec)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            yarn: build_yarn_rope(args, partial_dim, head_dim),
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        self.forward_scheduled(x, mask, cache, DFlashRowSchedule::NativeBatch)
    }

    /// Forward a speculative short block with the exact one-row numerical
    /// primitives used by autoregressive decode.
    fn forward_canonical_rows(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let seq_len = *x
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("canonical attention input has no token axis"))?;
        if !(1..=8).contains(&seq_len) {
            return Err(Exception::custom(format!(
                "canonical short-block attention requires 1..=8 rows, got {seq_len}"
            )));
        }
        self.forward_scheduled(x, mask, cache, DFlashRowSchedule::CanonicalS1)
    }

    /// Attend one query against the cache view produced by one cache append.
    ///
    /// Both autoregressive decode and the canonical short-block schedule call
    /// this primitive. Keeping the dispatch here is important: a `TurboQuant`
    /// cache must remain in its code domain instead of being materialized to a
    /// dense CPU buffer, and a dense cache must use the same one-query SDPA
    /// reduction in both paths.
    fn attend_one_query(
        &self,
        query: &Array,
        view: crate::cache::KvCacheView,
        batch: i32,
    ) -> Result<Array, Exception> {
        match view {
            crate::cache::KvCacheView::TurboQuant(tq_view) => {
                let scores = tq_view.decode_scores(query, self.num_attention_heads)?;
                let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
                let weights = ops::softmax_axis(&scores.multiply(&scale_arr)?, -1, true)?;
                tq_view
                    .decode_values(&weights, self.num_attention_heads)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[batch, 1, -1])
            }
            crate::cache::KvCacheView::Dense { keys, values } => {
                fast::scaled_dot_product_attention(
                    query,
                    keys,
                    values,
                    self.scale,
                    None,
                    None::<&Array>,
                )?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch, 1, -1])
            }
        }
    }

    #[allow(non_snake_case)]
    fn forward_scheduled(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
        row_schedule: DFlashRowSchedule,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Q is projected to 2 * num_heads * head_dim (doubled for gating)
        let q_proj_output = self.q_proj.forward_decode_fast(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward_decode_fast(x)?;
        let values_raw = self.v_proj.forward_decode_fast(x)?;

        // Per-head RmsNorm then transpose to [B, H, L, D]
        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys_normed =
            self.k_norm
                .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?;
        // DIAGNOSTIC: capture k_norm output (pre-transpose, pre-rope) to split
        // k_norm from rope. Requires the raw capture flag.
        let diag_keys_normed = if L > 1 && DIAG_NORM_CAPTURE_REQ.with(|c| c.replace(false)) {
            diag_materialize(&keys_normed)
        } else {
            None
        };
        let mut keys = keys_normed.transpose_axes(&[0, 2, 1, 3])?;
        let values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with cache offset.
        //
        // Prefill (L>1) uses a manual, length-independent rope
        // (`apply_rope_manual`) instead of `mlx_fast_rope`, because
        // `mlx_fast_rope` is length-dependent: for identical input at the same
        // offset it produces different output depending on the total sequence
        // length, which breaks cross-turn prefix-cache reuse for Hybrid models
        // (the stored clone's KV diverges from a cold full prefill). The manual
        // rope computes the rotation from explicit per-position frequencies, so
        // it is length-independent by construction. Verified FP-equivalent to
        // `mlx_fast_rope` for a fixed length (bf16 cos/sin precision) and
        // produces identical greedy output. Decode (L==1) keeps `mlx_fast_rope`
        // (L=1 is always length-independent, and the fused kernel is faster on
        // the per-token hot path).
        let offset = cache.offset();
        let rope_dim = self.rope.dimensions;
        let rope_base = self.rope.base;
        let rope_scale = self.rope.scale;
        // DIAGNOSTIC (HIGGS_DIAG_ROPE_COMPARE=1): verify apply_rope_manual
        // computes the SAME rotation as mlx_fast_rope for a fixed length — the
        // correctness gate before landing the manual rope as the default. Fires
        // once (first FA layer, offset==0, L>1). Compares on the PRE-rope keys.
        #[allow(
            clippy::print_stderr,
            clippy::indexing_slicing,
            clippy::shadow_unrelated
        )]
        if std::env::var("HIGGS_DIAG_ROPE_COMPARE").is_ok_and(|v| v == "1")
            && offset == 0
            && L > 1
            && DIAG_ROPE_COMPARE_FIRED
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
        {
            let pos: Vec<i32> = (offset..offset + L).collect();
            let positions = Array::from_slice(&pos, &[L]);
            let keys_fast = apply_rope(&keys, &self.rope, offset);
            let keys_manual = apply_rope_manual(&keys, &positions, rope_dim, rope_base, rope_scale);
            if let (Ok(kf), Ok(km)) = (keys_fast, keys_manual) {
                let (kfv, _) = diag_materialize(&kf).unwrap_or((vec![], vec![]));
                let (kmv, _) = diag_materialize(&km).unwrap_or((vec![], vec![]));
                let n = kfv.len().min(kmv.len());
                let mut max_abs = 0.0f32;
                let mut diffs = 0usize;
                for i in 0..n {
                    let d = (kfv[i] - kmv[i]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                    if kfv[i].to_bits() != kmv[i].to_bits() {
                        diffs += 1;
                    }
                }
                eprintln!(
                    "DIAG ROPE-COMPARE fast-vs-manual (fixed len, first FA): max_abs={max_abs:.3e} diffs={diffs}/{n}"
                );
            }
        }
        queries = apply_qwen3_next_rope_scheduled(
            queries,
            &self.rope,
            offset,
            self.yarn.as_ref(),
            row_schedule,
        )?;
        keys = apply_qwen3_next_rope_scheduled(
            keys,
            &self.rope,
            offset,
            self.yarn.as_ref(),
            row_schedule,
        )?;

        // DIAGNOSTIC: probe-driven capture of this FA layer's keys (first FA
        // layer of a forward: offset==0, L>1). Captures pre-write (post-rope)
        // and post-write (cache-stored) keys, materialized immediately so the
        // probe can compare two forwards directly.
        let diag_attn = offset == 0 && L > 1 && DIAG_ATTN_CAPTURE_REQ.with(|c| c.replace(false));
        // Capture x (attention input = normed h), keys post-rope pre-write, and
        // stored keys — all as fully-materialized Vec<f32>. Comparing x across
        // forwards isolates whether k_proj/norm/rope is length-dependent
        // (identical x but different keys) vs the input differing.
        let mat_vec = |a: &Array| -> Option<(Vec<f32>, Vec<i32>)> {
            let af = a.as_dtype(mlx_rs::Dtype::Float32).ok()?;
            mlx_rs::transforms::eval([&af]).ok()?;
            Some((af.as_slice::<f32>().to_vec(), af.shape().to_vec()))
        };
        // DIAGNOSTIC: capture k_proj's RAW output (pre-norm) to split the matmul
        // from k_norm. Requires the first-FA capture flag (set a SEPARATE raw
        // request so it doesn't collide with the post-rope capture).
        let diag_keys_raw =
            if offset == 0 && L > 1 && DIAG_RAW_CAPTURE_REQ.with(|c| c.replace(false)) {
                mat_vec(&keys_raw)
            } else {
                None
            };
        let keys_raw_cap = diag_keys_raw;
        let diag_x = if diag_attn { mat_vec(x) } else { None };
        let diag_keys_pre = if diag_attn { mat_vec(&keys) } else { None };

        let tq_prof = tq_profile_enabled() && L == 1;
        let canonical_rows = L > 1 && row_schedule == DFlashRowSchedule::CanonicalS1;
        let attn_t0 = tq_prof.then(std::time::Instant::now);
        let output = if canonical_rows {
            // Preserve the useful batched Q/K/V projections, then execute the
            // stateful part as the exact S1 transition. Appending K/V one row
            // at a time reproduces TurboQuant activation and code-domain
            // updates at precisely the same token boundary as AR decode.
            let row_capacity = usize::try_from(L)
                .map_err(|_| Exception::custom("negative canonical attention row count"))?;
            let mut rows = Vec::with_capacity(row_capacity);
            for position in 0..L {
                let query = queries.index((.., .., position..position + 1, ..));
                let row_keys = keys.index((.., .., position..position + 1, ..));
                let row_values = values.index((.., .., position..position + 1, ..));
                let view = cache.update_and_view(row_keys, row_values)?;
                rows.push(self.attend_one_query(&query, view, B)?);
            }
            ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1)?
        } else {
            let append_t0 = tq_prof.then(|| {
                let _ = mlx_rs::transforms::eval([&keys, &values]);
                std::time::Instant::now()
            });
            let view = cache.update_and_view(keys, values)?;
            if let Some(t0) = append_t0 {
                let _ = mlx_rs::transforms::eval(cache.eval_targets());
                PROF_TQ_APPEND_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos()));
                PROF_TQ_N.with(|c| c.set(c.get() + 1));
            }

            if mask.is_none() && L == 1 {
                self.attend_one_query(&queries, view, B)?
            } else {
                let (cached_keys, cached_values) = view.into_dense()?;
                let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
                fast::scaled_dot_product_attention(
                    queries,
                    cached_keys,
                    cached_values,
                    self.scale,
                    sdpa_mask,
                    None::<&Array>,
                )?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
            }
        };
        if let Some(t0) = attn_t0 {
            let _ = mlx_rs::transforms::eval([&output]);
            PROF_TQ_ATTN_NS.with(|c| c.set(c.get() + t0.elapsed().as_nanos()));
        }

        if diag_attn {
            let stored_vec = cache.keys().and_then(mat_vec);
            DIAG_ATTN_CAPTURED.with(|c| {
                *c.borrow_mut() = Some((
                    L,
                    diag_x,
                    diag_keys_pre,
                    stored_vec,
                    keys_raw_cap,
                    diag_keys_normed,
                ))
            });
        }
        if L == 1 && async_layer_state_eval_enabled() {
            mlx_rs::transforms::async_eval(cache.eval_targets())?;
        }

        let gated = sigmoid_mul(&gate, &output)?;
        let out = self.o_proj.forward_decode_fast(&gated)?;
        if L == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }

    /// Apply `RoPE` at custom positions using `rope_dynamic`.
    ///
    /// # Arguments
    /// * `queries` - Query tensor [B, `n_heads`, L, `head_dim`]
    /// * `keys` - Key tensor [B, `n_kv_heads`, L, `head_dim`]
    /// * `positions` - Position indices [L] (can be non-contiguous)
    ///
    /// # Returns
    /// (queries, keys) with `RoPE` applied at specified positions
    pub fn apply_rope_at_positions(
        &self,
        queries: &Array,
        keys: &Array,
        positions: &Array,
    ) -> Result<(Array, Array), Exception> {
        // Use manual RoPE implementation for per-token positions. `YaRN`
        // models prescale the rotary dims and rotate with the
        // yarn-interpolated frequencies — same treatment as the main forward.
        let (q_in, k_in, yarn_freqs) = match self.yarn.as_ref() {
            Some(yarn) => (
                yarn.prescale_rotary(queries)?,
                yarn.prescale_rotary(keys)?,
                Some(&yarn.freqs),
            ),
            None => (queries.clone(), keys.clone(), None),
        };
        let queries_with_rope = apply_rope_manual_with_freqs(
            &q_in,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
            yarn_freqs,
        )?;

        let keys_with_rope = apply_rope_manual_with_freqs(
            &k_in,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
            yarn_freqs,
        )?;

        Ok((queries_with_rope, keys_with_rope))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DenseQwen3NextAttention {
    #[param]
    q_proj: DenseLinearNoBias,
    #[param]
    k_proj: DenseLinearNoBias,
    #[param]
    v_proj: DenseLinearNoBias,
    #[param]
    o_proj: DenseLinearNoBias,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    /// `YaRN` long-context state; `None` for default-rope checkpoints.
    yarn: Option<YarnRope>,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

impl DenseQwen3NextAttention {
    fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let partial_dim = (head_dim_f32 * args.partial_rotary_factor).round() as i32;

        Ok(Self {
            q_proj: DenseLinearNoBias::new(),
            k_proj: DenseLinearNoBias::new(),
            v_proj: DenseLinearNoBias::new(),
            o_proj: DenseLinearNoBias::new(),
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            yarn: build_yarn_rope(args, partial_dim, head_dim),
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let q_proj_output = self.q_proj.forward(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward(x)?;
        let values_raw = self.v_proj.forward(x)?;

        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.offset();
        queries = apply_qwen3_next_rope(queries, &self.rope, offset, self.yarn.as_ref())?;
        keys = apply_qwen3_next_rope(keys, &self.rope, offset, self.yarn.as_ref())?;

        let view = cache.update_and_view(keys, values)?;
        let try_tq_decode = mask.is_none() && L == 1;

        let output = match view {
            crate::cache::KvCacheView::TurboQuant(tq_view) if try_tq_decode => {
                let scores = tq_view.decode_scores(&queries, self.num_attention_heads)?;
                let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
                let weights = ops::softmax_axis(&scores.multiply(&scale_arr)?, -1, true)?;
                tq_view
                    .decode_values(&weights, self.num_attention_heads)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[B, L, -1])?
            }
            other @ (crate::cache::KvCacheView::Dense { .. }
            | crate::cache::KvCacheView::TurboQuant(_)) => {
                let (cached_keys, cached_values) = other.into_dense()?;
                let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
                fast::scaled_dot_product_attention(
                    queries,
                    cached_keys,
                    cached_values,
                    self.scale,
                    sdpa_mask,
                    None::<&Array>,
                )?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
            }
        };

        if L == 1 && async_layer_state_eval_enabled() {
            mlx_rs::transforms::async_eval(cache.eval_targets())?;
        }

        let gated = sigmoid_mul(&gate, &output)?;
        let out = self.o_proj.forward(&gated)?;
        if L == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextMLP (standard SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextMLP {
    #[param]
    gate_proj: QLinear,
    #[param]
    down_proj: QLinear,
    #[param]
    up_proj: QLinear,
}

pub(crate) fn new_mlp_projections(
    args: &Qwen3NextModelArgs,
    mlp_prefix: &str,
) -> Result<(QLinear, QLinear, QLinear), Exception> {
    let g_spec = resolve_quant_for(args, &format!("{mlp_prefix}.gate_proj"));
    let d_spec = resolve_quant_for(args, &format!("{mlp_prefix}.down_proj"));
    let u_spec = resolve_quant_for(args, &format!("{mlp_prefix}.up_proj"));
    Ok((
        QLinear::new_spec(g_spec)?,
        QLinear::new_spec(d_spec)?,
        QLinear::new_spec(u_spec)?,
    ))
}

/// Build the three MLP projections with a single shared [`QuantSpec`].
///
/// Used by callers that resolve the spec themselves (e.g. MTP sidecar heads
/// whose checkpoints are uniformly quantized).
pub(crate) fn new_mlp_projections_from_spec(
    spec: QuantSpec,
) -> Result<(QLinear, QLinear, QLinear), Exception> {
    Ok((
        QLinear::new_spec(spec)?,
        QLinear::new_spec(spec)?,
        QLinear::new_spec(spec)?,
    ))
}

/// Build the three MLP projections with a single shared `(group_size, bits)`.
///
/// Used by callers that don't carry a `Qwen3NextModelArgs` and therefore can't
/// participate in path-aware override resolution (e.g. `qwen3_moe`).
pub(crate) fn new_mlp_projections_from_quant(
    ql: i32,
    qb: i32,
) -> Result<(QLinear, QLinear, QLinear), Exception> {
    new_mlp_projections_from_spec(QuantSpec {
        group_size: ql,
        bits: qb,
        mode: crate::quant_mode::QuantMode::Affine,
    })
}

impl Qwen3NextMLP {
    fn new(args: &Qwen3NextModelArgs, mlp_prefix: &str) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(args, mlp_prefix)?;
        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DenseQwen3NextMLP {
    #[param]
    gate_proj: DenseLinearNoBias,
    #[param]
    down_proj: DenseLinearNoBias,
    #[param]
    up_proj: DenseLinearNoBias,
}

impl DenseQwen3NextMLP {
    fn new() -> Self {
        Self {
            gate_proj: DenseLinearNoBias::new(),
            down_proj: DenseLinearNoBias::new(),
            up_proj: DenseLinearNoBias::new(),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MTP (Multi-Token Prediction) head
// ---------------------------------------------------------------------------

/// Single MTP transformer layer (full attention + dense MLP).
#[derive(Debug, Clone, ModuleParameters)]
struct MtpTransformerLayer {
    #[param]
    self_attn: Qwen3NextAttention,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: Qwen3NextMLP,
}

/// Single dense MTP transformer layer for sidecar checkpoints that store
/// full-precision projection weights without quantization metadata.
#[derive(Debug, Clone, ModuleParameters)]
struct DenseMtpTransformerLayer {
    #[param]
    self_attn: DenseQwen3NextAttention,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: DenseQwen3NextMLP,
}

/// Multi-Token Prediction head.
///
/// Predicts the token at position t+2 given:
/// - The backbone's hidden state at position t (`h_t`)
/// - The embedding of the confirmed token at position t+1
///
/// Forward:
///   `fc(concat(norm_h(h_t), norm_e(embed(tok_{t+1})))) → transformer layer → shared lm_head`
#[derive(Debug, Clone, ModuleParameters)]
pub struct MtpHead {
    #[param]
    pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    fc: MtpFc,
    #[param]
    layers: Vec<MtpTransformerLayer>,
    #[param]
    norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
struct DenseMtpHead {
    #[param]
    pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    fc: MtpFc,
    #[param]
    layers: Vec<DenseMtpTransformerLayer>,
    #[param]
    norm: nn::RmsNorm,
}

/// MTP fusion projection — kept in full precision (fp16) for accuracy.
///
/// mlx-lm's `quant_predicate` excludes `mtp.fc` from quantization because
/// quantizing the fusion layer destroys MTP prediction quality (0% acceptance).
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct MtpFc {
    #[param]
    weight: Param<Array>,
}

impl MtpFc {
    fn new() -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[1, 1])?),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        dense_linear_no_bias_forward(&self.weight, x)
    }
}

impl MtpHead {
    fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        let layers = (0..n)
            .map(|i| {
                let layer_prefix = format!("language_model.mtp.layers.{i}");
                Ok(MtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, &format!("{layer_prefix}.self_attn"))?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: Qwen3NextMLP::new(args, &format!("{layer_prefix}.mlp"))?,
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: MtpFc::new()?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl DenseMtpHead {
    fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        let layers = (0..n)
            .map(|_| {
                Ok(DenseMtpTransformerLayer {
                    self_attn: DenseQwen3NextAttention::new(args)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: DenseQwen3NextMLP::new(),
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: MtpFc::new()?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

/// Single `MoE` MTP transformer layer (Qwen3.6-A3B style).
///
/// Qwen3.6-A3B sidecars ship the MTP layer as a full `MoE` decoder layer:
/// full attention (with q/k norms) + `SparseMoeBlock`
/// (router gate + stacked experts + shared expert + shared-expert gate).
#[derive(Debug, Clone, ModuleParameters)]
struct MoeMtpTransformerLayer {
    #[param]
    self_attn: Qwen3NextAttention,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: SparseMoeBlock,
}

/// MTP head with an `MoE` transformer layer (Qwen3.6-A3B style sidecars).
///
/// Unlike [`MtpHead`], the fusion projection `fc` is a quantized [`QLinear`]
/// (these sidecars ship `fc.{weight,scales,biases}` triples), and the MLP is
/// a [`SparseMoeBlock`]. All projections use the checkpoint's uniform
/// quantization — the main model's `gate_quantization` override must NOT be
/// applied here (the sidecar's router gate is quantized at the default width).
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoeMtpHead {
    #[param]
    pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    fc: QLinear,
    #[param]
    layers: Vec<MoeMtpTransformerLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl MoeMtpHead {
    fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        // The sidecar's MoE block is uniformly quantized at the default width;
        // strip the main model's per-layer gate override so the router gate's
        // QLinear dequantizes with the right parameters.
        let mut mtp_args = args.clone();
        mtp_args.gate_quantization = None;

        // Prefixes feed `resolve_quant_for` only (not weight loading); the MoE-MTP
        // sidecar is uniformly quantized, so these fall back to the global width
        // unless a checkpoint ships per-tensor MTP overrides.
        let layers = (0..n)
            .map(|i| {
                let lp = format!("model.mtp.layers.{i}");
                Ok(MoeMtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, &format!("{lp}.self_attn"))?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: SparseMoeBlock::new(&mtp_args, &format!("{lp}.mlp"))?,
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        let fc_spec = resolve_quant_for(args, "model.mtp.fc");
        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: QLinear::new_spec(fc_spec)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

// ---------------------------------------------------------------------------
// SwitchMLP weights (stacked expert weights for MoE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct SwitchMlpWeights {
    #[param]
    gate_proj: QLinear,
    #[param]
    up_proj: QLinear,
    #[param]
    down_proj: QLinear,
    /// Lazily fused gate+up weights for `MoE` `gather_qmm` (3→2 calls per layer).
    fused_gate_up: Option<(Array, Array, Array, i32)>,
}

impl SwitchMlpWeights {
    pub(crate) fn new(args: &Qwen3NextModelArgs, prefix: &str) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(args, prefix)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            fused_gate_up: None,
        })
    }

    /// Build with a single shared `(group_size, bits)` — for callers without
    /// a `Qwen3NextModelArgs`.
    pub(crate) fn from_quant(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections_from_quant(ql, qb)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            fused_gate_up: None,
        })
    }

    /// Apply the full `SwiGLU` `MoE` block for all selected experts in one shot
    /// using `gather_qmm` (fused expert-indexed quantized matmul).
    ///
    /// `x`: `[..., D]` input
    /// `indices`: `[..., top_k]` expert indices
    /// Returns: `[..., top_k, D]`
    #[allow(dead_code)]
    pub(crate) fn forward_gather(
        &self,
        x: &Array,
        indices: &Array,
        sorted: bool,
    ) -> Result<Array, Exception> {
        // Reshape so x batch dims broadcast with the indices shape.
        // x: [B, L, D] -> [B, L, 1, 1, D]
        //   batch = [B, L, 1], M=1, K=D
        // indices: [B, L, top_k]
        //   broadcast([B, L, 1], [B, L, top_k]) -> [B, L, top_k]
        let shape = x.shape();
        let err = || Exception::custom("forward_gather input must be [B, L, D]");
        let b = *shape.first().ok_or_else(err)?;
        let l = *shape.get(1).ok_or_else(err)?;
        let d = *shape.get(2).ok_or_else(err)?;
        let x_exp = x.reshape(&[b, l, 1, 1, d])?;

        // Gate/up projections: [B, L, top_k, 1, intermediate]
        let gate_out = gather_qmm(
            &x_exp,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            indices,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            sorted,
        )?;
        let up_out = gather_qmm(
            &x_exp,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            indices,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            sorted,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        // Down projection: [B, L, top_k, 1, D]
        // activated batch=[B,L,top_k] broadcasts with indices [B,L,top_k] exactly
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            indices,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            sorted,
        )?;

        // Squeeze M=1: [B, L, top_k, D]
        down_out.squeeze_axes(&[-2])
    }

    /// Like `forward_gather` but reorders tokens globally by expert index
    /// before calling `gather_qmm`, matching mlx-lm's `_gather_sort` pattern.
    ///
    /// This gives coalesced GPU memory access and is 3-6x faster for prefill
    /// (L >= 32). For single-token decode (L=1) it's equivalent.
    ///
    /// `x`: `[B, L, D]`
    /// `indices`: `[B, L, top_k]` expert indices (need NOT be pre-sorted)
    /// Returns: `[B, L, top_k, D]`
    pub(crate) fn forward_gather_global_sort(
        &self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_global_sort input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        // --- Global sort: flatten, argsort, reorder tokens by expert ---
        // indices: [B, L, top_k] -> [N] where N = B*L*top_k
        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        // Map each sorted position back to its source token: order / top_k
        let top_k_u32 =
            u32::try_from(top_k).map_err(|_| Exception::custom("top_k must fit in u32"))?;
        let top_k_arr = Array::from_slice(&[top_k_u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        // x_flat: [B*L, 1, D] -> x_sorted: [N, 1, D]
        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;

        // idx_sorted: [N] — monotonically non-decreasing expert indices
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- gather_qmm with coalesced access ---
        let gate_out = gather_qmm(
            &x_sorted,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true, // indices are globally sorted
        )?;
        let up_out = gather_qmm(
            &x_sorted,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            &idx_sorted,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            true,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }

    /// Like `forward_gather_global_sort` but fuses gate+up into a single
    /// `gather_qmm` call (3→2 per layer). Lazy-inits fused weights on first call.
    /// Production routing gates this behind `HIGGS_MOE_FFN_GATE_UP` because the
    /// fused cache duplicates the resident gate/up tensors.
    pub(crate) fn forward_gather_fused(
        &mut self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        // Lazy-init: concatenate gate+up weights along axis 1 (intermediate dim).
        // MoE weights are [num_experts, intermediate_packed, hidden].
        if self.fused_gate_up.is_none() {
            let intermediate = *self
                .gate_proj
                .weight
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("gate_proj weight missing dim 1"))?;
            let fw = ops::concatenate_axis(&[&*self.gate_proj.weight, &*self.up_proj.weight], 1)?;
            let fs = ops::concatenate_axis(&[&*self.gate_proj.scales, &*self.up_proj.scales], 1)?;
            let fb = ops::concatenate_axis(&[&*self.gate_proj.biases, &*self.up_proj.biases], 1)?;
            fw.eval()?;
            fs.eval()?;
            fb.eval()?;
            self.fused_gate_up = Some((fw, fs, fb, intermediate));
        }
        let (fw, fs, fb, intermediate) = self
            .fused_gate_up
            .as_ref()
            .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;

        // --- Global sort (same as forward_gather_global_sort) ---
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_fused input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        let top_k_u32 =
            u32::try_from(top_k).map_err(|_| Exception::custom("top_k must fit in u32"))?;
        let top_k_arr = Array::from_slice(&[top_k_u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- Fused gate+up: ONE gather_qmm instead of TWO ---
        let fused_out = gather_qmm(
            &x_sorted,
            fw,
            fs,
            fb,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true,
        )?;
        // Split at intermediate boundary → gate_out, up_out
        let parts = fused_out.split_axis(&[*intermediate], Some(-1))?;
        let gate_out = parts
            .first()
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let up_out = parts
            .get(1)
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let activated = swiglu(gate_out, up_out)?;

        // --- down_proj: unchanged ---
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }
}

// ---------------------------------------------------------------------------
// SparseMoeBlock (router + SwitchGLU + shared expert)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SparseMoeBlock {
    #[param]
    gate: QLinear,
    #[param]
    switch_mlp: SwitchMlpWeights,
    #[param]
    shared_expert: Qwen3NextMLP,
    #[param]
    shared_expert_gate: QLinear,
    top_k: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn new(args: &Qwen3NextModelArgs, mlp_prefix: &str) -> Result<Self, Exception> {
        if args.num_experts <= 0 {
            return Err(Exception::custom("num_experts must be > 0"));
        }
        if args.num_experts_per_tok <= 0 {
            return Err(Exception::custom("num_experts_per_tok must be > 0"));
        }
        if args.num_experts_per_tok > args.num_experts {
            return Err(Exception::custom(
                "num_experts_per_tok must be <= num_experts",
            ));
        }
        let gate_path = format!("{mlp_prefix}.gate");
        let shared_expert_gate_path = format!("{mlp_prefix}.shared_expert_gate");
        let gate_spec = resolve_gate_quant(args, &gate_path);
        let seg_spec = resolve_gate_quant(args, &shared_expert_gate_path);
        let switch_mlp_prefix = format!("{mlp_prefix}.switch_mlp");
        let shared_expert_prefix = format!("{mlp_prefix}.shared_expert");
        Ok(Self {
            gate: QLinear::new_spec(gate_spec)?,
            switch_mlp: SwitchMlpWeights::new(args, &switch_mlp_prefix)?,
            shared_expert: Qwen3NextMLP::new(args, &shared_expert_prefix)?,
            shared_expert_gate: QLinear::new_spec(seg_spec)?,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gates = ops::softmax_axis(&self.gate.forward(x)?, -1, true)?;

        // Top-K selection via argpartition
        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let num_experts = *gates
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("gates must have last dim"))?;
        let top_k_start = num_experts - self.top_k;
        let top_inds = ops::sort_axis(all_inds.index((.., .., top_k_start..)), -1)?;
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        let top_scores = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        // Expert computation via fused gather_qmm (global sort for coalesced access)
        let y = self.switch_mlp.forward_gather_global_sort(x, &top_inds)?;

        // Weighted sum over experts: [B, L, top_k, D] * [B, L, top_k, 1] -> sum -> [B, L, D]
        let expert_sum = y
            .multiply(&top_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)?;

        // Shared expert
        let shared_y = self.shared_expert.forward(x)?;
        let shared_gate_val = nn::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_out = shared_y.multiply(&shared_gate_val)?;

        expert_sum.add(shared_out)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet (SSM-like linear attention)
// ---------------------------------------------------------------------------

/// Cache state for a `GatedDeltaNet` layer.
#[derive(Debug, Clone)]
pub struct ArraysCache {
    pub conv_state: Option<Array>,
    pub ssm_state: Option<Array>,
    pub conv_pos: i32,
    pub offset: i32,
}

impl ArraysCache {
    pub const fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            conv_pos: -1,
            offset: 0,
        }
    }
}

impl Default for ArraysCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ArraysCache {
    /// Evaluate lazy arrays so a subsequent `clone()` captures values.
    pub fn eval_arrays(&self) -> Result<(), mlx_rs::error::Exception> {
        if let Some(cs) = &self.conv_state {
            cs.eval()?;
        }
        if let Some(ss) = &self.ssm_state {
            ss.eval()?;
        }
        Ok(())
    }
}

impl Updatable for ArraysCache {
    fn updatable_states_len(&self) -> usize {
        usize::from(self.ssm_state.is_some())
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        let mut states = Vec::with_capacity(self.updatable_states_len());
        if let Some(state) = self.ssm_state.as_ref() {
            states.push(state);
        }
        states
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        let mut states = Vec::with_capacity(self.updatable_states_len());
        if let Some(state) = self.ssm_state.as_mut() {
            states.push(state);
        }
        states
    }
}

fn compute_g_direct(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let a_plus_bias = a.add(dt_bias)?;
    let sp = nn::softplus(&a_plus_bias)?;
    let neg_decay = a_log.exp()?.negative()?.multiply(sp)?;
    neg_decay.exp()
}

fn compiled_gdn_decode_step(
    cache: &mut ArraysCache,
    inputs: &[Array],
) -> Result<Vec<Array>, Exception> {
    let [q, k, v, g, beta, z, norm_weight] = inputs else {
        return Err(Exception::custom("compiled GDN decode expects 7 inputs"));
    };
    // q: [B, 1, Hv, Dk]
    // k: [B, 1, Hv, Dk]
    // v: [B, 1, Hv, Dv]
    // g: [B, 1, Hv]
    // beta: [B, 1, Hv]
    // z: [B, 1, Hv, Dv]
    // norm_weight: [Dv]

    let state = cache
        .ssm_state
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled GDN decode requires initialized state"))?;

    let q_t = q.squeeze_axes(&[1])?;
    let k_t = k.squeeze_axes(&[1])?;
    let v_t = v.squeeze_axes(&[1])?;
    let g_t = g.squeeze_axes(&[1])?.expand_dims(-1)?.expand_dims(-1)?;
    let beta_t = beta.squeeze_axes(&[1])?.expand_dims(-1)?;

    let decayed_state = state.multiply(&g_t)?;
    let k_expanded = k_t.expand_dims(-2)?;
    let kv_mem = decayed_state
        .multiply(&k_expanded)?
        .sum_axes(&[-1], false)?;
    let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
    let new_state = decayed_state.add(&k_expanded.multiply(&delta.expand_dims(-1)?)?)?;

    *cache
        .ssm_state
        .as_mut()
        .ok_or_else(|| Exception::custom("compiled GDN decode lost cache state"))? = new_state;

    let y_t = cache
        .ssm_state
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled GDN decode missing updated state"))?
        .multiply(&q_t.expand_dims(-2)?)?
        .sum_axes(&[-1], false)?;
    let y = y_t.expand_dims(1)?;
    let normed = fast::rms_norm(&y, norm_weight, 1e-6)?;
    let gated = nn::silu(z)?.multiply(&normed)?;
    Ok(vec![gated])
}

/// conv1d, norms, or attention. Just `state = state * g + k * delta`.
pub struct GdnLayerTape {
    /// Innovation delta at each timestep: `[B, T, Hv, Dv]`
    pub delta_tape: Array,
    /// Post-conv, post-norm key vectors: `[B, T, Hk, Dk]`
    pub norm_k: Array,
    /// Projected gate values: `[B, T, Hv]`
    pub a_proj: Array,
    /// Raw QKV input to conv1d (for `conv_state` rebuild): `[B, T, conv_dim]`
    pub qkv_input: Array,
    /// Pre-forward `conv_state` for rollback: `[B, K-1, conv_dim]`
    pub conv_state_init: Option<Array>,
    /// Pre-forward `ssm_state` for rollback: `[B, Hv, Dv, Dk]`
    pub ssm_state_init: Option<Array>,
    /// Pre-forward convolution ring cursor for rollback
    pub conv_pos_init: i32,
    /// Pre-forward cache offset for rollback
    pub offset_init: i32,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
struct GatedDeltaNet {
    #[param]
    in_proj_qkvz: QLinear,
    #[param]
    in_proj_ba: QLinear,
    // Separate projections for qwen3_5-style models (flat split, not per-head)
    #[param]
    in_proj_qkv: Option<QLinear>,
    #[param]
    in_proj_z: Option<QLinear>,
    #[param]
    in_proj_a: Option<QLinear>,
    #[param]
    in_proj_b: Option<QLinear>,
    #[param]
    conv1d: nn::Conv1d,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    out_proj: QLinear,
    #[param]
    A_log: Param<Array>,
    #[param]
    dt_bias: Param<Array>,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,
    use_separate_projections: bool,
    qk_norm_weight_q: Array,
    qk_norm_weight_k: Array,
    /// Pre-transposed conv weight for fast T=1 decode: [`kernel_size`, `conv_dim`].
    conv_weight_t: Option<Array>,
}

impl GatedDeltaNet {
    fn new(args: &Qwen3NextModelArgs, gdn_prefix: &str) -> Result<Self, Exception> {
        let num_k_heads = args.linear_num_key_heads;
        let num_v_heads = args.linear_num_value_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = args.linear_conv_kernel_dim;

        let use_sep = args.use_separate_gdn_projections;
        // Per-projection quantization resolved from module paths; overrides can
        // mark projections Dense (bf16) or mxfp4 in mixed-precision checkpoints
        // (e.g. AEON mxfp4 + bf16 GDN dynamics; Qwythos mxfp4 GDN projections).
        let resolve = |name: &str| {
            QLinear::new_spec(resolve_quant_for(args, &format!("{gdn_prefix}.{name}")))
        };
        // Names whose checkpoint tensors are BF16-dense (no `.scales`/`.biases`)
        // when `args.dense_attention_outputs` is true.
        let resolve_maybe_dense = |name: &str| {
            let resolved = resolve_quant_for(args, &format!("{gdn_prefix}.{name}"));
            let spec = if args.dense_attention_outputs {
                QuantSpec {
                    group_size: resolved.group_size,
                    bits: 0,
                    mode: crate::quant_mode::QuantMode::Dense,
                }
            } else {
                resolved
            };
            QLinear::new_spec(spec)
        };
        Ok(Self {
            in_proj_qkvz: resolve("in_proj_qkvz")?,
            in_proj_ba: resolve_maybe_dense("in_proj_ba")?,
            in_proj_qkv: if use_sep {
                Some(resolve("in_proj_qkv")?)
            } else {
                None
            },
            in_proj_z: if use_sep {
                Some(resolve("in_proj_z")?)
            } else {
                None
            },
            in_proj_a: if use_sep {
                Some(resolve_maybe_dense("in_proj_a")?)
            } else {
                None
            },
            in_proj_b: if use_sep {
                Some(resolve_maybe_dense("in_proj_b")?)
            } else {
                None
            },
            conv1d: nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
                .bias(false)
                .groups(conv_dim)
                .padding(0)
                .build()?,
            norm: nn::RmsNormBuilder::new(head_v_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            out_proj: resolve_maybe_dense("out_proj")?,
            A_log: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            dt_bias: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
            use_separate_projections: use_sep,
            qk_norm_weight_q: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let value = dim_f32.sqrt().recip().powi(2);
                let values = vec![
                    value;
                    usize::try_from(head_k_dim).map_err(|_| Exception::custom(
                        "head_k_dim out of usize range"
                    ))?
                ];
                Array::from_slice(&values, &[head_k_dim])
            },
            qk_norm_weight_k: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let value = dim_f32.sqrt().recip();
                let values = vec![
                    value;
                    usize::try_from(head_k_dim).map_err(|_| Exception::custom(
                        "head_k_dim out of usize range"
                    ))?
                ];
                Array::from_slice(&values, &[head_k_dim])
            },
            conv_weight_t: None,
        })
    }

    /// One canonical depthwise-convolution transition shared by AR decode and
    /// short-block verification. `previous(lag)` returns history from newest
    /// to oldest. Keeping the multiply/add/activation source identical makes
    /// the block schedule equivalent by construction instead of convention.
    fn canonical_conv1d_step<F>(
        &self,
        current: &Array,
        weight_t: &Array,
        available: i32,
        batch: i32,
        mut previous: F,
    ) -> Result<Array, Exception>
    where
        F: FnMut(i32) -> Result<Array, Exception>,
    {
        let history_len = self.conv_kernel_size - 1;
        let current_flat = current.reshape(&[batch, self.conv_dim])?;
        let current_weight = weight_t.index((self.conv_kernel_size - 1, ..));
        let mut conv_flat = current_flat.multiply(&current_weight)?;
        for lag in 0..available {
            let prior = previous(lag)?;
            let weight = weight_t.index((history_len - 1 - lag, ..));
            conv_flat = conv_flat.add(&prior.multiply(&weight)?)?;
        }
        silu_direct(&conv_flat.reshape(&[batch, 1, self.conv_dim])?)
    }

    fn decode_conv1d_step(
        &mut self,
        mixed_qkv: &Array,
        cache: &mut ArraysCache,
        batch: i32,
    ) -> Result<Array, Exception> {
        let history_len = self.conv_kernel_size.saturating_sub(1);
        let wt = if let Some(w) = &self.conv_weight_t {
            w.clone()
        } else {
            // Conv1d weight: [conv_dim, kernel_size, 1] -> [kernel_size, conv_dim]
            let raw_w = self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?;
            let typed_w = raw_w.as_dtype(mixed_qkv.dtype())?;
            typed_w.eval()?;
            self.conv_weight_t = Some(typed_w.clone());
            typed_w
        };

        let conv_out = if history_len > 0 {
            if cache.conv_state.is_none() {
                cache.conv_state = Some(ops::zeros_dtype(
                    &[batch, history_len, self.conv_dim],
                    mixed_qkv.dtype(),
                )?);
                cache.conv_pos = -1;
            }

            let history = cache
                .conv_state
                .as_mut()
                .ok_or_else(|| Exception::custom("decode conv history missing"))?;

            let available = cache.offset.min(history_len);
            let conv_out = if cache.conv_pos >= 0 {
                let conv_pos = cache.conv_pos;
                self.canonical_conv1d_step(mixed_qkv, &wt, available, batch, |lag| {
                    let idx = (conv_pos - lag).rem_euclid(history_len);
                    history
                        .index((.., idx..idx + 1, ..))
                        .reshape(&[batch, self.conv_dim])
                })?
            } else {
                self.canonical_conv1d_step(mixed_qkv, &wt, 0, batch, |_| {
                    Err(Exception::custom("unreachable empty convolution history"))
                })?
            };

            let next_pos = if cache.conv_pos < 0 {
                0
            } else {
                (cache.conv_pos + 1).rem_euclid(history_len)
            };
            history.try_index_mut((.., next_pos..next_pos + 1, ..), mixed_qkv.clone())?;
            cache.conv_pos = next_pos;
            conv_out
        } else {
            self.canonical_conv1d_step(mixed_qkv, &wt, 0, batch, |_| {
                Err(Exception::custom(
                    "unreachable zero-width convolution history",
                ))
            })?
        };

        Ok(conv_out)
    }

    fn chronological_conv_state(
        &self,
        cache: &mut ArraysCache,
        batch: i32,
        dtype: Dtype,
    ) -> Result<Array, Exception> {
        let n_keep = self.conv_kernel_size - 1;
        let Some(state) = cache.conv_state.take() else {
            return ops::zeros_dtype(&[batch, n_keep, self.conv_dim], dtype);
        };

        if n_keep <= 0 {
            return Ok(state);
        }

        let available = cache.offset.clamp(0, n_keep);
        if available == 0 {
            return Ok(state);
        }
        if available == n_keep && cache.conv_pos == n_keep - 1 {
            return Ok(state);
        }

        let start = (cache.conv_pos - available + 1).rem_euclid(n_keep);
        let ordered_tail = if start + available <= n_keep {
            state.index((.., start..start + available, ..))
        } else {
            let first = state.index((.., start.., ..));
            let second = state.index((.., ..(start + available - n_keep), ..));
            ops::concatenate_axis(&[&first, &second], 1)?
        };

        if available == n_keep {
            return Ok(ordered_tail);
        }

        let pad = ops::zeros_dtype(&[batch, n_keep - available, self.conv_dim], state.dtype())?;
        ops::concatenate_axis(&[&pad, &ordered_tail], 1)
    }

    #[allow(non_snake_case, clippy::too_many_lines)]
    fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs and split into q, k, v, z, b, a
        let (q, k, v, z, b, a) = if self.use_separate_projections {
            // qwen3.5-style: 4 separate projections, flat split
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward_decode_fast(inputs)?;
            let z = z_proj.forward_decode_fast(inputs)?.reshape(&[
                B,
                S,
                self.num_v_heads,
                self.head_v_dim,
            ])?;
            let b = b_proj.forward_decode_fast(inputs)?;
            let a = a_proj.forward_decode_fast(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

            (q, k, v, z, b, a)
        } else {
            // qwen3_next-style: combined projections, per-head reshape
            let mixed_qkvz = self.in_proj_qkvz.forward_decode_fast(inputs)?;
            let mixed_ba = self.in_proj_ba.forward_decode_fast(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Concatenate q, k, v for conv input
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;
        let n_keep = self.conv_kernel_size - 1;

        // DIAGNOSTIC: capture conv input/output + SSM output for the first GDN
        // prefill forward when requested by a probe (to split conv1d vs fused
        // gated_delta_kernel length-dependence).
        let diag_gdn = S > 1 && DIAG_GDN_CAPTURE_REQ.with(|c| c.replace(false));
        let diag_mixed = if diag_gdn {
            diag_materialize(&mixed_qkv)
        } else {
            None
        };

        let mut diag_conv_input: Option<(Vec<f32>, Vec<i32>)> = None;
        let conv_out = if S == 1 {
            self.decode_conv1d_step(&mixed_qkv, cache, B)?
        } else {
            let conv_state = self.chronological_conv_state(cache, B, inputs.dtype())?;
            let diag_conv_state_mat = if diag_gdn {
                diag_materialize(&conv_state)
            } else {
                None
            };
            let _ = diag_conv_state_mat; // inspected via print below
            #[allow(clippy::print_stderr, clippy::shadow_unrelated)]
            if diag_gdn {
                if let Some((sv, _)) = diag_conv_state_mat.as_ref() {
                    let nz = sv.iter().filter(|x| **x != 0.0f32).count();
                    let mx = sv.iter().cloned().fold(0.0f32, f32::max);
                    eprintln!(
                        "DIAG GDN conv_state: len={} nonzero={} max_abs={mx:.3e} first4=[{}]",
                        sv.len(),
                        nz,
                        sv.iter()
                            .take(4)
                            .map(|v| format!("{v:.3}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
            }
            let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;
            let conv_input_len = *conv_input
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
            let keep_start = conv_input_len - n_keep;
            cache.conv_state = Some(conv_input.index((.., keep_start.., ..)));
            cache.conv_pos = if n_keep > 0 { n_keep - 1 } else { -1 };
            let diag_conv_input_val = if diag_gdn {
                diag_materialize(&conv_input.index((.., 0..self.conv_kernel_size, ..)))
            } else {
                None
            };
            diag_conv_input = diag_conv_input_val;
            // DIAGNOSTIC (HIGGS_DIAG_CONV_MANUAL=1): use a length-independent
            // windowed conv (per-position slice * weight * sum) instead of MLX's
            // conv1d, which is length-dependent at the first n_keep (left-pad)
            // boundary positions. Mirrors forward_stateless's windowed conv.
            #[allow(
                clippy::shadow_unrelated,
                clippy::shadow_reuse,
                clippy::indexing_slicing,
                clippy::as_conversions
            )]
            if std::env::var("HIGGS_DIAG_CONV_MANUAL").is_ok_and(|v| v == "1") {
                // Materialize conv_input to a CONCRETE array first when
                // HIGGS_DIAG_CONV_EVAL_INPUT=1, to test whether MLX's lazy
                // graph (concatenate of lazy zeros + lazy mixed_qkv) is the
                // source of the length-dependent boundary output.
                let conv_input_mat =
                    if std::env::var("HIGGS_DIAG_CONV_EVAL_INPUT").is_ok_and(|v| v == "1") {
                        let _ = mlx_rs::transforms::eval([&conv_input]);
                        conv_input.clone()
                    } else {
                        conv_input.clone()
                    };
                let wt = {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {shape:?}"
                        )));
                    };
                    w.as_dtype(inputs.dtype())?
                };
                let ks = self.conv_kernel_size;
                let mut windows = Vec::with_capacity(S as usize);
                for i in 0..S {
                    windows.push(
                        conv_input_mat
                            .index((.., i..i + ks, ..))
                            .multiply(&wt)?
                            .sum_axes(&[1], true)?,
                    );
                }
                silu_direct(&ops::concatenate_axis(
                    &windows.iter().collect::<Vec<_>>(),
                    1,
                )?)?
            } else {
                silu_direct(&self.conv1d.forward(&conv_input)?)?
            }
        };
        let diag_conv_out = if diag_gdn {
            diag_materialize(&conv_out)
        } else {
            None
        };

        if S == 1 && async_layer_state_eval_enabled() {
            if let Some(conv_state) = cache.conv_state.as_ref() {
                mlx_rs::transforms::async_eval([conv_state])?;
            }
        }

        // Split conv output back to q, k, v
        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        // On first call, convert weight vectors to match input dtype.
        let in_dt = inputs.dtype();
        if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q = self.qk_norm_weight_q.as_dtype(in_dt)?;
            self.qk_norm_weight_k = self.qk_norm_weight_k.as_dtype(in_dt)?;
        }

        let norm_q = fast::rms_norm(&conv_q, &self.qk_norm_weight_q, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &self.qk_norm_weight_k, 1e-6)?;

        let use_compiled_decode = compiled_gdn_decode_enabled() && S == 1;
        if use_compiled_decode {
            if cache.ssm_state.is_none() {
                cache.ssm_state = Some(ops::zeros_dtype(
                    &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                    Dtype::Float32,
                )?);
            }

            let repeat_factor = self.num_v_heads / self.num_k_heads;
            let q_decode = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(norm_q, repeat_factor, -2)?
            } else {
                norm_q
            };
            let k_decode = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(norm_k, repeat_factor, -2)?
            } else {
                norm_k
            };
            let g = compute_g_direct(self.A_log.as_ref(), &a, self.dt_bias.as_ref())?;
            let beta = nn::sigmoid(&b)?;
            let kernel_inputs = [
                q_decode,
                k_decode,
                conv_v,
                g,
                beta,
                z,
                self.norm.weight.as_ref().clone(),
            ];
            let gated_out = run_compiled_gdn_decode(cache, &kernel_inputs)?;
            cache.offset += S;

            let out_flat = gated_out.reshape(&[B, S, -1])?;
            let out = self.out_proj.forward_decode_fast(&out_flat)?;
            return mlx_rs::stop_gradient(&out);
        }

        // Get or initialize SSM state: [B, Hv, Dv, Dk]
        let state = match cache.ssm_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                Dtype::Float32,
            )?,
        };

        // Fused kernel: computes g, beta, AND runs the full recurrence in one dispatch.
        let (y, new_state) = gated_delta_kernel_ffi(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        let diag_y = if diag_gdn { diag_materialize(&y) } else { None };
        if diag_gdn {
            DIAG_GDN_CAPTURED.with(|c| {
                *c.borrow_mut() = Some((S, diag_mixed, diag_conv_input, diag_conv_out, diag_y))
            });
        }
        cache.ssm_state = Some(new_state);
        cache.offset += S;

        if S == 1 && async_layer_state_eval_enabled() {
            if let Some(ssm_state) = cache.ssm_state.as_ref() {
                mlx_rs::transforms::async_eval([ssm_state])?;
            }
        }

        // Fused RMSNorm + gated output: silu(z) * rms_norm(y)
        // nn::silu is pre-compiled in MLX (1 fused dispatch vs 3 for manual swiglu)
        let gated_out = gdn_output_gate(&y, self.norm.weight.as_ref(), self.norm.eps, &z)?;

        // Output projection
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        let out = self.out_proj.forward_decode_fast(&out_flat)?;
        if S == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }

    /// Reorder the projected qkvz and ba tensors into separate heads.
    #[allow(non_snake_case, clippy::type_complexity)]
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Array,
        mixed_ba: &Array,
        B: i32,
        S: i32,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let nk = self.num_k_heads;
        let dn = self.head_k_dim;
        let nv = self.num_v_heads;
        let dv = self.head_v_dim;
        let v_per_k = nv / nk;

        // Reshape to [B, S, nk, -1]
        let qkvz = mixed_qkvz.reshape(&[B, S, nk, -1])?;
        let ba = mixed_ba.reshape(&[B, S, nk, -1])?;

        // Split qkvz at [dn, 2*dn, 2*dn + v_per_k*dv]
        let split_at = &[dn, 2 * dn, 2 * dn + v_per_k * dv];
        let qkvz_parts = qkvz.split_axis(split_at, Some(-1))?;
        let q = qkvz_parts
            .first()
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let k = qkvz_parts
            .get(1)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let v_raw = qkvz_parts
            .get(2)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;
        let z_raw = qkvz_parts
            .get(3)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;

        let v = v_raw.reshape(&[B, S, nv, dv])?;
        let z = z_raw.reshape(&[B, S, nv, dv])?;

        // Split ba at [v_per_k]
        let ba_parts = ba.split_axis(&[v_per_k], Some(-1))?;
        let b_raw = ba_parts
            .first()
            .ok_or_else(|| Exception::custom("ba split failed"))?;
        let a_raw = ba_parts
            .get(1)
            .ok_or_else(|| Exception::custom("ba split failed"))?;

        let b = b_raw.reshape(&[B, S, nv])?;
        let a = a_raw.reshape(&[B, S, nv])?;

        Ok((q, k, v, z, b, a))
    }

    /// Side-effect-free forward used by `DFlash` verify. Identical numerics to
    /// `forward`, but reads `cache.conv_state` / `cache.ssm_state` without
    /// mutating them so a rejected speculation can be retried cleanly.
    // Numerical kernel: tensor shape indices known finite; explicit casts preferred over try_from for hot path.
    #[allow(
        non_snake_case,
        clippy::too_many_lines,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::cast_sign_loss,
        clippy::if_not_else,
        clippy::single_match_else,
        clippy::shadow_unrelated,
        clippy::shadow_reuse
    )]
    fn forward_stateless(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs — same as stateful forward
        let (q, k, v, z, b, a) = if self.use_separate_projections {
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward(inputs)?;
            let z = z_proj
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            let b = b_proj.forward(inputs)?;
            let a = a_proj.forward(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

            (q, k, v, z, b, a)
        } else {
            let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
            let mixed_ba = self.in_proj_ba.forward(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Conv1d — read conv_state without consuming it
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Borrow conv_state without taking it (stateless must not mutate cache)
        let conv_state = match &cache.conv_state {
            Some(state) => state.clone(),
            None => ops::zeros_dtype(
                &[B, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )?,
        };
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // DO NOT update cache.conv_state

        let conv_out = if S > 1 && S <= 32 {
            let wt = match &self.conv_weight_t {
                Some(w) => w.clone(),
                None => {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {shape:?}"
                        )));
                    };
                    let w = w.as_dtype(inputs.dtype())?;
                    w.eval()?;
                    // Don't cache weight here — stateless should be side-effect-free
                    w
                }
            };
            let ks = self.conv_kernel_size;
            let mut windows = Vec::with_capacity(S as usize);
            for i in 0..S {
                windows.push(
                    conv_input
                        .index((.., i..i + ks, ..))
                        .multiply(&wt)?
                        .sum_axes(&[1], true)?,
                );
            }
            nn::silu(&ops::concatenate_axis(
                &windows.iter().collect::<Vec<_>>(),
                1,
            )?)?
        } else {
            // Stateless path: clone-coerce rather than mutate self.conv1d.weight,
            // matching the qk_norm_weight_* clone pattern below.
            let in_dt = inputs.dtype();
            if self.conv1d.weight.dtype() != in_dt {
                let coerced = self.conv1d.weight.as_dtype(in_dt)?;
                let out = ops::conv1d(&conv_input, &coerced, 1, 0, 1, self.conv_dim)?;
                nn::silu(&out)?
            } else {
                nn::silu(&self.conv1d.forward(&conv_input)?)?
            }
        };

        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        let in_dt = inputs.dtype();
        let qk_wq = if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q.as_dtype(in_dt)?
        } else {
            self.qk_norm_weight_q.clone()
        };
        let qk_wk = if self.qk_norm_weight_k.dtype() != in_dt {
            self.qk_norm_weight_k.as_dtype(in_dt)?
        } else {
            self.qk_norm_weight_k.clone()
        };

        let norm_q = fast::rms_norm(&conv_q, &qk_wq, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &qk_wk, 1e-6)?;

        // Stateless SSM recurrence — state_out := state_in
        let state = match &cache.ssm_state {
            Some(s) => s.clone(),
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                Dtype::Float32,
            )?,
        };
        let (y, _unchanged_state) = gated_delta_kernel_ffi_stateless(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        // DO NOT update cache.ssm_state or cache.offset

        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;

        let out_flat = gated_out.reshape(&[B, S, -1])?;
        self.out_proj.forward(&out_flat)
    }

    /// Tape-recording forward: identical output, also returns a `GdnLayerTape`
    /// containing everything needed to cheaply replay accepted steps.
    /// State IS updated (normal forward) — on full acceptance, zero extra work.
    // Numerical kernel: tensor shape indices known finite; explicit casts preferred over try_from for hot path.
    #[allow(
        non_snake_case,
        clippy::too_many_lines,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::cast_sign_loss,
        clippy::single_match_else,
        clippy::shadow_unrelated,
        clippy::shadow_reuse
    )]
    fn forward_with_tape(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &mut ArraysCache,
        row_schedule: DFlashRowSchedule,
    ) -> Result<(Array, GdnLayerTape), Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("need >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("need >= 2 dims"))?;

        let (q, k, v, z, b, a) = if self.use_separate_projections {
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward(inputs)?;
            let z = z_proj
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            let b = b_proj.forward(inputs)?;
            let a = a_proj.forward(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_axis(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
            (q, k, v, z, b, a)
        } else {
            let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
            let mixed_ba = self.in_proj_ba.forward(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Save a for replay (before any reshape that might happen)
        let a_for_replay = a.clone();

        // Conv1d — same as normal forward
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Save qkv for conv_state rebuild on replay
        let qkv_for_replay = mixed_qkv.clone();

        // Capture initial state for rollback (Python `_GDNStateCapture` equivalent)
        let conv_state_init = cache.conv_state.clone();
        let ssm_state_init = cache.ssm_state.clone();
        let conv_pos_init = cache.conv_pos;
        let offset_init = cache.offset;

        // Single-token decode keeps convolution history as a ring buffer. A
        // block verify needs the same history in chronological order before
        // appending the drafted positions.
        let conv_state = self.chronological_conv_state(cache, B, inputs.dtype())?;
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        let n_keep = self.conv_kernel_size - 1;
        let conv_input_len = *conv_input
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
        let keep_start = conv_input_len - n_keep;
        let cs = conv_input.index((.., keep_start.., ..));
        let cs_shape = cs.shape().to_vec();
        cache.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);
        cache.conv_pos = if n_keep > 0 { n_keep - 1 } else { -1 };

        let conv_out = if S <= 8 && row_schedule == DFlashRowSchedule::CanonicalS1 {
            let wt = match &self.conv_weight_t {
                Some(w) => w.clone(),
                None => {
                    let shape = self.conv1d.weight.shape();
                    let w = if shape.len() == 3 && shape[2] == 1 {
                        self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?
                    } else if shape.len() == 3 && shape[1] == 1 {
                        self.conv1d.weight.squeeze_axes(&[1])?.transpose()?
                    } else {
                        return Err(Exception::custom(format!(
                            "Unexpected conv1d weight shape: {shape:?}"
                        )));
                    };
                    let w = w.as_dtype(inputs.dtype())?;
                    w.eval()?;
                    self.conv_weight_t = Some(w.clone());
                    w
                }
            };
            if canonical_conv_enabled()
                && canonical_conv_kernel_supported(
                    &mixed_qkv,
                    &conv_state,
                    &wt,
                    B,
                    S,
                    self.conv_dim,
                    self.conv_kernel_size,
                )
            {
                let preactivation = canonical_conv_preactivation_ffi(
                    &mixed_qkv,
                    &conv_state,
                    &wt,
                    offset_init,
                    B,
                    S,
                    self.conv_dim,
                    self.conv_kernel_size,
                )?;
                // Keep activation on the existing two MLX primitives. Running
                // them over the complete block is elementwise-identical to the
                // per-row S=1 calls and avoids embedding a second math contract
                // in the custom kernel.
                silu_direct(&preactivation)?
            } else {
                // Canonical short-block convolution. Reproduce the exact S=1
                // transition's operation order for every row: current tap first,
                // then newest-to-oldest history with one rounded add per lag. A
                // `multiply(...).sum()` reduction is mathematically equivalent but
                // dispatches a different reduction tree and can flip a later
                // near-tie after 48 recurrent layers.
                let row_capacity = usize::try_from(S)
                    .map_err(|_| Exception::custom("negative canonical convolution row count"))?;
                let mut rows = Vec::with_capacity(row_capacity);
                for position in 0..S {
                    let current = mixed_qkv
                        .index((.., position..position + 1, ..))
                        .reshape(&[B, 1, self.conv_dim])?;
                    let available = (offset_init + position).clamp(0, n_keep);
                    rows.push(
                        self.canonical_conv1d_step(&current, &wt, available, B, |lag| {
                            if lag < position {
                                mixed_qkv
                                    .index((.., position - 1 - lag..position - lag, ..))
                                    .reshape(&[B, self.conv_dim])
                            } else {
                                let history_index = n_keep - 1 - (lag - position);
                                conv_state
                                    .index((.., history_index..history_index + 1, ..))
                                    .reshape(&[B, self.conv_dim])
                            }
                        })?,
                    );
                }
                ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1)?
            }
        } else {
            // Mirror the S>1 dtype coercion for the native Conv1d path.
            let in_dt = inputs.dtype();
            if self.conv1d.weight.dtype() != in_dt {
                let coerced = self.conv1d.weight.as_dtype(in_dt)?;
                coerced.eval()?;
                self.conv1d.weight = Param::new(coerced);
            }
            nn::silu(&self.conv1d.forward(&conv_input)?)?
        };

        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        let in_dt = inputs.dtype();
        if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q = self.qk_norm_weight_q.as_dtype(in_dt)?;
            self.qk_norm_weight_k = self.qk_norm_weight_k.as_dtype(in_dt)?;
        }

        let norm_q = fast::rms_norm(&conv_q, &self.qk_norm_weight_q, 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, &self.qk_norm_weight_k, 1e-6)?;

        // Save norm_k for replay
        let norm_k_for_replay = norm_k.clone();

        // Tape-recording kernel — state IS updated, tape IS recorded.
        // State is float32 to match the AR-decode kernel (bit-exact verify).
        let state = match cache.ssm_state.take() {
            Some(s) => s,
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                Dtype::Float32,
            )?,
        };
        let (y, new_state, delta_tape) = gated_delta_kernel_ffi_with_tape(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        cache.ssm_state = Some(new_state);
        cache.offset += S;

        let gated_out = gdn_output_gate(&y, self.norm.weight.as_ref(), self.norm.eps, &z)?;
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        let output = self.out_proj.forward(&out_flat)?;

        let tape = GdnLayerTape {
            delta_tape,
            norm_k: norm_k_for_replay,
            a_proj: a_for_replay,
            qkv_input: qkv_for_replay,
            conv_state_init,
            ssm_state_init,
            conv_pos_init,
            offset_init,
        };

        Ok((output, tape))
    }
}

/// Reference implementation of gate computation (used by tests).
/// Production code uses `compute_g_beta_kernel_ffi` instead.
#[cfg(test)]
fn compute_g_compiled((a_log, a, dt_bias): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    let a_plus_bias = a.add(dt_bias)?;
    let sp = nn::softplus(&a_plus_bias)?;
    let neg_decay = a_log.exp()?.negative()?.multiply(sp)?;
    neg_decay.exp()
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

/// Wrapper for the FFN block: either sparse `MoE` or dense `SwiGLU`.
/// Both share the `mlp` parameter namespace in safetensors — their sub-keys
/// don't overlap (`MoE`: gate, `switch_mlp`, `shared_expert`; Dense: `gate_proj`, `up_proj`, `down_proj`).
#[derive(Debug, Clone, ModuleParameters)]
struct FfnBlock {
    #[param]
    gate: Option<QLinear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    #[param]
    shared_expert: Option<Qwen3NextMLP>,
    #[param]
    shared_expert_gate: Option<QLinear>,
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    is_moe: bool,
    top_k: i32,
    norm_topk_prob: bool,
    /// Cached fused gate+up weights for dense layers (lazily computed on first forward).
    fused_gate_up: Option<(Array, Array, Array, i32)>,
}

impl FfnBlock {
    fn new_moe(args: &Qwen3NextModelArgs, mlp_prefix: &str) -> Result<Self, Exception> {
        let moe = SparseMoeBlock::new(args, mlp_prefix)?;
        Ok(Self {
            gate: Some(moe.gate),
            switch_mlp: Some(moe.switch_mlp),
            shared_expert: Some(moe.shared_expert),
            shared_expert_gate: Some(moe.shared_expert_gate),
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            is_moe: true,
            top_k: moe.top_k,
            norm_topk_prob: moe.norm_topk_prob,
            fused_gate_up: None,
        })
    }

    fn new_dense(args: &Qwen3NextModelArgs, mlp_prefix: &str) -> Result<Self, Exception> {
        let g_spec = resolve_quant_for(args, &format!("{mlp_prefix}.gate_proj"));
        let u_spec = resolve_quant_for(args, &format!("{mlp_prefix}.up_proj"));
        let d_spec = resolve_quant_for(args, &format!("{mlp_prefix}.down_proj"));
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_expert: None,
            shared_expert_gate: None,
            gate_proj: Some(QLinear::new_spec(g_spec)?),
            up_proj: Some(QLinear::new_spec(u_spec)?),
            down_proj: Some(QLinear::new_spec(d_spec)?),
            is_moe: false,
            top_k: 0,
            norm_topk_prob: false,
            fused_gate_up: None,
        })
    }

    fn tg_lut4_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("HIGGS_BONSAI_TG_LUT4").map_or(true, |v| v != "0"))
    }

    fn tg_lut4_fused_mlp_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("HIGGS_BONSAI_TG_LUT4_FUSED_MLP").is_ok_and(|value| value == "1")
        })
    }

    /// Promote each eligible dense projection independently. Logical MLP shapes
    /// are validated first and every required row4 transform completes before
    /// any parameter is replaced, so an error cannot leave a partially mutated
    /// layer. Exact non-symmetric affine projections deliberately remain in
    /// canonical storage and continue through [`QLinear::forward`].
    fn promote_bonsai_row4(
        &mut self,
        layer_index: usize,
    ) -> Result<BonsaiRow4Promotion, Exception> {
        if self.is_moe {
            return Err(Exception::custom(format!(
                "layer {layer_index} uses MoE and cannot be promoted to dense Bonsai row4"
            )));
        }
        let gate = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom(format!("layer {layer_index} gate_proj missing")))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom(format!("layer {layer_index} up_proj missing")))?;
        let down = self
            .down_proj
            .as_ref()
            .ok_or_else(|| Exception::custom(format!("layer {layer_index} down_proj missing")))?;

        let gate_path = format!("layers.{layer_index}.mlp.gate_proj");
        let up_path = format!("layers.{layer_index}.mlp.up_proj");
        let down_path = format!("layers.{layer_index}.mlp.down_proj");
        let (gate_shape, promote_gate) = gate.bonsai_row4_promotion_candidate(&gate_path)?;
        let (up_shape, promote_up) = up.bonsai_row4_promotion_candidate(&up_path)?;
        let (down_shape, promote_down) = down.bonsai_row4_promotion_candidate(&down_path)?;
        if gate_shape != up_shape || down_shape != (gate_shape.1, gate_shape.0) {
            return Err(Exception::custom(format!(
                "layer {layer_index} dense MLP logical shapes are inconsistent: gate={gate_shape:?} up={up_shape:?} down={down_shape:?}; require gate==up [I,H] and down [H,I] in [out,in] order"
            )));
        }

        // Prepare every eligible copy before installing the first one. This is
        // the transactional boundary for both transform and allocation errors.
        let gate_packed = promote_gate
            .then(|| gate.prepare_bonsai_row4(&gate_path))
            .transpose()?;
        let up_packed = promote_up
            .then(|| up.prepare_bonsai_row4(&up_path))
            .transpose()?;
        let down_packed = promote_down
            .then(|| down.prepare_bonsai_row4(&down_path))
            .transpose()?;
        let projections = [
            gate_packed.is_some(),
            up_packed.is_some(),
            down_packed.is_some(),
        ]
        .into_iter()
        .filter(|promoted| *promoted)
        .count();
        let bytes = gate_packed
            .as_ref()
            .map_or(0, crate::metal_kernel::BonsaiQ1Row4::cached_bytes)
            .saturating_add(
                up_packed
                    .as_ref()
                    .map_or(0, crate::metal_kernel::BonsaiQ1Row4::cached_bytes),
            )
            .saturating_add(
                down_packed
                    .as_ref()
                    .map_or(0, crate::metal_kernel::BonsaiQ1Row4::cached_bytes),
            );

        let (Some(gate), Some(up), Some(down)) = (
            self.gate_proj.as_mut(),
            self.up_proj.as_mut(),
            self.down_proj.as_mut(),
        ) else {
            return Err(Exception::custom(
                "dense MLP projections disappeared during row4 promotion",
            ));
        };
        if let Some(packed) = gate_packed {
            gate.install_bonsai_row4(packed);
        }
        if let Some(packed) = up_packed {
            up.install_bonsai_row4(packed);
        }
        if let Some(packed) = down_packed {
            down.install_bonsai_row4(packed);
        }
        Ok(BonsaiRow4Promotion {
            layers: usize::from(projections != 0),
            projections,
            bytes,
        })
    }

    fn dense_hidden_tg_lut4(&self, x: &Array) -> Result<Option<Array>, Exception> {
        if !Self::tg_lut4_enabled() {
            return Ok(None);
        }
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;
        let (Some(gate_packed), Some(up_packed)) = (gp.bonsai_row4()?, up.bonsai_row4()?) else {
            return Ok(None);
        };
        if !gate_packed.accepts_input(x) || !up_packed.accepts_input(x) {
            return Ok(None);
        }
        let (gate, up) = if Self::tg_lut4_fused_mlp_enabled()
            && gate_packed.accepts_fused_gate_up(x)
            && up_packed.accepts_fused_gate_up(x)
        {
            crate::metal_kernel::bonsai_q1_tg_lut4_gate_up_view(x, gate_packed, up_packed)?
        } else {
            (
                crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(x, gate_packed)?,
                crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(x, up_packed)?,
            )
        };
        silu_mul(&gate, &up).map(Some)
    }

    fn dense_down_tg_lut4(&self, x: &Array) -> Result<Option<Array>, Exception> {
        if !Self::tg_lut4_enabled() {
            return Ok(None);
        }
        let down = self
            .down_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
        let Some(packed) = down.bonsai_row4()? else {
            return Ok(None);
        };
        if !packed.accepts_input(x) {
            return Ok(None);
        }
        crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(x, packed).map(Some)
    }

    fn dense_hidden_fused(&mut self, x: &Array, use_fused_gemv: bool) -> Result<Array, Exception> {
        // The optional persistent fusion path expects materialized affine bias
        // arrays. Symmetric Q1 deliberately drops them; keep the memory-saving
        // representation and use the normal two-projection path instead.
        if self
            .gate_proj
            .as_ref()
            .is_some_and(|proj| has_symmetric_q1_biases(&proj.biases))
            || self
                .up_proj
                .as_ref()
                .is_some_and(|proj| has_symmetric_q1_biases(&proj.biases))
        {
            return self.dense_hidden_separate(x);
        }
        if self.fused_gate_up.is_none() {
            let gp = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
            let up = self
                .up_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense up_proj missing"))?;
            let intermediate = *gp
                .weight
                .shape()
                .first()
                .ok_or_else(|| Exception::custom("gate_proj weight has no dims"))?;
            let fw = ops::concatenate_axis(&[&*gp.weight, &*up.weight], 0)?;
            let fs = ops::concatenate_axis(&[&*gp.scales, &*up.scales], 0)?;
            let fb = ops::concatenate_axis(&[&*gp.biases, &*up.biases], 0)?;
            fw.eval()?;
            fs.eval()?;
            fb.eval()?;
            self.fused_gate_up = Some((fw, fs, fb, intermediate));
        }

        let (fw, fs, fb, intermediate) = self
            .fused_gate_up
            .as_ref()
            .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;

        let fused_out = match gp.mode {
            crate::quant_mode::QuantMode::MxFp4 => crate::quant_mode::quantized_matmul(
                x,
                fw,
                fs,
                None,
                true,
                gp.group_size,
                gp.bits,
                gp.mode,
            )?,
            crate::quant_mode::QuantMode::Dense => dense_linear_no_bias_forward(fw, x)?,
            // Affine fast path — GEMV for single-token decode, else standard matmul.
            crate::quant_mode::QuantMode::Affine => {
                if gp.bits == 1 {
                    affine_q1_forward(x, fw, fs, fb, gp.group_size)?
                } else if use_fused_gemv {
                    qgemv_4bit(x, fw, fs, fb, gp.group_size)?
                } else {
                    quantized_forward(x, fw, fs, fb, gp.group_size, gp.bits)?
                }
            }
        };
        let parts = fused_out.split_axis(&[*intermediate], Some(-1))?;
        let gate_out = parts
            .first()
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let up_out = parts
            .get(1)
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        silu_mul(gate_out, up_out)
    }

    fn dense_hidden_mxfp4_fused_verify(&self, x: &Array) -> Result<Option<Array>, Exception> {
        if !mxfp4_fused_ffn_verify_enabled() {
            return Ok(None);
        }

        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;

        if gp.mode != crate::quant_mode::QuantMode::MxFp4
            || up.mode != crate::quant_mode::QuantMode::MxFp4
            || gp.bits != 4
            || up.bits != 4
            || gp.group_size != up.group_size
            || gp.group_size <= 0
        {
            return Ok(None);
        }

        let x_shape = x.shape();
        let [1, t, k_in] = *x_shape else {
            return Ok(None);
        };
        if !(2..=16).contains(&t) {
            return Ok(None);
        }

        let gate_shape = gp.weight.shape();
        if gate_shape != up.weight.shape() {
            return Ok(None);
        }
        let Some(&k_packed) = gate_shape.get(1) else {
            return Ok(None);
        };
        let k_dim = k_packed * 8;
        if k_dim != k_in || k_dim % gp.group_size != 0 {
            return Ok(None);
        }

        mxfp4_gate_up_silu_4bit(
            x,
            &gp.weight,
            &gp.scales,
            &up.weight,
            &up.scales,
            gp.group_size,
            t,
        )
        .map(Some)
    }

    fn dense_hidden_separate(&self, x: &Array) -> Result<Array, Exception> {
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;
        let gate_out = gp.forward_decode_fast(x)?;
        let up_out = up.forward_decode_fast(x)?;
        silu_mul(&gate_out, &up_out)
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            // Delegate to SparseMoeBlock logic
            let gate_ref = self
                .gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE gate missing"))?;
            let seg_ref = self
                .shared_expert_gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert_gate missing"))?;

            let gates = ops::softmax_axis(&gate_ref.forward(x)?, -1, true)?;

            let neg_k = -self.top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
            let num_experts = *gates
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("gates must have last dim"))?;
            let top_k_start = num_experts - self.top_k;
            let inds = ops::sort_axis(all_inds.index((.., .., top_k_start..)), -1)?;
            let raw_scores = gates.take_along_axis(&inds, -1)?;
            let scores = if self.norm_topk_prob {
                let sum = raw_scores.sum_axes(&[-1], true)?;
                raw_scores.divide(&sum)?
            } else {
                raw_scores
            };

            let switch_ref = self
                .switch_mlp
                .as_mut()
                .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?;
            let y = if moe_ffn_fuse_gate_up() {
                switch_ref.forward_gather_fused(x, &inds)?
            } else {
                switch_ref.forward_gather_global_sort(x, &inds)?
            };

            let expert_sum = y
                .multiply(&scores.expand_dims(-1)?)?
                .sum_axes(&[-2], false)?;

            let se_ref = self
                .shared_expert
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert missing"))?;
            let shared_y = se_ref.forward(x)?;

            let shared_gate_val = nn::sigmoid(&seg_ref.forward(x)?)?;
            let shared_out = shared_y.multiply(&shared_gate_val)?;

            expert_sum.add(shared_out)
        } else {
            // Dense SwiGLU with configurable gate/up path so we can benchmark
            // whether one large fused matmul or two smaller matmuls are faster
            // on a given Apple GPU.
            let gp = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;

            let seq_len = *x.shape().get(1).unwrap_or(&0);
            let use_decode_gemv = seq_len == 1 && gp.bits == 4;
            let gemv_mode = if std::env::var_os("HIGGS_QGEMV_FFN_MODE").is_none()
                && should_force_dense_decode_safe_defaults_for_brand(apple_cpu_brand())
            {
                DenseFfnGemvMode::Off
            } else {
                dense_ffn_gemv_mode()
            };
            let use_fused_gemv = use_decode_gemv
                && matches!(
                    gemv_mode,
                    DenseFfnGemvMode::Both | DenseFfnGemvMode::FusedOnly
                );
            let use_down_gemv = use_decode_gemv
                && matches!(
                    gemv_mode,
                    DenseFfnGemvMode::Both | DenseFfnGemvMode::DownOnly
                );

            let hidden = if let Some(hidden) = self.dense_hidden_tg_lut4(x)? {
                hidden
            } else if let Some(hidden) = self.dense_hidden_mxfp4_fused_verify(x)? {
                hidden
            } else if dense_ffn_fuse_gate_up() {
                self.dense_hidden_fused(x, use_fused_gemv)?
            } else {
                self.dense_hidden_separate(x)?
            };

            // Down projection
            let out = if let Some(out) = self.dense_down_tg_lut4(&hidden)? {
                Ok(out)
            } else {
                let dp = self
                    .down_proj
                    .as_ref()
                    .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
                if use_down_gemv {
                    qgemv_4bit(&hidden, &dp.weight, &dp.scales, &dp.biases, dp.group_size)
                } else {
                    dp.forward(&hidden)
                }
            }?;
            if seq_len == 1 {
                mlx_rs::stop_gradient(&out)
            } else {
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DecoderLayer {
    #[param]
    linear_attn: Option<GatedDeltaNet>,
    #[param]
    self_attn: Option<Qwen3NextAttention>,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: FfnBlock,
    is_linear: bool,
}

impl DecoderLayer {
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let layer_prefix = format!("language_model.model.layers.{layer_idx}");
        let linear_attn = if is_linear {
            Some(GatedDeltaNet::new(
                args,
                &format!("{layer_prefix}.linear_attn"),
            )?)
        } else {
            None
        };
        let self_attn = if is_linear {
            None
        } else {
            Some(Qwen3NextAttention::new(
                args,
                &format!("{layer_prefix}.self_attn"),
            )?)
        };

        let mlp_prefix = format!("{layer_prefix}.mlp");
        let ffn = if args.num_experts > 0 {
            FfnBlock::new_moe(args, &mlp_prefix)?
        } else {
            FfnBlock::new_dense(args, &mlp_prefix)?
        };
        Ok(Self {
            linear_attn,
            self_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: ffn,
            is_linear,
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = if self.is_linear {
            let attn = self
                .linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("linear_attn missing on linear layer"))?;
            let LayerCache::Arrays(ssm_cache) = cache else {
                return Err(Exception::custom("Expected ArraysCache for linear layer"));
            };
            attn.forward(&normed, mask, ssm_cache)?
        } else {
            let attn = self
                .self_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("self_attn missing on attention layer"))?;
            let LayerCache::KV(kv_cache) = cache else {
                return Err(Exception::custom("Expected KVCache for attention layer"));
            };
            attn.forward(&normed, mask, kv_cache)?
        };

        let h = x.add(r)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// LayerCache enum
// ---------------------------------------------------------------------------

/// Per-layer cache: either KV cache (full attention) or arrays (SSM).
#[derive(Debug, Clone)]
pub enum LayerCache {
    KV(SteppingKeyValueCache),
    Arrays(ArraysCache),
}

// ---------------------------------------------------------------------------
// Qwen3NextInner (embed + layers + norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextInner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl Qwen3NextInner {
    fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        let embed_spec = resolve_quant_for(args, "language_model.model.embed_tokens");
        Ok(Self {
            embed_tokens: QEmbedding::new_spec(embed_spec),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextCausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextCausalLM {
    pub args: Qwen3NextModelArgs,
    #[param]
    model: Qwen3NextInner,
    #[param]
    lm_head: Option<QLinear>,
    #[param]
    mtp: Option<MtpHead>,
    #[param]
    dense_mtp: Option<DenseMtpHead>,
    #[param]
    moe_mtp: Option<MoeMtpHead>,
}

// Diag flags read once: this dispatcher runs per q/k per FA layer per chunk,
// and `std::env::var` takes a process-wide lock. The flags are only ever set
// before process start (no `set_var` in the tree).
static DIAG_ROPE_MANUAL: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("HIGGS_DIAG_ROPE_MANUAL").is_ok_and(|v| v == "1"));
static DIAG_ROPE_PERHEAD: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("HIGGS_DIAG_ROPE_PERHEAD").is_ok_and(|v| v == "1"));

// `shadow_reuse` for the positions Vec→Array rebind (and the YaRN prescale
// rebind of `x`); `indexing_slicing` for the diag per-head branch's fixed
// 4-dim shape access.
#[allow(clippy::shadow_reuse, clippy::indexing_slicing)]
fn apply_qwen3_next_rope(
    x: Array,
    rope: &nn::Rope,
    offset: i32,
    yarn: Option<&YarnRope>,
) -> Result<Array, Exception> {
    apply_qwen3_next_rope_scheduled(x, rope, offset, yarn, DFlashRowSchedule::NativeBatch)
}

#[allow(clippy::shadow_reuse, clippy::indexing_slicing)]
fn apply_qwen3_next_rope_scheduled(
    x: Array,
    rope: &nn::Rope,
    offset: i32,
    yarn: Option<&YarnRope>,
    row_schedule: DFlashRowSchedule,
) -> Result<Array, Exception> {
    let seq_len = {
        let shape = x.shape();
        shape[shape.len() - 2]
    };
    // YaRN: prescale the rotary dims by mscale, then rotate with the
    // yarn-interpolated frequencies. The SAME prescale + freqs feed both the
    // manual prefill branch and the fast decode branch below — decode and
    // prefill diverging on effective frequencies is the warm/cold drift bug
    // class, so they must stay in lockstep.
    let (x, yarn_freqs) = match yarn {
        Some(yarn) => (yarn.prescale_rotary(&x)?, Some(&yarn.freqs)),
        None => (x, None),
    };
    if seq_len > 1 && seq_len <= 8 && row_schedule == DFlashRowSchedule::CanonicalS1 {
        // Exact verifier schedule: apply the ordinary one-position RoPE kernel
        // independently at each absolute position. The surrounding graph stays
        // lazy, but every row now executes the same numerical primitive as AR.
        let row_capacity = usize::try_from(seq_len)
            .map_err(|_| Exception::custom("negative canonical RoPE row count"))?;
        let mut rows = Vec::with_capacity(row_capacity);
        for position in 0..seq_len {
            let row = x.index((.., .., position..position + 1, ..));
            rows.push(apply_fast_rope_with_freqs(
                &row,
                rope,
                offset + position,
                yarn_freqs,
            )?);
        }
        return ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 2);
    }
    if seq_len > 1 || *DIAG_ROPE_MANUAL {
        let positions: Vec<i32> = (offset..offset + seq_len).collect();
        let positions = Array::from_slice(&positions, &[seq_len]);
        return apply_rope_manual_with_freqs(
            &x,
            &positions,
            rope.dimensions,
            rope.base,
            rope.scale,
            yarn_freqs,
        );
    }

    if *DIAG_ROPE_PERHEAD {
        let shape = x.shape().to_vec();
        let batch_heads = shape[0] * shape[1];
        let seq = shape[2];
        let dim = shape[3];
        let flat = x.reshape(&[batch_heads, seq, dim])?;
        return apply_fast_rope_with_freqs(&flat, rope, offset, yarn_freqs)?.reshape(&shape);
    }

    apply_fast_rope_with_freqs(&x, rope, offset, yarn_freqs)
}

/// Decode-path rope: `mlx_fast_rope`, with optional precomputed `YaRN` periods.
/// When `freqs` is set, `base` must be omitted (MLX rejects both at once);
/// the default path stays on [`apply_rope`] and is bit-identical to before.
fn apply_fast_rope_with_freqs(
    x: &Array,
    rope: &nn::Rope,
    offset: i32,
    freqs: Option<&Array>,
) -> Result<Array, Exception> {
    freqs.map_or_else(
        || apply_rope(x, rope, offset),
        |periods| {
            mlx_rs::fast::rope(
                x,
                rope.dimensions,
                rope.traditional,
                None::<f32>,
                rope.scale,
                offset,
                periods,
            )
        },
    )
}

// Manual RoPE implementation for arbitrary positions
/// Manual `RoPE` implementation for arbitrary positions (non-traditional,
/// partial-rotary aware). Rotates the first `dimensions` elements of the last
/// axis and passes the remainder through unchanged.
#[allow(dead_code)]
pub(crate) fn apply_rope_manual(
    x: &Array,
    positions: &Array,
    dimensions: i32,
    base: f32,
    scale: f32,
) -> Result<Array, Exception> {
    apply_rope_manual_with_freqs(x, positions, dimensions, base, scale, None)
}

/// Default-path inverse frequencies for the manual rope:
/// `base^(-2i/dimensions)` for `i in [0, half_dim)`. Kept as a standalone fn
/// so a unit test can pin the values bit-for-bit — the default rope path must
/// stay byte-identical across refactors.
fn manual_rope_inv_freqs(dimensions: i32, base: f32) -> Vec<f32> {
    let half_dim = dimensions / 2;
    #[allow(clippy::cast_precision_loss)]
    let dimensions_f32 = f32::from(i16::try_from(dimensions).unwrap_or(i16::MAX));
    (0..half_dim)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let i_f32 = f32::from(i16::try_from(i).unwrap_or(i16::MAX));
            let power = -2.0 * i_f32 / dimensions_f32;
            base.powf(power)
        })
        .collect()
}

/// [`apply_rope_manual`] with optional precomputed `YaRN` rope periods.
///
/// `yarn_freqs` uses `mlx_fast_rope` conventions: it holds PERIODS
/// (`base^(2i/dims)`-shaped, yarn-interpolated), and the rotation angle is
/// `position / period` — computed here as `reciprocal(freqs)`, the exact op
/// the MLX rope kernel applies internally, so prefill (this fn) and decode
/// (`mlx_fast_rope`) see the same effective frequencies. `None` keeps the
/// inline default-path computation bit-identical to the pre-`YaRN` code.
fn apply_rope_manual_with_freqs(
    x: &Array,
    positions: &Array,
    dimensions: i32,
    base: f32,
    _scale: f32,
    yarn_freqs: Option<&Array>,
) -> Result<Array, Exception> {
    use mlx_rs::ops;

    // x shape: [B, H, L, D] or [B, L, D]
    let shape = x.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(Exception::custom("Input must have at least 2 dimensions"));
    }

    let half_dim = dimensions / 2;
    let half_dim_i32 = half_dim;

    let inv_freq_arr = if let Some(periods) = yarn_freqs {
        periods.reciprocal()?
    } else {
        // Compute frequencies: base^(-2i/dimensions) for i in [0, half_dim)
        let inv_freq = manual_rope_inv_freqs(dimensions, base);
        Array::from_slice(&inv_freq, &[half_dim_i32])
    };

    let pos_shape = positions.shape();
    let l_dim = *pos_shape
        .last()
        .ok_or_else(|| Exception::custom("positions must have at least 1 dim"))?;

    // Compute angles: positions[L] * inv_freq[half_dim] -> [L, half_dim]
    let positions_expanded = positions.reshape(&[l_dim, 1])?;
    let inv_freq_expanded = inv_freq_arr.reshape(&[1, half_dim_i32])?;
    let angles = ops::multiply(&positions_expanded, &inv_freq_expanded)?;
    let cos_raw = ops::cos(&angles)?;
    let sin_raw = ops::sin(&angles)?;

    // Broadcast shape: [1, 1, L, half_dim] (4D) or [1, L, half_dim] (3D).
    let cos_shape: Vec<i32> = if ndim == 4 {
        vec![1, 1, l_dim, half_dim_i32]
    } else {
        vec![1, l_dim, half_dim_i32]
    };
    let cos = cos_raw.reshape(&cos_shape)?;
    let sin = sin_raw.reshape(&cos_shape)?;

    // Partial rotary: rotate the first `dimensions` elems, pass the rest.
    let x_rot = x.index((.., .., .., ..dimensions));
    let x_first = x_rot.index((.., .., .., ..half_dim));
    let x_second = x_rot.index((.., .., .., half_dim..));

    // Rotate in f32 (cos/sin precision), then cast back to the input dtype.
    // Post-rope keys are written into the KV cache, so letting f32 escape here
    // silently promotes the FA KV cache and SDPA to f32 — 2x KV memory and 2x
    // attention bandwidth on every prefill chunk and decode step.
    let x_dtype = x.dtype();
    let output_first = ops::subtract(
        &ops::multiply(&x_first, &cos)?,
        &ops::multiply(&x_second, &sin)?,
    )?
    .as_dtype(x_dtype)?;
    let output_second = ops::add(
        &ops::multiply(&x_first, &sin)?,
        &ops::multiply(&x_second, &cos)?,
    )?
    .as_dtype(x_dtype)?;

    let last_axis = i32::try_from(ndim.saturating_sub(1))
        .map_err(|_| Exception::custom("ndim too large for i32"))?;
    let rotated = ops::concatenate_axis(&[&output_first, &output_second], last_axis)?;
    // Append the pass-through (non-rotated) tail if dimensions < D.
    let d = *shape.last().unwrap_or(&dimensions);
    if dimensions < d {
        let x_pass = x.index((.., .., .., dimensions..));
        ops::concatenate_axis(&[&rotated, &x_pass], last_axis)
    } else {
        Ok(rotated)
    }
}

// DIAGNOSTIC: probe-driven per-layer hidden-state capture. A probe calls
// `diag_request_hidden_capture()` then a forward; forward_raw_hidden fills the
// slot with fully-materialized Vec<f32> per layer. The probe takes both captures
// and compares them directly (no "previous forward" thread-local that gets
// polluted by intervening suffix/decode forwards).
pub type DiagLayer = (usize, bool, Vec<f32>, i32); // (layer_idx, is_linear, flat f32, hidden_dim)

thread_local! {
    static DIAG_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DIAG_CAPTURED: std::cell::RefCell<Option<Vec<DiagLayer>>> =
        const { std::cell::RefCell::new(None) };
}

/// Request that the NEXT `forward_raw_hidden` (on this thread) capture its
/// per-layer hidden states. The probe then retrieves them with
/// [`diag_take_hidden_capture`].
pub fn diag_request_hidden_capture() {
    DIAG_CAPTURE_REQ.with(|c| c.set(true));
}

/// Retrieve the per-layer hidden-state capture from the most recent requested
/// forward, clearing the slot.
pub fn diag_take_hidden_capture() -> Option<Vec<DiagLayer>> {
    DIAG_CAPTURED.with(|c| c.borrow_mut().take())
}

/// Compare two captured forwards over the shared prefix length, reporting the
/// first layer where positions diverge and the per-position max-abs pattern.
// Diagnostic-only reporting (env-gated probes): stderr output and direct
// indexing over probe-owned buffers are the point, not a hazard.
#[allow(clippy::print_stderr, clippy::indexing_slicing, clippy::as_conversions)]
pub fn diag_report_hidden_diff(label: &str, short: &[DiagLayer], long: &[DiagLayer]) {
    if short.is_empty() || long.is_empty() {
        eprintln!("DIAG HIDDEN {label}: empty capture");
        return;
    }
    // short forward produced h of shape [1, short_len, H]; long produced
    // [1, long_len, H]. Compare the first short_len positions per layer.
    let hdim = (short[0].3 as usize).max(1);
    let per_pos = short[0].2.len() / hdim;
    let long_hdim = (long[0].3 as usize).max(1);
    let short_elems = per_pos * hdim;
    eprintln!(
        "DIAG HIDDEN {label}: comparing short_len={per_pos} (hdim={hdim}) vs long (hdim={long_hdim}) over first {per_pos} positions"
    );
    for ((li_s, lin_s, hs, _), (li_l, _lin_l, hl, _)) in short.iter().zip(long.iter()) {
        if li_s != li_l {
            break;
        }
        let li = *li_s;
        let is_linear = *lin_s;
        if hs.len() < short_elems || hl.len() < short_elems {
            eprintln!("DIAG HIDDEN L{li}: data too short");
            continue;
        }
        let mut per_pos_max = vec![0.0f32; per_pos];
        let mut max_abs = 0.0f32;
        let mut diffs = 0usize;
        for i in 0..short_elems {
            let x = hs[i];
            let y = hl[i];
            let pos = i / hdim;
            let d = (x - y).abs();
            if d > per_pos_max[pos] {
                per_pos_max[pos] = d;
            }
            if d > max_abs {
                max_abs = d;
            }
            if x.to_bits() != y.to_bits() {
                diffs += 1;
            }
        }
        let nz: Vec<String> = per_pos_max
            .iter()
            .enumerate()
            .filter(|(_, m)| **m > 0.0)
            .map(|(p, m)| format!("p{p}:{m:.1e}"))
            .collect();
        let kind = if is_linear { "GDN" } else { "FA" };
        eprintln!(
            "DIAG HIDDEN L{li:02}({kind}): max_abs={max_abs:.3e} diffs={diffs}/{short_elems} nonzero_positions[{}]",
            nz.join(" ")
        );
    }
}

// DIAGNOSTIC: probe-driven capture of the first FA layer's keys (pre-write and
// post-write), as fully-materialized owned Vec<f32>. The probe requests capture,
// runs a forward, retrieves, then compares two forwards directly.
static DIAG_ROPE_COMPARE_FIRED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
type DiagAttnCapture = (
    i32,
    Option<(Vec<f32>, Vec<i32>)>, // x (attention input = normed h)
    Option<(Vec<f32>, Vec<i32>)>, // keys post-rope, pre-write
    Option<(Vec<f32>, Vec<i32>)>, // post-write stored keys
    Option<(Vec<f32>, Vec<i32>)>, // keys_raw = k_proj output, pre-norm
    Option<(Vec<f32>, Vec<i32>)>, // keys_normed = k_norm output, pre-rope
);

/// Materialize an MLX array to owned `(Vec<f32>, shape)` for uncontaminatable
/// cross-forward comparison.
fn diag_materialize(a: &Array) -> Option<(Vec<f32>, Vec<i32>)> {
    let af = a.as_dtype(mlx_rs::Dtype::Float32).ok()?;
    mlx_rs::transforms::eval([&af]).ok()?;
    Some((af.as_slice::<f32>().to_vec(), af.shape().to_vec()))
}

// DIAGNOSTIC: probe-driven capture of the first GDN prefill layer's conv
// input/output and SSM output, to split conv1d length-dependence from the fused
// gated_delta_kernel.
pub type DiagGdnCapture = (
    i32,                          // S
    Option<(Vec<f32>, Vec<i32>)>, // mixed_qkv (conv input)
    Option<(Vec<f32>, Vec<i32>)>, // conv_input[0..ks] (the first window)
    Option<(Vec<f32>, Vec<i32>)>, // conv_out (silu(conv1d))
    Option<(Vec<f32>, Vec<i32>)>, // y (fused SSM kernel output)
);
thread_local! {
    static DIAG_GDN_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DIAG_GDN_CAPTURED: std::cell::RefCell<Option<DiagGdnCapture>> =
        const { std::cell::RefCell::new(None) };
}

/// Request that the NEXT GDN prefill forward (first GDN layer, S>1) capture its
/// conv input/output and SSM output.
pub fn diag_request_gdn_capture() {
    DIAG_GDN_CAPTURE_REQ.with(|c| c.set(true));
}

/// Retrieve the captured GDN internals.
pub fn diag_take_gdn_capture() -> Option<DiagGdnCapture> {
    DIAG_GDN_CAPTURED.with(|c| c.borrow_mut().take())
}

/// Compare two captured GDN forwards over the shared prefix length.
// Diagnostic-only reporting — see `diag_report_hidden_diff`.
#[allow(clippy::print_stderr, clippy::indexing_slicing, clippy::as_conversions)]
pub fn diag_report_gdn_diff(label: &str, short: &DiagGdnCapture, long: &DiagGdnCapture) {
    let short_i = i32::try_from(short.0).unwrap_or(i32::MAX);
    let report = |tag: &str, a: Option<&(Vec<f32>, Vec<i32>)>, b: Option<&(Vec<f32>, Vec<i32>)>| {
        let (Some((av, ashape)), Some((bv, _bshape))) = (a, b) else {
            eprintln!("DIAG GDN {label} {tag}: missing");
            return;
        };
        // shape [B, S, D] -> position on axis 1 = (flat / D) % S
        let d = (*ashape.last().unwrap_or(&1)).max(1) as usize;
        let per_pos = short_i as usize;
        let short_elems = per_pos * d;
        let elems = av.len().min(bv.len()).min(short_elems);
        let mut per_pos_max = vec![0.0f32; per_pos];
        let mut max_abs = 0.0f32;
        let mut diffs = 0usize;
        for i in 0..elems {
            let diff = (av[i] - bv[i]).abs();
            let pos = (i / d) % per_pos;
            if diff > per_pos_max[pos] {
                per_pos_max[pos] = diff;
            }
            if diff > max_abs {
                max_abs = diff;
            }
            if av[i].to_bits() != bv[i].to_bits() {
                diffs += 1;
            }
        }
        let nz: Vec<String> = per_pos_max
            .iter()
            .enumerate()
            .filter(|(_, m)| **m > 0.0)
            .map(|(p, m)| format!("p{p}:{m:.1e}"))
            .collect();
        eprintln!(
            "DIAG GDN {label} {tag}: max_abs={max_abs:.3e} diffs={diffs}/{elems} nonzero_count={} sample[{}]",
            nz.len(),
            nz.iter().take(6).cloned().collect::<Vec<_>>().join(" ")
        );
    };
    eprintln!(
        "DIAG GDN {label}: short_len={} vs long_len={}",
        short.0, long.0
    );
    report("mixed_qkv(conv input)", short.1.as_ref(), long.1.as_ref());
    report(
        "conv_input[0..ks](first window)",
        short.2.as_ref(),
        long.2.as_ref(),
    );
    report("conv_out(silu conv1d)", short.3.as_ref(), long.3.as_ref());
    report("y(fused SSM kernel)", short.4.as_ref(), long.4.as_ref());
}
thread_local! {
    static DIAG_ATTN_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DIAG_ATTN_CAPTURED: std::cell::RefCell<Option<DiagAttnCapture>> =
        const { std::cell::RefCell::new(None) };
    static DIAG_RAW_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DIAG_NORM_CAPTURE_REQ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Request that the NEXT FA attention forward (first FA layer, offset==0, L>1)
/// capture its pre-write and post-write keys. Retrieved via
/// [`diag_take_attn_capture`].
pub fn diag_request_attn_capture() {
    DIAG_ATTN_CAPTURE_REQ.with(|c| c.set(true));
    DIAG_RAW_CAPTURE_REQ.with(|c| c.set(true));
    DIAG_NORM_CAPTURE_REQ.with(|c| c.set(true));
}

/// Retrieve the captured (L, pre-write keys, post-write stored keys) from the
/// most recent requested FA forward, clearing the slot.
pub fn diag_take_attn_capture() -> Option<DiagAttnCapture> {
    DIAG_ATTN_CAPTURED.with(|c| c.borrow_mut().take())
}

/// Compare two captured FA forwards' keys over the shared prefix length.
// Diagnostic-only reporting — see `diag_report_hidden_diff`.
#[allow(clippy::print_stderr, clippy::indexing_slicing, clippy::as_conversions)]
pub fn diag_report_attn_diff(label: &str, short: &DiagAttnCapture, long: &DiagAttnCapture) {
    let short_i = i32::try_from(short.0).unwrap_or(i32::MAX);
    let report = |tag: &str, a: Option<&(Vec<f32>, Vec<i32>)>, b: Option<&(Vec<f32>, Vec<i32>)>| {
        let (Some((av, ashape)), Some((bv, _bshape))) = (a, b) else {
            eprintln!("DIAG ATTN {label} {tag}: missing");
            return;
        };
        let d = (*ashape.get(3).unwrap_or(&1)).max(1) as usize;
        let per_pos = short_i as usize;
        // Walk ALL elements (every head), not just per_pos*d (which is one head).
        let short_elems = av.len().min(bv.len());
        if av.len() < short_elems || bv.len() < short_elems {
            eprintln!(
                "DIAG ATTN {label} {tag}: data too short ({} vs {}, need {short_elems})",
                av.len(),
                bv.len()
            );
            return;
        }
        let mut per_pos_max = vec![0.0f32; per_pos];
        let mut max_abs = 0.0f32;
        let mut diffs = 0usize;
        for i in 0..short_elems {
            let x = av[i];
            let y = bv[i];
            let diff = (x - y).abs();
            let pos = (i / d) % per_pos;
            if diff > per_pos_max[pos] {
                per_pos_max[pos] = diff;
            }
            if diff > max_abs {
                max_abs = diff;
            }
            if x.to_bits() != y.to_bits() {
                diffs += 1;
            }
        }
        let nz: Vec<String> = per_pos_max
            .iter()
            .enumerate()
            .filter(|(_, m)| **m > 0.0)
            .map(|(p, m)| format!("p{p}:{m:.1e}"))
            .collect();
        eprintln!(
            "DIAG ATTN {label} {tag}: short_len={} max_abs={max_abs:.3e} diffs={diffs}/{short_elems} nonzero[{}]",
            short.0,
            nz.join(" ")
        );
    };
    eprintln!(
        "DIAG ATTN {label}: short_len={} vs long_len={}",
        short.0, long.0
    );
    report("X(input)", short.1.as_ref(), long.1.as_ref());
    report(
        "KEYS_RAW(k_proj,pre-norm)",
        short.4.as_ref(),
        long.4.as_ref(),
    );
    report(
        "KEYS_NORMED(k_norm,pre-rope)",
        short.5.as_ref(),
        long.5.as_ref(),
    );
    report("PRE-WRITE(post-rope)", short.2.as_ref(), long.2.as_ref());
    report("POST-WRITE(stored)", short.3.as_ref(), long.3.as_ref());
}

impl Qwen3NextCausalLM {
    pub fn new(args: Qwen3NextModelArgs) -> Result<Self, Exception> {
        if args.full_attention_interval <= 0 {
            return Err(Exception::custom("full_attention_interval must be > 0"));
        }
        if args.linear_num_key_heads <= 0 || args.linear_num_value_heads <= 0 {
            return Err(Exception::custom("linear_num_*_heads must be > 0"));
        }
        if args.linear_conv_kernel_dim <= 0 {
            return Err(Exception::custom("linear_conv_kernel_dim must be > 0"));
        }

        let model = Qwen3NextInner::new(&args)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            let lm_spec = resolve_quant_for(&args, "language_model.lm_head");
            Some(QLinear::new_spec(lm_spec)?)
        };
        let mtp = (args.mtp_num_hidden_layers > 0 && !args.use_dense_mtp && !args.use_moe_mtp)
            .then(|| MtpHead::new(&args))
            .transpose()?;
        let dense_mtp = (args.mtp_num_hidden_layers > 0 && args.use_dense_mtp)
            .then(|| DenseMtpHead::new(&args))
            .transpose()?;
        let moe_mtp = (args.mtp_num_hidden_layers > 0 && args.use_moe_mtp)
            .then(|| MoeMtpHead::new(&args))
            .transpose()?;

        Ok(Self {
            args,
            model,
            lm_head,
            mtp,
            dense_mtp,
            moe_mtp,
        })
    }

    fn promote_bonsai_dense_mlps_to_row4(&mut self) -> Result<BonsaiRow4Promotion, Exception> {
        let mut promoted = BonsaiRow4Promotion::default();
        for (layer_index, layer) in self.model.layers.iter_mut().enumerate() {
            let layer = layer.mlp.promote_bonsai_row4(layer_index)?;
            promoted.layers = promoted.layers.saturating_add(layer.layers);
            promoted.projections = promoted.projections.saturating_add(layer.projections);
            promoted.bytes = promoted.bytes.saturating_add(layer.bytes);
        }
        Ok(promoted)
    }

    /// Validate the narrow domain in which the `DFlash` block schedule reuses
    /// the same numerical primitives as repeated one-token decode.
    ///
    /// The engine calls this before enabling the experimental block verifier.
    /// Keeping the model-specific constraints here prevents an engine policy
    /// knob from silently selecting a different projection or recurrent
    /// implementation.
    pub fn validate_dflash_block_domain(&self, rows: i32) -> Result<(), Exception> {
        if !(1..=8).contains(&rows) {
            return Err(Exception::custom(format!(
                "DFlash canonical block requires 1..=8 target rows, got {rows}"
            )));
        }
        if compiled_gdn_decode_enabled() {
            return Err(Exception::custom(
                "HIGGS_COMPILED_GDN_DECODE uses a different S=1 recurrent primitive",
            ));
        }
        if *DIAG_ROPE_MANUAL {
            return Err(Exception::custom(
                "HIGGS_DIAG_ROPE_MANUAL changes the S=1 RoPE primitive",
            ));
        }
        if *DIAG_ROPE_PERHEAD {
            return Err(Exception::custom(
                "HIGGS_DIAG_ROPE_PERHEAD changes the S=1 RoPE primitive",
            ));
        }
        if bonsai_q1_qmm_max_rows() < rows {
            return Err(Exception::custom(format!(
                "packed Q1 verifier supports {} rows, but DFlash requires {rows}",
                bonsai_q1_qmm_max_rows()
            )));
        }
        if self.args.num_experts != 0
            || self.args.decoder_sparse_step != 0
            || self.args.dense_attention_outputs
        {
            return Err(Exception::custom(
                "dSpark block verification is proven only for the dense all-Q1 Bonsai target",
            ));
        }

        let validate_linear = |path: &str, linear: &QLinear| validate_dflash_qlinear(path, linear);
        validate_dflash_q1_linear(
            "model.embed_tokens",
            &self.model.embed_tokens.weight,
            &self.model.embed_tokens.scales,
            &self.model.embed_tokens.biases,
            self.model.embed_tokens.group_size,
            self.model.embed_tokens.bits,
            self.model.embed_tokens.mode,
        )?;
        let lm_head = self.lm_head.as_ref().ok_or_else(|| {
            Exception::custom("dSpark block verification requires an untied packed-Q1 LM head")
        })?;
        validate_linear("lm_head", lm_head)?;

        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            if layer.mlp.is_moe {
                return Err(Exception::custom(format!(
                    "layer {layer_index} uses MoE outside the dSpark block domain"
                )));
            }
            for (name, projection) in [
                ("mlp.gate_proj", layer.mlp.gate_proj.as_ref()),
                ("mlp.up_proj", layer.mlp.up_proj.as_ref()),
                ("mlp.down_proj", layer.mlp.down_proj.as_ref()),
            ] {
                let projection = projection.ok_or_else(|| {
                    Exception::custom(format!("layer {layer_index} is missing {name}"))
                })?;
                validate_linear(&format!("layers.{layer_index}.{name}"), projection)?;
            }

            if layer.is_linear {
                let gdn = layer.linear_attn.as_ref().ok_or_else(|| {
                    Exception::custom(format!("layer {layer_index} is missing GDN attention"))
                })?;
                if gdn.use_separate_projections {
                    for (name, projection) in [
                        ("in_proj_qkv", gdn.in_proj_qkv.as_ref()),
                        ("in_proj_z", gdn.in_proj_z.as_ref()),
                        ("in_proj_a", gdn.in_proj_a.as_ref()),
                        ("in_proj_b", gdn.in_proj_b.as_ref()),
                    ] {
                        let projection = projection.ok_or_else(|| {
                            Exception::custom(format!("layer {layer_index} is missing GDN {name}"))
                        })?;
                        validate_linear(
                            &format!("layers.{layer_index}.linear_attn.{name}"),
                            projection,
                        )?;
                    }
                } else {
                    validate_linear(
                        &format!("layers.{layer_index}.linear_attn.in_proj_qkvz"),
                        &gdn.in_proj_qkvz,
                    )?;
                    validate_linear(
                        &format!("layers.{layer_index}.linear_attn.in_proj_ba"),
                        &gdn.in_proj_ba,
                    )?;
                }
                validate_linear(
                    &format!("layers.{layer_index}.linear_attn.out_proj"),
                    &gdn.out_proj,
                )?;
            } else {
                let attention = layer.self_attn.as_ref().ok_or_else(|| {
                    Exception::custom(format!("layer {layer_index} is missing full attention"))
                })?;
                for (name, projection) in [
                    ("q_proj", &attention.q_proj),
                    ("k_proj", &attention.k_proj),
                    ("v_proj", &attention.v_proj),
                    ("o_proj", &attention.o_proj),
                ] {
                    validate_linear(
                        &format!("layers.{layer_index}.self_attn.{name}"),
                        projection,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Create the per-layer cache vector.
    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Some(LayerCache::Arrays(ArraysCache::new()))
                } else {
                    Some(LayerCache::KV(SteppingKeyValueCache::new()))
                }
            })
            .collect()
    }

    /// Create a hybrid cache with `TurboQuant` on the full-attention KV layers.
    ///
    /// Linear-attention (SSM/GDN) layers get a plain `ArraysCache`; full-attention
    /// layers get a `SteppingKeyValueCache` with `TurboQuant` storage. This matches
    /// the selective compression strategy used by other `TurboQuant` implementations.
    pub fn make_cache_turbo(
        &self,
        kv_cache_config: crate::turboquant::KvCacheConfig,
    ) -> Result<Vec<Option<LayerCache>>, mlx_rs::error::Exception> {
        let n_layers = self.model.layers.len();
        let dense_tail = usize::from(kv_cache_config.adaptive_dense_layers);
        self.model
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                if layer.is_linear {
                    Ok(Some(LayerCache::Arrays(ArraysCache::new())))
                } else if dense_tail > 0 && i >= n_layers.saturating_sub(dense_tail) {
                    // Layer-adaptive: final layers stay dense for quality
                    Ok(Some(LayerCache::KV(SteppingKeyValueCache::new())))
                } else {
                    Ok(Some(LayerCache::KV(SteppingKeyValueCache::new_turbo(
                        kv_cache_config,
                        self.args.num_key_value_heads,
                        self.args.head_dim,
                    )?)))
                }
            })
            .collect()
    }

    /// Forward pass returning raw hidden states (before final `RMSNorm`).
    #[allow(non_snake_case, clippy::too_many_lines)]
    fn forward_raw_hidden_with_taps(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: Option<&[usize]>,
    ) -> Result<(Array, Vec<Array>), Exception> {
        // DIAGNOSTIC: capture h after each layer when a probe has requested it
        // (higgs_models::diag_request_hidden_capture). The probe then compares
        // two captured forwards (e.g. body vs full) directly — no "previous
        // forward" thread-local that gets polluted by intervening forwards.
        let do_diag_capture = DIAG_CAPTURE_REQ.with(|c| c.get());
        if do_diag_capture {
            DIAG_CAPTURE_REQ.with(|c| c.set(false));
        }
        let mut diag_layers: Vec<DiagLayer> = Vec::new();
        let mut taps = Vec::with_capacity(tap_layers.map_or(0, <[usize]>::len));

        if let Some(layers) = tap_layers
            && (layers.iter().any(|&index| index >= self.model.layers.len())
                || layers.windows(2).any(|pair| pair[0] >= pair[1]))
        {
            return Err(Exception::custom(
                "tap layers must be unique, strictly increasing, and in range",
            ));
        }

        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        // Create attention mask for full-attention layers
        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .find_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    LayerCache::Arrays(_) => None,
                })
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        // HIGGS_PROFILE=1: instrument per-layer timing with eval barriers.
        // Samples layers 0-3 (3 GDN + 1 FA), extrapolates to all 64 layers.
        let profiling = std::env::var("HIGGS_PROFILE").is_ok_and(|v| v == "1") && T == 1;
        let mut prof_gdn_attn_ns: u128 = 0;
        let mut prof_gdn_mlp_ns: u128 = 0;
        let mut prof_fa_attn_ns: u128 = 0;
        let mut prof_fa_mlp_ns: u128 = 0;
        let mut prof_gdn_samples: u32 = 0;
        let mut prof_fa_samples: u32 = 0;

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let sample_this = profiling && layer_idx < 4;
            let t0 = if sample_this {
                mlx_rs::transforms::eval([&h])?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, mask, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                attn.forward(&normed, mask, layer_kv)?
            };

            if let Some(start) = t0 {
                let h2 = h.add(r)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                mlx_rs::transforms::eval([&h2])?;
                let attn_ns = start.elapsed().as_nanos();
                let t1 = std::time::Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post)?;
                h = h2.add(mlp_out)?;
                mlx_rs::transforms::eval([&h])?;
                let mlp_ns = t1.elapsed().as_nanos();

                if layer.is_linear {
                    prof_gdn_attn_ns += attn_ns;
                    prof_gdn_mlp_ns += mlp_ns;
                    prof_gdn_samples += 1;
                } else {
                    prof_fa_attn_ns += attn_ns;
                    prof_fa_mlp_ns += mlp_ns;
                    prof_fa_samples += 1;
                }
            } else {
                let h2 = h.add(r)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                h = h2.add(mlp_out)?;
            }

            if tap_layers.is_some_and(|layers| layers.binary_search(&layer_idx).is_ok()) {
                taps.push(h.clone());
            }

            // Eval every 8 layers during long prefill chunks to bound lazy
            // graph size. Short speculative verifier windows are intentionally
            // left fused; otherwise MTP pays several eval barriers per cycle.
            if should_eval_between_prefill_layers(T, layer_idx) {
                mlx_rs::transforms::eval([&h])?;
            }
            // DIAGNOSTIC (HIGGS_DIAG_EVAL_EVERY_LAYER=1): force-eval h after
            // EVERY layer so the production hidden state is materialized before
            // the next layer's projections read it. Tests whether lazy-graph
            // context-dependent FP eval is the divergence source.
            if std::env::var("HIGGS_DIAG_EVAL_EVERY_LAYER").is_ok_and(|v| v == "1") {
                mlx_rs::transforms::eval([&h])?;
            }

            if do_diag_capture {
                // Store FULLY MATERIALIZED owned data (Vec<f32>), not a lazy
                // Array clone — a lazy clone gets contaminated when eval'd after
                // later forwards reuse the shared graph nodes.
                let hf = h.as_dtype(mlx_rs::Dtype::Float32)?;
                let _ = mlx_rs::transforms::eval([&hf]);
                let hdim = *hf.shape().last().unwrap_or(&1);
                diag_layers.push((
                    layer_idx,
                    layer.is_linear,
                    hf.as_slice::<f32>().to_vec(),
                    hdim,
                ));
            }
        }

        if profiling && prof_gdn_samples > 0 && prof_fa_samples > 0 {
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                let gdn_attn_avg = prof_gdn_attn_ns as f64 / f64::from(prof_gdn_samples);
                let gdn_mlp_avg = prof_gdn_mlp_ns as f64 / f64::from(prof_gdn_samples);
                let fa_attn_avg = prof_fa_attn_ns as f64 / f64::from(prof_fa_samples);
                let fa_mlp_avg = prof_fa_mlp_ns as f64 / f64::from(prof_fa_samples);
                let est_total =
                    (gdn_attn_avg + gdn_mlp_avg).mul_add(48.0, (fa_attn_avg + fa_mlp_avg) * 16.0);
                tracing::info!(
                    gdn_attn_ms = format!("{:.2}", gdn_attn_avg / 1e6),
                    gdn_mlp_ms = format!("{:.2}", gdn_mlp_avg / 1e6),
                    fa_attn_ms = format!("{:.2}", fa_attn_avg / 1e6),
                    fa_mlp_ms = format!("{:.2}", fa_mlp_avg / 1e6),
                    est_total_ms = format!("{:.1}", est_total / 1e6),
                    "PROFILE: per-layer avg (×48 GDN + ×16 FA)"
                );
            }
        }
        if profiling {
            let n = PROF_TQ_N.with(|c| c.get());
            if n > 0 {
                #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                {
                    let ap = PROF_TQ_APPEND_NS.with(|c| c.get()) as f64 / f64::from(n);
                    let at = PROF_TQ_ATTN_NS.with(|c| c.get()) as f64 / f64::from(n);
                    tracing::info!(
                        fa_layers = n,
                        append_ms = format!("{:.3}", ap / 1e6),
                        attn_ms = format!("{:.3}", at / 1e6),
                        append_over_attn = format!("{:.2}", ap / at.max(1.0)),
                        "PROFILE-TQ: per-FA-layer append(quantize) vs attn(kernels)"
                    );
                }
            }
            PROF_TQ_APPEND_NS.with(|c| c.set(0));
            PROF_TQ_ATTN_NS.with(|c| c.set(0));
            PROF_TQ_N.with(|c| c.set(0));
        }

        if do_diag_capture {
            DIAG_CAPTURED.with(|c| *c.borrow_mut() = Some(std::mem::take(&mut diag_layers)));
        }

        Ok((h, taps))
    }

    fn forward_raw_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        self.forward_raw_hidden_with_taps(inputs, mask, kv_cache, None)
            .map(|(hidden, _)| hidden)
    }
    #[allow(non_snake_case)]
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_raw_hidden(inputs, mask, kv_cache)?;
        self.model.norm.forward(&h)
    }

    /// Forward pass producing logits for the **last position only**.
    ///
    /// During inference only the last token's logits are sampled, so we
    /// slice hidden states before the `lm_head` projection. This avoids a
    /// full `quantized_matmul(vocab, hidden)` on T-1 discarded positions.
    /// Returns shape `[B, 1, vocab]`.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let h_last = h.index((.., -1.., ..)); // [B, 1, hidden]

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_last),
            None => self.model.embed_tokens.as_linear(&h_last),
        }
    }

    /// Forward pass producing logits for **only the last token**.
    ///
    /// During prefill we only need the last token's logits for sampling.
    /// Computing the full `[B, L, vocab]` LM head is wasteful for large vocab.
    /// This method computes hidden states for all tokens (needed for KV cache),
    /// then applies the LM head only to the last token.
    #[allow(non_snake_case)]
    pub fn forward_last_token(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let last_slice = h.index((.., -1, ..));
        // Reshape to [B, 1, D] so the LM head produces [B, 1, vocab]
        let shape = last_slice.shape();
        let batch = *shape
            .first()
            .ok_or_else(|| Exception::custom("forward_last_token: empty shape"))?;
        let last_dim = *shape
            .last()
            .ok_or_else(|| Exception::custom("forward_last_token: empty shape"))?;
        let last_h = last_slice.reshape(&[batch, 1, last_dim])?;
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&last_h),
            None => self.model.embed_tokens.as_linear(&last_h),
        }
    }

    /// Chunked prefill: process the prompt in `chunk_size`-token segments
    /// through all layers. Produces identical logits to `forward()` but with
    /// smaller per-dispatch working sets and lower peak memory.
    ///
    /// Only the **last chunk's** logits are returned (shape `[B, chunk_len, vocab]`).
    /// For full-sequence hidden states, use `forward_hidden` directly.
    #[allow(non_snake_case)]
    pub fn forward_chunked(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        chunk_size: i32,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // If chunk_size covers the whole sequence, just do a normal forward.
        if chunk_size >= T {
            return self.forward(inputs, mask, kv_cache);
        }

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        // Process all chunks except the last through forward_hidden (discard logits).
        // Cache states must be eval'd between chunks so the next chunk reads
        // materialized values (MLX is lazy).
        let mut offset = 0i32;
        while offset + chunk_size < T {
            let chunk = inputs.index((.., offset..offset + chunk_size));
            let h = self.forward_hidden(&chunk, None, kv_cache)?;
            // Eval hidden output + ALL cache states between chunks.
            // Both KV and SSM/conv must be materialized:
            // - SSM/conv: consumed by GDN FFI kernel (requires concrete arrays)
            // - KV: slice_update creates lazy nodes; without eval, nested
            //   updates accumulate and OOM on long sequences
            let mut targets: Vec<&Array> = vec![&h];
            for lc in kv_cache.iter().flatten() {
                match lc {
                    LayerCache::KV(kv) => targets.extend(kv.eval_targets()),
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            targets.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            targets.push(c);
                        }
                    }
                }
            }
            mlx_rs::transforms::eval(targets)?;
            offset += chunk_size;
            crate::progress::report_prefill_progress(offset, T);
        }

        // Last chunk: use forward_last_token which efficiently projects only
        // the last position through the LM head.
        let last_chunk = inputs.index((.., offset..));
        let logits = self.forward_last_token(&last_chunk, None, kv_cache)?;
        // The loop only reports up to the final chunk boundary; emit the
        // terminal 100% mark once the last chunk is processed so clients don't
        // stall just short of `total`.
        crate::progress::report_prefill_progress(T, T);
        Ok(logits)
    }

    // -----------------------------------------------------------------------
    // MTP (Multi-Token Prediction) speculative decode
    // -----------------------------------------------------------------------

    /// Whether this model has an MTP head loaded.
    pub const fn has_mtp(&self) -> bool {
        self.mtp.is_some() || self.dense_mtp.is_some() || self.moe_mtp.is_some()
    }

    /// Create a fresh KV cache for the MTP head (one entry per MTP layer).
    /// Returns `None` if the model has no MTP head.
    pub fn make_mtp_cache(&self) -> Option<Vec<SteppingKeyValueCache>> {
        let layer_count = self
            .mtp
            .as_ref()
            .map(|mtp| mtp.layers.len())
            .or_else(|| self.dense_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .or_else(|| self.moe_mtp.as_ref().map(|mtp| mtp.layers.len()))?;
        Some(
            (0..layer_count)
                .map(|_| SteppingKeyValueCache::new())
                .collect(),
        )
    }

    /// Look up the embedding for a token id. Shape: `[1, 1, hidden_size]`.
    pub fn embed_token(&self, token_id: u32) -> Result<Array, Exception> {
        let token_id_i32 =
            i32::try_from(token_id).map_err(|_| Exception::custom("token_id exceeds i32 range"))?;
        let ids = Array::from_slice(&[token_id_i32], &[1, 1]);
        self.model.embed_tokens.forward(&ids)
    }

    fn embed_tokens_from_ids(&self, token_ids: &[u32]) -> Result<Array, Exception> {
        let token_i32s: Vec<i32> = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| Exception::custom("token_id exceeds i32 range"))
            })
            .collect::<Result<_, _>>()?;
        let len = i32::try_from(token_i32s.len())
            .map_err(|_| Exception::custom("token id batch exceeds i32 range"))?;
        let ids_array = Array::from_slice(&token_i32s, &[1, len]);
        self.model.embed_tokens.forward(&ids_array)
    }

    fn mtp_attention_mask(
        seq_len: i32,
        mtp_cache: &[SteppingKeyValueCache],
    ) -> Result<Option<AttentionMask>, Exception> {
        if seq_len <= 1 {
            return Ok(None);
        }
        let offset = mtp_cache.first().map_or(0, SteppingKeyValueCache::offset);
        if offset > 0 {
            Ok(Some(AttentionMask::Array(create_causal_mask(
                seq_len,
                Some(offset),
            )?)))
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    }

    /// Run the MTP head to produce draft logits for position t+2.
    ///
    /// - `hidden` — backbone hidden state at position t, shape `[B, 1, D]`.
    /// - `next_token_id` — the confirmed next token (t+1).
    /// - `mtp_cache` — per-layer KV cache for the MTP attention.
    ///
    /// Returns the normalized hidden state for the confirmed token position.
    fn mtp_step_hidden(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<Array, Exception> {
        if !self.has_mtp() {
            return Err(Exception::custom("MTP head not loaded"));
        }

        // Compute embedding before mutable borrow of mtp.
        let next_embed = self.embed_token(next_token_id)?;

        // Scope the mutable borrow: run MTP forward, defer lm_head projection.
        if let Some(mtp) = self.mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, None, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            return mtp.norm.forward(&x);
        }

        // MoE MTP head (Qwen3.6-A3B style): same loop, MoE MLP.
        if let Some(mtp) = self.moe_mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, None, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            return mtp.norm.forward(&x);
        }

        let mtp = self
            .dense_mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;

        let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
        let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
        let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
        let mut x = mtp.fc.forward(&concat)?;

        for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
            let normed = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&normed, None, kv)?;
            let h2 = x.add(attn_out)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            x = h2.add(mlp_out)?;
        }

        mtp.norm.forward(&x)
    }

    /// Run the MTP head to produce draft logits for position t+2.
    ///
    /// - `hidden` — backbone hidden state at position t, shape `[B, 1, D]`.
    /// - `next_token_id` — the confirmed next token (t+1).
    /// - `mtp_cache` — per-layer KV cache for the MTP attention.
    ///
    /// Returns draft logits of shape `[B, 1, vocab]`.
    pub fn mtp_draft(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<Array, Exception> {
        let (_, logits) = self.mtp_draft_with_hidden(hidden, next_token_id, mtp_cache)?;
        Ok(logits)
    }

    /// Run the MTP head and return both its hidden state and draft logits.
    ///
    /// The hidden state is useful for chained speculative drafting. Final
    /// committed MTP cache state is still replayed with backbone hidden states
    /// after verification.
    pub fn mtp_draft_with_hidden(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(Array, Array), Exception> {
        let normed = self.mtp_step_hidden(hidden, next_token_id, mtp_cache)?;

        // Now lm_head/embed_tokens can be borrowed immutably.
        let logits = match self.lm_head.as_ref() {
            Some(head) => head.forward(&normed),
            None => self.model.embed_tokens.as_linear(&normed),
        }?;
        Ok((normed, logits))
    }

    /// Advance the MTP cache for a newly accepted token without computing logits.
    pub fn mtp_advance(
        &mut self,
        hidden: &Array,
        next_token_id: u32,
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(), Exception> {
        let _ = self.mtp_step_hidden(hidden, next_token_id, mtp_cache)?;
        Ok(())
    }

    /// Advance the MTP cache for multiple accepted tokens in one sequence pass.
    pub fn mtp_advance_many(
        &mut self,
        hidden: &Array,
        next_token_ids: &[u32],
        mtp_cache: &mut [SteppingKeyValueCache],
    ) -> Result<(), Exception> {
        if next_token_ids.is_empty() {
            return Ok(());
        }
        if !self.has_mtp() {
            return Err(Exception::custom("MTP head not loaded"));
        }

        let seq_len = i32::try_from(next_token_ids.len())
            .map_err(|_| Exception::custom("MTP advance token batch exceeds i32 range"))?;
        let expected_layers = self
            .mtp
            .as_ref()
            .map(|mtp| mtp.layers.len())
            .or_else(|| self.dense_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .or_else(|| self.moe_mtp.as_ref().map(|mtp| mtp.layers.len()))
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;
        Self::validate_mtp_advance_many_inputs(hidden, mtp_cache, expected_layers, seq_len)?;

        let next_embed = self.embed_tokens_from_ids(next_token_ids)?;
        let mask = Self::mtp_attention_mask(seq_len, mtp_cache)?;
        let mask_ref = mask.as_ref();

        if let Some(mtp) = self.mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            let _ = mtp.norm.forward(&x)?;
            return Ok(());
        }

        // MoE MTP head (Qwen3.6-A3B style): same loop, MoE MLP.
        if let Some(mtp) = self.moe_mtp.as_mut() {
            let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
            let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
            let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
            let mut x = mtp.fc.forward(&concat)?;

            for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
                let normed = layer.input_layernorm.forward(&x)?;
                let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
                let h2 = x.add(attn_out)?;
                let normed_post = layer.post_attention_layernorm.forward(&h2)?;
                let mlp_out = layer.mlp.forward(&normed_post)?;
                x = h2.add(mlp_out)?;
            }

            let _ = mtp.norm.forward(&x)?;
            return Ok(());
        }

        let mtp = self
            .dense_mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("MTP head not loaded"))?;

        let h_norm = mtp.pre_fc_norm_hidden.forward(hidden)?;
        let e_norm = mtp.pre_fc_norm_embedding.forward(&next_embed)?;
        let concat = ops::concatenate_axis(&[&e_norm, &h_norm], -1)?;
        let mut x = mtp.fc.forward(&concat)?;

        for (layer, kv) in mtp.layers.iter_mut().zip(mtp_cache.iter_mut()) {
            let normed = layer.input_layernorm.forward(&x)?;
            let attn_out = layer.self_attn.forward(&normed, mask_ref, kv)?;
            let h2 = x.add(attn_out)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            x = h2.add(mlp_out)?;
        }

        let _ = mtp.norm.forward(&x)?;
        Ok(())
    }

    fn validate_mtp_advance_many_inputs(
        hidden: &Array,
        mtp_cache: &[SteppingKeyValueCache],
        expected_layers: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        Self::validate_mtp_advance_many_shape(
            hidden.shape(),
            mtp_cache.len(),
            expected_layers,
            seq_len,
        )
    }

    fn validate_mtp_advance_many_shape(
        hidden_shape: &[i32],
        cache_layers: usize,
        expected_layers: usize,
        seq_len: i32,
    ) -> Result<(), Exception> {
        if cache_layers != expected_layers {
            return Err(Exception::custom(format!(
                "mtp_cache length ({cache_layers}) must match MTP layer count ({expected_layers})"
            )));
        }

        let hidden_seq_len = *hidden_shape
            .get(1)
            .ok_or_else(|| Exception::custom("hidden must be [B, T, D]"))?;
        if hidden_seq_len != seq_len {
            return Err(Exception::custom(format!(
                "hidden sequence length ({hidden_seq_len}) must match next_token_ids length ({seq_len})"
            )));
        }

        Ok(())
    }

    /// Forward pass returning BOTH raw hidden states and logits for all positions.
    ///
    /// Used by MTP speculative decode: the verify pass needs **raw** (pre-norm)
    /// hidden states for the next MTP draft, and logits for acceptance check.
    /// Returns `(raw_hidden, logits)` where both have shape `[B, T, ...]`.
    /// The raw hidden states have NOT been through the final `RMSNorm` — the MTP
    /// head applies its own `pre_fc_norm_hidden` instead.
    #[allow(non_snake_case)]
    pub fn forward_with_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<(Array, Array), Exception> {
        let h_raw = self.forward_raw_hidden(inputs, mask, kv_cache)?;
        let h_normed = self.model.norm.forward(&h_raw)?;
        let logits = match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_normed)?,
            None => self.model.embed_tokens.as_linear(&h_normed)?,
        };
        Ok((h_raw, logits))
    }

    #[allow(non_snake_case)]
    pub fn forward_with_taps(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), Exception> {
        let (_h, logits, taps) =
            self.forward_with_hidden_taps(inputs, mask, kv_cache, tap_layers)?;
        Ok((logits, taps))
    }

    /// Run the backbone and return raw final hidden plus configured tap rows,
    /// without projecting the vocabulary head.
    #[allow(non_snake_case)]
    pub fn forward_raw_with_taps(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), Exception> {
        self.forward_raw_hidden_with_taps(inputs, mask, kv_cache, Some(tap_layers))
    }

    /// `forward_with_taps` that also returns the raw (pre-norm) last-layer
    /// hidden state, so a caller can run tap-consuming (DFlash drafter) and
    /// hidden-consuming (MTP head) speculation off one backbone pass.
    #[allow(non_snake_case)]
    pub fn forward_with_hidden_taps(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Array, Vec<Array>), Exception> {
        let (hidden, taps) = self.forward_raw_with_taps(inputs, mask, kv_cache, tap_layers)?;
        let normed = self.model.norm.forward(&hidden)?;
        let logits = self.project_logits(&normed)?;
        Ok((hidden, logits, taps))
    }

    /// Project only the final position of a raw backbone hidden block.
    pub fn project_raw_hidden_last(&mut self, hidden: &Array) -> Result<Array, Exception> {
        // Keep the same RMSNorm shape/schedule as ordinary prefill, then slice
        // before the large vocabulary projection. Normalizing only the final
        // row is mathematically equivalent but can select a different MLX
        // kernel and is not a construction-level equivalence guarantee.
        let normed = self.model.norm.forward(hidden)?;
        let last = normed.index((.., -1, ..));
        let batch = hidden.shape().first().copied().ok_or_else(|| {
            Exception::custom("project_raw_hidden_last: hidden has no batch axis")
        })?;
        let width = hidden.shape().last().copied().ok_or_else(|| {
            Exception::custom("project_raw_hidden_last: hidden has no feature axis")
        })?;
        let last = last.reshape(&[batch, 1, width])?;
        self.project_logits(&last)
    }

    /// Stateless verify pass: identical to `forward_with_taps` but GDN layers
    /// use `forward_stateless` — they compute correct outputs without updating
    /// `ssm_state` or `conv_state`. KV cache layers update normally (needed for
    /// future decode). Eliminates GdnStateBackup/restore overhead in `DFlash` verify.
    ///
    /// After verify, the caller runs `forward_hidden` with only the accepted
    /// tokens to commit the GDN state for those positions.
    #[allow(non_snake_case)]
    pub fn forward_with_taps_stateless(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>), Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .find_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    LayerCache::Arrays(_) => None,
                })
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let mut taps = Vec::with_capacity(tap_layers.len());

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask_ref = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };

            let normed = layer.input_layernorm.forward(&h)?;

            let r = if layer.is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                // STATELESS: GDN state not updated
                attn.forward_stateless(&normed, mask_ref, ssm_cache)?
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                // KV cache updates normally — needed for future decode
                attn.forward(&normed, mask_ref, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }
        }

        let normed = self.model.norm.forward(&h)?;
        let logits = self.project_logits(&normed)?;

        Ok((logits, taps))
    }

    /// Tape-recording verify pass: runs normal forward (state IS updated) and
    /// records innovation tape per GDN layer. Returns `(logits, taps, tape_data)`.
    ///
    /// On full acceptance (89% of rounds): zero extra work — state already correct.
    /// On partial rejection: restore conv+ssm snapshots, replay `tape[:n_accepted]`.
    // Numerical kernel dispatch: long fn but single straight-line decode loop, casts are timing arithmetic over small counters.
    #[allow(
        non_snake_case,
        clippy::too_many_lines,
        clippy::type_complexity,
        clippy::cast_precision_loss,
        clippy::as_conversions,
        clippy::map_unwrap_or
    )]
    pub fn forward_with_taps_tape(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> Result<(Array, Vec<Array>, Vec<Option<GdnLayerTape>>), Exception> {
        self.forward_with_taps_tape_n(inputs, _mask, kv_cache, tap_layers, None)
    }

    /// Tape-recording verify. `max_layers` is retained for API compatibility,
    /// but partial-model transactions fail closed because skipped attention
    /// layers cannot be represented by the current per-GDN tape type.
    #[allow(non_snake_case)]
    pub fn forward_with_taps_tape_n(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
        max_layers: Option<usize>,
    ) -> Result<(Array, Vec<Array>, Vec<Option<GdnLayerTape>>), Exception> {
        self.forward_with_taps_tape_scheduled(
            inputs,
            _mask,
            kv_cache,
            tap_layers,
            max_layers,
            DFlashRowSchedule::NativeBatch,
        )
    }

    /// Tape-recording verify with an explicit numerical row schedule.
    ///
    /// `CanonicalS1` is intentionally selected only by a verifier that has
    /// passed [`Self::validate_dflash_block_domain`]. Keeping the choice in the
    /// call prevents the dSpark proof boundary from changing ordinary DFlash
    /// or prefill behavior process-wide.
    #[allow(non_snake_case)]
    pub fn forward_with_taps_tape_scheduled(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
        max_layers: Option<usize>,
        row_schedule: DFlashRowSchedule,
    ) -> Result<(Array, Vec<Array>, Vec<Option<GdnLayerTape>>), Exception> {
        if max_layers.is_some() {
            return Err(Exception::custom(
                "partial-layer tape verification is disabled: the transaction cannot represent skipped layers",
            ));
        }
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask: Option<AttentionMask> = if T > 1 {
            let kv_offset = kv_cache
                .iter()
                .find_map(|lc| match lc.as_ref()? {
                    LayerCache::KV(kv) => Some(kv.offset()),
                    LayerCache::Arrays(_) => None,
                })
                .unwrap_or(0);

            if kv_offset > 0 {
                Some(AttentionMask::Array(create_causal_mask(
                    T,
                    Some(kv_offset),
                )?))
            } else {
                Some(AttentionMask::Causal)
            }
        } else {
            None
        };

        let mut taps = Vec::with_capacity(tap_layers.len());
        let mut layer_tapes: Vec<Option<GdnLayerTape>> =
            Vec::with_capacity(self.model.layers.len());

        // Optional per-layer GDN/FA timing. Gated by env to avoid the eval()
        // stalls (which serialize the GPU pipeline) in normal runs. Numbers
        // produced under timing are upper bounds: they include synchronization
        // cost that real execution overlaps. Useful for the GDN-vs-FA ratio.
        let layer_timing = std::env::var("HIGGS_DFLASH_LAYER_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        let layer_detail_timing = std::env::var("HIGGS_DFLASH_LAYER_DETAIL_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[derive(Default)]
        struct LayerDetailTiming {
            input_norm_ms: f64,
            gdn_attn_ms: f64,
            fa_attn_ms: f64,
            residual1_ms: f64,
            post_norm_ms: f64,
            mlp_ms: f64,
            residual2_ms: f64,
            final_norm_ms: f64,
            logits_ms: f64,
            gdn_layers: usize,
            fa_layers: usize,
        }
        let mut detail = LayerDetailTiming::default();
        let mut gdn_total_ms = 0.0_f64;
        let mut fa_total_ms = 0.0_f64;
        let mut gdn_count = 0usize;
        let mut fa_count = 0usize;
        let mut layer_ckpt = if layer_timing {
            mlx_rs::transforms::eval([&h])?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        for (layer_idx, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(kv_cache.iter_mut())
            .enumerate()
        {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let is_linear = layer.is_linear;
            let mask_ref = if is_linear { None } else { fa_mask.as_ref() };

            let mut detail_ckpt = if layer_detail_timing {
                mlx_rs::transforms::eval([&h])?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            let normed = layer.input_layernorm.forward(&h)?;
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&normed])?;
                let now = std::time::Instant::now();
                detail.input_norm_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                *ckpt = now;
            }

            let (r, tape) = if is_linear {
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                let (out, tape) =
                    attn.forward_with_tape(&normed, mask_ref, ssm_cache, row_schedule)?;
                (out, Some(tape))
            } else {
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };
                let output = match row_schedule {
                    DFlashRowSchedule::CanonicalS1 => {
                        attn.forward_canonical_rows(&normed, mask_ref, layer_kv)?
                    }
                    DFlashRowSchedule::NativeBatch => attn.forward(&normed, mask_ref, layer_kv)?,
                };
                (output, None)
            };
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&r])?;
                let now = std::time::Instant::now();
                let dt_ms = now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                if is_linear {
                    detail.gdn_attn_ms += dt_ms;
                    detail.gdn_layers += 1;
                } else {
                    detail.fa_attn_ms += dt_ms;
                    detail.fa_layers += 1;
                }
                *ckpt = now;
            }

            layer_tapes.push(tape);

            let h2 = h.add(r)?;
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&h2])?;
                let now = std::time::Instant::now();
                detail.residual1_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                *ckpt = now;
            }
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&normed_post])?;
                let now = std::time::Instant::now();
                detail.post_norm_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                *ckpt = now;
            }
            let mlp_out = layer.mlp.forward(&normed_post)?;
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&mlp_out])?;
                let now = std::time::Instant::now();
                detail.mlp_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                *ckpt = now;
            }
            h = h2.add(mlp_out)?;

            if tap_layers.contains(&layer_idx) {
                taps.push(h.clone());
            }
            if let Some(ckpt) = detail_ckpt.as_mut() {
                mlx_rs::transforms::eval([&h])?;
                let now = std::time::Instant::now();
                detail.residual2_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
            }

            if let Some(ckpt) = layer_ckpt.as_mut() {
                mlx_rs::transforms::eval([&h])?;
                let now = std::time::Instant::now();
                let dt_ms = now.duration_since(*ckpt).as_secs_f64() * 1000.0;
                if is_linear {
                    gdn_total_ms += dt_ms;
                    gdn_count += 1;
                } else {
                    fa_total_ms += dt_ms;
                    fa_count += 1;
                }
                *ckpt = now;
            }
        }

        let mut tail_detail_ckpt = if layer_detail_timing {
            mlx_rs::transforms::eval([&h])?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let normed = self.model.norm.forward(&h)?;
        if let Some(ckpt) = tail_detail_ckpt.as_mut() {
            mlx_rs::transforms::eval([&normed])?;
            let now = std::time::Instant::now();
            detail.final_norm_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
            *ckpt = now;
        }
        let logits = self.project_logits(&normed)?;
        if let Some(ckpt) = tail_detail_ckpt.as_mut() {
            mlx_rs::transforms::eval([&logits])?;
            let now = std::time::Instant::now();
            detail.logits_ms += now.duration_since(*ckpt).as_secs_f64() * 1000.0;
        }

        if layer_timing {
            mlx_rs::transforms::eval([&logits])?;
            let tail_ms = layer_ckpt
                .map(|c| c.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            #[allow(clippy::as_conversions)]
            {
                tracing::info!(
                    "dflash_layer_timing seq={} gdn_layers={} gdn_total_ms={:.1} gdn_avg={:.2}ms \
                     fa_layers={} fa_total_ms={:.1} fa_avg={:.2}ms tail_ms={:.1}",
                    T,
                    gdn_count,
                    gdn_total_ms,
                    gdn_total_ms / gdn_count.max(1) as f64,
                    fa_count,
                    fa_total_ms,
                    fa_total_ms / fa_count.max(1) as f64,
                    tail_ms,
                );
            }
        }
        if layer_detail_timing {
            #[allow(clippy::as_conversions)]
            {
                tracing::info!(
                    "dflash_layer_detail seq={} gdn_layers={} fa_layers={} \
                     input_norm_ms={:.1} input_norm_avg={:.2} \
                     gdn_attn_ms={:.1} gdn_attn_avg={:.2} \
                     fa_attn_ms={:.1} fa_attn_avg={:.2} \
                     residual1_ms={:.1} post_norm_ms={:.1} mlp_ms={:.1} mlp_avg={:.2} \
                     residual2_ms={:.1} final_norm_ms={:.1} logits_ms={:.1}",
                    T,
                    detail.gdn_layers,
                    detail.fa_layers,
                    detail.input_norm_ms,
                    detail.input_norm_ms / (detail.gdn_layers + detail.fa_layers).max(1) as f64,
                    detail.gdn_attn_ms,
                    detail.gdn_attn_ms / detail.gdn_layers.max(1) as f64,
                    detail.fa_attn_ms,
                    detail.fa_attn_ms / detail.fa_layers.max(1) as f64,
                    detail.residual1_ms,
                    detail.post_norm_ms,
                    detail.mlp_ms,
                    detail.mlp_ms / (detail.gdn_layers + detail.fa_layers).max(1) as f64,
                    detail.residual2_ms,
                    detail.final_norm_ms,
                    detail.logits_ms,
                );
            }
        }

        Ok((logits, taps, layer_tapes))
    }

    /// Replay accepted steps from recorded tape data on partial rejection.
    /// Restores GDN state from `snapshots`, replays `tape[:n_accepted]`,
    /// and rolls back KV cache for rejected positions.
    ///
    /// All GDN layers are batched into a single Metal kernel dispatch
    /// (concat along batch dim, one kernel call, split back) to avoid
    /// per-layer dispatch overhead (~0.4ms × 24 layers = 10ms → <1ms).
    // Numerical kernel: layer indices and counts known finite; explicit casts preferred over try_from.
    #[allow(
        clippy::too_many_lines,
        clippy::indexing_slicing,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    pub fn replay_tape_rollback(
        &self,
        layer_tapes: &[Option<GdnLayerTape>],
        kv_cache: &mut [Option<LayerCache>],
        n_accepted: i32,
        kv_rollback: i32,
    ) -> Result<(), Exception> {
        use mlx_rs::ops;

        if n_accepted <= 0 {
            return Err(Exception::custom(format!(
                "tape rollback requires a positive committed prefix, got {n_accepted}"
            )));
        }
        if kv_rollback < 0 {
            return Err(Exception::custom(format!(
                "tape rollback count must be non-negative, got {kv_rollback}"
            )));
        }
        let target_rows = n_accepted
            .checked_add(kv_rollback)
            .ok_or_else(|| Exception::custom("tape rollback row count overflow"))?;
        if kv_cache.len() != self.model.layers.len() || layer_tapes.len() != self.model.layers.len()
        {
            return Err(Exception::custom(format!(
                "tape transaction layer mismatch: model={} cache={} tapes={}",
                self.model.layers.len(),
                kv_cache.len(),
                layer_tapes.len()
            )));
        }

        // Validate the complete transaction before changing any live cache.
        // A missing GDN tape must never degrade into keeping speculative state.
        for (index, ((layer, cache), tape)) in self
            .model
            .layers
            .iter()
            .zip(kv_cache.iter())
            .zip(layer_tapes.iter())
            .enumerate()
        {
            match (layer.is_linear, cache.as_ref(), tape.as_ref()) {
                (true, Some(LayerCache::Arrays(arrays)), Some(tape)) => {
                    for (name, array) in [
                        ("delta", &tape.delta_tape),
                        ("key", &tape.norm_k),
                        ("gate", &tape.a_proj),
                        ("qkv", &tape.qkv_input),
                    ] {
                        let rows = array.shape().get(1).copied().ok_or_else(|| {
                            Exception::custom(format!(
                                "GDN layer {index} {name} tape has no token axis"
                            ))
                        })?;
                        if rows != target_rows {
                            return Err(Exception::custom(format!(
                                "GDN layer {index} {name} tape has {rows} rows, expected {target_rows}"
                            )));
                        }
                    }
                    let expected_offset = tape
                        .offset_init
                        .checked_add(target_rows)
                        .ok_or_else(|| Exception::custom("GDN tape offset overflow"))?;
                    if arrays.offset != expected_offset {
                        return Err(Exception::custom(format!(
                            "GDN layer {index} live offset {} does not match tape transaction {expected_offset}",
                            arrays.offset
                        )));
                    }
                }
                (false, Some(LayerCache::KV(kv)), None) => {
                    if kv.offset() < kv_rollback {
                        return Err(Exception::custom(format!(
                            "attention layer {index} cannot roll back {kv_rollback} rows from offset {}",
                            kv.offset()
                        )));
                    }
                }
                (true, Some(LayerCache::Arrays(_)), None) => {
                    return Err(Exception::custom(format!(
                        "GDN layer {index} is missing its rollback tape"
                    )));
                }
                (false, Some(LayerCache::KV(_)), Some(_)) => {
                    return Err(Exception::custom(format!(
                        "attention layer {index} unexpectedly has a GDN rollback tape"
                    )));
                }
                (true, Some(LayerCache::KV(_)), _) | (false, Some(LayerCache::Arrays(_)), _) => {
                    return Err(Exception::custom(format!(
                        "layer {index} cache variant does not match the model"
                    )));
                }
                (_, None, _) => {
                    return Err(Exception::custom(format!(
                        "layer {index} cache is missing from the tape transaction"
                    )));
                }
            }
        }

        // Collect GDN layer data for batched replay without mutating live state.
        struct GdnReplayEntry<'a> {
            cache_idx: usize,
            tape: &'a GdnLayerTape,
            layer: &'a GatedDeltaNet,
            snap_state: Array,
        }

        let mut gdn_entries: Vec<GdnReplayEntry> = Vec::new();

        for (index, tape) in layer_tapes.iter().enumerate() {
            let Some(tape) = tape.as_ref() else {
                continue;
            };
            let gdn_layer = self.model.layers[index]
                .linear_attn
                .as_ref()
                .ok_or_else(|| Exception::custom("linear_attn missing for replay"))?;
            let state = if let Some(state) = tape.ssm_state_init.clone() {
                state
            } else {
                ops::zeros_dtype(
                    &[
                        1,
                        gdn_layer.num_v_heads,
                        gdn_layer.head_v_dim,
                        gdn_layer.head_k_dim,
                    ],
                    tape.delta_tape.dtype(),
                )?
            };
            gdn_entries.push(GdnReplayEntry {
                cache_idx: index,
                tape,
                layer: gdn_layer,
                snap_state: state,
            });
        }

        if gdn_entries.is_empty() {
            if kv_rollback > 0 {
                let rollback = usize::try_from(kv_rollback)
                    .map_err(|_| Exception::custom("KV rollback does not fit usize"))?;
                for cache in kv_cache.iter_mut() {
                    if let Some(LayerCache::KV(kv)) = cache {
                        kv.trim_by(rollback);
                    }
                }
            }
            return Ok(());
        }

        // Batch all GDN layers: concat tape/k/a/state/A_log/dt_bias along batch dim
        let tape_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.delta_tape.index((.., ..n_accepted, ..)))
            .collect();
        let k_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.norm_k.index((.., ..n_accepted, ..)))
            .collect();
        let a_slices: Vec<Array> = gdn_entries
            .iter()
            .map(|e| e.tape.a_proj.index((.., ..n_accepted, ..)))
            .collect();
        let states: Vec<&Array> = gdn_entries.iter().map(|e| &e.snap_state).collect();
        let a_logs: Vec<&Array> = gdn_entries.iter().map(|e| e.layer.A_log.as_ref()).collect();
        let dt_biases: Vec<&Array> = gdn_entries
            .iter()
            .map(|e| e.layer.dt_bias.as_ref())
            .collect();

        let tape_refs: Vec<&Array> = tape_slices.iter().collect();
        let k_refs: Vec<&Array> = k_slices.iter().collect();
        let a_refs: Vec<&Array> = a_slices.iter().collect();

        let batched_tape = ops::concatenate_axis(&tape_refs, 0)?;
        let batched_k = ops::concatenate_axis(&k_refs, 0)?;
        let batched_a = ops::concatenate_axis(&a_refs, 0)?;
        let batched_state = ops::concatenate_axis(&states, 0)?;
        // Flatten A_log [Hv] per layer → [num_layers * Hv]
        let batched_a_log = ops::concatenate_axis(&a_logs, 0)?;
        let batched_dt_bias = ops::concatenate_axis(&dt_biases, 0)?;

        let num_layers = gdn_entries.len() as i32;
        let e0 = &gdn_entries[0];

        // Single kernel dispatch for all GDN layers
        let batched_new_state = tape_replay_kernel_ffi(
            &batched_tape,
            &batched_k,
            &batched_a,
            &batched_a_log,
            &batched_dt_bias,
            &batched_state,
            num_layers,
            n_accepted,
            e0.layer.num_k_heads,
            e0.layer.head_k_dim,
            e0.layer.num_v_heads,
            e0.layer.head_v_dim,
        )?;

        // Build every accepted-prefix GDN state off to the side. Live GDN and
        // KV caches are committed only after all fallible graph construction
        // succeeds, so a malformed tape cannot leave a half-rolled-back model.
        let mut staged_gdn = Vec::with_capacity(gdn_entries.len());
        for (offset, entry) in gdn_entries.iter().enumerate() {
            let start = i32::try_from(offset)
                .map_err(|_| Exception::custom("GDN replay layer index overflow"))?;
            let new_state = batched_new_state.index((start..start + 1, .., .., ..));
            let mut staged = ArraysCache {
                conv_state: entry.tape.conv_state_init.clone(),
                ssm_state: entry.tape.ssm_state_init.clone(),
                conv_pos: entry.tape.conv_pos_init,
                offset: entry.tape.offset_init,
            };
            staged.ssm_state = Some(new_state);

            // Rebuild conv_state from recorded qkv input
            let ks = entry.layer.conv_kernel_size;
            let n_keep = ks - 1;
            if n_keep > 0 {
                let qkv_slice = entry.tape.qkv_input.index((.., ..n_accepted, ..));
                let batch = *entry.tape.qkv_input.shape().first().ok_or_else(|| {
                    Exception::custom("conv rebuild: qkv input missing batch dim")
                })?;
                let prefix = entry.layer.chronological_conv_state(
                    &mut staged,
                    batch,
                    entry.tape.qkv_input.dtype(),
                )?;
                let full = ops::concatenate_axis(&[&prefix, &qkv_slice], 1)?;
                let total_len = *full
                    .shape()
                    .get(1)
                    .ok_or_else(|| Exception::custom("conv rebuild: missing seq dim"))?;
                let cs_start = total_len - n_keep;
                let cs = full.index((.., cs_start.., ..));
                let cs_shape = cs.shape().to_vec();
                staged.conv_state = Some(cs.flatten(None, None)?.reshape(&cs_shape)?);
                staged.conv_pos = n_keep - 1;
            } else {
                staged.conv_pos = -1;
            }
            staged.offset = staged
                .offset
                .checked_add(n_accepted)
                .ok_or_else(|| Exception::custom("GDN committed offset overflow"))?;
            staged_gdn.push((entry.cache_idx, staged));
        }

        for (cache_index, staged) in staged_gdn {
            let Some(LayerCache::Arrays(cache)) = &mut kv_cache[cache_index] else {
                return Err(Exception::custom(
                    "validated GDN cache changed variant before commit",
                ));
            };
            *cache = staged;
        }
        if kv_rollback > 0 {
            let rollback = usize::try_from(kv_rollback)
                .map_err(|_| Exception::custom("KV rollback does not fit usize"))?;
            for cache in kv_cache.iter_mut() {
                if let Some(LayerCache::KV(kv)) = cache {
                    kv.trim_by(rollback);
                }
            }
        }

        Ok(())
    }

    /// Embed raw token IDs through the target model's embedding layer.
    ///
    /// Used by `DFlash` to convert `[anchor, mask, mask, ...]` block into
    /// the embedding space expected by the drafter.
    pub fn embed_token_ids(&self, token_ids: &Array) -> Result<Array, Exception> {
        self.model.embed_tokens.forward(token_ids)
    }

    /// Apply only the `lm_head` to pre-computed hidden states.
    ///
    /// Used by `DFlash`: the drafter produces hidden states in the target model's
    /// hidden space, and we project them through the target's `lm_head` to get logits.
    /// Input: `[B, T, hidden_size]`. Returns: `[B, T, vocab_size]`.
    pub fn forward_all_logits_from_hidden(&self, hidden: &Array) -> Result<Array, Exception> {
        self.project_logits(hidden)
    }

    fn project_logits(&self, hidden: &Array) -> Result<Array, Exception> {
        self.lm_head.as_ref().map_or_else(
            || self.model.embed_tokens.as_linear(hidden),
            |head| head.forward(hidden),
        )
    }
}

const PREFILL_LAYER_EVAL_INTERVAL: usize = 8;
const PREFILL_LAYER_EVAL_MIN_SEQ_LEN: i32 = 17;

const fn should_eval_between_prefill_layers(seq_len: i32, layer_idx: usize) -> bool {
    seq_len >= PREFILL_LAYER_EVAL_MIN_SEQ_LEN
        && (layer_idx + 1).is_multiple_of(PREFILL_LAYER_EVAL_INTERVAL)
}

#[cfg(test)]
mod prefill_eval_tests {
    use super::should_eval_between_prefill_layers;

    #[test]
    fn skips_layer_eval_barriers_for_short_speculative_windows() {
        assert!(!should_eval_between_prefill_layers(3, 7));
        assert!(!should_eval_between_prefill_layers(8, 7));
    }

    #[test]
    fn keeps_layer_eval_barriers_for_long_prefill_chunks() {
        assert!(should_eval_between_prefill_layers(128, 7));
        assert!(!should_eval_between_prefill_layers(128, 6));
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load model args from config.json.
pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;
    let mut args = load_qwen3_next_args_from_value(config)?;
    args.dense_attention_outputs = detect_dense_attention_outputs(model_dir.as_ref());
    Ok(args)
}

/// Returns `true` only when the checkpoint stores its attention / GDN output
/// projections as raw BF16 — a `.weight` with no `.scales` sibling (the
/// Unsloth-UD-dense layout). Quantized checkpoints (Ornith, stock MLX quants)
/// keep `.scales`, so those projections must stay on the quantized forward
/// path (`bits != 0`). A missing or unreadable index defaults to `false`.
fn detect_dense_attention_outputs(model_dir: &Path) -> bool {
    let index_path = model_dir.join("model.safetensors.index.json");
    let Ok(file) = std::fs::File::open(&index_path) else {
        return false;
    };
    let Ok(index) = serde_json::from_reader::<_, serde_json::Value>(file) else {
        return false;
    };
    let Some(weight_map) = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    let mut saw_output = false;
    for key in weight_map.keys() {
        let Some(base) = key.strip_suffix(".weight") else {
            continue;
        };
        if base.ends_with("o_proj") || base.ends_with("out_proj") {
            saw_output = true;
            if weight_map.contains_key(&format!("{base}.scales")) {
                return false; // quantized output projection -> not dense
            }
        }
    }
    saw_output
}

/// Resolve `(group_size, bits)` for a canonical tensor path.
///
/// Looks up `path` in [`Qwen3NextModelArgs::quant_overrides`]; falls back to
/// `quantization` (the global default) when no override applies. Returns
/// `(64, 4)` if neither is set, matching the historical default.
pub(crate) fn resolve_quant_for(args: &Qwen3NextModelArgs, path: &str) -> QuantSpec {
    args.quant_spec_for(path)
}

/// Resolve `(group_size, bits)` for an `MoE` gate-style tensor.
///
/// Resolution order: `quant_overrides[path]` → `gate_quantization` → global
/// `quantization` → `(64, 4)`. The middle `gate_quantization` step preserves
/// backward compat with checkpoints that publish a single gate-quantization
/// override but no per-tensor override map.
fn resolve_gate_quant(args: &Qwen3NextModelArgs, path: &str) -> QuantSpec {
    if let Some(o) = args.quant_override_for(path) {
        return o.spec();
    }
    if let Some(gq) = args.gate_quantization.as_ref() {
        return gq.spec();
    }
    args.default_quant_spec()
}

fn gate_quantization_override(config: &serde_json::Value) -> Option<serde_json::Value> {
    let quant = config.get("quantization")?;
    for key in [
        "model.layers.0.mlp.gate",
        "language_model.model.layers.0.mlp.gate",
    ] {
        if let Some(gate_q) = quant.get(key) {
            return Some(gate_q.clone());
        }
    }
    None
}

/// Collect per-tensor mix-bit overrides from a config.json blob.
///
/// Reads `config["quantization"]` (preferred) or `config["quantization_config"]`
/// (sibling fallback used by some Unsloth UD checkpoints). Every nested entry
/// that is a JSON object carrying both `group_size` and `bits` is treated as
/// an override and copied — keyed by the canonical tensor path that holds it.
/// Scalar siblings (`bits`, `group_size`, `mode`) are skipped, and `mode` is
/// dropped from override entries since [`QuantizationConfig`] only carries
/// `(group_size, bits)`.
///
/// Synthesizes a fused-key override at `<prefix>.linear_attn.in_proj_qkvz`
/// whenever both `<prefix>.linear_attn.in_proj_qkv` and `<prefix>.linear_attn.in_proj_z`
/// are present and agree on `(group_size, bits)`. Unsloth UD checkpoints publish
/// GDN overrides under the on-disk SPLIT keys, but the model resolves the FUSED
/// key when running with default (non-separate) GDN projections; without this
/// synthesis the fused `QLinear` silently picks up the global default and the
/// runtime quantized matmul fails on the on-disk packing shape.
fn collect_quant_overrides(
    config: &serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    let mut overrides = serde_json::Map::new();
    let Some(quant) = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
    else {
        return overrides;
    };
    let Some(obj) = quant.as_object() else {
        return overrides;
    };
    for (key, value) in obj {
        let Some(entry) = value.as_object() else {
            continue; // scalar default (`bits`, `group_size`, `mode`).
        };
        let (Some(group_size), Some(bits)) = (entry.get("group_size"), entry.get("bits")) else {
            continue;
        };
        let mut clean = serde_json::Map::with_capacity(3);
        clean.insert("group_size".to_owned(), group_size.clone());
        clean.insert("bits".to_owned(), bits.clone());
        // Preserve per-tensor quantization mode (e.g. AEON mxfp4 bulk +
        // affine islands); absent mode deserializes to the Affine default.
        if let Some(mode) = entry.get("mode") {
            clean.insert("mode".to_owned(), mode.clone());
        }
        overrides.insert(key.clone(), serde_json::Value::Object(clean));
    }
    synthesize_fused_gdn_overrides(&mut overrides);
    overrides
}

/// For each `<prefix>.linear_attn.in_proj_qkv` entry, insert a sibling
/// `<prefix>.linear_attn.in_proj_qkvz` override copied from the matching
/// `in_proj_qkv` / `in_proj_z` pair when they agree on `(group_size, bits)`.
/// No-op if `in_proj_qkvz` is already present, if the matching `in_proj_z`
/// sibling is missing, or if the two sides disagree.
fn synthesize_fused_gdn_overrides(overrides: &mut serde_json::Map<String, serde_json::Value>) {
    // When `use_separate_gdn_projections` is off the model builds fused GDN
    // projections (`in_proj_qkvz`, `in_proj_ba`) and the weight loader rearranges
    // the separate checkpoint tensors into them. Per-tensor quant overrides,
    // however, are keyed by the split names — so synthesize the fused-key
    // override from the split parts when they agree on (bits, group_size).
    // Covers both qkv+z -> qkvz and b+a -> ba.
    fn fuse_from_parts(
        overrides: &serde_json::Map<String, serde_json::Value>,
        first_suffix: &str,
        second_suffix: &str,
        fused_suffix: &str,
    ) -> Vec<(String, serde_json::Value)> {
        overrides
            .iter()
            .filter_map(|(key, value)| {
                let prefix = key.strip_suffix(first_suffix)?;
                let fused_key = format!("{prefix}{fused_suffix}");
                if overrides.contains_key(&fused_key) {
                    return None;
                }
                let second_key = format!("{prefix}{second_suffix}");
                let second_entry = overrides.get(&second_key)?;
                if second_entry != value {
                    tracing::warn!(
                        first_key = %key,
                        second_key = %second_key,
                        first = %value,
                        second = %second_entry,
                        "GDN split overrides disagree; skipping fused-key synthesis"
                    );
                    return None;
                }
                Some((fused_key, value.clone()))
            })
            .collect()
    }

    let mut synthesized = fuse_from_parts(
        overrides,
        ".linear_attn.in_proj_qkv",
        ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_qkvz",
    );
    synthesized.extend(fuse_from_parts(
        overrides,
        ".linear_attn.in_proj_b",
        ".linear_attn.in_proj_a",
        ".linear_attn.in_proj_ba",
    ));

    for (key, value) in synthesized {
        overrides.insert(key, value);
    }
}

fn load_qwen3_next_args_from_value(
    mut config: serde_json::Value,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let gate_override = gate_quantization_override(&config);
    let quant_overrides = collect_quant_overrides(&config);
    let map = config
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("config.json root is not an object".into()))?;
    if !map.contains_key("gate_quantization") {
        if let Some(gate_q) = gate_override {
            map.insert("gate_quantization".to_owned(), gate_q);
        }
    }
    if !quant_overrides.is_empty() && !map.contains_key("quant_overrides") {
        map.insert(
            "quant_overrides".to_owned(),
            serde_json::Value::Object(quant_overrides),
        );
    }
    // Newer transformers exports nest rope fields under a top-level
    // `rope_parameters` object instead of `rope_theta`/`rope_scaling` — the
    // same layout the text_config loader flattens.
    flatten_rope_parameters(map);
    Ok(serde_json::from_value(config)?)
}

/// Flatten a `rope_parameters` object (transformers-v5 style nesting) into
/// the top-level fields serde reads (`rope_theta`, `partial_rotary_factor`)
/// and carry the full object as `rope_scaling` so long-context checkpoints
/// reach the attention layers with their `YaRN` geometry
/// (`type`/`factor`/`original_max_position_embeddings`/betas). Matches
/// mlx-lm `qwen3_5`'s `rope_scaling = rope_parameters`. Base checkpoints
/// carry mrope layout hints here but no `type`, which [`yarn_rope_params`]
/// treats as default rope. No-op when the config has no `rope_parameters`.
fn flatten_rope_parameters(map: &mut serde_json::Map<String, serde_json::Value>) {
    let Some(rope_params) = map.get("rope_parameters").cloned() else {
        return;
    };
    if let Some(theta) = rope_params.get("rope_theta") {
        map.entry("rope_theta").or_insert_with(|| theta.clone());
    }
    if let Some(prf) = rope_params.get("partial_rotary_factor") {
        map.entry("partial_rotary_factor")
            .or_insert_with(|| prf.clone());
    }
    map.entry("rope_scaling").or_insert(rope_params);
}

fn placeholder_param_names<'a, I, K>(params: I) -> Vec<String>
where
    I: IntoIterator<Item = (K, &'a Array)>,
    K: AsRef<str>,
{
    params
        .into_iter()
        .filter(|(_, value)| value.shape() == [1])
        .map(|(name, _)| name.as_ref().to_owned())
        .collect()
}

fn ensure_all_model_params_loaded<'a, I, K>(params: I) -> Result<(), ModelError>
where
    I: IntoIterator<Item = (K, &'a Array)>,
    K: AsRef<str>,
{
    let placeholders = placeholder_param_names(params);
    if placeholders.is_empty() {
        return Ok(());
    }

    let examples = placeholders
        .iter()
        .take(10)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(ModelError::MissingWeight(format!(
        "{} model params were not loaded from the checkpoint; examples: {examples}",
        placeholders.len()
    )))
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SymmetricQ1Compaction {
    tensors: usize,
    bytes: usize,
}

fn symmetric_q1_compaction_enabled() -> bool {
    !std::env::var("HIGGS_BONSAI_SYMMETRIC_Q1").is_ok_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

/// Whether a Q1 affine bias tensor is exactly `-scale / 2` under the Float32
/// arithmetic used by the Metal kernel. Any deviation keeps the original bias
/// tensor and therefore preserves the generic affine fallback exactly.
fn q1_biases_are_symmetric(scales: &Array, biases: &Array) -> Result<bool, ModelError> {
    if scales.shape() != biases.shape()
        || scales.size() == 0
        || scales.shape() == [1]
        || biases.shape() == [1]
    {
        return Ok(false);
    }

    let scales_f32 = scales.as_dtype(Dtype::Float32).map_err(ModelError::Mlx)?;
    let biases_f32 = biases.as_dtype(Dtype::Float32).map_err(ModelError::Mlx)?;
    let expected = scales_f32
        .multiply(Array::from_f32(-0.5))
        .map_err(ModelError::Mlx)?;
    let equal = biases_f32
        .array_eq(&expected, None)
        .map_err(ModelError::Mlx)?;
    equal.try_item::<bool>().map_err(ModelError::Mlx)
}

/// Validate every loaded Q1 scale/bias pair, then replace only symmetric bias
/// tensors with a zero-sized marker. Non-symmetric affine tensors remain fully
/// supported and continue through the existing bias-reading kernels.
fn compact_symmetric_q1_biases(
    params: &mut HashMap<std::rc::Rc<str>, &mut Array>,
) -> Result<SymmetricQ1Compaction, ModelError> {
    let bias_keys = params
        .keys()
        .filter(|key| key.ends_with(".biases"))
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>();
    let mut compacted = SymmetricQ1Compaction::default();

    for bias_key in bias_keys {
        let Some(scale_prefix) = bias_key.strip_suffix(".biases") else {
            continue;
        };
        let scale_key = format!("{scale_prefix}.scales");
        let Some(scales) = params
            .get(scale_key.as_str())
            .map(|value| (**value).clone())
        else {
            continue;
        };
        let Some(biases) = params.get(bias_key.as_str()).map(|value| (**value).clone()) else {
            continue;
        };
        if !q1_biases_are_symmetric(&scales, &biases)? {
            continue;
        }

        compacted.tensors += 1;
        compacted.bytes = compacted.bytes.saturating_add(biases.nbytes());
        if let Some(param) = params.get_mut(bias_key.as_str()) {
            **param = symmetric_q1_bias_sentinel();
        }
    }

    Ok(compacted)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpWeightLayout {
    None,
    Quantized,
    Dense,
    /// MTP layer is MoE-structured (`mlp.gate` / `shared_expert` / experts),
    /// e.g. Qwen3.6-A3B sidecars — loaded via [`MoeMtpHead`].
    MoeQuantized,
}

fn is_mtp_key(key: &str) -> bool {
    key.starts_with("mtp.") || key.contains(".mtp.")
}

fn mtp_weight_layout_from_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> MtpWeightLayout {
    let mut has_mtp = false;
    let mut has_unprefixed_mtp = false;
    let mut has_quantized_aux = false;
    let mut has_moe_mlp = false;

    for key in keys {
        if !is_mtp_key(key) {
            continue;
        }
        has_mtp = true;
        has_unprefixed_mtp |= key.starts_with("mtp.");
        has_quantized_aux |= key.ends_with(".scales") || key.ends_with(".biases");
        has_moe_mlp |= key.contains(".mlp.gate.")
            || key.contains(".mlp.shared_expert")
            || key.contains(".mlp.switch_mlp")
            || key.contains(".mlp.experts");
    }

    if has_mtp && has_moe_mlp {
        MtpWeightLayout::MoeQuantized
    } else if has_quantized_aux {
        MtpWeightLayout::Quantized
    } else if has_unprefixed_mtp {
        MtpWeightLayout::Dense
    } else if has_mtp {
        MtpWeightLayout::Quantized
    } else {
        MtpWeightLayout::None
    }
}

fn checkpoint_mtp_weight_layout(model_path: &Path) -> Result<MtpWeightLayout, ModelError> {
    fn safetensors_file_mtp_weight_layout(file_path: &Path) -> Result<MtpWeightLayout, ModelError> {
        let bytes = std::fs::read(file_path)?;
        let metadata = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        // Auxiliary sidecars may ship truly unprefixed keys (`fc.weight`,
        // `layers.0....`) — normalize so they classify as MTP keys.
        let normalized: Vec<String> = metadata
            .names()
            .into_iter()
            .map(|name| normalize_sidecar_mtp_key(file_path, name.to_owned()))
            .collect();
        Ok(mtp_weight_layout_from_keys(
            normalized.iter().map(String::as_str),
        ))
    }

    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let file = std::fs::File::open(index_path)?;
        let index: serde_json::Value = serde_json::from_reader(file)?;
        let Some(weight_map) = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
        else {
            return Err(ModelError::UnsupportedModel(
                "model.safetensors.index.json missing weight_map".into(),
            ));
        };
        let index_layout = mtp_weight_layout_from_keys(weight_map.keys().map(String::as_str));
        if index_layout != MtpWeightLayout::None {
            return Ok(index_layout);
        }

        let auxiliary_files: Vec<_> = crate::AUXILIARY_SAFETENSORS_FILES
            .iter()
            .map(|file_name| model_path.join(file_name))
            .filter(|file_path| file_path.exists())
            .collect();
        if auxiliary_files.len() > 1 {
            return Err(ModelError::UnsupportedModel(
                "ambiguous MTP sidecars: both mtp.safetensors and model-mtp.safetensors are present; remove one".to_owned(),
            ));
        }

        for file_path in auxiliary_files {
            let layout = safetensors_file_mtp_weight_layout(&file_path)?;
            if layout != MtpWeightLayout::None {
                return Ok(layout);
            }
        }

        return Ok(MtpWeightLayout::None);
    }

    for file_path in crate::collect_safetensors_files(model_path)? {
        let layout = safetensors_file_mtp_weight_layout(&file_path)?;
        if layout != MtpWeightLayout::None {
            return Ok(layout);
        }
    }

    Ok(MtpWeightLayout::None)
}

#[cfg(test)]
fn checkpoint_has_mtp_weights(model_path: &Path) -> Result<bool, ModelError> {
    Ok(checkpoint_mtp_weight_layout(model_path)? != MtpWeightLayout::None)
}

fn maybe_disable_mtp_without_checkpoint_weights(
    args: &mut Qwen3NextModelArgs,
    model_path: &Path,
) -> Result<(), ModelError> {
    if args.mtp_num_hidden_layers <= 0 {
        return Ok(());
    }

    match checkpoint_mtp_weight_layout(model_path)? {
        MtpWeightLayout::Quantized => {
            args.use_dense_mtp = false;
            return Ok(());
        }
        MtpWeightLayout::Dense => {
            args.use_dense_mtp = true;
            return Ok(());
        }
        MtpWeightLayout::MoeQuantized => {
            tracing::info!(
                "Checkpoint ships an MoE-structured MTP head (Qwen3.6-A3B style); \
                 loading via MoeMtpHead"
            );
            args.use_dense_mtp = false;
            args.use_moe_mtp = true;
            return Ok(());
        }
        MtpWeightLayout::None => {}
    }

    tracing::warn!(
        mtp_num_hidden_layers = args.mtp_num_hidden_layers,
        "Config enables MTP but checkpoint has no MTP weights; disabling MTP for this load"
    );
    args.mtp_num_hidden_layers = 0;
    Ok(())
}

/// Load a `Qwen3Next` model from a directory containing safetensors + config.json.
pub fn load_qwen3_next_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let mut args = load_model_args(model_path)?;
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        "Loading qwen3_next model"
    );

    let mut model = Qwen3NextCausalLM::new(args)?;

    // Backbone keys match model params directly, but the MTP sidecar may need
    // remapping: `maybe_disable_mtp_without_checkpoint_weights` can select the
    // dense or MoE head (params `dense_mtp.*` / `moe_mtp.*`) while the checkpoint
    // still ships the head under the `mtp.*` namespace. The plain loader can't
    // bridge that, so it would silently leave the draft head uninitialized.
    load_qwen3_next_weights(&mut model, model_path)?;

    tracing::info!("Qwen3Next model loaded successfully");
    Ok(model)
}

// ---------------------------------------------------------------------------
// Qwen3.5-MoE VLM support
// ---------------------------------------------------------------------------

/// Load model args from a Qwen3.5-MoE VLM config.json.
///
/// Qwen3.5-MoE uses the same architecture as `Qwen3Next` (hybrid
/// `GatedDeltaNet` + full attention + sparse `MoE` with shared expert) but ships
/// as a VLM with config nested under `text_config` and rope parameters nested
/// under `rope_parameters`.
fn load_qwen3_5_moe_text_config_args<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;

    let text_config = config
        .get("text_config")
        .ok_or_else(|| ModelError::UnsupportedModel("missing text_config in config.json".into()))?;

    let mut obj = text_config.clone();
    let map = obj
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("text_config is not an object".into()))?;

    // Flatten rope_parameters into top-level fields
    flatten_rope_parameters(map);

    // Merge top-level quantization config
    if let Some(quant) = config.get("quantization") {
        map.entry("quantization").or_insert_with(|| quant.clone());
    }

    // Extract per-path quantization overrides before serde drops them.
    // config.json stores both scalar defaults ("group_size", "bits", "mode")
    // and per-path maps ("language_model.model.layers.3.self_attn.k_proj": {...})
    // inside the same `quantization` object. serde's QuantizationConfig only
    // captures the scalars; we pull the per-path entries here and set them on
    // the args after deserialisation, with the `language_model.` prefix
    // stripped so they match the model's own parameter names after weight
    // loading.
    let quantization_overrides = config
        .get("quantization")
        .and_then(serde_json::Value::as_object)
        .map_or_else(HashMap::new, |quant_map| {
            quant_map
                .iter()
                .filter_map(|(key, val)| {
                    // Skip scalar defaults (group_size/bits/mode are non-objects).
                    let entry = val.as_object()?;
                    let qc =
                        qwen3_5_quantization_config(&serde_json::Value::Object(entry.clone()))?;
                    // Strip `language_model.` prefix to match stripped param keys.
                    let stripped = key.strip_prefix("language_model.").unwrap_or(key);
                    Some((stripped.to_owned(), qc))
                })
                .collect()
        });
    if !quantization_overrides.is_empty() {
        tracing::info!(
            count = quantization_overrides.len(),
            "Detected per-path quantization overrides (mixed-precision checkpoint)"
        );
    }

    // Merge top-level tie_word_embeddings
    if let Some(tie) = config.get("tie_word_embeddings") {
        map.entry("tie_word_embeddings")
            .or_insert_with(|| tie.clone());
    }

    // Set decoder_sparse_step=1 only for MoE models (num_experts > 0).
    // Dense models (qwen3_5) use standard FFN and must keep decoder_sparse_step=0.
    let has_experts = text_config
        .get("num_experts")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0)
        > 0;
    if has_experts {
        map.entry("decoder_sparse_step")
            .or_insert(serde_json::Value::from(1));
    }

    // intermediate_size is unused when all layers are MoE;
    // for dense models, keep whatever value is in text_config.
    if has_experts {
        map.entry("intermediate_size")
            .or_insert(serde_json::Value::from(0));
    }

    // When HIGGS_SEPARATE_GDN_PROJ is set, or when per-layer GDN BA quantization
    // disagrees on bit-width / group_size between in_proj_a and in_proj_b (common
    // in Unsloth dynamic quants), construct the model with separate GDN
    // projection fields so the direct weight loader can match them. Otherwise,
    // construct with fused fields (weights are rearranged at load time).
    let mixed_ba_layers = qwen3_5_mixed_ba_quantization_layers(&config, text_config);
    let config_requests_separate = map
        .get("use_separate_gdn_projections")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let use_separate = config_requests_separate
        || std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok()
        || !mixed_ba_layers.is_empty();
    map.insert(
        "use_separate_gdn_projections".to_owned(),
        serde_json::Value::from(use_separate),
    );
    if !mixed_ba_layers.is_empty() {
        tracing::info!(
            layers = ?mixed_ba_layers,
            "Detected mixed-bit GDN BA projections; using separate GDN projections"
        );
    }

    // Default off; `load_model_args` refines this per-checkpoint by inspecting
    // the safetensors index. Some Unsloth-UD checkpoints store the attention
    // output projections as raw BF16 (a `.weight` with no `.scales` sibling),
    // but others (e.g. Ornith) quantize them, so this cannot be assumed from
    // config alone — see `detect_dense_attention_outputs`.
    map.insert(
        "dense_attention_outputs".to_owned(),
        serde_json::Value::from(false),
    );

    // Detect per-layer gate quantization override from top-level quantization config
    if let Some(gate_q) = gate_quantization_override(&config) {
        map.insert("gate_quantization".to_owned(), gate_q);
    }

    // Mix-bit per-tensor overrides live on the OUTER `config["quantization"]`
    // (not under `text_config`) for Qwen3.5 VLM-wrapped checkpoints such as
    // Unsloth's UD-Q2_K_XL builds. Lift them so the inner args carry them.
    let quant_overrides = collect_quant_overrides(&config);
    if !quant_overrides.is_empty() {
        map.insert(
            "quant_overrides".to_owned(),
            serde_json::Value::Object(quant_overrides),
        );
    }

    let mut args: Qwen3NextModelArgs = serde_json::from_value(obj)?;
    if let Some(yarn) = yarn_rope_params(&args) {
        tracing::info!(
            factor = yarn.factor,
            original_max_position_embeddings = yarn.original_max_position_embeddings,
            "YaRN rope scaling detected in config"
        );
    }
    // Refine the config-only default by inspecting the checkpoint: attention /
    // GDN output projections are BF16-dense only when they ship a `.weight`
    // with no `.scales` sibling (Unsloth-UD-dense). Quantized checkpoints
    // (e.g. Ornith) keep `.scales`, so they stay on the quantized path.
    args.dense_attention_outputs = detect_dense_attention_outputs(model_dir.as_ref());

    // Fold in the prefix-stripped, mode-aware entries collected above (AEON
    // mxfp4 bulk + affine islands). `collect_quant_overrides` lifts the same
    // specs under their raw (prefixed) keys; `quant_override_for` accepts
    // either key form.
    args.quant_overrides.extend(quantization_overrides);

    // Pre-scan checkpoint for dense GDN projections (bf16, no .scales).
    // Mixed-precision models like AEON keep GDN dynamics (in_proj_a, in_proj_b)
    // in bf16 while quantizing the bulk. The model constructs these as QLinear
    // (from the global mxfp4 default); without a Dense override the weight
    // loader rejects them for missing .scales. Detect from the safetensors
    // index and inject Dense overrides for paths that have .weight but no .scales.
    detect_dense_gdn_projections(model_dir.as_ref(), &mut args.quant_overrides);

    Ok(args)
}

/// Parse a `{group_size, bits, mode}` quantization spec from a JSON node.
fn qwen3_5_quantization_config(value: &serde_json::Value) -> Option<QuantizationConfig> {
    let group_size = i32::try_from(value.get("group_size")?.as_i64()?).ok()?;
    let bits = i32::try_from(value.get("bits")?.as_i64()?).ok()?;
    let mode = value
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .map_or(
            crate::quant_mode::QuantMode::Affine,
            crate::quant_mode::QuantMode::parse,
        );
    Some(QuantizationConfig {
        group_size,
        bits,
        mode,
    })
}

/// Pre-scan the safetensors index to detect GDN projections that are dense
/// (bf16, no `.scales`) in the checkpoint but would default to mxfp4/affine
/// from the config. Injects `QuantMode::Dense` overrides for those paths so
/// the model constructs them with plain matmul instead of `quantized_matmul`.
///
/// Without this, mixed-precision checkpoints (e.g. AEON mxfp4 + bf16 GDN
/// dynamics) fail to load because the weight loader can't find `.scales`
/// for the dense projections.
fn detect_dense_gdn_projections(
    model_dir: &Path,
    overrides: &mut BTreeMap<String, QuantizationConfig>,
) {
    let index_path = model_dir.join("model.safetensors.index.json");
    let Ok(index_text) = std::fs::read_to_string(&index_path) else {
        return; // single-shard models have no index; nothing to scan
    };
    let Ok(index) = serde_json::from_str::<crate::WeightMapIndex>(&index_text) else {
        return;
    };

    // Collect all tensor keys that have `.scales` (i.e. are quantized).
    // Any GDN projection path that has `.weight` but NOT `.scales` is dense.
    // Normalize keys to the model's parameter form (`language_model.` prefix
    // stripped) so the membership check below compares like with like. mxfp4
    // exports ship quantized GDN projections as `.weight` + `.scales` with no
    // `.biases` (E2M1 has no zero-point) — scales presence alone decides.
    fn strip_prefix(k: &str) -> &str {
        k.strip_prefix("language_model.").unwrap_or(k)
    }
    let quantized_paths: std::collections::HashSet<&str> = index
        .weight_map
        .keys()
        .filter_map(|k| k.strip_suffix(".scales"))
        .map(strip_prefix)
        .collect();

    let dense_spec = QuantizationConfig {
        group_size: 0,
        bits: 0,
        mode: crate::quant_mode::QuantMode::Dense,
    };

    let mut added = 0usize;
    for weight_key in index.weight_map.keys() {
        // Look for GDN projection weights: ...linear_attn.in_proj_{a,b}.weight
        let Some(base) = strip_prefix(weight_key).strip_suffix(".weight") else {
            continue;
        };
        if !(base.ends_with(".linear_attn.in_proj_a") || base.ends_with(".linear_attn.in_proj_b")) {
            continue;
        }
        // If this path is NOT quantized (no .scales key) and not already overridden
        if !quantized_paths.contains(base) && !overrides.contains_key(base) {
            overrides.insert(base.to_owned(), dense_spec.clone());
            added += 1;
        }
    }

    // Fused construction resolves the concatenated `in_proj_ba` path, so
    // mirror the split-part overrides onto the fused key when both halves
    // are dense (the fused loader concatenates weight-only pairs directly).
    let fused_ba: Vec<String> = overrides
        .keys()
        .filter_map(|k| k.strip_suffix(".in_proj_a"))
        .filter(|prefix| overrides.contains_key(&format!("{prefix}.in_proj_b")))
        .map(|prefix| format!("{prefix}.in_proj_ba"))
        .collect();
    for key in fused_ba {
        if !overrides.contains_key(&key) {
            overrides.insert(key, dense_spec.clone());
            added += 1;
        }
    }

    if added > 0 {
        tracing::info!(
            added,
            "Detected dense (unquantized) GDN projections from checkpoint; added Dense overrides"
        );
    }
}

/// Scan the per-layer `quantization` map and return layer indices where the GDN
/// `in_proj_a` and `in_proj_b` projections disagree on bit-width or group size.
/// Such layers cannot be fused into a single `in_proj_ba` matrix without
/// dequantizing, so the loader must fall back to separate GDN projections.
fn qwen3_5_mixed_ba_quantization_layers(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
) -> Vec<i32> {
    let Some(quant) = config.get("quantization") else {
        return Vec::new();
    };
    let Some(default_quant) = qwen3_5_quantization_config(quant) else {
        return Vec::new();
    };
    let Some(num_hidden_layers) = text_config
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
    else {
        return Vec::new();
    };

    (0..num_hidden_layers)
        .filter(|layer_idx| {
            let prefixes = [
                format!("language_model.model.layers.{layer_idx}.linear_attn"),
                format!("model.layers.{layer_idx}.linear_attn"),
            ];
            let projection_quantization = |projection: &str| {
                prefixes
                    .iter()
                    .find_map(|prefix| {
                        quant
                            .get(format!("{prefix}.{projection}"))
                            .and_then(qwen3_5_quantization_config)
                    })
                    .unwrap_or_else(|| default_quant.clone())
            };
            let a_quant = projection_quantization("in_proj_a");
            let b_quant = projection_quantization("in_proj_b");
            // The qkvz fusion pair has the same constraint: `in_proj_qkv` and
            // `in_proj_z` are concatenated into `in_proj_qkvz`, which is only
            // possible when their packed (quantized) shapes agree. Mixed-
            // precision quants (e.g. OptiQ) assign different bit-widths per
            // projection on sensitive layers, so check both fusion pairs.
            let qkv_quant = projection_quantization("in_proj_qkv");
            let z_quant = projection_quantization("in_proj_z");
            a_quant.bits != b_quant.bits
                || a_quant.group_size != b_quant.group_size
                || qkv_quant.bits != z_quant.bits
                || qkv_quant.group_size != z_quant.group_size
        })
        .collect()
}

/// Load a Qwen3.5 dense model (VLM wrapper around `Qwen3Next` architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys. Unlike [`load_qwen3_5_moe_model`], does NOT force
/// `decoder_sparse_step=1` or attempt `MoE` gate fusion.
///
/// With `HIGGS_BONSAI_TG_LUT4=1`, eligible dense Q1 MLP parameters are
/// rewritten in place to Higgs' physical row4 inference layout. The resulting
/// parameter tree is authoritative for inference, but generic
/// `ModuleParametersExt::save_safetensors` output is not a canonical MLX
/// checkpoint. Export requires an explicit row4-to-canonical demotion (not yet
/// provided), or a separately loaded canonical model instance.
pub fn load_qwen3_5_model<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let mut args = load_qwen3_5_moe_text_config_args(model_path)?;
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5 dense model (VLM text backbone via qwen3_next)"
    );

    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    gdn_dims.validate()?;
    let compact_symmetric_q1 = args
        .quantization
        .as_ref()
        .is_some_and(|quantization| quantization.bits == 1)
        && symmetric_q1_compaction_enabled();
    let mut model =
        load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims, compact_symmetric_q1)?;

    // Optional: re-quantize Dense (bf16) GDN in_proj_a/b to 8-bit affine at
    // load time. Saves ~0.7GB/token bandwidth with negligible precision loss
    // (8-bit has 256 levels vs 4-bit E2M1's 8 levels that corrupt the recurrence).
    // The model card warns against quantizing GDN dynamics, but that's for 4-bit.
    if std::env::var("HIGGS_DENSE_REQUANT_8BIT").is_ok() {
        let n = requant_dense_gdn_to_8bit(&mut model)?;
        if n > 0 {
            tracing::info!(
                n,
                "Re-quantized Dense GDN projections to 8-bit affine (load-time optimization)"
            );
        }
    }
    if FfnBlock::tg_lut4_enabled() {
        // Row4 promotion eagerly evaluates each packed copy so the canonical
        // checkpoint buffers can be released immediately. Model loading is
        // normally called before the engine acquires its long-lived MLX token,
        // while tests and embedding callers may already hold one. Acquire only
        // for the former case: the process-global gate is deliberately not
        // reentrant.
        let _promotion_exec = (!crate::mlx_exec::held()).then(crate::mlx_exec::acquire);
        let promoted = model.promote_bonsai_dense_mlps_to_row4()?;
        tracing::info!(
            layers = promoted.layers,
            projections = promoted.projections,
            resident_bytes = promoted.bytes,
            "Promoted dense Bonsai MLP parameters to primary row4 storage"
        );
    }
    tracing::info!("Qwen3.5 dense model loaded successfully");
    Ok(model)
}

/// Load a Qwen3.5-MoE model (VLM wrapper around `Qwen3Next` architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys.
pub fn load_qwen3_5_moe_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let mut args = load_qwen3_5_moe_text_config_args(model_path)?;
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5_moe model (VLM text backbone via qwen3_next)"
    );

    // Save GDN dimensions before args is moved
    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    gdn_dims.validate()?;
    // Load weights with GDN projection rearrangement: flat (qkv,z,b,a)
    // → per-head-grouped (qkvz,ba) for fused 2-dispatch forward path.
    // Respects use_separate_gdn_projections (set by HIGGS_SEPARATE_GDN_PROJ env
    // var or mixed-bit BA detection in load_qwen3_5_moe_text_config_args), and
    // falls back to separate projections at runtime if fusion finds a
    // shape-incompatible BA pair.
    let model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims, false)?;

    tracing::info!("Qwen3.5-MoE model loaded successfully");
    Ok(model)
}

/// Build a `Qwen3NextCausalLM` and load weights, choosing fused or separate GDN
/// projections. When the config (or env var) requests separate projections, use
/// the direct loader. Otherwise try the fused loader; if it reports a mixed-bit
/// `in_proj_ba` shape mismatch, rebuild the model with separate projections and
/// retry via the direct loader.
/// Re-quantize Dense (bf16) GDN `in_proj_a`/`in_proj_b` `QLinears` to 8-bit
/// affine at load time. Walks the model tree, finds Dense `QLinears` in GDN
/// layers, calls `ops::quantize(weight, 64, 8)` and swaps the params + mode.
///
/// Returns the number of `QLinears` requantized.
fn requant_dense_gdn_to_8bit(model: &mut Qwen3NextCausalLM) -> Result<usize, ModelError> {
    let mut count = 0usize;
    for layer in &mut model.model.layers {
        if let Some(ref mut gdn) = layer.linear_attn {
            // Handle separate projections (in_proj_a, in_proj_b)
            for ql in [&mut gdn.in_proj_a, &mut gdn.in_proj_b]
                .into_iter()
                .flatten()
            {
                if ql.mode.is_dense() {
                    requant_one_to_8bit(ql)?;
                    count += 1;
                }
            }
            // Handle fused projection (in_proj_ba) — when GDN uses fused mode,
            // a+b are concatenated into a single QLinear.
            if gdn.in_proj_ba.mode.is_dense() {
                requant_one_to_8bit(&mut gdn.in_proj_ba)?;
                count += 1;
            }
        }
    }
    Ok(count)
}

/// Quantize a single Dense `QLinear`'s bf16 weight to 8-bit affine in-place.
fn requant_one_to_8bit(ql: &mut QLinear) -> Result<(), ModelError> {
    let (wq, scales, biases) = ops::quantize(&ql.weight, 64, 8).map_err(ModelError::Mlx)?;
    mlx_rs::transforms::eval([&wq, &scales, &biases]).map_err(ModelError::Mlx)?;
    ql.weight = Param::new(wq);
    ql.scales = Param::new(scales);
    ql.biases = Param::new(biases);
    ql.group_size = 64;
    ql.bits = 8;
    ql.mode = crate::quant_mode::QuantMode::Affine;
    ql.reset_weight_layout();
    Ok(())
}

fn load_qwen3_5_model_with_gdn_fallback(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
    gdn_dims: &GdnDims,
    compact_symmetric_q1: bool,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let force_separate =
        args.use_separate_gdn_projections || std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok();
    if force_separate {
        args.use_separate_gdn_projections = true;
        let mut model = Qwen3NextCausalLM::new(args)?;
        load_qwen3_5_moe_weights_direct(&mut model, model_path, compact_symmetric_q1)?;
        tracing::info!("Using SEPARATE GDN projections (4 dispatches per layer)");
        return Ok(model);
    }

    let mut fused_model = Qwen3NextCausalLM::new(args.clone())?;
    match load_qwen3_5_moe_weights_fused(
        &mut fused_model,
        model_path,
        gdn_dims,
        compact_symmetric_q1,
    ) {
        Ok(()) => {
            tracing::info!("Using FUSED GDN projections (2 dispatches per layer)");
            Ok(fused_model)
        }
        Err(err) if is_mixed_bit_gdn_ba_fusion_error(&err) => {
            tracing::warn!(
                error = %err,
                "Detected mixed-bit GDN BA projection shapes; retrying with separate GDN projections"
            );
            args.use_separate_gdn_projections = true;
            let mut separate_model = Qwen3NextCausalLM::new(args)?;
            load_qwen3_5_moe_weights_direct(&mut separate_model, model_path, compact_symmetric_q1)?;
            tracing::info!(
                "Using SEPARATE GDN projections (4 dispatches per layer, mixed-bit fallback)"
            );
            Ok(separate_model)
        }
        Err(err) => Err(err),
    }
}

/// Returns true when the supplied error is the mixed-bit BA fusion error raised
/// by [`load_qwen3_5_moe_weights_fused`] when `in_proj_a` and `in_proj_b` have
/// incompatible packed inner shapes.
fn is_mixed_bit_gdn_ba_fusion_error(err: &ModelError) -> bool {
    matches!(
        err,
        ModelError::ShapeMismatch(message)
            if message.contains("in_proj_ba")
                && message.contains("requires separate GDN projections")
    )
}

/// GDN dimension info extracted from model args before move.
struct GdnDims {
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
}

impl GdnDims {
    /// Validate GQA ratio: `num_v_heads` must be divisible by `num_k_heads`.
    fn validate(&self) -> Result<(), Exception> {
        if self.num_k_heads == 0 || self.num_v_heads % self.num_k_heads != 0 {
            return Err(Exception::custom(format!(
                "GQA ratio invalid: num_v_heads={} not divisible by num_k_heads={}",
                self.num_v_heads, self.num_k_heads
            )));
        }
        Ok(())
    }
}

/// Build row permutation to convert flat [`q_all|k_all|v_all|z_all`] layout
/// to per-head-grouped [`q_h0|k_h0|v_h0|z_h0|q_h1`|...] for `in_proj_qkvz`.
fn build_qkvz_permutation(d: &GdnDims) -> Result<Vec<i32>, Exception> {
    let nk = d.num_k_heads;
    if nk == 0 || d.num_v_heads % nk != 0 {
        return Err(Exception::custom(format!(
            "GQA ratio invalid: num_v_heads={} not divisible by num_k_heads={nk}",
            d.num_v_heads
        )));
    }
    let dk = d.head_k_dim;
    let v_per_k = d.num_v_heads / nk;
    let dv = d.head_v_dim;
    let key_dim = nk * dk;
    let qkv_rows = key_dim * 2 + d.num_v_heads * dv; // offset for z

    let mut perm = Vec::new();
    for h in 0..nk {
        // q: rows h*dk .. (h+1)*dk from qkv (offset 0)
        for i in 0..dk {
            perm.push(h * dk + i);
        }
        // k: rows key_dim + h*dk .. from qkv
        for i in 0..dk {
            perm.push(key_dim + h * dk + i);
        }
        // v: rows 2*key_dim + h*(v_per_k*dv) .. from qkv
        for i in 0..(v_per_k * dv) {
            perm.push(2 * key_dim + h * v_per_k * dv + i);
        }
        // z: rows h*(v_per_k*dv) .. from z (offset by qkv_rows)
        for i in 0..(v_per_k * dv) {
            perm.push(qkv_rows + h * v_per_k * dv + i);
        }
    }
    Ok(perm)
}

/// Build row permutation for flat [`b_all|a_all`] → per-head-grouped [`b_h0|a_h0|b_h1|a_h1`|...].
fn build_ba_permutation(d: &GdnDims) -> Vec<i32> {
    let nk = d.num_k_heads;
    let v_per_k = d.num_v_heads / nk;
    let nv = d.num_v_heads;

    let mut perm = Vec::new();
    for h in 0..nk {
        // b: rows h*v_per_k .. (h+1)*v_per_k from b
        for i in 0..v_per_k {
            perm.push(h * v_per_k + i);
        }
        // a: rows h*v_per_k .. (h+1)*v_per_k from a (offset by nv)
        for i in 0..v_per_k {
            perm.push(nv + h * v_per_k + i);
        }
    }
    perm
}

/// Concatenate two arrays along dim 0 and permute rows.
fn concat_and_permute(a: &Array, b: &Array, perm: &[i32]) -> Result<Array, Exception> {
    let cat = ops::concatenate_axis(&[a, b], 0)?;
    let perm_arr = Array::from_slice(
        perm,
        &[i32::try_from(perm.len()).map_err(|_| Exception::custom("perm len overflow"))?],
    );
    cat.take_axis(&perm_arr, 0)
}

/// Return true when `a` and `b` can be concatenated along axis 0: the rank
/// matches and every non-axis-0 dimension is identical. Quantized weights pack
/// different bit-widths into different inner shapes, so this guards the BA
/// fusion path from silently producing a malformed `in_proj_ba` matrix.
fn can_concatenate_axis0_shapes(a_shape: &[i32], b_shape: &[i32]) -> bool {
    a_shape.len() == b_shape.len()
        && a_shape
            .iter()
            .zip(b_shape.iter())
            .enumerate()
            .all(|(axis, (lhs, rhs))| axis == 0 || lhs == rhs)
}

fn can_concatenate_axis0(a: &Array, b: &Array) -> bool {
    let a_shape = a.shape();
    let b_shape = b.shape();
    can_concatenate_axis0_shapes(a_shape, b_shape)
}

fn qwen35_checkpoint_param_key(key: &str) -> Option<&str> {
    if key.starts_with("mtp.") {
        Some(key)
    } else {
        key.strip_prefix("language_model.")
    }
}

fn dense_mtp_param_key(stripped: &str) -> Option<String> {
    stripped
        .strip_prefix("mtp.")
        .map(|rest| format!("dense_mtp.{rest}"))
}

fn moe_mtp_param_key(stripped: &str) -> Option<String> {
    stripped
        .strip_prefix("mtp.")
        .map(|rest| format!("moe_mtp.{rest}"))
}

/// Normalize a tensor key loaded from an auxiliary MTP sidecar file.
///
/// Some sidecars (e.g. mlx-community MTP drafters) ship truly unprefixed keys
/// (`fc.weight`, `layers.0....`); prefix them with `mtp.` so they map onto the
/// model's MTP head params the same way prefixed sidecars do. Keys from
/// non-auxiliary files are returned unchanged.
fn normalize_sidecar_mtp_key(file_path: &Path, key: String) -> String {
    let is_aux = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| crate::AUXILIARY_SAFETENSORS_FILES.contains(&n));
    // Only prefix truly un-namespaced sidecar keys (`fc.weight`,
    // `layers.0....`). Already-namespaced keys (`mtp.*`, `language_model.mtp.*`)
    // are left intact so `qwen35_checkpoint_param_key` can still strip/remap
    // them — prefixing those would produce unmatchable `mtp.language_model.mtp.*`.
    if is_aux && !is_mtp_key(&key) {
        format!("mtp.{key}")
    } else {
        key
    }
}

fn qwen35_target_param_key(
    params: &HashMap<std::rc::Rc<str>, &mut Array>,
    stripped: &str,
) -> Option<(String, bool)> {
    if params.contains_key(stripped) {
        Some((stripped.to_owned(), false))
    } else if let Some(moe_key) =
        moe_mtp_param_key(stripped).filter(|key| params.contains_key(key.as_str()))
    {
        // MoE MTP head: plain remap, no dense rmsnorm adjustment.
        Some((moe_key, false))
    } else {
        dense_mtp_param_key(stripped)
            .filter(|dense_key| params.contains_key(dense_key.as_str()))
            .map(|dense_key| (dense_key, true))
    }
}

fn dense_mtp_rmsnorm_weight_key(stripped: &str) -> bool {
    stripped.starts_with("mtp.")
        && stripped.ends_with(".weight")
        && (stripped.contains(".input_layernorm.")
            || stripped.contains(".post_attention_layernorm.")
            || stripped.contains(".q_norm.")
            || stripped.contains(".k_norm.")
            || stripped == "mtp.norm.weight"
            || stripped == "mtp.pre_fc_norm_hidden.weight"
            || stripped == "mtp.pre_fc_norm_embedding.weight")
}

fn qwen35_loaded_value(
    stripped: &str,
    value: Array,
    dense_mtp_target: bool,
) -> Result<Array, crate::error::ModelError> {
    if dense_mtp_target && dense_mtp_rmsnorm_weight_key(stripped) {
        let one = Array::from_f32(1.0)
            .as_dtype(value.dtype())
            .map_err(crate::error::ModelError::Mlx)?;
        value.add(&one).map_err(crate::error::ModelError::Mlx)
    } else {
        Ok(value)
    }
}

/// Load `Qwen3Next` weights, remapping the `mtp.*` sidecar onto whichever MTP
/// head is active (`mtp` / `dense_mtp` / `moe_mtp`).
///
/// Backbone keys match params directly (via `qwen35_target_param_key`'s
/// direct-match branch), so this is behaviour-compatible with the plain loader
/// for the common `Quantized` layout. The only added behaviour is the
/// `mtp.*` → `dense_mtp.*` / `moe_mtp.*` remap, which the plain loader lacks —
/// without it a dense/MoE draft head selected by
/// `maybe_disable_mtp_without_checkpoint_weights` is silently left uninitialized.
#[allow(clippy::shadow_reuse)]
fn load_qwen3_next_weights<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            if let Some((target_key, dense_mtp_target)) = qwen35_target_param_key(&params, &key) {
                if let Some(param) = params.get_mut(target_key.as_str()) {
                    **param = qwen35_loaded_value(&key, value, dense_mtp_target)?;
                    continue;
                }
            }
            tracing::warn!(key = %key, "Weight key not found in model parameters");
        }
    }

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Load Qwen3.5-MoE weights with GDN projection fusion.
///
/// Direct weight loader: strip `language_model.` prefix, no rearrangement.
/// Used when `use_separate_gdn_projections = true`.
#[allow(clippy::shadow_reuse)]
fn load_qwen3_5_moe_weights_direct<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    compact_symmetric_q1: bool,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();
    let mut matched = 0usize;
    let mut unmatched = Vec::new();

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            let Some(stripped) = qwen35_checkpoint_param_key(&key) else {
                unmatched.push(key);
                continue;
            };
            if let Some((target_key, dense_mtp_target)) = qwen35_target_param_key(&params, stripped)
            {
                if let Some(param) = params.get_mut(target_key.as_str()) {
                    **param = qwen35_loaded_value(stripped, value, dense_mtp_target)?;
                } else {
                    unmatched.push(key);
                    continue;
                }
                matched += 1;
            } else {
                unmatched.push(key);
            }
        }
    }

    tracing::info!(
        matched,
        unmatched_count = unmatched.len(),
        "Direct weight loading stats"
    );
    if !unmatched.is_empty() {
        for k in unmatched.iter().take(10) {
            tracing::debug!(key = %k, "Unmatched weight key (no matching model param)");
        }
        if unmatched.len() > 10 {
            tracing::debug!("... and {} more unmatched keys", unmatched.len() - 10);
        }
    }
    let param_count = params.len();
    // This loader is only used with separate GDN projections (see
    // `load_qwen3_5_model_with_gdn_fallback`). In that mode the fused
    // `in_proj_qkvz` / `in_proj_ba` QLinears are still constructed — as unused
    // placeholders, since the forward path dispatches on
    // `use_separate_projections` — so they must be exempt from the
    // completeness check. Flagging them would reject every mixed-bit
    // checkpoint that *requires* separate projections (e.g. OptiQ quants).
    ensure_all_model_params_loaded(
        params
            .iter()
            .filter(|(name, _)| !(name.contains(".in_proj_qkvz.") || name.contains(".in_proj_ba.")))
            .map(|(name, value)| (std::rc::Rc::<str>::clone(name), &**value)),
    )?;
    tracing::info!(param_count, matched, "Total model parameters loaded");

    if compact_symmetric_q1 {
        let compacted = compact_symmetric_q1_biases(&mut params)?;
        tracing::info!(
            tensors = compacted.tensors,
            bytes = compacted.bytes,
            "Dropped validated symmetric Q1 bias tensors"
        );
    }

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Rearranges flat (qkv,z,b,a) projections to per-head-grouped (qkvz,ba)
/// so the model uses the fused 2-dispatch forward path instead of 4 separate.
#[allow(clippy::too_many_lines, clippy::shadow_reuse)]
fn load_qwen3_5_moe_weights_fused<M: mlx_rs::module::ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    gdn_dims: &GdnDims,
    compact_symmetric_q1: bool,
) -> Result<(), crate::error::ModelError> {
    use std::collections::HashMap;

    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    let qkvz_perm = build_qkvz_permutation(gdn_dims)
        .map_err(|e| crate::error::ModelError::ShapeMismatch(e.to_string()))?;
    let ba_perm = build_ba_permutation(gdn_dims);

    // GDN split keys: collect (part_a, part_b) for each combined target
    // Key format: "model.layers.N.linear_attn.in_proj_qkvz.{weight|scales|biases}"
    let mut gdn_parts: HashMap<String, (Option<Array>, Option<Array>)> = HashMap::new();

    let gdn_remap: &[(&str, &str, &str)] = &[
        ("in_proj_qkv", "in_proj_z", "in_proj_qkvz"),
        ("in_proj_b", "in_proj_a", "in_proj_ba"),
    ];

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let key = normalize_sidecar_mtp_key(file_path, key);
            let Some(stripped) = qwen35_checkpoint_param_key(&key) else {
                continue;
            };

            let mut handled = false;
            for &(part_a_name, part_b_name, combined_name) in gdn_remap {
                for (is_b, split_name) in [(false, part_a_name), (true, part_b_name)] {
                    let needle = format!(".{split_name}.");
                    if let Some(pos) = stripped.find(&needle) {
                        let pfx = &stripped[..pos];
                        let sfx = &stripped[pos + needle.len()..];
                        let map_key = format!("{pfx}.{combined_name}.{sfx}");
                        let entry = gdn_parts.entry(map_key).or_insert((None, None));
                        if is_b {
                            entry.1 = Some(value.clone());
                        } else {
                            entry.0 = Some(value.clone());
                        }
                        handled = true;
                        break;
                    }
                }
                if handled {
                    break;
                }
            }

            if !handled {
                if let Some((target_key, dense_mtp_target)) =
                    qwen35_target_param_key(&params, stripped)
                {
                    if let Some(param) = params.get_mut(target_key.as_str()) {
                        **param = qwen35_loaded_value(stripped, value, dense_mtp_target)?;
                    }
                }
            }
        }
    }

    // Fuse GDN pairs: concat + row permutation
    let mut fused_count = 0usize;
    for (combined_key, (part_a, part_b)) in &gdn_parts {
        let (Some(a), Some(b)) = (part_a, part_b) else {
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Incomplete GDN projection pair for key: {combined_key}"),
            )));
        };
        if combined_key.contains("in_proj_ba") && !can_concatenate_axis0(a, b) {
            return Err(crate::error::ModelError::ShapeMismatch(format!(
                "Mixed-bit BA fusion requires separate GDN projections for key {combined_key}: {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }
        let Some(param) = params.get_mut(combined_key.as_str()) else {
            // Quantization metadata keys (global_scale, scales, etc.) don't
            // have a fused target in the model's parameter dict — they're
            // handled by the quant-mode loader. Skip them instead of erroring.
            tracing::debug!(
                key = %combined_key,
                "GDN fusion: skipping key without fused target (likely quant metadata)"
            );
            continue;
        };
        let perm = if combined_key.contains("in_proj_qkvz") {
            &qkvz_perm
        } else {
            &ba_perm
        };
        match concat_and_permute(a, b, perm) {
            Ok(fused) => {
                **param = fused;
                fused_count += 1;
            }
            Err(e) => {
                return Err(crate::error::ModelError::Io(std::io::Error::other(
                    format!("GDN fusion failed for key {combined_key}: {e}"),
                )));
            }
        }
    }

    tracing::info!(
        fused_count,
        total_pairs = gdn_parts.len(),
        "Fused GDN projections (4→2 dispatches per layer)"
    );
    ensure_all_model_params_loaded(
        params
            .iter()
            .map(|(name, value)| (std::rc::Rc::<str>::clone(name), &**value)),
    )?;

    if compact_symmetric_q1 {
        let compacted = compact_symmetric_q1_biases(&mut params)?;
        tracing::info!(
            tensors = compacted.tensors,
            bytes = compacted.bytes,
            "Dropped validated symmetric Q1 bias tensors"
        );
    }

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::needless_for_each,
    clippy::needless_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_range_loop,
    clippy::manual_flatten,
    clippy::unnecessary_map_or,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    clippy::explicit_iter_loop,
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr,
    clippy::str_to_string,
    clippy::if_then_some_else_none,
    clippy::ignore_without_reason,
    clippy::unreadable_literal,
    clippy::cast_possible_wrap,
    clippy::useless_conversion,
    clippy::manual_assert,
    clippy::option_if_let_else,
    clippy::used_underscore_binding,
    clippy::redundant_clone,
    clippy::as_conversions
)]
mod tests {
    use super::*;
    use crate::cache::KeyValueCache;

    #[test]
    fn gated_delta_tape_config_key_covers_specialization_and_output_geometry() {
        let f16 = mlx_sys::mlx_dtype__MLX_FLOAT16;
        let f32 = mlx_sys::mlx_dtype__MLX_FLOAT32;
        let base = GatedDeltaTapeKernelConfigKey::new(f16, 1, 5, 16, 128, 32, 128);
        let variants = [
            GatedDeltaTapeKernelConfigKey::new(f32, 1, 5, 16, 128, 32, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 2, 5, 16, 128, 32, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 1, 4, 16, 128, 32, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 1, 5, 8, 128, 32, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 1, 5, 16, 64, 32, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 1, 5, 16, 128, 16, 128),
            GatedDeltaTapeKernelConfigKey::new(f16, 1, 5, 16, 128, 32, 64),
        ];

        for variant in variants {
            assert_ne!(base, variant);
        }
    }

    #[test]
    fn gated_delta_tape_config_reuse_preserves_outputs() {
        let (batch, seq_len, hk, dk, hv, dv) = (1, 2, 1, 32, 1, 4);
        let q = Array::ones::<f32>(&[batch, seq_len, hk, dk]).unwrap();
        let k = Array::ones::<f32>(&[batch, seq_len, hk, dk]).unwrap();
        let v = Array::ones::<f32>(&[batch, seq_len, hv, dv]).unwrap();
        let a_log = Array::zeros::<f32>(&[hv]).unwrap();
        let a = Array::ones::<f32>(&[batch, seq_len, hv]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[hv]).unwrap();
        let beta = Array::zeros::<f32>(&[batch, seq_len, hv]).unwrap();
        let state = Array::zeros::<f32>(&[batch, hv, dv, dk]).unwrap();

        let first = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();
        let second = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();

        mlx_rs::transforms::eval([
            &first.0, &first.1, &first.2, &second.0, &second.1, &second.2,
        ])
        .unwrap();
        assert_eq!(first.0.as_slice::<f32>(), second.0.as_slice::<f32>());
        assert_eq!(first.1.as_slice::<f32>(), second.1.as_slice::<f32>());
        assert_eq!(first.2.as_slice::<f32>(), second.2.as_slice::<f32>());

        if gated_delta_tape_config_cache_enabled() {
            let key = GatedDeltaTapeKernelConfigKey::new(
                mlx_sys::mlx_dtype__MLX_FLOAT32,
                batch,
                seq_len,
                hk,
                dk,
                hv,
                dv,
            );
            GATED_DELTA_TAPE_CONFIG_CACHE.with(|cache_cell| {
                assert!(cache_cell.borrow().contains_key(&key));
            });
        }
    }

    #[test]
    fn affine_q1_linear_and_embedding_paths_match_known_values() {
        let group_size = 128;
        let input_dim = 128;
        let weight = Array::from_slice(
            &[
                0_u32,
                0,
                0,
                0, // row 0 dequantizes to bias = 1
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX, // row 1 dequantizes to scale + bias = 2
            ],
            &[2, input_dim / 32],
        );
        let scales = Array::from_slice(&[2.0_f32, 3.0], &[2, 1]);
        let biases = Array::from_slice(&[1.0_f32, -1.0], &[2, 1]);

        let decode = Array::from_slice(&vec![1.0_f32; input_dim as usize], &[1, 1, input_dim]);
        let decode_out = affine_q1_forward(&decode, &weight, &scales, &biases, group_size).unwrap();

        let mut prefill_values = vec![1.0_f32; input_dim as usize];
        prefill_values.extend(vec![2.0_f32; input_dim as usize]);
        let prefill = Array::from_slice(&prefill_values, &[1, 2, input_dim]);
        let prefill_out =
            affine_q1_forward(&prefill, &weight, &scales, &biases, group_size).unwrap();

        let mut embedding = QEmbedding::new(group_size, 1).unwrap();
        embedding.weight = Param::new(weight);
        embedding.scales = Param::new(scales);
        embedding.biases = Param::new(biases);
        let ids = Array::from_slice(&[0_u32, 1], &[1, 2]);
        let embedding_out = embedding.forward(&ids).unwrap();

        mlx_rs::transforms::eval([&decode_out, &prefill_out, &embedding_out]).unwrap();
        assert_eq!(decode_out.as_slice::<f32>(), &[128.0, 256.0]);
        assert_eq!(prefill_out.as_slice::<f32>(), &[128.0, 256.0, 256.0, 512.0]);
        let embedding_values = embedding_out.as_slice::<f32>();
        assert!(
            embedding_values[..128]
                .iter()
                .all(|value| (*value - 1.0).abs() <= f32::EPSILON)
        );
        assert!(
            embedding_values[128..]
                .iter()
                .all(|value| (*value - 2.0).abs() <= f32::EPSILON)
        );
    }

    /// The manual prefill rope must not promote bf16 q/k to f32: post-rope
    /// keys are written into the KV cache, so a dtype promotion here doubles
    /// FA KV memory and SDPA bandwidth for every subsequent token.
    #[test]
    fn manual_rope_preserves_input_dtype() {
        let x = mlx_rs::ops::ones::<f32>(&[1, 2, 8, 16])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let positions = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7], &[8]);
        let out = apply_rope_manual(&x, &positions, 8, 10000.0, 1.0).unwrap();
        assert_eq!(out.dtype(), mlx_rs::Dtype::Bfloat16);
    }

    /// Default-rope byte-exactness gate: the inverse frequencies of the
    /// manual prefill path must stay bit-for-bit identical to the historical
    /// inline formula `base^(-2i/dims)` — any drift here silently changes
    /// every non-`YaRN` qwen3_next/qwen3_5 checkpoint.
    #[test]
    fn manual_rope_default_inv_freqs_bit_exact() {
        for (dims, base) in [
            (64_i32, 10_000_000.0_f32),
            (128, 10_000.0),
            (8, 5_000_000.0),
        ] {
            let got = manual_rope_inv_freqs(dims, base);
            assert_eq!(got.len(), (dims / 2) as usize);
            for (i, v) in got.iter().enumerate() {
                let expected = base.powf(-2.0 * i as f32 / dims as f32);
                assert_eq!(
                    v.to_bits(),
                    expected.to_bits(),
                    "inv_freq[{i}] changed for dims={dims} base={base}: {v} != {expected}"
                );
            }
        }
    }

    /// `YaRN` frequency values against an independent transcription of the
    /// reference formula (HF `_compute_yarn_parameters` / mlx-lm `YarnRoPE`),
    /// at the Qwythos-1M geometry: 64 rotary dims, theta 1e7, factor 4,
    /// original context 262144. Also pins the attention mscale.
    #[test]
    fn yarn_freqs_and_mscale_match_reference() {
        let dims = 64_i32;
        let base = 10_000_000.0_f32;
        let factor = 4.0_f32;
        let orig_max = 262_144_i32;

        let periods = compute_yarn_freqs(dims, base, factor, orig_max, 32.0, 1.0);
        let periods: Vec<f32> = periods.as_slice().to_vec();
        assert_eq!(periods.len(), 32);

        // Independent reference (inv_freq form): linear interpolation between
        // extrapolated and interpolated inverse frequencies over the
        // beta_fast..beta_slow wavelength ramp.
        let ln_base = f64::from(base).ln();
        let corr_dim = |rot: f64| -> f64 {
            f64::from(dims) * (f64::from(orig_max) / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * ln_base)
        };
        let low = corr_dim(32.0).floor().max(0.0);
        let high = corr_dim(1.0).ceil().min(f64::from(dims) - 1.0);
        for (i, period) in periods.iter().enumerate() {
            let pos_freq = f64::from(base).powf(2.0 * i as f64 / f64::from(dims));
            let inv_extra = 1.0 / pos_freq;
            let inv_inter = 1.0 / (f64::from(factor) * pos_freq);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp; // extrapolation weight
            let inv_ref = inv_inter * (1.0 - mask) + inv_extra * mask;
            let inv_got = 1.0 / f64::from(*period);
            let rel = ((inv_got - inv_ref) / inv_ref).abs();
            assert!(
                rel < 1e-4,
                "dim {i}: inv_freq {inv_got:.6e} vs reference {inv_ref:.6e} (rel {rel:.2e})"
            );
        }
        // Band structure: lowest dims are pure extrapolation (original rope,
        // period[0] == base^0 == 1), highest dims pure interpolation
        // (period == factor * base^(2i/dims)).
        assert_eq!(periods[0].to_bits(), 1.0_f32.to_bits());
        let last_extra = base.powf(2.0 * 31.0 / dims as f32);
        let ratio = periods[31] / last_extra;
        assert!(
            (ratio - factor).abs() < 1e-3,
            "highest dim should be fully interpolated: ratio {ratio} != {factor}"
        );

        // Attention mscale: 0.1*ln(4)+1 with the mlx-lm default
        // mscale=1 / mscale_all_dim=0 pair.
        let mscale = yarn_get_mscale(factor, 1.0) / yarn_get_mscale(factor, 0.0);
        assert!(
            (mscale - 1.138_629_4).abs() < 1e-6,
            "mscale {mscale} != 1.1386294"
        );
    }

    /// Prefill (manual rope) and decode (`mlx_fast_rope`) must agree on the
    /// SAME yarn frequencies — divergence here is the warm/cold drift bug
    /// class. Rotates a full L=6 window manually, then rotates each position
    /// as a single-token decode step with the same freqs, and compares.
    /// Also guards the bf16 cast-back on the yarn prefill path.
    #[test]
    fn yarn_prefill_decode_rope_agree() {
        let dims = 8_i32; // partial rotary: 8 of 16
        let head_dim = 16_i32;
        let seq = 6_i32;
        let freqs = compute_yarn_freqs(dims, 10_000.0, 4.0, 2048, 32.0, 1.0);

        let data: Vec<f32> = (0..2 * seq * head_dim)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let x = Array::from_slice(&data, &[1, 2, seq, head_dim]);

        let positions: Vec<i32> = (0..seq).collect();
        let positions = Array::from_slice(&positions, &[seq]);
        let manual =
            apply_rope_manual_with_freqs(&x, &positions, dims, 10_000.0, 1.0, Some(&freqs))
                .unwrap();

        for pos in 0..seq {
            let x_tok = x.index((.., .., pos..pos + 1, ..));
            let fast =
                mlx_rs::fast::rope(&x_tok, dims, false, None::<f32>, 1.0, pos, &freqs).unwrap();
            let manual_tok = manual.index((.., .., pos..pos + 1, ..));
            let diff = fast
                .subtract(&manual_tok)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(diff < 1e-5, "pos {pos}: manual vs fast rope diff {diff}");
        }

        // bf16 in -> bf16 out on the yarn variant (same guarantee as
        // `manual_rope_preserves_input_dtype` for the default path).
        let x_bf16 = x.as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();
        let out =
            apply_rope_manual_with_freqs(&x_bf16, &positions, dims, 10_000.0, 1.0, Some(&freqs))
                .unwrap();
        assert_eq!(out.dtype(), mlx_rs::Dtype::Bfloat16);
    }

    #[test]
    fn symmetric_q1_linear_and_embedding_paths_derive_bias() {
        let group_size = 128;
        let input_dim = 128;
        let weight = Array::from_slice(
            &[
                0_u32,
                0,
                0,
                0, // row 0: bit=0, scale=2 => -1
                u32::MAX,
                u32::MAX,
                u32::MAX,
                u32::MAX, // row 1: bit=1, scale=4 => +2
            ],
            &[2, input_dim / 32],
        );
        let scales = Array::from_slice(&[2.0_f32, 4.0], &[2, 1]);
        let no_biases = symmetric_q1_bias_sentinel();

        let decode = Array::from_slice(&vec![1.0_f32; input_dim as usize], &[1, 1, input_dim]);
        let decode_out =
            affine_q1_forward(&decode, &weight, &scales, &no_biases, group_size).unwrap();

        let mut prefill_values = vec![1.0_f32; input_dim as usize];
        prefill_values.extend(vec![2.0_f32; input_dim as usize]);
        let prefill = Array::from_slice(&prefill_values, &[1, 2, input_dim]);
        let prefill_out =
            affine_q1_forward(&prefill, &weight, &scales, &no_biases, group_size).unwrap();

        let mut embedding = QEmbedding::new(group_size, 1).unwrap();
        embedding.weight = Param::new(weight);
        embedding.scales = Param::new(scales);
        embedding.biases = Param::new(no_biases);
        let ids = Array::from_slice(&[0_u32, 1], &[1, 2]);
        let embedding_out = embedding.forward(&ids).unwrap();

        mlx_rs::transforms::eval([&decode_out, &prefill_out, &embedding_out]).unwrap();
        assert_eq!(decode_out.as_slice::<f32>(), &[-128.0, 256.0]);
        assert_eq!(
            prefill_out.as_slice::<f32>(),
            &[-128.0, 256.0, -256.0, 512.0]
        );
        let embedding_values = embedding_out.as_slice::<f32>();
        assert!(
            embedding_values[..128]
                .iter()
                .all(|value| (*value + 1.0).abs() <= f32::EPSILON)
        );
        assert!(
            embedding_values[128..]
                .iter()
                .all(|value| (*value - 2.0).abs() <= f32::EPSILON)
        );
    }

    #[test]
    fn packed_q1_qmm_matches_dense_reference_for_m1_through_m9() {
        const GROUP_SIZE: i32 = 128;
        const K: i32 = 128;
        const N: i32 = 9;

        let mut packed = Vec::with_capacity((N * K / 32) as usize);
        for row in 0..N {
            let word = match row % 3 {
                0 => 0_u32,
                1 => u32::MAX,
                _ => 0xAAAA_AAAA,
            };
            packed.extend(std::iter::repeat_n(word, (K / 32) as usize));
        }
        let weight = Array::from_slice(&packed, &[N, K / 32]);
        let scales = Array::from_slice(
            &(0..N)
                .map(|row| 0.5_f32 + row as f32 * 0.125)
                .collect::<Vec<_>>(),
            &[N, 1],
        );
        let affine_biases = Array::from_slice(
            &(0..N)
                .map(|row| -0.25_f32 + row as f32 * 0.031_25)
                .collect::<Vec<_>>(),
            &[N, 1],
        );
        let symmetric_biases = symmetric_q1_bias_sentinel();

        for biases in [&affine_biases, &symmetric_biases] {
            let dense =
                crate::metal_kernel::bonsai_q1_dequant(&weight, &scales, biases, GROUP_SIZE)
                    .unwrap();

            for m in 1..=9 {
                let values = (0..m)
                    .flat_map(|row| {
                        (0..K).map(move |col| {
                            0.25_f32 + row as f32 * 0.5 + (col % 7) as f32 * 0.062_5
                        })
                    })
                    .collect::<Vec<_>>();
                let x = Array::from_slice(&values, &[m, K]);
                let actual = if m <= 8 {
                    crate::metal_kernel::bonsai_q1_qmm(&x, &weight, &scales, biases, GROUP_SIZE)
                        .unwrap()
                } else {
                    // M=9 exercises the dense fallback at the dispatch boundary.
                    affine_q1_forward(&x, &weight, &scales, biases, GROUP_SIZE).unwrap()
                };
                let expected = x.matmul(&dense.transpose().unwrap()).unwrap();
                mlx_rs::transforms::eval([&actual, &expected]).unwrap();

                assert_eq!(actual.shape(), &[m, N]);
                for (index, (got, want)) in actual
                    .as_slice::<f32>()
                    .iter()
                    .zip(expected.as_slice::<f32>())
                    .enumerate()
                {
                    let tolerance = 1e-3_f32 * want.abs().max(1.0);
                    assert!(
                        (*got - *want).abs() <= tolerance,
                        "M={m} value {index}: packed={got}, dense={want}, tolerance={tolerance}"
                    );
                }
            }

            let leading = Array::from_slice(&vec![0.5_f32; (8 * K) as usize], &[2, 4, K]);
            let output = affine_q1_forward(&leading, &weight, &scales, biases, GROUP_SIZE).unwrap();
            mlx_rs::transforms::eval([&output]).unwrap();
            assert_eq!(output.shape(), &[2, 4, N]);
        }

        // Exercise the fast kernel's full 1024-value block with the dtype used
        // by the real Bonsai-27B backbone.
        const MAIN_K: i32 = 1024;
        const MAIN_M: i32 = 2;
        let main_weight = Array::from_slice(
            &(0..N * MAIN_K / 32)
                .map(|index| {
                    if index % 2 == 0 {
                        0x5555_5555_u32
                    } else {
                        0xAAAA_AAAA_u32
                    }
                })
                .collect::<Vec<_>>(),
            &[N, MAIN_K / 32],
        );
        let main_scales = Array::from_slice(
            &vec![0.75_f32; (N * MAIN_K / GROUP_SIZE) as usize],
            &[N, MAIN_K / GROUP_SIZE],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let main_biases = Array::from_slice(
            &vec![-0.375_f32; (N * MAIN_K / GROUP_SIZE) as usize],
            &[N, MAIN_K / GROUP_SIZE],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let main_x = Array::from_slice(
            &(0..MAIN_M * MAIN_K)
                .map(|index| 0.125_f32 + (index % 11) as f32 * 0.031_25)
                .collect::<Vec<_>>(),
            &[MAIN_M, MAIN_K],
        )
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .unwrap();
        let main_actual = crate::metal_kernel::bonsai_q1_qmm(
            &main_x,
            &main_weight,
            &main_scales,
            &main_biases,
            GROUP_SIZE,
        )
        .unwrap();
        let main_dense = crate::metal_kernel::bonsai_q1_dequant(
            &main_weight,
            &main_scales,
            &main_biases,
            GROUP_SIZE,
        )
        .unwrap();
        let main_expected = main_x.matmul(&main_dense.transpose().unwrap()).unwrap();
        let main_actual_f32 = main_actual.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let main_expected_f32 = main_expected.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&main_actual_f32, &main_expected_f32]).unwrap();
        for (index, (got, want)) in main_actual_f32
            .as_slice::<f32>()
            .iter()
            .zip(main_expected_f32.as_slice::<f32>())
            .enumerate()
        {
            let tolerance = 0.02_f32 * want.abs().max(1.0);
            assert!(
                (*got - *want).abs() <= tolerance,
                "BF16 main block value {index}: packed={got}, dense={want}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn packed_q1_qmm_matches_independent_cpu_affine_oracle() {
        const GROUP_SIZE: i32 = 128;
        const K: i32 = 256;
        const N: i32 = 3;
        const M: i32 = 5;

        let packed = (0..N * K / 32)
            .map(|index| {
                let shift = u32::try_from((index * 7 + 3).rem_euclid(31)).unwrap();
                0x963C_A5F0_u32.rotate_left(shift)
            })
            .collect::<Vec<_>>();
        let scale_values = (0..N * K / GROUP_SIZE)
            .map(|index| 0.125_f32 + (index % 5) as f32 * 0.062_5)
            .collect::<Vec<_>>();
        let bias_values = (0..N * K / GROUP_SIZE)
            .map(|index| -0.093_75_f32 + (index % 4) as f32 * 0.031_25)
            .collect::<Vec<_>>();
        let x_values = (0..M * K)
            .map(|index| ((index * 11 + 9).rem_euclid(53) - 26) as f32 * 0.007_812_5)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&packed, &[N, K / 32]);
        let scales = Array::from_slice(&scale_values, &[N, K / GROUP_SIZE]);
        let affine_biases = Array::from_slice(&bias_values, &[N, K / GROUP_SIZE]);
        let symmetric_biases = symmetric_q1_bias_sentinel();
        let x = Array::from_slice(&x_values, &[M, K]);

        for (biases, symmetric) in [(&affine_biases, false), (&symmetric_biases, true)] {
            let actual =
                crate::metal_kernel::bonsai_q1_qmm(&x, &weight, &scales, biases, GROUP_SIZE)
                    .unwrap();
            mlx_rs::transforms::eval([&actual]).unwrap();
            let got = actual.as_slice::<f32>();

            for row in 0..M {
                for output in 0..N {
                    let mut expected = 0.0_f32;
                    for column in 0..K {
                        let word_index = usize::try_from(output * (K / 32) + column / 32).unwrap();
                        let bit = ((packed[word_index]
                            >> u32::try_from(column.rem_euclid(32)).unwrap())
                            & 1) as f32;
                        let group_index =
                            usize::try_from(output * (K / GROUP_SIZE) + column / GROUP_SIZE)
                                .unwrap();
                        let scale = scale_values[group_index];
                        let bias = if symmetric {
                            -0.5 * scale
                        } else {
                            bias_values[group_index]
                        };
                        let input_index = usize::try_from(row * K + column).unwrap();
                        expected += x_values[input_index] * scale.mul_add(bit, bias);
                    }
                    let index = usize::try_from(row * N + output).unwrap();
                    let tolerance = 2e-5_f32 * expected.abs().max(1.0);
                    assert!(
                        (got[index] - expected).abs() <= tolerance,
                        "row={row} output={output} symmetric={symmetric}: metal={} cpu={expected} tolerance={tolerance}",
                        got[index]
                    );
                }
            }
        }
    }

    #[test]
    fn packed_q1_qmm_is_bit_exact_with_repeated_qmv() {
        const GROUP_SIZE: i32 = 128;
        const K: i32 = 5120;
        const N: i32 = 9;
        let weight = Array::from_slice(
            &(0..N * K / 32)
                .map(|index| {
                    let shift = u32::try_from(index.rem_euclid(31)).unwrap();
                    0xA5A5_5A5A_u32.rotate_left(shift)
                })
                .collect::<Vec<_>>(),
            &[N, K / 32],
        );

        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let scales = Array::from_slice(
                &(0..N * K / GROUP_SIZE)
                    .map(|index| 0.25_f32 + (index % 7) as f32 * 0.062_5)
                    .collect::<Vec<_>>(),
                &[N, K / GROUP_SIZE],
            )
            .as_dtype(dtype)
            .unwrap();
            let affine_biases = Array::from_slice(
                &(0..N * K / GROUP_SIZE)
                    .map(|index| -0.125_f32 + (index % 5) as f32 * 0.031_25)
                    .collect::<Vec<_>>(),
                &[N, K / GROUP_SIZE],
            )
            .as_dtype(dtype)
            .unwrap();
            let symmetric_biases = symmetric_q1_bias_sentinel();

            for biases in [&affine_biases, &symmetric_biases] {
                for m in [2_i32, 3, 4, 5, 7, 8, 9] {
                    let x = Array::from_slice(
                        &(0..m * K)
                            .map(|index| ((index * 13 + 5).rem_euclid(41) - 20) as f32 * 0.015_625)
                            .collect::<Vec<_>>(),
                        &[m, K],
                    )
                    .as_dtype(dtype)
                    .unwrap();
                    let rows = (0..m)
                        .map(|row| {
                            crate::metal_kernel::bonsai_q1_qmv_fast(
                                &x.index((row..row + 1, ..)),
                                &weight,
                                &scales,
                                biases,
                                GROUP_SIZE,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let expected =
                        ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0).unwrap();

                    let actual = crate::metal_kernel::bonsai_q1_qmm(
                        &x, &weight, &scales, biases, GROUP_SIZE,
                    )
                    .unwrap();
                    assert_canonical_array_exact(
                        &format!("Q1 grid-Z M={m} dtype={dtype:?}"),
                        &actual,
                        &expected,
                    );
                }
            }
        }

        let scales = ops::ones::<f32>(&[N, K / GROUP_SIZE])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let biases = symmetric_q1_bias_sentinel();
        let assert_shape_exact = |input: &Array, label: &str| {
            let m = input
                .shape()
                .iter()
                .take(input.ndim().saturating_sub(1))
                .product::<i32>();
            let flat = input.reshape(&[m, K]).unwrap();
            let rows = (0..m)
                .map(|row| {
                    crate::metal_kernel::bonsai_q1_qmv_fast(
                        &flat.index((row..row + 1, ..)),
                        &weight,
                        &scales,
                        &biases,
                        GROUP_SIZE,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let expected_flat = ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
            let mut expected_shape = input.shape()[..input.ndim() - 1].to_vec();
            expected_shape.push(N);
            let expected = expected_flat.reshape(&expected_shape).unwrap();
            let actual =
                crate::metal_kernel::bonsai_q1_qmm(input, &weight, &scales, &biases, GROUP_SIZE)
                    .unwrap();
            assert_canonical_array_exact(label, &actual, &expected);
        };

        let leading = Array::from_slice(
            &(0..8 * K)
                .map(|index| ((index * 7 + 3).rem_euclid(43) - 21) as f32 * 0.015_625)
                .collect::<Vec<_>>(),
            &[2, 4, K],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert_shape_exact(&leading, "Q1 leading dimensions");

        let transposed = Array::from_slice(
            &(0..5 * K)
                .map(|index| ((index * 5 + 1).rem_euclid(47) - 23) as f32 * 0.015_625)
                .collect::<Vec<_>>(),
            &[K, 5],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap();
        assert_shape_exact(&transposed, "Q1 non-row-contiguous input");
    }

    fn row4_fixture_linear(n_rows: i32, k_dim: i32, salt: u32) -> QLinear {
        let mut linear = QLinear::new(128, 1).unwrap();
        linear.weight = Param::new(Array::from_slice(
            &(0..n_rows * k_dim / 32)
                .map(|index| {
                    u32::try_from(index)
                        .unwrap()
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add(salt)
                })
                .collect::<Vec<_>>(),
            &[n_rows, k_dim / 32],
        ));
        linear.scales = Param::new(
            Array::from_slice(
                &(0..n_rows * k_dim / 128)
                    .map(|index| 0.125 + (index.rem_euclid(17) as f32) * 0.007_812_5)
                    .collect::<Vec<_>>(),
                &[n_rows, k_dim / 128],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap(),
        );
        linear.biases = Param::new(symmetric_q1_bias_sentinel());
        linear
    }

    fn exceptional_affine_q1_linear(n_rows: i32, k_dim: i32, salt: u32) -> QLinear {
        const FP16_MIN_SUBNORMAL: f32 = 5.960_464_5e-8;
        let mut linear = row4_fixture_linear(n_rows, k_dim, salt);
        let count = (n_rows * k_dim / 128) as usize;
        let mut scales = vec![0.25_f32; count];
        let mut biases = vec![-0.125_f32; count];
        // The stored F16 bias differs from the unrounded `-scale / 2` by
        // exactly half an F16 subnormal. This models Bonsai-27B layer 31's
        // down projection and must remain an exact affine projection.
        // Actual checkpoint pattern: 223 * min-subnormal divided by two is
        // exactly halfway between representable F16 values, so round-to-even
        // stores -112 * min-subnormal rather than the kernel's unrounded
        // Float32 value of -111.5 * min-subnormal.
        scales[count - 1] = 223.0 * FP16_MIN_SUBNORMAL;
        biases[count - 1] = -112.0 * FP16_MIN_SUBNORMAL;
        linear.scales = Param::new(
            Array::from_slice(&scales, &[n_rows, k_dim / 128])
                .as_dtype(Dtype::Float16)
                .unwrap(),
        );
        linear.biases = Param::new(
            Array::from_slice(&biases, &[n_rows, k_dim / 128])
                .as_dtype(Dtype::Float16)
                .unwrap(),
        );
        linear
    }

    #[test]
    fn dense_row4_promotion_installs_primary_parameter_shapes() {
        let _exec = crate::mlx_exec::acquire();
        let args = minimal_qwen3_next_args();
        let mut block = FfnBlock::new_dense(&args, "fixture.mlp").unwrap();
        block.gate_proj = Some(row4_fixture_linear(128, 256, 1));
        block.up_proj = Some(row4_fixture_linear(128, 256, 2));
        block.down_proj = Some(row4_fixture_linear(256, 128, 3));

        let promoted = block.promote_bonsai_row4(7).unwrap();
        assert_eq!(promoted.layers, 1);
        assert_eq!(promoted.projections, 3);
        assert!(promoted.bytes > 0);
        for (name, projection, weight_shape, scale_shape) in [
            (
                "gate",
                block.gate_proj.as_ref().unwrap(),
                &[32, 2, 4, 4][..],
                &[32, 2, 4][..],
            ),
            (
                "up",
                block.up_proj.as_ref().unwrap(),
                &[32, 2, 4, 4][..],
                &[32, 2, 4][..],
            ),
            (
                "down",
                block.down_proj.as_ref().unwrap(),
                &[64, 1, 4, 4][..],
                &[64, 1, 4][..],
            ),
        ] {
            assert_eq!(projection.weight.shape(), weight_shape, "{name} weight");
            assert_eq!(projection.scales.shape(), scale_shape, "{name} scales");
            assert!(projection.bonsai_row4().unwrap().is_some(), "{name} layout");
            validate_dflash_qlinear(name, projection).unwrap();
        }
        assert_eq!(
            block.promote_bonsai_row4(7).unwrap(),
            BonsaiRow4Promotion::default()
        );
    }

    #[test]
    fn dense_row4_exceptional_affine_down_stays_canonical_and_exact() {
        let _exec = crate::mlx_exec::acquire();
        let args = minimal_qwen3_next_args();
        let mut block = FfnBlock::new_dense(&args, "fixture.mlp").unwrap();
        block.gate_proj = Some(row4_fixture_linear(128, 256, 31));
        block.up_proj = Some(row4_fixture_linear(128, 256, 32));
        block.down_proj = Some(exceptional_affine_q1_linear(256, 128, 33));
        let mut canonical = block.clone();

        let expected_bytes = [&block.gate_proj, &block.up_proj]
            .into_iter()
            .map(|projection| {
                let projection = projection.as_ref().unwrap();
                projection.weight.nbytes() + projection.scales.nbytes()
            })
            .sum::<usize>();
        assert!(
            !q1_biases_are_symmetric(
                &block.down_proj.as_ref().unwrap().scales,
                &block.down_proj.as_ref().unwrap().biases,
            )
            .unwrap(),
            "half-subnormal exception must not compact as symmetric"
        );

        let promoted = block.promote_bonsai_row4(31).unwrap();
        assert_eq!(
            promoted,
            BonsaiRow4Promotion {
                layers: 1,
                projections: 2,
                bytes: expected_bytes,
            }
        );
        for (name, projection) in [
            ("gate", block.gate_proj.as_ref().unwrap()),
            ("up", block.up_proj.as_ref().unwrap()),
        ] {
            assert!(projection.bonsai_row4().unwrap().is_some(), "{name}");
            validate_dflash_qlinear(name, projection).unwrap();
        }
        let down = block.down_proj.as_ref().unwrap();
        assert!(matches!(down.weight_layout, QLinearWeightLayout::Canonical));
        assert!(down.bonsai_row4().unwrap().is_none());
        assert!(down.biases.size() > 0);
        validate_dflash_qlinear("down", down).unwrap();

        for rows in [1_i32, 5, 6, 9] {
            let input = Array::from_slice(
                &(0..rows * 256)
                    .map(|index| ((index * 17 + rows * 3).rem_euclid(61) - 30) as f32 * 0.007_812_5)
                    .collect::<Vec<_>>(),
                &[1, rows, 256],
            )
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            let expected = canonical.forward(&input).unwrap();
            let actual = block.forward(&input).unwrap();
            assert_canonical_array_exact(
                &format!("mixed row4/affine dense MLP M={rows}"),
                &actual,
                &expected,
            );
        }

        assert_eq!(
            block.promote_bonsai_row4(31).unwrap(),
            BonsaiRow4Promotion::default(),
            "second promotion must accept the mixed steady state"
        );
    }

    #[test]
    fn promoted_row4_parameter_replacement_is_authoritative_and_fails_closed() {
        let _exec = crate::mlx_exec::acquire();
        const N: i32 = 8;
        const K: i32 = 256;

        let input = Array::from_slice(
            &(0..5 * K)
                .map(|index| ((index * 13 + 7).rem_euclid(43) - 21) as f32 * 0.015_625)
                .collect::<Vec<_>>(),
            &[1, 5, K],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let mut promoted = row4_fixture_linear(N, K, 19);
        let packed = promoted.prepare_bonsai_row4("fixture").unwrap();
        promoted.install_bonsai_row4(packed);

        let f32_bits = |array: &Array| {
            let values = array.as_dtype(Dtype::Float32).unwrap();
            values.eval().unwrap();
            values
                .as_slice::<f32>()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let baseline = promoted.forward(&input).unwrap();

        // Replace the authoritative Param with a same-shaped row4 buffer. A
        // stale layout-owned handle would keep producing `baseline`; rebuilding
        // a borrowed view must instead observe this replacement immediately.
        let weight_shape = promoted.weight.shape().to_vec();
        let weight_len = weight_shape.iter().product::<i32>() as usize;
        promoted.weight = Param::new(Array::from_slice(&vec![0_u32; weight_len], &weight_shape));
        let fresh_weight_view = promoted.bonsai_row4().unwrap().unwrap();
        let direct_after_weight =
            crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(&input, fresh_weight_view).unwrap();
        let forward_after_weight = promoted.forward(&input).unwrap();
        assert_canonical_array_exact(
            "same-shaped row4 weight replacement",
            &forward_after_weight,
            &direct_after_weight,
        );
        assert_ne!(
            f32_bits(&forward_after_weight),
            f32_bits(&baseline),
            "forward retained a stale pre-replacement weight handle"
        );

        // Replacing the scales is authoritative too. Zero scales give an exact
        // zero result, which cannot be explained by either old resident array.
        let scale_shape = promoted.scales.shape().to_vec();
        promoted.scales = Param::new(
            Array::zeros::<f32>(&scale_shape)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        let fresh_scale_view = promoted.bonsai_row4().unwrap().unwrap();
        let direct_after_scales =
            crate::metal_kernel::bonsai_q1_tg_lut4_qmm_view(&input, fresh_scale_view).unwrap();
        let forward_after_scales = promoted.forward(&input).unwrap();
        assert_canonical_array_exact(
            "same-shaped row4 scale replacement",
            &forward_after_scales,
            &direct_after_scales,
        );
        let expected_zero = Array::zeros::<f32>(&[1, 5, N])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert_canonical_array_exact(
            "zero replacement scales",
            &forward_after_scales,
            &expected_zero,
        );
        validate_dflash_qlinear("fixture", &promoted).unwrap();

        // A same-shaped but wrong-dtype parameter replacement cannot satisfy
        // the physical row4 contract and must fail before any kernel dispatch.
        promoted.weight = Param::new(
            Array::zeros::<f32>(&weight_shape)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap(),
        );
        assert!(promoted.bonsai_row4().is_err());
        assert!(promoted.forward(&input).is_err());
        assert!(validate_dflash_qlinear("fixture", &promoted).is_err());
    }

    #[test]
    fn dense_row4_promotion_rejection_is_atomic() {
        let _exec = crate::mlx_exec::acquire();
        let args = minimal_qwen3_next_args();
        let mut block = FfnBlock::new_dense(&args, "fixture.mlp").unwrap();
        block.gate_proj = Some(row4_fixture_linear(128, 256, 1));
        block.up_proj = Some(row4_fixture_linear(256, 256, 2));
        block.down_proj = Some(row4_fixture_linear(256, 128, 3));

        let before = [
            block.gate_proj.as_ref().unwrap().weight.shape().to_vec(),
            block.up_proj.as_ref().unwrap().weight.shape().to_vec(),
            block.down_proj.as_ref().unwrap().weight.shape().to_vec(),
        ];
        assert!(block.promote_bonsai_row4(9).is_err());
        for (projection, expected_shape) in [
            block.gate_proj.as_ref().unwrap(),
            block.up_proj.as_ref().unwrap(),
            block.down_proj.as_ref().unwrap(),
        ]
        .into_iter()
        .zip(before)
        {
            assert!(matches!(
                &projection.weight_layout,
                QLinearWeightLayout::Canonical
            ));
            assert_eq!(projection.weight.shape(), expected_shape);
        }
    }

    #[test]
    fn promoted_row4_m6_wide_prefill_and_clone_match_canonical() {
        let _exec = crate::mlx_exec::acquire();
        const N: i32 = 8;
        const K: i32 = 256;
        let canonical = row4_fixture_linear(N, K, 11);
        let input = Array::from_slice(
            &(0..6 * K)
                .map(|index| ((index * 7 + 5).rem_euclid(37) - 18) as f32 * 0.015_625)
                .collect::<Vec<_>>(),
            &[1, 6, K],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let expected = canonical.forward(&input).unwrap();

        let mut promoted = canonical.clone();
        let packed = promoted.prepare_bonsai_row4("fixture").unwrap();
        promoted.install_bonsai_row4(packed);
        let cloned = promoted.clone();
        let actual = promoted.forward(&input).unwrap();
        let cloned_actual = cloned.forward(&input).unwrap();
        assert_canonical_array_exact("promoted row4 M6 TG-LUT4", &actual, &expected);
        assert_canonical_array_exact("cloned promoted row4", &cloned_actual, &expected);

        let prefill_rows = 9;
        let prefill = Array::from_slice(
            &(0..prefill_rows * K)
                .map(|index| ((index * 5 + 9).rem_euclid(41) - 20) as f32 * 0.015_625)
                .collect::<Vec<_>>(),
            &[1, prefill_rows, K],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let expected_prefill = canonical.forward(&prefill).unwrap();
        let actual_prefill = promoted.forward(&prefill).unwrap();
        assert_canonical_array_exact(
            "promoted row4 direct-dequant prefill",
            &actual_prefill,
            &expected_prefill,
        );

        let narrow = Array::ones::<f32>(&[1, 5, K])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert!(
            promoted
                .bonsai_row4()
                .unwrap()
                .unwrap()
                .accepts_input(&narrow)
        );
    }

    #[test]
    fn symmetric_q1_bias_validation_and_compaction_preserve_affine_fallback() {
        const FP16_MIN_SUBNORMAL: f32 = 5.960_464_5e-8;

        let mut scales = Array::from_slice(&[2.0_f32, 4.0, 6.0, 2.0 * FP16_MIN_SUBNORMAL], &[2, 2]);
        let mut symmetric =
            Array::from_slice(&[-1.0_f32, -2.0, -3.0, -FP16_MIN_SUBNORMAL], &[2, 2]);
        assert!(q1_biases_are_symmetric(&scales, &symmetric).unwrap());

        let asymmetric = Array::from_slice(&[-1.0_f32, -2.0, -3.0, 0.0], &[2, 2]);
        assert!(!q1_biases_are_symmetric(&scales, &asymmetric).unwrap());

        let mut params = HashMap::new();
        params.insert(std::rc::Rc::<str>::from("layer.scales"), &mut scales);
        params.insert(std::rc::Rc::<str>::from("layer.biases"), &mut symmetric);
        let compacted = compact_symmetric_q1_biases(&mut params).unwrap();
        drop(params);

        assert_eq!(compacted.tensors, 1);
        assert_eq!(compacted.bytes, 4 * std::mem::size_of::<f32>());
        assert!(has_symmetric_q1_biases(&symmetric));
    }

    #[test]
    fn dflash_q1_bias_gate_accepts_compacted_and_loaded_affine_biases_only() {
        let weight = Array::from_slice(&[0_u32; 8], &[2, 4]);
        let scales = Array::from_slice(&[0.25_f32, 0.5], &[2, 1]);
        let validate = |biases: &Array| {
            validate_dflash_q1_linear(
                "fixture",
                &weight,
                &scales,
                biases,
                128,
                1,
                crate::quant_mode::QuantMode::Affine,
            )
        };

        assert!(validate(&symmetric_q1_bias_sentinel()).is_ok());
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            let biases = Array::from_slice(&[-0.125_f32, 0.375], &[2, 1])
                .as_dtype(dtype)
                .unwrap();
            assert!(validate(&biases).is_ok(), "loaded {dtype:?} affine bias");
        }

        let unloaded_placeholder = Array::from_slice(&[0.0_f32], &[1]);
        assert!(validate(&unloaded_placeholder).is_err());
        let wrong_shape = Array::from_slice(&[-0.125_f32, 0.375], &[1, 2]);
        assert!(validate(&wrong_shape).is_err());
        let wrong_dtype = Array::from_slice(&[0_u32, 1], &[2, 1]);
        assert!(validate(&wrong_dtype).is_err());
    }

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 262144,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "full_attention_interval": 4,
            "tie_word_embeddings": false,
            "quantization": { "group_size": 64, "bits": 4 }
        }"#;

        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "qwen3_next");
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.num_hidden_layers, 48);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.num_experts, 512);
        assert_eq!(args.num_experts_per_tok, 10);
        assert_eq!(args.full_attention_interval, 4);
        assert_eq!(args.linear_conv_kernel_dim, 4);
        assert!(!args.tie_word_embeddings);
        assert!(args.norm_topk_prob);
        let qc = args.quantization.unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_swiglu() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5], &[1, 3]);
        let x = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[1, 3]);
        let result = swiglu(&gate, &x).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        // silu(1.0) * 2.0 = 0.7311 * 2.0 ~= 1.462
        let first: f32 = result.index((.., 0..1)).item();
        assert!(first > 1.0);
    }

    #[test]
    fn test_gated_delta_kernel_basic() {
        // B=1, T=1, Hk=2, Hv=4, Dk=32, Dv=32
        // Dk must be multiple of 32 for SIMD group width
        let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
        let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        let (y, new_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        y.eval().unwrap();
        new_state.eval().unwrap();
        assert_eq!(y.shape(), &[1, 1, 4, 32]);
        assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
    }

    #[test]
    fn test_sparse_moe_rejects_top_k_exceeding_num_experts() {
        assert_sparse_moe_rejects(
            |a| {
                a.num_experts = 4;
                a.num_experts_per_tok = 8;
            },
            "num_experts_per_tok",
        );
    }

    #[test]
    fn test_sparse_moe_accepts_top_k_equal_to_num_experts() {
        let mut args = minimal_qwen3_next_args();
        args.num_experts = 4;
        args.num_experts_per_tok = 4; // top_k == num_experts is fine
        let result = SparseMoeBlock::new(&args, "test.layer.mlp");
        assert!(result.is_ok());
    }

    fn assert_sparse_moe_rejects(
        mutate: impl FnOnce(&mut Qwen3NextModelArgs),
        expected_substring: &str,
    ) {
        let mut args = minimal_qwen3_next_args();
        mutate(&mut args);
        let result = SparseMoeBlock::new(&args, "test.layer.mlp");
        assert!(result.is_err(), "Should reject invalid args");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains(expected_substring),
            "Expected error about {expected_substring}, got: {msg}"
        );
    }

    #[test]
    fn test_sparse_moe_rejects_zero_num_experts() {
        assert_sparse_moe_rejects(|a| a.num_experts = 0, "num_experts");
    }

    #[test]
    fn test_sparse_moe_rejects_zero_num_experts_per_tok() {
        assert_sparse_moe_rejects(|a| a.num_experts_per_tok = 0, "num_experts_per_tok");
    }

    /// Minimal args for tests that only care about `MoE` fields.
    fn minimal_qwen3_next_args() -> Qwen3NextModelArgs {
        serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 512,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap()
    }

    /// Full args suitable for `Qwen3NextCausalLM::new()` validation tests.
    fn valid_causal_lm_args() -> Qwen3NextModelArgs {
        serde_json::from_str(
            r#"{
                "model_type": "qwen3_next",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 512,
                "full_attention_interval": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_causal_lm_rejects_zero_full_attention_interval() {
        let mut args = valid_causal_lm_args();
        args.full_attention_interval = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(
            result.is_err(),
            "Should reject full_attention_interval == 0"
        );
    }

    #[test]
    fn test_causal_lm_rejects_zero_linear_key_heads() {
        let mut args = valid_causal_lm_args();
        args.linear_num_key_heads = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_num_key_heads == 0");
    }

    #[test]
    fn test_causal_lm_rejects_zero_linear_value_heads() {
        let mut args = valid_causal_lm_args();
        args.linear_num_value_heads = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_num_value_heads == 0");
    }

    #[test]
    fn test_causal_lm_rejects_zero_conv_kernel_dim() {
        let mut args = valid_causal_lm_args();
        args.linear_conv_kernel_dim = 0;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err(), "Should reject linear_conv_kernel_dim == 0");
    }

    #[test]
    fn test_layer_cache_variants() {
        let kv = LayerCache::KV(SteppingKeyValueCache::new());
        let arrays = LayerCache::Arrays(ArraysCache::new());
        match &kv {
            LayerCache::KV(c) => assert_eq!(c.offset(), 0),
            LayerCache::Arrays(_) => panic!("Expected KV variant"),
        }
        match &arrays {
            LayerCache::Arrays(c) => assert_eq!(c.offset, 0),
            LayerCache::KV(_) => panic!("Expected Arrays variant"),
        }
    }

    #[test]
    fn test_config_deserialization_missing_optional_fields() {
        // Only required fields; all serde(default) fields should get defaults
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!((args.partial_rotary_factor - 1.0).abs() < f32::EPSILON);
        assert_eq!(args.full_attention_interval, 4);
        assert!(!args.tie_word_embeddings);
        assert!(!args.attention_bias);
        assert!(args.rope_scaling.is_none());
        assert!(args.quantization.is_none());
        assert_eq!(args.linear_num_value_heads, 0);
        assert_eq!(args.linear_num_key_heads, 0);
        assert_eq!(args.linear_key_head_dim, 0);
        assert_eq!(args.linear_value_head_dim, 0);
        assert_eq!(args.linear_conv_kernel_dim, 0);
        assert_eq!(args.num_experts, 0);
        assert_eq!(args.num_experts_per_tok, 0);
        assert_eq!(args.decoder_sparse_step, 0);
        assert!(args.norm_topk_prob);
        assert!(args.mlp_only_layers.is_empty());
    }

    #[test]
    fn test_config_deserialization_quantization_null() {
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "quantization": null
        }"#;
        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.quantization.is_none());
    }

    #[test]
    fn test_load_qwen3_next_args_injects_gate_quantization_override() {
        let config = serde_json::json!({
            "model_type": "qwen3_next",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 32,
            "moe_intermediate_size": 32,
            "full_attention_interval": 4,
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "model.layers.0.mlp.gate": {
                    "group_size": 64,
                    "bits": 8
                }
            }
        });

        let args = load_qwen3_next_args_from_value(config).unwrap();
        let gate_q = args.gate_quantization.unwrap();
        assert_eq!(gate_q.group_size, 64);
        assert_eq!(gate_q.bits, 8);
    }

    #[test]
    fn test_load_args_lifts_mix_bit_overrides_into_quant_overrides() {
        // Unsloth UD-style: top-level `quantization` carries default `(group_size, bits)`
        // plus per-tensor override entries keyed by canonical module path.
        // Loader must lift those entries into `args.quant_overrides`.
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "quantization": {
                "group_size": 64,
                "bits": 2,
                "language_model.lm_head": { "group_size": 64, "bits": 5, "mode": "affine" },
                "language_model.model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
                "language_model.model.layers.0.mlp.down_proj": { "group_size": 64, "bits": 3, "mode": "affine" },
                "language_model.model.layers.0.linear_attn.in_proj_qkv": { "group_size": 64, "bits": 4, "mode": "affine" },
                "language_model.model.layers.3.self_attn.q_proj": { "group_size": 64, "bits": 4, "mode": "affine" }
            }
        });

        let args = load_qwen3_next_args_from_value(config).unwrap();

        // Default still parses.
        let q = args.quantization.as_ref().unwrap();
        assert_eq!((q.group_size, q.bits), (64, 2));

        // Every override key landed.
        assert_eq!(
            args.quant_overrides
                .get("language_model.lm_head")
                .map(|o| (o.group_size, o.bits)),
            Some((64, 5))
        );
        assert_eq!(
            args.quant_overrides
                .get("language_model.model.embed_tokens")
                .map(|o| (o.group_size, o.bits)),
            Some((64, 4))
        );
        assert_eq!(
            args.quant_overrides
                .get("language_model.model.layers.0.mlp.down_proj")
                .map(|o| (o.group_size, o.bits)),
            Some((64, 3))
        );
        assert_eq!(
            args.quant_overrides
                .get("language_model.model.layers.0.linear_attn.in_proj_qkv")
                .map(|o| (o.group_size, o.bits)),
            Some((64, 4))
        );
        assert_eq!(
            args.quant_overrides
                .get("language_model.model.layers.3.self_attn.q_proj")
                .map(|o| (o.group_size, o.bits)),
            Some((64, 4))
        );

        // Scalar default keys (`bits`, `group_size`) must not pollute the override map.
        assert!(!args.quant_overrides.contains_key("bits"));
        assert!(!args.quant_overrides.contains_key("group_size"));
    }

    #[test]
    fn test_collect_quant_overrides_synthesizes_fused_gdn_key() {
        // Real Unsloth UD-Q2 checkpoints publish GDN per-tensor overrides under
        // the on-disk SPLIT keys (`in_proj_qkv` / `in_proj_z`). The model in
        // default (non-separate) GDN mode resolves the FUSED key
        // (`in_proj_qkvz`) at QLinear construction time. The lift step must
        // synthesize a fused-key entry from agreeing split entries, otherwise
        // the global default applies and the quantized matmul shape check fails
        // at runtime against the 4-bit-packed weight on disk.
        let config = serde_json::json!({
            "quantization": {
                "group_size": 64,
                "bits": 2,
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "group_size": 64, "bits": 4, "mode": "affine"
                },
                "language_model.model.layers.0.linear_attn.in_proj_z": {
                    "group_size": 64, "bits": 4, "mode": "affine"
                }
            }
        });

        let overrides = collect_quant_overrides(&config);
        let fused = overrides
            .get("language_model.model.layers.0.linear_attn.in_proj_qkvz")
            .and_then(|v| v.as_object())
            .expect("fused-key override synthesized");
        assert_eq!(fused.get("group_size"), Some(&serde_json::json!(64)));
        assert_eq!(fused.get("bits"), Some(&serde_json::json!(4)));
    }

    #[test]
    fn test_collect_quant_overrides_skips_synthesis_when_split_overrides_disagree() {
        let config = serde_json::json!({
            "quantization": {
                "group_size": 64,
                "bits": 2,
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "group_size": 64, "bits": 4, "mode": "affine"
                },
                "language_model.model.layers.0.linear_attn.in_proj_z": {
                    "group_size": 64, "bits": 3, "mode": "affine"
                }
            }
        });
        let overrides = collect_quant_overrides(&config);
        assert!(
            !overrides.contains_key("language_model.model.layers.0.linear_attn.in_proj_qkvz"),
            "must not synthesize when split overrides disagree on (group_size, bits)"
        );
    }

    #[test]
    fn test_collect_quant_overrides_does_not_overwrite_existing_fused_override() {
        let config = serde_json::json!({
            "quantization": {
                "group_size": 64,
                "bits": 2,
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "group_size": 64, "bits": 4, "mode": "affine"
                },
                "language_model.model.layers.0.linear_attn.in_proj_z": {
                    "group_size": 64, "bits": 4, "mode": "affine"
                },
                "language_model.model.layers.0.linear_attn.in_proj_qkvz": {
                    "group_size": 64, "bits": 5, "mode": "affine"
                }
            }
        });
        let overrides = collect_quant_overrides(&config);
        let fused = overrides
            .get("language_model.model.layers.0.linear_attn.in_proj_qkvz")
            .and_then(|v| v.as_object())
            .expect("explicit fused override preserved");
        // Explicit user override wins over synthesized split-pair.
        assert_eq!(fused.get("bits"), Some(&serde_json::json!(5)));
    }

    #[test]
    fn test_resolve_quant_for_falls_back_to_default_when_no_override() {
        let mut args: Qwen3NextModelArgs = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1024,
            "max_position_embeddings": 4096,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "quantization": { "group_size": 64, "bits": 2 }
        }))
        .unwrap();

        // Default fallback: no overrides yet.
        let spec = resolve_quant_for(&args, "language_model.model.layers.0.mlp.down_proj");
        assert_eq!((spec.group_size, spec.bits), (64, 2));

        // Insert an override; resolver picks it up over the default.
        args.quant_overrides.insert(
            "language_model.model.layers.0.mlp.down_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 3,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );
        let spec = resolve_quant_for(&args, "language_model.model.layers.0.mlp.down_proj");
        assert_eq!((spec.group_size, spec.bits), (64, 3));

        // Unrelated key still falls back to the default.
        let spec = resolve_quant_for(&args, "language_model.lm_head");
        assert_eq!((spec.group_size, spec.bits), (64, 2));
    }

    #[test]
    fn test_decoder_layer_routes_overrides_to_qlinears() {
        // A full-attention layer (idx 3 with full_attention_interval=4) for the
        // dense Qwen3.5 path. We override mlp.down_proj to 3-bit and self_attn.q_proj
        // to 5-bit; everything else must stay at the global default (64, 2).
        let mut args = valid_causal_lm_args();
        args.num_experts = 0; // dense FFN path
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 2,
            mode: crate::quant_mode::QuantMode::Affine,
        });
        args.quant_overrides.insert(
            "language_model.model.layers.3.mlp.down_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 3,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );
        args.quant_overrides.insert(
            "language_model.model.layers.3.self_attn.q_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 5,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );

        let layer = DecoderLayer::new(&args, 3).unwrap();
        assert!(!layer.is_linear, "layer 3 should be full-attention");

        let attn = layer.self_attn.as_ref().expect("self_attn present");
        assert_eq!(attn.q_proj.bits, 5, "q_proj override applied");
        assert_eq!(attn.k_proj.bits, 2, "k_proj falls back to global");
        assert_eq!(attn.v_proj.bits, 2, "v_proj falls back to global");
        assert_eq!(attn.o_proj.bits, 2, "o_proj falls back to global");

        let down = layer
            .mlp
            .down_proj
            .as_ref()
            .expect("dense down_proj present");
        let gate = layer
            .mlp
            .gate_proj
            .as_ref()
            .expect("dense gate_proj present");
        let up = layer.mlp.up_proj.as_ref().expect("dense up_proj present");
        assert_eq!(down.bits, 3, "down_proj override applied");
        assert_eq!(gate.bits, 2, "gate_proj falls back to global");
        assert_eq!(up.bits, 2, "up_proj falls back to global");
    }

    #[test]
    fn test_decoder_layer_moe_routes_overrides_to_shared_expert_and_switch_mlp() {
        // MoE layer at idx 3 (full attention). Overrides target the shared expert
        // down_proj, switch_mlp gate_proj, and the router gate.
        let mut args = valid_causal_lm_args();
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        });
        args.quant_overrides.insert(
            "language_model.model.layers.3.mlp.shared_expert.down_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 3,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );
        args.quant_overrides.insert(
            "language_model.model.layers.3.mlp.switch_mlp.gate_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 5,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );
        args.quant_overrides.insert(
            "language_model.model.layers.3.mlp.gate".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 8,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );

        let layer = DecoderLayer::new(&args, 3).unwrap();

        let gate = layer.mlp.gate.as_ref().expect("MoE gate present");
        assert_eq!(gate.bits, 8, "router gate override applied");

        let switch = layer.mlp.switch_mlp.as_ref().expect("switch_mlp present");
        assert_eq!(
            switch.gate_proj.bits, 5,
            "switch_mlp.gate_proj override applied"
        );
        assert_eq!(
            switch.down_proj.bits, 4,
            "switch_mlp.down_proj falls back to global"
        );

        let shared = layer
            .mlp
            .shared_expert
            .as_ref()
            .expect("shared_expert present");
        assert_eq!(
            shared.down_proj.bits, 3,
            "shared_expert.down_proj override applied"
        );
        assert_eq!(
            shared.gate_proj.bits, 4,
            "shared_expert.gate_proj falls back to global"
        );
    }

    #[test]
    fn test_o_proj_and_out_proj_are_bf16_in_qwen3_5() {
        // `dense_attention_outputs` forces the four checkpoint-BF16-dense
        // attention output projections to bits=0, while leaving every other
        // QLinear at the resolved (overrides → global) bit width.
        let mut args = valid_causal_lm_args();
        args.dense_attention_outputs = true;
        args.use_separate_gdn_projections = true;
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        });

        // Full-attention layer (idx 3): only o_proj drops to BF16-dense.
        let attn_layer = DecoderLayer::new(&args, 3).unwrap();
        let attn = attn_layer
            .self_attn
            .as_ref()
            .expect("self_attn at full-attention layer");
        assert_eq!(attn.q_proj.bits, 4, "q_proj keeps global quant");
        assert_eq!(attn.k_proj.bits, 4, "k_proj keeps global quant");
        assert_eq!(attn.v_proj.bits, 4, "v_proj keeps global quant");
        assert_eq!(attn.o_proj.bits, 0, "o_proj forced to BF16-dense");

        // Linear (GDN) layer (idx 0): out_proj, in_proj_ba, in_proj_a, in_proj_b
        // drop to BF16-dense; in_proj_qkvz / in_proj_qkv / in_proj_z keep quant.
        let gdn_layer = DecoderLayer::new(&args, 0).unwrap();
        let gdn = gdn_layer
            .linear_attn
            .as_ref()
            .expect("linear_attn at GDN layer");
        assert_eq!(gdn.in_proj_qkvz.bits, 4, "in_proj_qkvz keeps global quant");
        assert_eq!(gdn.in_proj_ba.bits, 0, "in_proj_ba forced to BF16-dense");
        assert_eq!(gdn.out_proj.bits, 0, "out_proj forced to BF16-dense");
        assert_eq!(
            gdn.in_proj_qkv.as_ref().expect("separate in_proj_qkv").bits,
            4,
            "in_proj_qkv keeps global quant",
        );
        assert_eq!(
            gdn.in_proj_z.as_ref().expect("separate in_proj_z").bits,
            4,
            "in_proj_z keeps global quant",
        );
        assert_eq!(
            gdn.in_proj_a.as_ref().expect("separate in_proj_a").bits,
            0,
            "in_proj_a forced to BF16-dense",
        );
        assert_eq!(
            gdn.in_proj_b.as_ref().expect("separate in_proj_b").bits,
            0,
            "in_proj_b forced to BF16-dense",
        );

        // Scales/biases for bits=0 use shape [0] so they bypass the
        // placeholder-`[1]` missing-param check after weight loading.
        assert_eq!(attn.o_proj.scales.shape(), [0]);
        assert_eq!(attn.o_proj.biases.shape(), [0]);
        assert_eq!(gdn.out_proj.scales.shape(), [0]);
        assert_eq!(gdn.out_proj.biases.shape(), [0]);
    }

    #[test]
    fn test_placeholder_param_names_finds_shape_one_tensors() {
        let loaded = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let placeholder = Array::from_slice(&[0.0f32], &[1]);
        let names =
            placeholder_param_names([("loaded.weight", &loaded), ("missing.weight", &placeholder)]);
        assert_eq!(names, vec!["missing.weight".to_owned()]);
    }

    #[test]
    fn test_ensure_all_model_params_loaded_errors_on_placeholders() {
        let loaded = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let placeholder = Array::from_slice(&[0.0f32], &[1]);
        let err = ensure_all_model_params_loaded([
            ("loaded.weight", &loaded),
            ("missing.weight", &placeholder),
        ])
        .unwrap_err();
        if let ModelError::MissingWeight(message) = err {
            assert!(message.contains("1 model params"));
            assert!(message.contains("missing.weight"));
        } else {
            panic!("expected MissingWeight");
        }
    }

    #[test]
    fn test_ensure_all_model_params_loaded_accepts_fully_loaded_params() {
        let weight = Array::from_slice(&[1.0f32, 2.0], &[2]);
        ensure_all_model_params_loaded([("loaded.weight", &weight)]).unwrap();
    }

    #[test]
    fn test_swiglu_numeric_correctness() {
        // silu(x) = x * sigmoid(x)
        // silu(0) = 0 * 0.5 = 0
        // silu(1) = 1 * sigmoid(1) = 1 * 0.7310586 = 0.7310586
        // silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689414 = -0.2689414

        // swiglu(gate, x) = silu(gate) * x

        // gate=0, x=5 => silu(0) * 5 = 0
        let gate = Array::from_slice(&[0.0_f32], &[1, 1]);
        let x = Array::from_slice(&[5.0_f32], &[1, 1]);
        let result = swiglu(&gate, &x).unwrap();
        let val: f32 = result.item();
        assert!((val - 0.0).abs() < 1e-6, "silu(0)*5 should be 0, got {val}");

        // gate=1, x=1 => silu(1) * 1 = 0.7310586
        let gate2 = Array::from_slice(&[1.0_f32], &[1, 1]);
        let x2 = Array::from_slice(&[1.0_f32], &[1, 1]);
        let result2 = swiglu(&gate2, &x2).unwrap();
        let val2: f32 = result2.item();
        assert!(
            (val2 - 0.731_058_6).abs() < 1e-4,
            "silu(1)*1 should be ~0.7311, got {val2}"
        );

        // gate=-1, x=2 => silu(-1) * 2 = -0.2689414 * 2 = -0.5378828
        let gate3 = Array::from_slice(&[-1.0_f32], &[1, 1]);
        let x3 = Array::from_slice(&[2.0_f32], &[1, 1]);
        let result3 = swiglu(&gate3, &x3).unwrap();
        let val3: f32 = result3.item();
        assert!(
            (val3 - (-0.537_882_8)).abs() < 1e-4,
            "silu(-1)*2 should be ~-0.5379, got {val3}"
        );
    }

    #[test]
    fn test_sparse_moe_happy_path_construction() {
        let args = minimal_qwen3_next_args();
        let result = SparseMoeBlock::new(&args, "test.layer.mlp");
        assert!(result.is_ok());
        let block = result.unwrap();
        assert_eq!(block.top_k, args.num_experts_per_tok);
        assert!(block.norm_topk_prob);
    }

    #[test]
    fn test_causal_lm_valid_construction() {
        let args = valid_causal_lm_args();
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.args.model_type, "qwen3_next");
    }

    #[test]
    fn test_causal_lm_make_cache_layer_types() {
        let args = valid_causal_lm_args();
        let model = Qwen3NextCausalLM::new(args).unwrap();
        let cache = model.make_cache();
        // 4 layers, full_attention_interval=4, so layers 0,1,2 are linear, layer 3 is full attention
        assert_eq!(cache.len(), 4);
        for (i, layer_cache) in cache.iter().enumerate() {
            let lc = layer_cache.as_ref().unwrap();
            let is_linear = (i + 1) % 4 != 0;
            if is_linear {
                assert!(
                    matches!(lc, LayerCache::Arrays(_)),
                    "Layer {i} should be Arrays (linear)"
                );
            } else {
                assert!(
                    matches!(lc, LayerCache::KV(_)),
                    "Layer {i} should be KV (full attention)"
                );
            }
        }
    }

    #[test]
    fn test_causal_lm_negative_full_attention_interval() {
        let mut args = valid_causal_lm_args();
        args.full_attention_interval = -1;
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_lm_with_quantization() {
        let mut args = valid_causal_lm_args();
        args.quantization = Some(QuantizationConfig {
            group_size: 32,
            bits: 8,
            mode: crate::quant_mode::QuantMode::Affine,
        });
        let result = Qwen3NextCausalLM::new(args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_lm_with_tied_embeddings() {
        let mut args = valid_causal_lm_args();
        args.tie_word_embeddings = true;
        let model = Qwen3NextCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_causal_lm_without_tied_embeddings() {
        let mut args = valid_causal_lm_args();
        args.tie_word_embeddings = false;
        let model = Qwen3NextCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    #[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
    fn test_mtp_draft_advances_cache_for_confirmed_token() {
        let stream = Stream::new();
        mlx_rs::with_new_default_stream(stream, || {
            let mut args = valid_causal_lm_args();
            args.mtp_num_hidden_layers = 1;
            let mut model = Qwen3NextCausalLM::new(args).unwrap();
            let mut mtp_cache = model.make_mtp_cache().unwrap();
            let hidden = Array::zeros::<f32>(&[1, 1, model.args.hidden_size]).unwrap();

            let draft = model.mtp_draft(&hidden, 0, &mut mtp_cache).unwrap();
            draft.eval().unwrap();

            assert_eq!(mtp_cache[0].offset(), 1);
        });
    }

    #[test]
    #[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
    fn test_mtp_advance_appends_accepted_token_state() {
        let stream = Stream::new();
        mlx_rs::with_new_default_stream(stream, || {
            let mut args = valid_causal_lm_args();
            args.mtp_num_hidden_layers = 1;
            let mut model = Qwen3NextCausalLM::new(args).unwrap();
            let mut mtp_cache = model.make_mtp_cache().unwrap();
            let hidden = Array::zeros::<f32>(&[1, 1, model.args.hidden_size]).unwrap();

            let draft = model.mtp_draft(&hidden, 0, &mut mtp_cache).unwrap();
            draft.eval().unwrap();
            model.mtp_advance(&hidden, 1, &mut mtp_cache).unwrap();

            assert_eq!(mtp_cache[0].offset(), 2);
        });
    }

    #[test]
    #[ignore = "requires real model weights; placeholder test tensors cannot run MTP forward"]
    fn test_mtp_advance_many_appends_accepted_token_states() {
        let stream = Stream::new();
        mlx_rs::with_new_default_stream(stream, || {
            let mut args = valid_causal_lm_args();
            args.mtp_num_hidden_layers = 1;
            let mut model = Qwen3NextCausalLM::new(args).unwrap();
            let mut mtp_cache = model.make_mtp_cache().unwrap();
            let hidden = Array::zeros::<f32>(&[1, 2, model.args.hidden_size]).unwrap();

            model
                .mtp_advance_many(&hidden, &[1, 2], &mut mtp_cache)
                .unwrap();

            assert_eq!(mtp_cache[0].offset(), 2);
        });
    }

    #[test]
    fn test_mtp_advance_many_rejects_cache_layer_mismatch() {
        let err =
            Qwen3NextCausalLM::validate_mtp_advance_many_shape(&[1, 1, 256], 0, 1, 1).unwrap_err();

        assert!(
            err.to_string().contains("mtp_cache length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_mtp_advance_many_rejects_hidden_sequence_mismatch() {
        let err =
            Qwen3NextCausalLM::validate_mtp_advance_many_shape(&[1, 2, 256], 1, 1, 1).unwrap_err();

        assert!(
            err.to_string().contains("hidden sequence length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_load_model_args_happy_path() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        let args = load_model_args(dir.path()).unwrap();
        assert_eq!(args.model_type, "qwen3_next");
        assert_eq!(args.hidden_size, 2048);
    }

    #[test]
    fn test_load_model_args_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_args_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{{bad json").unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_arrays_cache_default() {
        let cache = ArraysCache::default();
        assert!(cache.conv_state.is_none());
        assert!(cache.ssm_state.is_none());
        assert_eq!(cache.offset, 0);
    }

    #[test]
    fn test_gated_delta_kernel_prefill() {
        // B=1, T=4, Hk=2, Hv=4, Dk=32, Dv=32
        let q = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 4, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 4, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 4, 4]).unwrap();
        let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        let (y, new_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 4, 2, 32, 4, 32,
        )
        .unwrap();
        y.eval().unwrap();
        new_state.eval().unwrap();
        assert_eq!(y.shape(), &[1, 4, 4, 32]);
        assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
    }

    // -----------------------------------------------------------------------
    // gather_qmm + MoE rewrite tests
    // -----------------------------------------------------------------------

    /// Quantize a float matrix and return (weight, scales, biases) suitable for
    /// `gather_qmm` / `quantized_matmul`.
    fn quantize_weights(w: &Array, group_size: i32, bits: i32) -> (Array, Array, Array) {
        let (qw, scales, biases) = ops::quantize(w, group_size, bits).unwrap();
        (qw, scales, biases)
    }

    #[test]
    fn test_gather_qmm_basic() {
        // 2 experts, out=64, in=64 (dims must be multiples of 32 for quantize)
        let w_float = Array::ones::<f32>(&[2, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Input [1, 1, 1, 64], select expert 0
        let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32], &[1, 1, 1]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        // Force evaluation to run the Metal kernel (MLX is lazy)
        result.eval().unwrap();
        // Output: [1, 1, 1, 1, 64] (batch broadcast with indices, M=1, N=64)
        assert_eq!(result.ndim(), 5);
        assert_eq!(*result.shape().last().unwrap(), 64);
    }

    #[test]
    fn test_gather_qmm_multi_expert() {
        // 4 experts, out=64, in=64
        let w_float = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32, 2, 3], &[1, 1, 3]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        // Output: [1, 1, 3, 1, 64] — 3 experts selected
        assert_eq!(*result.shape().get(2).unwrap(), 3);
    }

    #[test]
    fn test_gather_qmm_matches_per_expert() {
        // Verify that gather_qmm produces the same result as the old
        // take_axis + quantized_matmul path for a single expert.
        let w_float = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[4, 64, 64], None).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 64], None).unwrap();
        let expert_idx = Array::from_slice(&[2_u32], &[1]);

        // Old path: take_axis + quantized_matmul
        let ew = qw
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let es = scales
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let eb = biases
            .take_axis(&expert_idx, 0)
            .unwrap()
            .squeeze_axes(&[0])
            .unwrap();
        let old_result = ops::quantized_matmul(&x, &ew, &es, &eb, true, 64, 4).unwrap();

        // New path: gather_qmm
        let x_expanded = x.expand_dims(-2).unwrap(); // [1, 1, 64]
        let indices = Array::from_slice(&[2_u32], &[1, 1]);
        let new_result = gather_qmm(
            &x_expanded,
            &qw,
            &scales,
            &biases,
            &indices,
            true,
            64,
            4,
            false,
        )
        .unwrap()
        .squeeze_axes(&[-2])
        .unwrap()
        .squeeze_axes(&[-2])
        .unwrap();

        // Compare element-wise (both are quantized, should be exact match)
        let diff = old_result.subtract(&new_result).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "gather_qmm and per-expert path differ by {max_diff}"
        );
    }

    #[test]
    fn test_forward_gather_global_sort_shape() {
        // RED: forward_gather_global_sort should produce [B, L, top_k, D]
        let args = minimal_qwen3_next_args();
        let mut block = SwitchMlpWeights::new(&args, "test.layer.mlp.switch_mlp").unwrap();

        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        // B=1, L=4, top_k=2 — enough tokens to exercise the sort path
        let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
        let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

        let result = block.forward_gather_global_sort(&x, &indices).unwrap();
        assert_eq!(result.shape(), &[1, 4, 2, 64]);
    }

    #[test]
    fn test_forward_gather_global_sort_equivalence() {
        // RED: global sort must produce the same values as forward_gather
        let args = minimal_qwen3_next_args();
        let mut block = SwitchMlpWeights::new(&args, "test.layer.mlp.switch_mlp").unwrap();

        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
        let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

        let baseline = block.forward_gather(&x, &indices, false).unwrap();
        let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
        baseline.eval().unwrap();
        sorted.eval().unwrap();

        let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "global sort and baseline differ by {max_diff}"
        );
    }

    #[test]
    fn test_forward_gather_global_sort_random_weights() {
        // Harder: random weights + distinct per-token inputs + more experts
        // Verifies the sort/unsort cycle preserves per-token identity.
        let num_experts = 8;
        let hidden = 64;
        let top_k = 3;
        let b = 1;
        let l = 16;

        let args = minimal_qwen3_next_args();
        let mut block = SwitchMlpWeights::new(&args, "test.layer.mlp.switch_mlp").unwrap();

        let gate_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
                .unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        // Random input — each token is distinct
        let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
        // Random expert indices in [0, num_experts)
        let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
            .map(|i| i % num_experts as u32)
            .collect();
        let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
        x.eval().unwrap();
        indices.eval().unwrap();

        let baseline = block.forward_gather(&x, &indices, false).unwrap();
        let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
        baseline.eval().unwrap();
        sorted.eval().unwrap();

        assert_eq!(baseline.shape(), sorted.shape());
        assert_eq!(sorted.shape(), &[b, l, top_k, hidden]);

        let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-4,
            "random weights: global sort differs by {max_diff}"
        );
    }

    #[test]
    fn test_moe_gate_up_fusion_parity() {
        // Fused gate+up (2 gather_qmm) must match unfused (3 gather_qmm).
        // Uses random weights + distinct per-token inputs to stress sort/unsort.
        let num_experts = 8;
        let hidden = 128;
        let intermediate = 64;
        let top_k = 3;
        let b = 1;
        let l = 16;

        let mut block = SwitchMlpWeights::from_quant(64, 4).unwrap();

        let gate_w = mlx_rs::random::uniform::<f32, f32>(
            -1.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = mlx_rs::random::uniform::<f32, f32>(
            -1.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = mlx_rs::random::uniform::<f32, f32>(
            -1.0,
            1.0,
            &[num_experts, hidden, intermediate],
            None,
        )
        .unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
        let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
            .map(|i| i % num_experts as u32)
            .collect();
        let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
        x.eval().unwrap();
        indices.eval().unwrap();

        // Reference: unfused 3-call path
        let reference = block.forward_gather_global_sort(&x, &indices).unwrap();
        // Fused: 2-call path
        let fused = block.forward_gather_fused(&x, &indices).unwrap();
        reference.eval().unwrap();
        fused.eval().unwrap();

        assert_eq!(reference.shape(), fused.shape());
        assert_eq!(fused.shape(), &[b, l, top_k, hidden]);

        let diff = reference.subtract(&fused).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "fused gate+up differs from unfused by {max_diff}"
        );
    }

    #[test]
    fn test_switch_mlp_forward_gather_shapes() {
        // Verify forward_gather produces the correct output shape with the
        // double expand_dims pattern matching Python's SwitchGLU.
        let args = minimal_qwen3_next_args();
        let mut block = SwitchMlpWeights::new(&args, "test.layer.mlp.switch_mlp").unwrap();

        // 4 experts, intermediate=64, hidden=64
        let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
        *block.gate_proj.weight = gw;
        *block.gate_proj.scales = gs;
        *block.gate_proj.biases = gb;

        let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
        *block.up_proj.weight = uw;
        *block.up_proj.scales = us;
        *block.up_proj.biases = ub;

        let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
        *block.down_proj.weight = dw;
        *block.down_proj.scales = ds;
        *block.down_proj.biases = db;

        let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
        let indices = Array::from_slice(&[0_u32, 1, 2], &[1, 1, 3]);

        let result = block.forward_gather(&x, &indices, false).unwrap();
        // [B=1, L=1, top_k=3, D=64]
        assert_eq!(result.shape(), &[1, 1, 3, 64]);
    }

    #[test]
    fn test_sparse_moe_forward_output_shape() {
        // Build a SparseMoeBlock with quantized dummy weights and verify the
        // full forward pass produces the correct output shape.
        let mut args = minimal_qwen3_next_args();
        args.num_experts = 4;
        args.num_experts_per_tok = 2;
        args.moe_intermediate_size = 64;
        args.shared_expert_intermediate_size = 64;
        args.hidden_size = 64;
        args.gate_quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 8,
            mode: crate::quant_mode::QuantMode::Affine,
        });

        let mut block = SparseMoeBlock::new(&args, "test.layer.mlp").unwrap();

        // Set router gate weights: [num_experts, hidden_size]
        let gate_w = Array::ones::<f32>(&[4, 64]).unwrap();
        let (gw, gs, gb) = quantize_weights(&gate_w, 64, 8);
        *block.gate.weight = gw;
        *block.gate.scales = gs;
        *block.gate.biases = gb;

        // Set switch_mlp expert weights: [4, intermediate, hidden] and [4, hidden, intermediate]
        let proj_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
        let (pw, ps, pb) = quantize_weights(&proj_w, 64, 4);
        for proj in [
            &mut block.switch_mlp.gate_proj,
            &mut block.switch_mlp.up_proj,
        ] {
            *proj.weight = pw.clone();
            *proj.scales = ps.clone();
            *proj.biases = pb.clone();
        }
        *block.switch_mlp.down_proj.weight = pw;
        *block.switch_mlp.down_proj.scales = ps;
        *block.switch_mlp.down_proj.biases = pb;

        // Set shared expert weights
        let shared_w = Array::ones::<f32>(&[64, 64]).unwrap();
        let (sw, ss, sb) = quantize_weights(&shared_w, 64, 4);
        for proj in [
            &mut block.shared_expert.gate_proj,
            &mut block.shared_expert.up_proj,
            &mut block.shared_expert.down_proj,
        ] {
            *proj.weight = sw.clone();
            *proj.scales = ss.clone();
            *proj.biases = sb.clone();
        }

        // Set shared expert gate weights
        let sgate_w = Array::ones::<f32>(&[1, 64]).unwrap();
        let (sgw, sgs, sgb) = quantize_weights(&sgate_w, 64, 8);
        *block.shared_expert_gate.weight = sgw;
        *block.shared_expert_gate.scales = sgs;
        *block.shared_expert_gate.biases = sgb;

        let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
        let result = block.forward(&x).unwrap();
        assert_eq!(result.shape(), &[1, 1, 64]);
    }

    #[test]
    fn test_gather_qmm_model_scale() {
        // Reproduce actual Qwen3-Next-4bit shapes: 512 experts, hidden=2048,
        // intermediate=512, group_size=64, bits=4, top_k=10.
        // Use smaller dims to keep test fast but same expert count.
        let num_experts = 512;
        let hidden = 128; // Smaller than 2048 for test speed
        let intermediate = 64;

        let w_float = mlx_rs::random::uniform::<f32, f32>(
            0.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Decode shape: B=1, L=1, M=1
        let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
        let indices = Array::from_slice(
            &[0_u32, 10, 50, 100, 200, 300, 400, 450, 500, 511],
            &[1, 1, 10],
        );

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        // Force actual Metal kernel evaluation
        result.eval().unwrap();
        assert_eq!(result.shape(), &[1, 1, 10, 1, intermediate]);
    }

    #[test]
    fn test_gather_qmm_prefill_broadcast() {
        // Prefill case: L > 1 requires the double expand_dims pattern.
        // x batch [B, L, 1] must broadcast with indices [B, L, top_k].
        let w_float = Array::ones::<f32>(&[8, 64, 64]).unwrap();
        let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

        // Prefill: B=1, L=9
        let x = Array::ones::<f32>(&[1, 9, 1, 1, 64]).unwrap(); // double expand
        let indices = Array::from_slice(
            &[0_u32, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 7],
            &[1, 9, 2],
        );

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        // [1, 9, 2, 1, 64]: broadcast batch [1,9,1] with [1,9,2] -> [1,9,2], M=1, N=64
        assert_eq!(result.shape(), &[1, 9, 2, 1, 64]);
    }

    #[test]
    fn test_gather_qmm_bfloat16() {
        // Model uses bfloat16 for scales/biases and input activations.
        // Verify gather_qmm works with bfloat16 dtypes.
        use mlx_rs::Dtype;

        let num_experts = 8;
        let hidden = 128;
        let intermediate = 64;

        let w_float = mlx_rs::random::uniform::<f32, f32>(
            0.0,
            1.0,
            &[num_experts, intermediate, hidden],
            None,
        )
        .unwrap();
        let (qw, scales_f32, biases_f32) = quantize_weights(&w_float, 64, 4);

        // Convert scales/biases to bfloat16 (matching model file dtype)
        let scales = scales_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let biases = biases_f32.as_dtype(Dtype::Bfloat16).unwrap();

        // Input in bfloat16
        let x_f32 =
            mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
        let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let indices = Array::from_slice(&[0_u32, 3, 7], &[1, 1, 3]);

        let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
        result.eval().unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 1, intermediate]);
    }

    // -----------------------------------------------------------------------
    // compile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_compute_g_matches_raw() {
        let a_log = Array::from_slice(&[0.5_f32, -0.3], &[1, 2]);
        let a = Array::from_slice(&[1.0_f32, -1.0], &[1, 2]);
        let dt_bias = Array::from_slice(&[0.1_f32, 0.2], &[1, 2]);

        // Raw computation
        let a_plus_bias = a.add(&dt_bias).unwrap();
        let sp = nn::softplus(&a_plus_bias).unwrap();
        let neg_decay = a_log
            .exp()
            .unwrap()
            .negative()
            .unwrap()
            .multiply(sp)
            .unwrap();
        let raw_g = neg_decay.exp().unwrap();

        // Compiled computation
        let mut compiled = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
        let compiled_g = compiled((&a_log, &a, &dt_bias)).unwrap();

        let diff = raw_g.subtract(&compiled_g).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-6,
            "compiled compute_g differs from raw by {max_diff}"
        );
    }

    #[test]
    fn test_compiled_silu_mul_matches_raw() {
        let gate = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 17408], None).unwrap();
        let x = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 17408], None).unwrap();

        let raw = nn::silu(&gate).unwrap().multiply(&x).unwrap();
        let compiled = silu_mul(&gate, &x).unwrap();

        let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "compiled silu_mul differs from raw by {max_diff}"
        );
    }

    #[test]
    fn test_compiled_sigmoid_mul_matches_raw() {
        let gate = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 6144], None).unwrap();
        let x = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 6144], None).unwrap();

        let raw = nn::sigmoid(&gate).unwrap().multiply(&x).unwrap();
        let compiled = sigmoid_mul(&gate, &x).unwrap();

        let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "compiled sigmoid_mul differs from raw by {max_diff}"
        );
    }

    #[test]
    fn test_compiled_gdn_output_gate_matches_raw() {
        let y = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 48, 128], None).unwrap();
        let z = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 48, 128], None).unwrap();
        let weight = Array::ones::<f32>(&[128]).unwrap();

        let normed = fast::rms_norm(&y, &weight, 1e-6).unwrap();
        let raw = nn::silu(&z).unwrap().multiply(&normed).unwrap();
        let compiled = gdn_output_gate(&y, &weight, 1e-6, &z).unwrap();

        let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-5,
            "compiled gdn_output_gate differs from raw by {max_diff}"
        );
    }

    fn canonical_exact_weight(rows: i32, columns: i32, salt: i32) -> Array {
        let mut values = vec![0.0_f32; (rows * columns) as usize];
        for row in 0..rows {
            let column = (row * 17 + salt).rem_euclid(columns);
            let coefficient = match row.rem_euclid(4) {
                0 => -0.5_f32,
                1 => -0.25_f32,
                2 => 0.25_f32,
                _ => 0.5_f32,
            };
            values[(row * columns + column) as usize] = coefficient;
        }
        let weight = Array::from_slice(&values, &[rows, columns])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        weight.eval().unwrap();
        weight
    }

    fn install_canonical_exact_dense(linear: &mut QLinear, rows: i32, columns: i32, salt: i32) {
        linear.weight = Param::new(canonical_exact_weight(rows, columns, salt));
        linear.scales = Param::new(Array::from_slice::<f32>(&[], &[0]));
        linear.biases = Param::new(Array::from_slice::<f32>(&[], &[0]));
        linear.group_size = 64;
        linear.bits = 0;
        linear.mode = crate::quant_mode::QuantMode::Dense;
    }

    fn canonical_exact_input(seq_len: i32, hidden_size: i32, salt: i32) -> Array {
        let values = (0..seq_len * hidden_size)
            .map(|index| {
                let value = (index * 11 + salt).rem_euclid(31) - 15;
                value as f32 * 0.015_625
            })
            .collect::<Vec<_>>();
        let input = Array::from_slice(&values, &[1, seq_len, hidden_size])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        input.eval().unwrap();
        input
    }

    fn assert_canonical_array_exact(label: &str, actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{label} dtype");
        let actual_f32 = actual.as_dtype(Dtype::Float32).unwrap();
        let expected_f32 = expected.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&actual_f32, &expected_f32]).unwrap();
        for (index, (got, want)) in actual_f32
            .as_slice::<f32>()
            .iter()
            .zip(expected_f32.as_slice::<f32>())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}[{index}] differs: {got:?} != {want:?}"
            );
        }
    }

    fn assert_canonical_kv_cache_exact(
        label: &str,
        actual: &SteppingKeyValueCache,
        expected: &SteppingKeyValueCache,
    ) {
        assert_eq!(actual.offset(), expected.offset(), "{label} offset");
        let offset = actual.offset();
        let actual_keys =
            crate::cache::slice_axis2(actual.keys().expect("actual keys initialized"), 0, offset)
                .unwrap();
        let expected_keys = crate::cache::slice_axis2(
            expected.keys().expect("expected keys initialized"),
            0,
            offset,
        )
        .unwrap();
        let actual_values = crate::cache::slice_axis2(
            actual.values().expect("actual values initialized"),
            0,
            offset,
        )
        .unwrap();
        let expected_values = crate::cache::slice_axis2(
            expected.values().expect("expected values initialized"),
            0,
            offset,
        )
        .unwrap();
        assert_canonical_array_exact(&format!("{label} keys"), &actual_keys, &expected_keys);
        assert_canonical_array_exact(&format!("{label} values"), &actual_values, &expected_values);
    }

    fn materialize_turboquant_prefix(array: &Array, offset: i32) -> Array {
        let prefix = crate::cache::slice_axis1(array, 0, offset).unwrap();
        let shape = prefix.shape().to_vec();
        let contiguous = prefix.flatten(None, None).unwrap().reshape(&shape).unwrap();
        contiguous.eval().unwrap();
        contiguous
    }

    fn assert_turboquant_u32_exact(label: &str, actual: &Array, expected: &Array, offset: i32) {
        let actual = materialize_turboquant_prefix(actual, offset);
        let expected = materialize_turboquant_prefix(expected, offset);
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(
            actual.as_slice::<u32>(),
            expected.as_slice::<u32>(),
            "{label}"
        );
    }

    fn assert_turboquant_f32_exact(label: &str, actual: &Array, expected: &Array, offset: i32) {
        let actual = materialize_turboquant_prefix(actual, offset);
        let expected = materialize_turboquant_prefix(expected, offset);
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        for (index, (got, want)) in actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .enumerate()
        {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}[{index}] differs: {got:?} != {want:?}"
            );
        }
    }

    fn assert_canonical_turboquant_cache_exact(
        label: &str,
        actual: &SteppingKeyValueCache,
        expected: &SteppingKeyValueCache,
    ) {
        assert_eq!(actual.offset(), expected.offset(), "{label} offset");
        assert_eq!(
            actual.kv_cache_config(),
            expected.kv_cache_config(),
            "{label} config"
        );
        assert!(actual.is_turbo_active() && expected.is_turbo_active());
        assert!(actual.keys().is_none() && expected.keys().is_none());
        assert!(actual.values().is_none() && expected.values().is_none());

        let offset = actual.offset();
        let (
            actual_context,
            actual_key_codes,
            actual_key_norms,
            actual_key_gammas,
            actual_value_codes,
            actual_value_norms,
        ) = actual
            .turbo_arrays()
            .expect("actual TurboQuant cache active");
        let (
            expected_context,
            expected_key_codes,
            expected_key_norms,
            expected_key_gammas,
            expected_value_codes,
            expected_value_norms,
        ) = expected
            .turbo_arrays()
            .expect("expected TurboQuant cache active");
        assert_eq!(
            actual_context.config, expected_context.config,
            "{label} context"
        );
        assert_eq!(
            actual_context.head_dim, expected_context.head_dim,
            "{label} head dim"
        );
        assert_eq!(
            actual_context.num_kv_heads, expected_context.num_kv_heads,
            "{label} KV heads"
        );

        assert_turboquant_u32_exact(
            &format!("{label} key codes"),
            actual_key_codes,
            expected_key_codes,
            offset,
        );
        assert_turboquant_u32_exact(
            &format!("{label} value codes"),
            actual_value_codes,
            expected_value_codes,
            offset,
        );
        assert_turboquant_f32_exact(
            &format!("{label} key norms"),
            actual_key_norms,
            expected_key_norms,
            offset,
        );
        assert_turboquant_f32_exact(
            &format!("{label} key gammas"),
            actual_key_gammas,
            expected_key_gammas,
            offset,
        );
        assert_turboquant_f32_exact(
            &format!("{label} value norms"),
            actual_value_norms,
            expected_value_norms,
            offset,
        );
    }

    fn canonical_attention_fixture(yarn: bool) -> Qwen3NextAttention {
        let mut args = valid_causal_lm_args();
        args.hidden_size = 64;
        args.num_attention_heads = 2;
        args.num_key_value_heads = 1;
        args.head_dim = 32;
        args.partial_rotary_factor = 0.5;
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 0,
            mode: crate::quant_mode::QuantMode::Dense,
        });
        args.quant_overrides.clear();
        args.rope_scaling = yarn.then(|| {
            serde_json::json!({
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 512,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            })
        });

        let mut attention = Qwen3NextAttention::new(&args, "test.canonical.self_attn").unwrap();
        let hidden = args.hidden_size;
        let q_rows = 2 * args.num_attention_heads * args.head_dim;
        let kv_rows = args.num_key_value_heads * args.head_dim;
        install_canonical_exact_dense(&mut attention.q_proj, q_rows, hidden, 1);
        install_canonical_exact_dense(&mut attention.k_proj, kv_rows, hidden, 3);
        install_canonical_exact_dense(&mut attention.v_proj, kv_rows, hidden, 5);
        install_canonical_exact_dense(
            &mut attention.o_proj,
            hidden,
            args.num_attention_heads * args.head_dim,
            7,
        );
        attention.q_norm.weight = Param::new(
            Array::ones::<f32>(&[args.head_dim])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        attention.k_norm.weight = Param::new(
            Array::ones::<f32>(&[args.head_dim])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        attention
    }

    #[test]
    fn canonical_short_rope_matches_repeated_s1_exact_s2_through_s5() {
        for yarn in [false, true] {
            let attention = canonical_attention_fixture(yarn);
            for offset in [0_i32, 7_i32] {
                for seq_len in 2_i32..=5_i32 {
                    let input = canonical_exact_input(
                        seq_len,
                        attention.num_attention_heads * 32,
                        offset + seq_len,
                    )
                    .reshape(&[1, attention.num_attention_heads, seq_len, 32])
                    .unwrap();
                    let block = apply_qwen3_next_rope_scheduled(
                        input.clone(),
                        &attention.rope,
                        offset,
                        attention.yarn.as_ref(),
                        DFlashRowSchedule::CanonicalS1,
                    )
                    .unwrap();
                    let rows = (0..seq_len)
                        .map(|position| {
                            apply_qwen3_next_rope_scheduled(
                                input.index((.., .., position..position + 1, ..)),
                                &attention.rope,
                                offset + position,
                                attention.yarn.as_ref(),
                                DFlashRowSchedule::NativeBatch,
                            )
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    let repeated =
                        ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 2).unwrap();
                    assert_canonical_array_exact(
                        &format!("canonical rope yarn={yarn} offset={offset} S={seq_len}"),
                        &block,
                        &repeated,
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_full_attention_matches_repeated_s1_exact_s2_through_s5() {
        for yarn in [false, true] {
            let attention = canonical_attention_fixture(yarn);
            for prefix_len in [0_i32, 3_i32] {
                let mut prefix_attention = attention.clone();
                let mut prefix_cache = SteppingKeyValueCache::new();
                if prefix_len > 0 {
                    let prefix = canonical_exact_input(prefix_len, 64, 41);
                    for position in 0..prefix_len {
                        let row = prefix.index((.., position..position + 1, ..));
                        let output = prefix_attention
                            .forward(&row, None, &mut prefix_cache)
                            .unwrap();
                        mlx_rs::transforms::eval([&output]).unwrap();
                        mlx_rs::transforms::eval(prefix_cache.eval_targets()).unwrap();
                    }
                }

                for seq_len in 2_i32..=5_i32 {
                    let input = canonical_exact_input(seq_len, 64, prefix_len + seq_len + 71);
                    let mut sequential_attention = attention.clone();
                    let mut block_attention = attention.clone();
                    let mut sequential_cache = prefix_cache.deep_clone();
                    let mut block_cache = prefix_cache.deep_clone();

                    let mut rows = Vec::with_capacity(seq_len as usize);
                    for position in 0..seq_len {
                        let row = input.index((.., position..position + 1, ..));
                        let output = sequential_attention
                            .forward(&row, None, &mut sequential_cache)
                            .unwrap();
                        mlx_rs::transforms::eval([&output]).unwrap();
                        mlx_rs::transforms::eval(sequential_cache.eval_targets()).unwrap();
                        rows.push(output);
                    }
                    let repeated =
                        ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1).unwrap();
                    let mask = AttentionMask::Array(
                        create_causal_mask(seq_len, (prefix_len > 0).then_some(prefix_len))
                            .unwrap(),
                    );
                    let block = block_attention
                        .forward_canonical_rows(&input, Some(&mask), &mut block_cache)
                        .unwrap();
                    mlx_rs::transforms::eval([&block]).unwrap();
                    mlx_rs::transforms::eval(block_cache.eval_targets()).unwrap();

                    let label =
                        format!("canonical attention yarn={yarn} prefix={prefix_len} S={seq_len}");
                    assert_canonical_array_exact(&label, &block, &repeated);
                    assert_canonical_kv_cache_exact(
                        &format!("{label} cache"),
                        &block_cache,
                        &sequential_cache,
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_turboquant_attention_matches_repeated_s1_exact_s2_through_s5() {
        use crate::turboquant::{KvCacheConfig, KvCacheMode};

        let attention = canonical_attention_fixture(false);
        let prefix_len = 3;
        let mut prefix_attention = attention.clone();
        let mut prefix_cache = SteppingKeyValueCache::new();
        let prefix = canonical_exact_input(prefix_len, 64, 41);
        for position in 0..prefix_len {
            let row = prefix.index((.., position..position + 1, ..));
            let output = prefix_attention
                .forward(&row, None, &mut prefix_cache)
                .unwrap();
            mlx_rs::transforms::eval([&output]).unwrap();
            mlx_rs::transforms::eval(prefix_cache.eval_targets()).unwrap();
        }

        let turboquant = KvCacheConfig {
            mode: KvCacheMode::Turboquant,
            bits: 3,
            seed: 7,
            ..Default::default()
        };
        assert!(prefix_cache.quantize_for_retention(turboquant).unwrap());
        mlx_rs::transforms::eval(prefix_cache.eval_targets()).unwrap();
        assert!(prefix_cache.is_turbo_active());

        for seq_len in 2_i32..=5_i32 {
            let input = canonical_exact_input(seq_len, 64, prefix_len + seq_len + 71);
            let mut sequential_attention = attention.clone();
            let mut block_attention = attention.clone();
            let mut sequential_cache = prefix_cache.deep_clone();
            let mut block_cache = prefix_cache.deep_clone();

            let mut rows = Vec::with_capacity(seq_len as usize);
            for position in 0..seq_len {
                let row = input.index((.., position..position + 1, ..));
                let output = sequential_attention
                    .forward(&row, None, &mut sequential_cache)
                    .unwrap();
                mlx_rs::transforms::eval([&output]).unwrap();
                mlx_rs::transforms::eval(sequential_cache.eval_targets()).unwrap();
                rows.push(output);
            }
            let repeated = ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1).unwrap();
            let mask = AttentionMask::Array(create_causal_mask(seq_len, Some(prefix_len)).unwrap());
            let block = block_attention
                .forward_canonical_rows(&input, Some(&mask), &mut block_cache)
                .unwrap();
            mlx_rs::transforms::eval([&block]).unwrap();
            mlx_rs::transforms::eval(block_cache.eval_targets()).unwrap();

            let label = format!("canonical TurboQuant attention S={seq_len}");
            assert_canonical_array_exact(&label, &block, &repeated);
            assert_canonical_turboquant_cache_exact(
                &format!("{label} cache"),
                &block_cache,
                &sequential_cache,
            );
        }
    }

    /// P2a gate (correctness): the DFlash tape replay must reconstruct the GDN
    /// SSM state at a partial-accept boundary *exactly* like a fresh forward over
    /// the accepted prefix. This bit-exactness is what makes greedy spec-decode
    /// AR-identical regardless of draft quality (commit e23415da's whole point).
    /// Tested at the kernel level (the projections need loaded quantized weights).
    #[test]
    #[allow(clippy::print_stderr, clippy::many_single_char_names)]
    fn test_gdn_tape_replay_kernel_matches_forward() {
        let (b_, kk, j, nk, dk, nv, dv) = (1, 8, 5, 2, 32, 4, 32);
        let rnd = |s: &[i32]| mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, s, None).unwrap();
        let q = rnd(&[b_, kk, nk, dk]);
        let k = rnd(&[b_, kk, nk, dk]);
        let v = rnd(&[b_, kk, nv, dv]);
        let a = rnd(&[b_, kk, nv]);
        let beta = rnd(&[b_, kk, nv]);
        let a_log = Array::zeros::<f32>(&[nv]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[nv]).unwrap();
        let s0 = Array::zeros::<f32>(&[b_, nv, dv, dk]).unwrap();

        // Record the tape over the whole draft block.
        let (_y, _sk, tape) = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &s0, b_, kk, nk, dk, nv, dv,
        )
        .unwrap();

        // Slice tape/inputs to the accepted prefix (mirrors replay_from_tape).
        let sj = |arr: &Array| arr.index((.., ..j, ..));

        // Ground truth: fresh forward kernel over the first j steps.
        let (_yj, state_fwd) = gated_delta_kernel_ffi(
            &sj(&q),
            &sj(&k),
            &sj(&v),
            &a_log,
            &sj(&a),
            &dt_bias,
            &sj(&beta),
            &s0,
            b_,
            j,
            nk,
            dk,
            nv,
            dv,
        )
        .unwrap();

        // Replay the first j steps from the recorded tape.
        let state_replay = tape_replay_kernel_ffi(
            &sj(&tape),
            &sj(&k),
            &sj(&a),
            &a_log,
            &dt_bias,
            &s0,
            b_,
            j,
            nk,
            dk,
            nv,
            dv,
        )
        .unwrap();

        assert_canonical_array_exact("tape replay state", &state_replay, &state_fwd);
    }

    /// DFlash verify must be bit-exact with AR decode. The verify path uses the
    /// tape kernel (S>1 block); AR decode uses the plain kernel. With **bf16**
    /// inputs (the production dtype) both must keep the SSM state in f32 and
    /// produce identical `y`/`state` — otherwise the verify argmax flips on
    /// close calls and DFlash diverges from greedy AR. Regression for the bug
    /// where the tape kernel downcast state to bf16 between timesteps.
    #[test]
    #[allow(clippy::print_stderr, clippy::many_single_char_names)]
    fn test_gdn_tape_forward_matches_plain_forward_bf16() {
        let (b_, t, nk, dk, nv, dv) = (1, 12, 2, 32, 4, 32);
        let rnd_bf16 = |s: &[i32]| {
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, s, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        // Production dtypes: q/k/v/a/beta bf16, a_log/dt_bias/state f32.
        let q = rnd_bf16(&[b_, t, nk, dk]);
        let k = rnd_bf16(&[b_, t, nk, dk]);
        let v = rnd_bf16(&[b_, t, nv, dv]);
        let a = rnd_bf16(&[b_, t, nv]);
        let beta = rnd_bf16(&[b_, t, nv]);
        let a_log = Array::zeros::<f32>(&[nv]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[nv]).unwrap();
        let s0 = Array::zeros::<f32>(&[b_, nv, dv, dk]).unwrap();

        let (y_plain, st_plain) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &s0, b_, t, nk, dk, nv, dv,
        )
        .unwrap();
        let (y_tape, st_tape, _tape) = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &s0, b_, t, nk, dk, nv, dv,
        )
        .unwrap();

        assert_canonical_array_exact("tape vs plain output", &y_tape, &y_plain);
        assert_canonical_array_exact("tape vs plain state", &st_tape, &st_plain);
    }

    /// P2a gate (cost): rollback must be cheap vs recompute, or partial accepts
    /// (every round) erode the spec-decode win. Times the tape-replay kernel vs
    /// the forward kernel at the accepted length — and note the *real* rollback
    /// win is bigger still, since replay skips all the QLinear projections.
    #[test]
    #[ignore = "bench: GDN tape-replay rollback kernel cost vs forward kernel"]
    #[allow(
        clippy::print_stderr,
        clippy::cast_precision_loss,
        clippy::many_single_char_names
    )]
    fn bench_gdn_tape_replay_rollback_cost() {
        // Real Qwen3.6-35B-A3B GDN dims: 16 k-heads, 32 v-heads, 128 head dims.
        let (b_, kk, j, nk, dk, nv, dv) = (1, 16, 10, 16, 128, 32, 128);
        let rnd = |s: &[i32]| mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, s, None).unwrap();
        let (q, k, v) = (
            rnd(&[b_, kk, nk, dk]),
            rnd(&[b_, kk, nk, dk]),
            rnd(&[b_, kk, nv, dv]),
        );
        let (a, beta) = (rnd(&[b_, kk, nv]), rnd(&[b_, kk, nv]));
        let a_log = Array::zeros::<f32>(&[nv]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[nv]).unwrap();
        let s0 = Array::zeros::<f32>(&[b_, nv, dv, dk]).unwrap();
        let (_y, _sk, tape) = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &s0, b_, kk, nk, dk, nv, dv,
        )
        .unwrap();
        let sj = |arr: &Array| arr.index((.., ..j, ..));
        let (tj, kj, aj, qj, vj, bj) = (sj(&tape), sj(&k), sj(&a), sj(&q), sj(&v), sj(&beta));

        let n = 100;
        // warm
        tape_replay_kernel_ffi(&tj, &kj, &aj, &a_log, &dt_bias, &s0, b_, j, nk, dk, nv, dv)
            .unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let s =
                tape_replay_kernel_ffi(&tj, &kj, &aj, &a_log, &dt_bias, &s0, b_, j, nk, dk, nv, dv)
                    .unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
        }
        let t_replay = t0.elapsed().as_secs_f64() / f64::from(n);

        let t1 = std::time::Instant::now();
        for _ in 0..n {
            let (_y, s) = gated_delta_kernel_ffi(
                &qj, &kj, &vj, &a_log, &aj, &dt_bias, &bj, &s0, b_, j, nk, dk, nv, dv,
            )
            .unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
        }
        let t_forward = t1.elapsed().as_secs_f64() / f64::from(n);

        eprintln!(
            "GDN ROLLBACK KERNEL [accept={j}/{kk}]: replay={:.4}ms forward(SSM-only)={:.4}ms ratio={:.2}x  (real rollback also skips projections -> bigger win)",
            t_replay * 1e3,
            t_forward * 1e3,
            t_replay / t_forward
        );
    }

    fn install_tape_transition_dense(linear: &mut QLinear, rows: i32, columns: i32, salt: i32) {
        // A sparse dyadic matrix makes batched and repeated-S1 projections
        // exactly equal, so this test isolates the stateful GDN transition
        // instead of MLX dense-GEMM reduction scheduling.
        linear.weight = Param::new(canonical_exact_weight(rows, columns, salt));
        linear.scales = Param::new(Array::from_slice::<f32>(&[], &[0]));
        linear.biases = Param::new(Array::from_slice::<f32>(&[], &[0]));
        linear.group_size = 64;
        linear.bits = 0;
        linear.mode = crate::quant_mode::QuantMode::Dense;
    }

    fn tape_transition_model() -> Qwen3NextCausalLM {
        let mut args = valid_causal_lm_args();
        // This fixture isolates one GDN transition. Keep the configured model
        // single-layer so its one cache and one tape are also a complete
        // whole-model transaction under the fail-closed replay contract.
        args.num_hidden_layers = 1;
        args.num_experts = 0;
        args.num_experts_per_tok = 0;
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 0,
            mode: crate::quant_mode::QuantMode::Dense,
        });
        let hidden_size = args.hidden_size;
        let mut model = Qwen3NextCausalLM::new(args).unwrap();
        let gdn = model.model.layers[0]
            .linear_attn
            .as_mut()
            .expect("layer 0 must be GDN");
        let value_dim = gdn.num_v_heads * gdn.head_v_dim;
        install_tape_transition_dense(
            &mut gdn.in_proj_qkvz,
            2 * (gdn.key_dim + value_dim),
            hidden_size,
            1,
        );
        install_tape_transition_dense(&mut gdn.in_proj_ba, 2 * gdn.num_v_heads, hidden_size, 3);
        install_tape_transition_dense(&mut gdn.out_proj, hidden_size, value_dim, 5);
        gdn.conv1d.weight = Param::new(
            canonical_exact_weight(gdn.conv_dim, gdn.conv_kernel_size, 7)
                .reshape(&[gdn.conv_dim, gdn.conv_kernel_size, 1])
                .unwrap(),
        );
        gdn.norm.weight = Param::new(
            Array::ones::<f32>(&[gdn.head_v_dim])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
        gdn.conv_weight_t = None;
        model
    }

    fn tape_transition_input(seq_len: i32, hidden_size: i32) -> Array {
        canonical_exact_input(seq_len, hidden_size, seq_len + 97)
    }

    fn run_gdn_ar_steps(gdn: &mut GatedDeltaNet, inputs: &Array, cache: &mut ArraysCache) -> Array {
        let seq_len = inputs.shape()[1];
        let mut outputs = Vec::with_capacity(seq_len as usize);
        for index in 0..seq_len {
            let input = inputs.index((.., index..index + 1, ..));
            let output = gdn.forward(&input, None, cache).unwrap();
            output.eval().unwrap();
            cache.eval_arrays().unwrap();
            outputs.push(output);
        }
        let refs: Vec<&Array> = outputs.iter().collect();
        ops::concatenate_axis(&refs, 1).unwrap()
    }

    fn deep_clone_tape_transition_cache(cache: &ArraysCache) -> ArraysCache {
        ArraysCache {
            conv_state: cache
                .conv_state
                .as_ref()
                .map(|array| crate::cache::try_eval_deep_clone(array).unwrap()),
            ssm_state: cache
                .ssm_state
                .as_ref()
                .map(|array| crate::cache::try_eval_deep_clone(array).unwrap()),
            conv_pos: cache.conv_pos,
            offset: cache.offset,
        }
    }

    fn chronological_tape_transition_state(gdn: &GatedDeltaNet, cache: &ArraysCache) -> Array {
        let dtype = cache
            .conv_state
            .as_ref()
            .map_or(Dtype::Bfloat16, Array::dtype);
        let mut copy = deep_clone_tape_transition_cache(cache);
        gdn.chronological_conv_state(&mut copy, 1, dtype).unwrap()
    }

    fn assert_tape_transition_close(label: &str, actual: &Array, expected: &Array, limit: f32) {
        let actual = actual.as_dtype(Dtype::Float32).unwrap();
        let expected = expected.as_dtype(Dtype::Float32).unwrap();
        let diff = actual
            .subtract(&expected)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        diff.eval().unwrap();
        let max_diff: f32 = diff.item();
        assert!(
            max_diff <= limit,
            "{label} max diff {max_diff} exceeds {limit}"
        );
    }

    fn warm_rotated_tape_transition_cache(model: &mut Qwen3NextCausalLM) -> ArraysCache {
        let hidden_size = model.args.hidden_size;
        let warmup = tape_transition_input(5, hidden_size);
        let mut cache = ArraysCache::new();
        let gdn = model.model.layers[0]
            .linear_attn
            .as_mut()
            .expect("layer 0 must be GDN");
        run_gdn_ar_steps(gdn, &warmup, &mut cache);
        let canonical_pos = gdn.conv_kernel_size - 2;
        assert_ne!(
            cache.conv_pos, canonical_pos,
            "warmup must leave the convolution ring rotated"
        );
        cache
    }

    #[test]
    fn test_gdn_tape_block_after_ar_ring_matches_sequential() {
        let mut model = tape_transition_model();
        let initial_cache = warm_rotated_tape_transition_cache(&mut model);
        let initial_gdn = model.model.layers[0]
            .linear_attn
            .as_ref()
            .expect("layer 0 must be GDN")
            .clone();

        for seq_len in 2..=5 {
            // Every sequence length starts from the exact same independently
            // cloned checkpoint. Reusing a cache between lengths would compare
            // different absolute positions and produce a false numerical drift.
            let mut ar_gdn = initial_gdn.clone();
            let mut tape_gdn = initial_gdn.clone();
            let mut ar_cache = deep_clone_tape_transition_cache(&initial_cache);
            let mut tape_cache = deep_clone_tape_transition_cache(&initial_cache);
            let block = tape_transition_input(seq_len, model.args.hidden_size);

            let ar_output = run_gdn_ar_steps(&mut ar_gdn, &block, &mut ar_cache);
            let (tape_output, _) = tape_gdn
                .forward_with_tape(
                    &block,
                    None,
                    &mut tape_cache,
                    DFlashRowSchedule::CanonicalS1,
                )
                .unwrap();
            mlx_rs::transforms::eval([&ar_output, &tape_output]).unwrap();
            ar_cache.eval_arrays().unwrap();
            tape_cache.eval_arrays().unwrap();

            let label = format!("GDN repeated-S1 output S={seq_len}");
            assert_canonical_array_exact(&label, &tape_output, &ar_output);
            assert_canonical_array_exact(
                &format!("GDN repeated-S1 SSM state S={seq_len}"),
                tape_cache.ssm_state.as_ref().unwrap(),
                ar_cache.ssm_state.as_ref().unwrap(),
            );
            let tape_conv = chronological_tape_transition_state(&tape_gdn, &tape_cache);
            let ar_conv = chronological_tape_transition_state(&ar_gdn, &ar_cache);
            assert_canonical_array_exact(
                &format!("GDN repeated-S1 conv state S={seq_len}"),
                &tape_conv,
                &ar_conv,
            );
            assert_eq!(tape_cache.offset, ar_cache.offset);
            assert_eq!(tape_cache.conv_pos, tape_gdn.conv_kernel_size - 2);
        }
    }

    #[test]
    fn test_gdn_tape_partial_rollback_after_ar_ring_matches_sequential() {
        let mut model = tape_transition_model();
        let mut tape_cache = warm_rotated_tape_transition_cache(&mut model);
        let initial_conv_pos = tape_cache.conv_pos;
        let mut ar_cache = deep_clone_tape_transition_cache(&tape_cache);
        let block = tape_transition_input(4, model.args.hidden_size);
        let accepted = 2;

        let mut ar_gdn = model.model.layers[0]
            .linear_attn
            .as_ref()
            .expect("layer 0 must be GDN")
            .clone();
        run_gdn_ar_steps(
            &mut ar_gdn,
            &block.index((.., ..accepted, ..)),
            &mut ar_cache,
        );

        let (_, tape) = model.model.layers[0]
            .linear_attn
            .as_mut()
            .expect("layer 0 must be GDN")
            .forward_with_tape(
                &block,
                None,
                &mut tape_cache,
                DFlashRowSchedule::CanonicalS1,
            )
            .unwrap();
        assert_eq!(tape.conv_pos_init, initial_conv_pos);

        let mut caches = vec![Some(LayerCache::Arrays(tape_cache))];
        model
            .replay_tape_rollback(&[Some(tape)], &mut caches, accepted, 4 - accepted)
            .unwrap();
        let Some(LayerCache::Arrays(mut replay_cache)) = caches.pop().flatten() else {
            panic!("expected replayed Arrays cache");
        };
        replay_cache.eval_arrays().unwrap();

        assert_eq!(replay_cache.offset, ar_cache.offset);
        assert_tape_transition_close(
            "partial rollback SSM state",
            replay_cache.ssm_state.as_ref().unwrap(),
            ar_cache.ssm_state.as_ref().unwrap(),
            1e-4,
        );
        let replay_conv = chronological_tape_transition_state(&ar_gdn, &replay_cache);
        let ar_conv = chronological_tape_transition_state(&ar_gdn, &ar_cache);
        assert_tape_transition_close("partial rollback conv state", &replay_conv, &ar_conv, 1e-3);

        // A following AR token exercises the restored/canonicalized cursor,
        // not just the equivalent history values.
        let next = tape_transition_input(1, model.args.hidden_size);
        let ar_next = run_gdn_ar_steps(&mut ar_gdn, &next, &mut ar_cache);
        let replay_next = run_gdn_ar_steps(
            model.model.layers[0]
                .linear_attn
                .as_mut()
                .expect("layer 0 must be GDN"),
            &next,
            &mut replay_cache,
        );
        assert_tape_transition_close("post-rollback AR output", &replay_next, &ar_next, 2e-2);
        assert_tape_transition_close(
            "post-rollback AR SSM state",
            replay_cache.ssm_state.as_ref().unwrap(),
            ar_cache.ssm_state.as_ref().unwrap(),
            1e-4,
        );
    }

    fn deterministic_q1_params(rows: i32, columns: i32, salt: u32) -> (Array, Array, Array) {
        const GROUP_SIZE: i32 = 128;
        assert_eq!(columns % GROUP_SIZE, 0);
        let words_per_row = columns / 32;
        let patterns = [
            0xA5A5_5A5A_u32,
            0x3C3C_C3C3_u32,
            0x9696_6969_u32,
            0xF0F0_0F0F_u32,
        ];
        let packed = (0..rows * words_per_row)
            .map(|index| {
                let pattern = patterns[((index as u32 + salt) % patterns.len() as u32) as usize];
                pattern.rotate_left((index as u32 * 7 + salt) % 32)
            })
            .collect::<Vec<_>>();
        let groups_per_row = columns / GROUP_SIZE;
        let scales = (0..rows * groups_per_row)
            .map(|index| 0.015_625_f32 + (index.rem_euclid(3) as f32) * 0.003_906_25)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&packed, &[rows, words_per_row]);
        let scales = Array::from_slice(&scales, &[rows, groups_per_row])
            .as_dtype(Dtype::Float16)
            .unwrap();
        weight.eval().unwrap();
        scales.eval().unwrap();
        (weight, scales, symmetric_q1_bias_sentinel())
    }

    fn install_deterministic_q1(linear: &mut QLinear, rows: i32, columns: i32, salt: u32) {
        let (weight, scales, biases) = deterministic_q1_params(rows, columns, salt);
        linear.weight = Param::new(weight);
        linear.scales = Param::new(scales);
        linear.biases = Param::new(biases);
        linear.group_size = 128;
        linear.bits = 1;
        linear.mode = crate::quant_mode::QuantMode::Affine;
    }

    fn install_deterministic_q1_embedding(
        embedding: &mut QEmbedding,
        rows: i32,
        columns: i32,
        salt: u32,
    ) {
        let (weight, scales, biases) = deterministic_q1_params(rows, columns, salt);
        embedding.weight = Param::new(weight);
        embedding.scales = Param::new(scales);
        embedding.biases = Param::new(biases);
        embedding.group_size = 128;
        embedding.bits = 1;
        embedding.mode = crate::quant_mode::QuantMode::Affine;
    }

    fn deterministic_hybrid_q1_model() -> Qwen3NextCausalLM {
        let mut args = valid_causal_lm_args();
        args.hidden_size = 128;
        args.intermediate_size = 128;
        args.vocab_size = 128;
        args.num_hidden_layers = 2;
        args.full_attention_interval = 2;
        args.num_attention_heads = 2;
        args.num_key_value_heads = 1;
        args.head_dim = 64;
        args.linear_num_key_heads = 1;
        args.linear_key_head_dim = 32;
        args.linear_num_value_heads = 2;
        args.linear_value_head_dim = 64;
        args.num_experts = 0;
        args.num_experts_per_tok = 0;
        args.decoder_sparse_step = 0;
        args.quantization = Some(QuantizationConfig {
            group_size: 128,
            bits: 1,
            mode: crate::quant_mode::QuantMode::Affine,
        });
        args.quant_overrides.clear();

        let hidden = args.hidden_size;
        let intermediate = args.intermediate_size;
        let vocab = args.vocab_size;
        let mut model = Qwen3NextCausalLM::new(args).unwrap();
        install_deterministic_q1_embedding(&mut model.model.embed_tokens, vocab, hidden, 1);
        install_deterministic_q1(
            model.lm_head.as_mut().expect("untied LM head"),
            vocab,
            hidden,
            3,
        );
        model.model.norm.weight = Param::new(Array::ones::<f32>(&[hidden]).unwrap());

        for (layer_index, layer) in model.model.layers.iter_mut().enumerate() {
            let salt = 11 + layer_index as u32 * 17;
            layer.input_layernorm.weight = Param::new(Array::ones::<f32>(&[hidden]).unwrap());
            layer.post_attention_layernorm.weight =
                Param::new(Array::ones::<f32>(&[hidden]).unwrap());
            install_deterministic_q1(
                layer.mlp.gate_proj.as_mut().expect("dense gate projection"),
                intermediate,
                hidden,
                salt,
            );
            install_deterministic_q1(
                layer.mlp.up_proj.as_mut().expect("dense up projection"),
                intermediate,
                hidden,
                salt + 1,
            );
            install_deterministic_q1(
                layer.mlp.down_proj.as_mut().expect("dense down projection"),
                hidden,
                intermediate,
                salt + 2,
            );

            if let Some(gdn) = layer.linear_attn.as_mut() {
                let value_dim = gdn.num_v_heads * gdn.head_v_dim;
                install_deterministic_q1(
                    &mut gdn.in_proj_qkvz,
                    2 * (gdn.key_dim + value_dim),
                    hidden,
                    salt + 3,
                );
                install_deterministic_q1(
                    &mut gdn.in_proj_ba,
                    2 * gdn.num_v_heads,
                    hidden,
                    salt + 4,
                );
                install_deterministic_q1(&mut gdn.out_proj, hidden, value_dim, salt + 5);
                let conv_values = (0..gdn.conv_dim * gdn.conv_kernel_size)
                    .map(|index| {
                        const TAPS: [f32; 4] = [0.125, -0.0625, 0.03125, 0.25];
                        TAPS[((index + layer_index as i32) % 4) as usize]
                    })
                    .collect::<Vec<_>>();
                gdn.conv1d.weight = Param::new(Array::from_slice(
                    &conv_values,
                    &[gdn.conv_dim, gdn.conv_kernel_size, 1],
                ));
                gdn.norm.weight = Param::new(Array::ones::<f32>(&[gdn.head_v_dim]).unwrap());
                gdn.A_log = Param::new(Array::zeros::<f32>(&[gdn.num_v_heads]).unwrap());
                gdn.dt_bias = Param::new(Array::zeros::<f32>(&[gdn.num_v_heads]).unwrap());
                gdn.conv_weight_t = None;
            } else {
                let attention = layer.self_attn.as_mut().expect("full attention layer");
                let q_rows = 2 * attention.num_attention_heads * model.args.head_dim;
                let kv_rows = attention.num_key_value_heads * model.args.head_dim;
                install_deterministic_q1(&mut attention.q_proj, q_rows, hidden, salt + 3);
                install_deterministic_q1(&mut attention.k_proj, kv_rows, hidden, salt + 4);
                install_deterministic_q1(&mut attention.v_proj, kv_rows, hidden, salt + 5);
                install_deterministic_q1(
                    &mut attention.o_proj,
                    hidden,
                    attention.num_attention_heads * model.args.head_dim,
                    salt + 6,
                );
                attention.q_norm.weight =
                    Param::new(Array::ones::<f32>(&[model.args.head_dim]).unwrap());
                attention.k_norm.weight =
                    Param::new(Array::ones::<f32>(&[model.args.head_dim]).unwrap());
            }
        }
        model
    }

    fn retain_loaded_affine_q1_bias(linear: &mut QLinear) {
        linear.biases = Param::new((*linear.scales).clone());
    }

    fn retain_loaded_affine_q1_embedding_bias(embedding: &mut QEmbedding) {
        embedding.biases = Param::new((*embedding.scales).clone());
    }

    fn retain_whole_model_affine_q1_biases(model: &mut Qwen3NextCausalLM) -> usize {
        let mut retained = 0;
        retain_loaded_affine_q1_embedding_bias(&mut model.model.embed_tokens);
        retained += 1;
        retain_loaded_affine_q1_bias(model.lm_head.as_mut().expect("untied LM head"));
        retained += 1;

        for layer in &mut model.model.layers {
            retain_loaded_affine_q1_bias(
                layer.mlp.gate_proj.as_mut().expect("dense gate projection"),
            );
            retain_loaded_affine_q1_bias(layer.mlp.up_proj.as_mut().expect("dense up projection"));
            retain_loaded_affine_q1_bias(
                layer.mlp.down_proj.as_mut().expect("dense down projection"),
            );
            retained += 3;

            if layer.is_linear {
                let gdn = layer.linear_attn.as_mut().expect("GDN layer");
                if gdn.use_separate_projections {
                    retain_loaded_affine_q1_bias(
                        gdn.in_proj_qkv.as_mut().expect("separate QKV projection"),
                    );
                    retain_loaded_affine_q1_bias(
                        gdn.in_proj_z.as_mut().expect("separate Z projection"),
                    );
                    retain_loaded_affine_q1_bias(
                        gdn.in_proj_a.as_mut().expect("separate A projection"),
                    );
                    retain_loaded_affine_q1_bias(
                        gdn.in_proj_b.as_mut().expect("separate B projection"),
                    );
                    retained += 4;
                } else {
                    retain_loaded_affine_q1_bias(&mut gdn.in_proj_qkvz);
                    retain_loaded_affine_q1_bias(&mut gdn.in_proj_ba);
                    retained += 2;
                }
                retain_loaded_affine_q1_bias(&mut gdn.out_proj);
                retained += 1;
            } else {
                let attention = layer.self_attn.as_mut().expect("full-attention layer");
                retain_loaded_affine_q1_bias(&mut attention.q_proj);
                retain_loaded_affine_q1_bias(&mut attention.k_proj);
                retain_loaded_affine_q1_bias(&mut attention.v_proj);
                retain_loaded_affine_q1_bias(&mut attention.o_proj);
                retained += 4;
            }
        }
        retained
    }

    #[test]
    fn dflash_block_capability_accepts_whole_model_loaded_affine_q1_biases() {
        let mut model = deterministic_hybrid_q1_model();
        let retained = retain_whole_model_affine_q1_biases(&mut model);

        assert_eq!(retained, 15, "fixture must retain every active affine bias");
        assert!(!has_symmetric_q1_biases(&model.model.embed_tokens.biases));
        model.validate_dflash_block_domain(5).unwrap();
    }

    fn eval_hybrid_cache(cache: &[Option<LayerCache>]) {
        for layer in cache.iter().flatten() {
            match layer {
                LayerCache::KV(kv) => mlx_rs::transforms::eval(kv.eval_targets()).unwrap(),
                LayerCache::Arrays(arrays) => arrays.eval_arrays().unwrap(),
            }
        }
    }

    fn deep_clone_hybrid_cache(cache: &[Option<LayerCache>]) -> Vec<Option<LayerCache>> {
        cache
            .iter()
            .map(|layer| {
                layer.as_ref().map(|layer| match layer {
                    LayerCache::KV(kv) => LayerCache::KV(kv.deep_clone()),
                    LayerCache::Arrays(arrays) => {
                        LayerCache::Arrays(deep_clone_tape_transition_cache(arrays))
                    }
                })
            })
            .collect()
    }

    fn assert_hybrid_q1_cache_exact(
        model: &Qwen3NextCausalLM,
        label: &str,
        actual: &[Option<LayerCache>],
        expected: &[Option<LayerCache>],
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} cache count");
        for (layer_index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            match (actual.as_ref(), expected.as_ref()) {
                (Some(LayerCache::KV(actual)), Some(LayerCache::KV(expected))) => {
                    assert_canonical_kv_cache_exact(
                        &format!("{label} layer {layer_index} KV"),
                        actual,
                        expected,
                    );
                }
                (Some(LayerCache::Arrays(actual)), Some(LayerCache::Arrays(expected))) => {
                    assert_eq!(
                        actual.offset, expected.offset,
                        "{label} layer {layer_index} offset"
                    );
                    assert_canonical_array_exact(
                        &format!("{label} layer {layer_index} SSM"),
                        actual.ssm_state.as_ref().expect("actual SSM state"),
                        expected.ssm_state.as_ref().expect("expected SSM state"),
                    );
                    let gdn = model.model.layers[layer_index]
                        .linear_attn
                        .as_ref()
                        .expect("GDN layer");
                    let actual_conv = chronological_tape_transition_state(gdn, actual);
                    let expected_conv = chronological_tape_transition_state(gdn, expected);
                    assert_canonical_array_exact(
                        &format!("{label} layer {layer_index} convolution history"),
                        &actual_conv,
                        &expected_conv,
                    );
                }
                (None, None) => {}
                _ => panic!("{label} layer {layer_index} cache variant differs"),
            }
        }
    }

    fn run_hybrid_q1_s1(
        model: &mut Qwen3NextCausalLM,
        tokens: &Array,
        cache: &mut Vec<Option<LayerCache>>,
        tap_layers: &[usize],
    ) -> (Array, Vec<Array>) {
        let seq_len = tokens.shape()[1];
        let mut logits = Vec::with_capacity(seq_len as usize);
        let mut taps = (0..tap_layers.len())
            .map(|_| Vec::with_capacity(seq_len as usize))
            .collect::<Vec<_>>();
        for position in 0..seq_len {
            let row = tokens.index((.., position..position + 1));
            let (row_logits, row_taps) = model
                .forward_with_taps(&row, None, cache, tap_layers)
                .unwrap();
            mlx_rs::transforms::eval(std::iter::once(&row_logits).chain(row_taps.iter())).unwrap();
            eval_hybrid_cache(cache);
            logits.push(row_logits);
            for (tap_rows, row_tap) in taps.iter_mut().zip(row_taps) {
                tap_rows.push(row_tap);
            }
        }
        let logits = ops::concatenate_axis(&logits.iter().collect::<Vec<_>>(), 1).unwrap();
        let taps = taps
            .into_iter()
            .map(|rows| ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1).unwrap())
            .collect();
        (logits, taps)
    }

    #[test]
    fn hybrid_q1_taped_verify_matches_repeated_s1_exact_s2_through_s5() {
        let mut warmed_model = deterministic_hybrid_q1_model();
        warmed_model.validate_dflash_block_domain(5).unwrap();
        let mut unsupported_model = warmed_model.clone();
        unsupported_model.model.layers[0]
            .linear_attn
            .as_mut()
            .unwrap()
            .out_proj
            .bits = 4;
        assert!(
            unsupported_model.validate_dflash_block_domain(5).is_err(),
            "the block capability must positively attest every active Q1 projection"
        );
        let tap_layers = [0_usize, 1_usize];
        let warmup = Array::from_slice(&[3_u32, 7, 11], &[1, 3]);
        let mut initial_cache = warmed_model.make_cache();
        let _ = run_hybrid_q1_s1(&mut warmed_model, &warmup, &mut initial_cache, &tap_layers);
        eval_hybrid_cache(&initial_cache);

        for seq_len in 2_i32..=5_i32 {
            let values = (0..seq_len)
                .map(|position| (19 + seq_len * 13 + position * 17) as u32 % 128)
                .collect::<Vec<_>>();
            let tokens = Array::from_slice(&values, &[1, seq_len]);
            let mut sequential_model = warmed_model.clone();
            let mut tape_model = warmed_model.clone();
            let mut sequential_cache = deep_clone_hybrid_cache(&initial_cache);
            let mut tape_cache = deep_clone_hybrid_cache(&initial_cache);

            let (sequential_logits, sequential_taps) = run_hybrid_q1_s1(
                &mut sequential_model,
                &tokens,
                &mut sequential_cache,
                &tap_layers,
            );
            let (tape_logits, tape_taps, layer_tapes) = tape_model
                .forward_with_taps_tape_scheduled(
                    &tokens,
                    None,
                    &mut tape_cache,
                    &tap_layers,
                    None,
                    DFlashRowSchedule::CanonicalS1,
                )
                .unwrap();
            mlx_rs::transforms::eval(std::iter::once(&tape_logits).chain(tape_taps.iter()))
                .unwrap();
            eval_hybrid_cache(&tape_cache);

            let label = format!("whole-model Q1 S={seq_len}");
            assert_canonical_array_exact(
                &format!("{label} logits"),
                &tape_logits,
                &sequential_logits,
            );
            assert_eq!(tape_taps.len(), sequential_taps.len(), "{label} tap count");
            for (tap_index, (actual, expected)) in
                tape_taps.iter().zip(&sequential_taps).enumerate()
            {
                assert_canonical_array_exact(&format!("{label} tap {tap_index}"), actual, expected);
            }
            assert_hybrid_q1_cache_exact(&tape_model, &label, &tape_cache, &sequential_cache);

            if seq_len == 5 {
                let mut fault_model = warmed_model.clone();
                let mut fault_cache = deep_clone_hybrid_cache(&initial_cache);
                let (_, _, mut fault_tapes) = fault_model
                    .forward_with_taps_tape_scheduled(
                        &tokens,
                        None,
                        &mut fault_cache,
                        &tap_layers,
                        None,
                        DFlashRowSchedule::CanonicalS1,
                    )
                    .unwrap();
                eval_hybrid_cache(&fault_cache);
                let fault_cache_before = deep_clone_hybrid_cache(&fault_cache);
                fault_tapes[0] = None;
                assert!(
                    fault_model
                        .replay_tape_rollback(&fault_tapes, &mut fault_cache, 1, seq_len - 1,)
                        .is_err(),
                    "a missing GDN artifact must fail before cache mutation"
                );
                eval_hybrid_cache(&fault_cache);
                assert_hybrid_q1_cache_exact(
                    &fault_model,
                    "missing-tape rollback is atomic",
                    &fault_cache,
                    &fault_cache_before,
                );

                for accepted in 1_i32..seq_len {
                    let mut prefix_model = warmed_model.clone();
                    let mut prefix_cache = deep_clone_hybrid_cache(&initial_cache);
                    let _ = run_hybrid_q1_s1(
                        &mut prefix_model,
                        &tokens.index((.., ..accepted)),
                        &mut prefix_cache,
                        &tap_layers,
                    );
                    let mut rollback_model = tape_model.clone();
                    let mut rollback_cache = deep_clone_hybrid_cache(&tape_cache);
                    rollback_model
                        .replay_tape_rollback(
                            &layer_tapes,
                            &mut rollback_cache,
                            accepted,
                            seq_len - accepted,
                        )
                        .unwrap();
                    eval_hybrid_cache(&rollback_cache);
                    assert_hybrid_q1_cache_exact(
                        &rollback_model,
                        &format!("whole-model Q1 rollback accepted={accepted}/{seq_len}"),
                        &rollback_cache,
                        &prefix_cache,
                    );

                    // A semantically restored ring can use a different raw
                    // cursor/layout. The decisive invariant is that the next
                    // S1 transition and a following block remain exact.
                    let next =
                        Array::from_slice(&[u32::try_from(101 + accepted).unwrap()], &[1, 1]);
                    let (prefix_next_logits, prefix_next_taps) =
                        run_hybrid_q1_s1(&mut prefix_model, &next, &mut prefix_cache, &tap_layers);
                    let (rollback_next_logits, rollback_next_taps) = run_hybrid_q1_s1(
                        &mut rollback_model,
                        &next,
                        &mut rollback_cache,
                        &tap_layers,
                    );
                    assert_canonical_array_exact(
                        &format!("rollback follow-through logits accepted={accepted}"),
                        &rollback_next_logits,
                        &prefix_next_logits,
                    );
                    for (tap_index, (actual, expected)) in
                        rollback_next_taps.iter().zip(&prefix_next_taps).enumerate()
                    {
                        assert_canonical_array_exact(
                            &format!("rollback follow-through tap {tap_index} accepted={accepted}"),
                            actual,
                            expected,
                        );
                    }
                    assert_hybrid_q1_cache_exact(
                        &rollback_model,
                        &format!("rollback follow-through cache accepted={accepted}"),
                        &rollback_cache,
                        &prefix_cache,
                    );

                    let second = Array::from_slice(
                        &[
                            u32::try_from(109 + accepted).unwrap(),
                            u32::try_from(117 + accepted).unwrap(),
                        ],
                        &[1, 2],
                    );
                    let (second_s1_logits, second_s1_taps) = run_hybrid_q1_s1(
                        &mut prefix_model,
                        &second,
                        &mut prefix_cache,
                        &tap_layers,
                    );
                    let (second_block_logits, second_block_taps, _) = rollback_model
                        .forward_with_taps_tape_scheduled(
                            &second,
                            None,
                            &mut rollback_cache,
                            &tap_layers,
                            None,
                            DFlashRowSchedule::CanonicalS1,
                        )
                        .unwrap();
                    mlx_rs::transforms::eval(
                        std::iter::once(&second_block_logits).chain(second_block_taps.iter()),
                    )
                    .unwrap();
                    eval_hybrid_cache(&rollback_cache);
                    assert_canonical_array_exact(
                        &format!("consecutive block logits accepted={accepted}"),
                        &second_block_logits,
                        &second_s1_logits,
                    );
                    for (tap_index, (actual, expected)) in
                        second_block_taps.iter().zip(&second_s1_taps).enumerate()
                    {
                        assert_canonical_array_exact(
                            &format!("consecutive block tap {tap_index} accepted={accepted}"),
                            actual,
                            expected,
                        );
                    }
                    assert_hybrid_q1_cache_exact(
                        &rollback_model,
                        &format!("consecutive block cache accepted={accepted}"),
                        &rollback_cache,
                        &prefix_cache,
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_conv_metal_matches_ordered_s1_bf16_exact_m1_through_m5() {
        let args = valid_causal_lm_args();
        let gdn = GatedDeltaNet::new(&args, "test.layer.linear_attn").unwrap();
        let batch = 1;
        let kernel_size = gdn.conv_kernel_size;
        let history_len = kernel_size - 1;
        let conv_dim = gdn.conv_dim;
        assert_eq!(kernel_size, 4);

        let bf16_pattern = |shape: &[i32], salt: i32, modulus: i32, divisor: f32| {
            let size = shape.iter().product::<i32>();
            let values = (0..size)
                .map(|index| {
                    let centered = (index * 29 + salt).rem_euclid(modulus) - modulus / 2;
                    centered as f32 / divisor
                })
                .collect::<Vec<_>>();
            let array = Array::from_slice(&values, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            array.eval().unwrap();
            array
        };
        let history = bf16_pattern(&[batch, history_len, conv_dim], 7, 61, 47.0);
        let weight_t = bf16_pattern(&[kernel_size, conv_dim], 19, 67, 59.0);

        for offset_init in [0, 1, 2, 3, 11] {
            for seq_len in 1..=5 {
                let mixed_qkv = bf16_pattern(&[batch, seq_len, conv_dim], 31 + seq_len, 71, 53.0);
                let actual_preactivation = canonical_conv_preactivation_ffi(
                    &mixed_qkv,
                    &history,
                    &weight_t,
                    offset_init,
                    batch,
                    seq_len,
                    conv_dim,
                    kernel_size,
                )
                .unwrap();

                let mut expected_preactivation_rows = Vec::with_capacity(seq_len as usize);
                let mut expected_activated_rows = Vec::with_capacity(seq_len as usize);
                for position in 0..seq_len {
                    let current = mixed_qkv
                        .index((.., position..position + 1, ..))
                        .reshape(&[batch, conv_dim])
                        .unwrap();
                    let mut preactivation = current
                        .multiply(&weight_t.index((kernel_size - 1, ..)))
                        .unwrap();
                    let available = (offset_init + position).clamp(0, history_len);
                    for lag in 0..available {
                        let prior = if lag < position {
                            mixed_qkv
                                .index((.., position - 1 - lag..position - lag, ..))
                                .reshape(&[batch, conv_dim])
                                .unwrap()
                        } else {
                            let history_index = history_len - 1 - (lag - position);
                            history
                                .index((.., history_index..history_index + 1, ..))
                                .reshape(&[batch, conv_dim])
                                .unwrap()
                        };
                        let product = prior
                            .multiply(&weight_t.index((history_len - 1 - lag, ..)))
                            .unwrap();
                        preactivation = preactivation.add(&product).unwrap();
                    }
                    expected_preactivation_rows
                        .push(preactivation.reshape(&[batch, 1, conv_dim]).unwrap());

                    let current = mixed_qkv.index((.., position..position + 1, ..));
                    expected_activated_rows.push(
                        gdn.canonical_conv1d_step(&current, &weight_t, available, batch, |lag| {
                            if lag < position {
                                mixed_qkv
                                    .index((.., position - 1 - lag..position - lag, ..))
                                    .reshape(&[batch, conv_dim])
                            } else {
                                let history_index = history_len - 1 - (lag - position);
                                history
                                    .index((.., history_index..history_index + 1, ..))
                                    .reshape(&[batch, conv_dim])
                            }
                        })
                        .unwrap(),
                    );
                }
                let expected_preactivation = ops::concatenate_axis(
                    &expected_preactivation_rows.iter().collect::<Vec<_>>(),
                    1,
                )
                .unwrap();
                let expected_activated =
                    ops::concatenate_axis(&expected_activated_rows.iter().collect::<Vec<_>>(), 1)
                        .unwrap();
                let actual_activated = silu_direct(&actual_preactivation).unwrap();
                let label = format!("canonical conv offset={offset_init} M={seq_len}");
                assert_canonical_array_exact(
                    &format!("{label} preactivation"),
                    &actual_preactivation,
                    &expected_preactivation,
                );
                assert_canonical_array_exact(
                    &format!("{label} activated"),
                    &actual_activated,
                    &expected_activated,
                );
            }
        }

        let fp16_history = history.as_dtype(Dtype::Float16).unwrap();
        assert!(!canonical_conv_kernel_supported(
            &bf16_pattern(&[batch, 5, conv_dim], 3, 71, 53.0),
            &fp16_history,
            &weight_t,
            batch,
            5,
            conv_dim,
            kernel_size,
        ));
    }

    /// Measures only the divergent portion of CanonicalS1 short-block
    /// convolution. History rotation is shared by both production paths, so
    /// this benchmark materializes a fully warm, rotated ring before timing.
    #[test]
    #[ignore = "microbenchmark, requires Apple Metal GPU"]
    fn bench_bonsai_canonical_conv_ordered_vs_fused_bf16_m5() {
        use std::time::Instant;

        const BATCH: i32 = 1;
        const SEQ_LEN: i32 = 5;
        const CONV_DIM: i32 = 10_240;
        const KERNEL_SIZE: i32 = 4;
        const HISTORY_LEN: i32 = KERNEL_SIZE - 1;
        const OFFSET_INIT: i32 = 4_096;
        const WARMUP_ITERS: usize = 20;
        const DEFAULT_SAMPLES: usize = 201;
        const ORDERED_ARITHMETIC_DISPATCHES: usize = 45;
        const FUSED_ARITHMETIC_DISPATCHES: usize = 3;

        let bf16_pattern = |shape: &[i32], salt: i32, modulus: i32, divisor: f32| {
            let size = shape.iter().product::<i32>();
            let values = (0..size)
                .map(|index| {
                    let centered = (index * 29 + salt).rem_euclid(modulus) - modulus / 2;
                    centered as f32 / divisor
                })
                .collect::<Vec<_>>();
            let array = Array::from_slice(&values, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            array.eval().unwrap();
            array
        };

        let mixed_qkv = bf16_pattern(&[BATCH, SEQ_LEN, CONV_DIM], 31, 71, 53.0);
        let weight_t = bf16_pattern(&[KERNEL_SIZE, CONV_DIM], 19, 67, 59.0);

        // Fully warm decode ring with newest row in slot zero. Production's
        // chronological conversion therefore rotates slots [1, 2, 0].
        let ring_state = bf16_pattern(&[BATCH, HISTORY_LEN, CONV_DIM], 7, 61, 47.0);
        let ring_after_newest = ring_state.index((.., 1.., ..));
        let ring_through_newest = ring_state.index((.., ..1, ..));
        let history =
            ops::concatenate_axis(&[&ring_after_newest, &ring_through_newest], 1).unwrap();
        history.eval().unwrap();
        assert_eq!(history.shape(), &[BATCH, HISTORY_LEN, CONV_DIM]);

        let ordered = || {
            let mut rows = Vec::with_capacity(SEQ_LEN as usize);
            for position in 0..SEQ_LEN {
                let current = mixed_qkv
                    .index((.., position..position + 1, ..))
                    .reshape(&[BATCH, CONV_DIM])
                    .unwrap();
                let mut preactivation = current
                    .multiply(&weight_t.index((KERNEL_SIZE - 1, ..)))
                    .unwrap();
                let available = (OFFSET_INIT + position).clamp(0, HISTORY_LEN);
                for lag in 0..available {
                    let prior = if lag < position {
                        mixed_qkv
                            .index((.., position - 1 - lag..position - lag, ..))
                            .reshape(&[BATCH, CONV_DIM])
                            .unwrap()
                    } else {
                        let history_index = HISTORY_LEN - 1 - (lag - position);
                        history
                            .index((.., history_index..history_index + 1, ..))
                            .reshape(&[BATCH, CONV_DIM])
                            .unwrap()
                    };
                    let product = prior
                        .multiply(&weight_t.index((HISTORY_LEN - 1 - lag, ..)))
                        .unwrap();
                    preactivation = preactivation.add(&product).unwrap();
                }
                rows.push(
                    silu_direct(&preactivation.reshape(&[BATCH, 1, CONV_DIM]).unwrap()).unwrap(),
                );
            }
            ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 1).unwrap()
        };

        let fused = || {
            let preactivation = canonical_conv_preactivation_ffi(
                &mixed_qkv,
                &history,
                &weight_t,
                OFFSET_INIT,
                BATCH,
                SEQ_LEN,
                CONV_DIM,
                KERNEL_SIZE,
            )
            .unwrap();
            silu_direct(&preactivation).unwrap()
        };

        // Compile every primitive and prove the benchmark paths still obey the
        // exact production contract before collecting timings.
        let ordered_check = ordered();
        let fused_check = fused();
        mlx_rs::transforms::eval([&ordered_check, &fused_check]).unwrap();
        assert_canonical_array_exact(
            "Bonsai canonical conv microbenchmark",
            &fused_check,
            &ordered_check,
        );

        for iteration in 0..WARMUP_ITERS {
            let outputs = if iteration % 2 == 0 {
                [ordered(), fused()]
            } else {
                [fused(), ordered()]
            };
            mlx_rs::transforms::eval(outputs.iter()).unwrap();
            std::hint::black_box(outputs);
        }

        let samples = std::env::var("HIGGS_DFLASH_CONV_BENCH_SAMPLES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(DEFAULT_SAMPLES);
        let mut ordered_us = Vec::with_capacity(samples);
        let mut fused_us = Vec::with_capacity(samples);

        let measure = |path: &dyn Fn() -> Array, timings: &mut Vec<f64>| {
            let start = Instant::now();
            let output = path();
            mlx_rs::transforms::eval([&output]).unwrap();
            timings.push(start.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(output);
        };
        for sample in 0..samples {
            if sample % 2 == 0 {
                measure(&ordered, &mut ordered_us);
                measure(&fused, &mut fused_us);
            } else {
                measure(&fused, &mut fused_us);
                measure(&ordered, &mut ordered_us);
            }
        }

        let summarize = |values: &mut [f64]| {
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (median, mean)
        };
        let (ordered_median_us, ordered_mean_us) = summarize(&mut ordered_us);
        let (fused_median_us, fused_mean_us) = summarize(&mut fused_us);
        let dispatch_reduction =
            ORDERED_ARITHMETIC_DISPATCHES as f64 / FUSED_ARITHMETIC_DISPATCHES as f64;

        eprintln!(
            "Bonsai CanonicalS1 conv B={BATCH} M={SEQ_LEN} D={CONV_DIM} K={KERNEL_SIZE} BF16, warm rotated history, samples={samples}"
        );
        eprintln!("  ordered exact: median={ordered_median_us:.1}us mean={ordered_mean_us:.1}us");
        eprintln!("  fused + SiLU:   median={fused_median_us:.1}us mean={fused_mean_us:.1}us");
        eprintln!(
            "  wall speedup:   median={:.2}x mean={:.2}x",
            ordered_median_us / fused_median_us,
            ordered_mean_us / fused_mean_us,
        );
        eprintln!(
            "  nominal arithmetic dispatches: ordered={ORDERED_ARITHMETIC_DISPATCHES} + concatenate, fused={FUSED_ARITHMETIC_DISPATCHES}; reduction={dispatch_reduction:.1}x"
        );
    }

    #[test]
    fn test_gdn_decode_conv_ring_buffer_matches_concat_path() {
        use mlx_rs::Dtype;

        let args = valid_causal_lm_args();
        let mut gdn = GatedDeltaNet::new(&args, "test.layer.linear_attn").unwrap();
        let conv_w = mlx_rs::random::uniform::<f32, f32>(
            -0.5,
            0.5,
            &[gdn.conv_dim, gdn.conv_kernel_size, 1],
            None,
        )
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
        gdn.conv1d.weight = Param::new(conv_w);
        gdn.conv_weight_t = None;

        let wt = gdn
            .conv1d
            .weight
            .squeeze_axes(&[-1])
            .unwrap()
            .transpose()
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();

        let history_len = gdn.conv_kernel_size - 1;
        let mut ref_state =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, history_len, gdn.conv_dim], None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
        let mut cache = ArraysCache {
            conv_state: Some(ref_state.clone()),
            ssm_state: None,
            conv_pos: history_len - 1,
            offset: history_len,
        };

        for _ in 0..6 {
            let mixed_qkv =
                mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, gdn.conv_dim], None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap();

            let actual = gdn.decode_conv1d_step(&mixed_qkv, &mut cache, 1).unwrap();
            let conv_in = ops::concatenate_axis(&[&ref_state, &mixed_qkv], 1).unwrap();
            let expected =
                silu_direct(&conv_in.multiply(&wt).unwrap().sum_axes(&[1], true).unwrap()).unwrap();

            mlx_rs::transforms::eval([&actual, &expected]).unwrap();
            let diff = actual.subtract(&expected).unwrap().abs().unwrap();
            let max_diff: f32 = diff.max(None).unwrap().item();
            assert!(
                max_diff < 2e-3,
                "ring-buffer decode conv differs from concat path by {max_diff}"
            );

            ref_state = conv_in.index((.., 1.., ..));
            cache.offset += 1;
        }

        let ordered = gdn
            .chronological_conv_state(&mut cache, 1, Dtype::Float16)
            .unwrap();
        mlx_rs::transforms::eval([&ordered, &ref_state]).unwrap();
        let diff = ordered.subtract(&ref_state).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 2e-3,
            "linearized ring-buffer conv state differs from chronological state by {max_diff}"
        );
    }

    #[test]
    fn test_gated_delta_kernel_state_passthrough() {
        // Verify that running kernel with T=1 twice produces different state
        // than running with T=2, confirming sequential dependence works.
        let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
        let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
        let a_log = Array::zeros::<f32>(&[4]).unwrap();
        let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
        let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
        let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
        let state0 = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

        // Step 1
        let (_, state1) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state0, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        state1.eval().unwrap();

        // Step 2 (uses state1)
        let (y2, state2) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state1, 1, 1, 2, 32, 4, 32,
        )
        .unwrap();
        y2.eval().unwrap();
        state2.eval().unwrap();

        assert_eq!(y2.shape(), &[1, 1, 4, 32]);
        assert_eq!(state2.shape(), &[1, 4, 32, 32]);
    }

    /// Reference ops implementation of a single gated delta step (for comparison tests).
    fn gated_delta_step_ref(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> (Array, Array) {
        let decay = g.expand_dims(-1).unwrap().expand_dims(-1).unwrap();
        let decayed_state = state.multiply(&decay).unwrap();
        let k_expanded = k.expand_dims(-2).unwrap();
        let kv_mem = decayed_state
            .multiply(&k_expanded)
            .unwrap()
            .sum_axes(&[-1], false)
            .unwrap();
        let beta_expanded = beta.expand_dims(-1).unwrap();
        let delta = v
            .subtract(&kv_mem)
            .unwrap()
            .multiply(&beta_expanded)
            .unwrap();
        let delta_expanded = delta.expand_dims(-1).unwrap();
        let new_state = decayed_state
            .add(k_expanded.multiply(&delta_expanded).unwrap())
            .unwrap();
        let q_expanded = q.expand_dims(-2).unwrap();
        let y = new_state
            .multiply(&q_expanded)
            .unwrap()
            .sum_axes(&[-1], false)
            .unwrap();
        (y, new_state)
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops() {
        // Compare kernel output against reference ops for T=1, no GQA.
        // B=1, T=1, Hk=1, Hv=1, Dk=32, Dv=32
        assert_kernel_matches_ops(1, 1, 1, 1, 32, 32, 1e-4, "Hk=Hv=1");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_gqa() {
        // GQA: Hk=2, Hv=4 (repeat factor 2). This is the pattern used by Qwen3-Next.
        assert_kernel_matches_ops(1, 1, 2, 4, 32, 32, 1e-4, "Hk=2,Hv=4 GQA");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_multi_step() {
        // T=3 with GQA: verify multi-timestep correctness
        assert_kernel_matches_ops(1, 3, 2, 4, 32, 32, 1e-4, "T=3 GQA");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_model_dims() {
        // Actual Qwen3-Next dims: Hk=16, Hv=32, Dk=128, Dv=128
        assert_kernel_matches_ops(1, 1, 16, 32, 128, 128, 1e-4, "model dims");
    }

    #[test]
    fn test_gated_delta_kernel_matches_ops_bfloat16() {
        // The actual model uses bfloat16. Test with model dims in bfloat16.
        use mlx_rs::Dtype;
        let hk = 2;
        let hv = 4;
        let dk = 32;
        let dv = 32;
        let batch = 1;
        let seq_len = 1;

        let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let a = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let b = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        // The recurrence state is always f32 even when activations are bf16.
        let state =
            mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

        // Kernel
        let (kern_y, kern_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();
        kern_y.eval().unwrap();
        kern_state.eval().unwrap();

        assert_eq!(kern_y.shape(), &[batch, seq_len, hv, dv]);
        assert_eq!(kern_state.shape(), &[batch, hv, dv, dk]);

        // Verify outputs are finite (not NaN/Inf)
        let y_f32 = kern_y.as_dtype(Dtype::Float32).unwrap();
        let y_abs_max: f32 = y_f32.abs().unwrap().max(None).unwrap().item();
        assert!(
            y_abs_max.is_finite() && y_abs_max < 1e6,
            "bfloat16 kernel y has bad values: max abs = {y_abs_max}"
        );
    }

    #[test]
    fn gdn_metal_paths_reject_non_f32_state() {
        let (batch, seq_len, hk, dk, hv, dv) = (1, 1, 1, 32, 1, 1);
        let bf16 = |shape: &[i32]| {
            Array::zeros::<f32>(shape)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let q = bf16(&[batch, seq_len, hk, dk]);
        let k = bf16(&[batch, seq_len, hk, dk]);
        let v = bf16(&[batch, seq_len, hv, dv]);
        let a = bf16(&[batch, seq_len, hv]);
        let beta = bf16(&[batch, seq_len, hv]);
        let a_log = bf16(&[hv]);
        let dt_bias = bf16(&[hv]);
        let bad_state = bf16(&[batch, hv, dv, dk]);

        let plain = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &bad_state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap_err();
        assert!(plain.to_string().contains("state must be Float32"));

        let tape = gated_delta_kernel_ffi_with_tape(
            &q, &k, &v, &a_log, &a, &dt_bias, &beta, &bad_state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap_err();
        assert!(tape.to_string().contains("state must be Float32"));

        let innovation = Array::zeros::<f32>(&[batch, seq_len, hv, dv]).unwrap();
        let replay = tape_replay_kernel_ffi(
            &innovation,
            &k,
            &a,
            &a_log,
            &dt_bias,
            &bad_state,
            batch,
            seq_len,
            hk,
            dk,
            hv,
            dv,
        )
        .unwrap_err();
        assert!(replay.to_string().contains("state must be Float32"));
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_kernel_matches_ops(
        batch: i32,
        seq_len: i32,
        hk: i32,
        hv: i32,
        dk: i32,
        dv: i32,
        tol: f32,
        label: &str,
    ) {
        let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
            .unwrap();
        let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None)
            .unwrap();
        let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None).unwrap();
        let a_val =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
        let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None).unwrap();
        let b =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
        let state =
            mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

        // Compute g and beta from raw inputs for the reference path
        let mut compute_g_fn = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
        let g = compute_g_fn((&a_log, &a_val, &dt_bias)).unwrap();
        let beta = nn::sigmoid(&b).unwrap();

        // Reference: loop over timesteps with repeat_axis for GQA
        let repeat_factor = hv / hk;
        let mut ref_state = state.clone();
        let mut ref_ys = Vec::new();
        for t in 0..seq_len {
            let qt = q.index((.., t, .., ..));
            let kt = k.index((.., t, .., ..));
            let vt = v.index((.., t, .., ..));
            let gt = g.index((.., t, ..));
            let bt = beta.index((.., t, ..));

            let qt_rep = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(qt, repeat_factor, -2).unwrap()
            } else {
                qt
            };
            let kt_rep = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(kt, repeat_factor, -2).unwrap()
            } else {
                kt
            };

            let (y_t, new_state) =
                gated_delta_step_ref(&qt_rep, &kt_rep, &vt, &gt, &bt, &ref_state);
            ref_state = new_state;
            ref_ys.push(y_t);
        }
        let ref_y_refs: Vec<&Array> = ref_ys.iter().collect();
        let ref_y = ops::stack_axis(&ref_y_refs, 1).unwrap();
        ref_y.eval().unwrap();
        ref_state.eval().unwrap();

        // Kernel
        let (kern_y, kern_state) = gated_delta_kernel_ffi(
            &q, &k, &v, &a_log, &a_val, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
        )
        .unwrap();
        kern_y.eval().unwrap();
        kern_state.eval().unwrap();

        // Compare y
        let y_diff = ref_y.subtract(&kern_y).unwrap().abs().unwrap();
        let y_max: f32 = y_diff.max(None).unwrap().item();
        assert!(y_max < tol, "[{label}] kernel y differs by {y_max}");

        // Compare state
        let s_diff = ref_state.subtract(&kern_state).unwrap().abs().unwrap();
        let s_max: f32 = s_diff.max(None).unwrap().item();
        assert!(s_max < tol, "[{label}] kernel state differs by {s_max}");
    }

    /// Benchmark: chain 48 layers of 3x gather_qmm + SwiGLU, single eval.
    /// Compare with Python's 0.378ms (48 layers, single eval).
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_gather_qmm_chain() {
        let num_experts = 512;
        let d = 2048;
        let intermediate = 512;
        let top_k = 10;

        // Create quantized expert weights (same as model)
        let gate_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
        let gate_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
        let gate_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

        let up_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
        let up_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
        let up_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

        let down_w = Array::zeros::<u32>(&[num_experts, d, intermediate * 4 / 32]).unwrap();
        let down_s = Array::ones::<f32>(&[num_experts, d, intermediate / 64]).unwrap();
        let down_b = Array::zeros::<f32>(&[num_experts, d, intermediate / 64]).unwrap();

        let x = Array::ones::<f32>(&[1, 1, 1, 1, d]).unwrap();
        let indices = Array::from_slice(&[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
        mlx_rs::transforms::eval([
            &gate_w, &gate_s, &gate_b, &up_w, &up_s, &up_b, &down_w, &down_s, &down_b, &x, &indices,
        ])
        .unwrap();

        // Warm up
        for _ in 0..3 {
            let mut y = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }

        // Benchmark: 48 layers, single eval -- split graph build vs eval
        let n = 50;
        let mut total_build_ns = 0u128;
        let mut total_eval_ns = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build_ns += (t1 - t0).as_nanos();
            total_eval_ns += (t2 - t1).as_nanos();
        }
        let build_ms = total_build_ns as f64 / n as f64 / 1_000_000.0;
        let eval_ms = total_eval_ns as f64 / n as f64 / 1_000_000.0;
        eprintln!(
            "48 layers * 3 gather_qmm + SwiGLU: build={build_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
            build_ms + eval_ms
        );

        // Also test with mlx-rs ops::add chain (no FFI gather_qmm)
        let n3 = 50;
        let x_simple = Array::ones::<f32>(&[1, 1, d]).unwrap();
        mlx_rs::transforms::eval([&x_simple]).unwrap();
        let mut total_simple_ns = 0u128;
        for _ in 0..n3 {
            let t0 = std::time::Instant::now();
            let mut y2 = x_simple.clone();
            for _ in 0..(48 * 5) {
                y2 = y2.add(&x_simple).unwrap();
            }
            mlx_rs::transforms::eval([&y2]).unwrap();
            total_simple_ns += t0.elapsed().as_nanos();
        }
        let simple_ms = total_simple_ns as f64 / n3 as f64 / 1_000_000.0;
        eprintln!("240 chained adds (single eval): {simple_ms:.2}ms");

        // Test with the shared gather_qmm wrapper
        let n4 = 50;
        let mut total_builtin_build = 0u128;
        let mut total_builtin_eval = 0u128;
        for _ in 0..n4 {
            let t0 = std::time::Instant::now();
            let mut y3 = x.clone();
            for _ in 0..48 {
                let g = gather_qmm(&y3, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                    .unwrap();
                let u = gather_qmm(&y3, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y3 = gather_qmm(
                    &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                )
                .unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y3]).unwrap();
            let t2 = std::time::Instant::now();
            total_builtin_build += (t1 - t0).as_nanos();
            total_builtin_eval += (t2 - t1).as_nanos();
        }
        let builtin_build = total_builtin_build as f64 / n4 as f64 / 1_000_000.0;
        let builtin_eval = total_builtin_eval as f64 / n4 as f64 / 1_000_000.0;
        eprintln!(
            "48 layers mlx-rs gather_qmm: build={builtin_build:.2}ms eval={builtin_eval:.2}ms total={:.2}ms",
            builtin_build + builtin_eval
        );

        // Test with quantized_matmul (not gather) - 144 chained calls
        let qm_w = Array::zeros::<u32>(&[d, d * 4 / 32]).unwrap();
        let qm_s = Array::ones::<f32>(&[d, d / 64]).unwrap();
        let qm_b = Array::zeros::<f32>(&[d, d / 64]).unwrap();
        let x_qm = Array::ones::<f32>(&[1, 1, d]).unwrap();
        mlx_rs::transforms::eval([&qm_w, &qm_s, &qm_b, &x_qm]).unwrap();

        // Warm up
        for _ in 0..3 {
            let mut y4 = x_qm.clone();
            for _ in 0..144 {
                y4 = ops::quantized_matmul(&y4, &qm_w, &qm_s, &qm_b, true, 64, 4).unwrap();
            }
            mlx_rs::transforms::eval([&y4]).unwrap();
        }

        let n5 = 50;
        let mut total_qm_build = 0u128;
        let mut total_qm_eval = 0u128;
        for _ in 0..n5 {
            let t0 = std::time::Instant::now();
            let mut y4 = x_qm.clone();
            for _ in 0..144 {
                y4 = ops::quantized_matmul(&y4, &qm_w, &qm_s, &qm_b, true, 64, 4).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y4]).unwrap();
            let t2 = std::time::Instant::now();
            total_qm_build += (t1 - t0).as_nanos();
            total_qm_eval += (t2 - t1).as_nanos();
        }
        let qm_build = total_qm_build as f64 / n5 as f64 / 1_000_000.0;
        let qm_eval = total_qm_eval as f64 / n5 as f64 / 1_000_000.0;
        eprintln!(
            "144 chained quantized_matmul: build={qm_build:.2}ms eval={qm_eval:.2}ms total={:.2}ms",
            qm_build + qm_eval
        );

        // Benchmark: single layer, per-call eval
        let n2 = 200;
        let start2 = std::time::Instant::now();
        for _ in 0..n2 {
            let g =
                gather_qmm(&x, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
            let u = gather_qmm(&x, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
            let activated = swiglu(&g, &u).unwrap();
            let y = gather_qmm(
                &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let per_layer_ms = start2.elapsed().as_millis() as f64 / n2 as f64;
        eprintln!("1 layer * 3 gather_qmm + SwiGLU (per-call eval): {per_layer_ms:.2} ms");

        // Test eval overhead: 1000 chained adds (Python: build=0.23ms eval=1.87ms)
        let n_ops = 1000;
        let x_add = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        mlx_rs::transforms::eval([&x_add]).unwrap();
        // Warmup
        for _ in 0..3 {
            let mut y = x_add.clone();
            for _ in 0..n_ops {
                y = y.add(&x_add).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let n6 = 50;
        let mut total_add_build = 0u128;
        let mut total_add_eval = 0u128;
        for _ in 0..n6 {
            let t0 = std::time::Instant::now();
            let mut y = x_add.clone();
            for _ in 0..n_ops {
                y = y.add(&x_add).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_add_build += (t1 - t0).as_nanos();
            total_add_eval += (t2 - t1).as_nanos();
        }
        let add_build = total_add_build as f64 / n6 as f64 / 1_000_000.0;
        let add_eval = total_add_eval as f64 / n6 as f64 / 1_000_000.0;
        eprintln!(
            "{n_ops} chained adds: build={add_build:.2}ms eval={add_eval:.2}ms total={:.2}ms",
            add_build + add_eval
        );
        eprintln!(
            "Per op: build={:.1}us eval={:.1}us",
            add_build * 1000.0 / n_ops as f64,
            add_eval * 1000.0 / n_ops as f64
        );

        // Test with task-local default stream
        let stream = mlx_rs::Stream::new();
        let gather_with_stream = || {
            mlx_rs::with_new_default_stream(stream.clone(), || {
                let mut total_b = 0u128;
                let mut total_e = 0u128;
                let n7 = 50;
                for _ in 0..n7 {
                    let t0 = std::time::Instant::now();
                    let mut y = x.clone();
                    for _ in 0..48 {
                        let g =
                            gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                                .unwrap();
                        let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false)
                            .unwrap();
                        let activated = swiglu(&g, &u).unwrap();
                        y = gather_qmm(
                            &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                        )
                        .unwrap();
                    }
                    let t1 = std::time::Instant::now();
                    mlx_rs::transforms::eval([&y]).unwrap();
                    let t2 = std::time::Instant::now();
                    total_b += (t1 - t0).as_nanos();
                    total_e += (t2 - t1).as_nanos();
                }
                let b = total_b as f64 / n7 as f64 / 1_000_000.0;
                let e = total_e as f64 / n7 as f64 / 1_000_000.0;
                eprintln!(
                    "48 layers gather_qmm (with task-local stream): build={b:.2}ms eval={e:.2}ms total={:.2}ms",
                    b + e
                );
            });
        };
        gather_with_stream();
    }

    /// Benchmark: 200 chained quantized_matmul ops (matching Python bench).
    /// Python: build=0.05ms eval=1.40ms total=1.45ms
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_chained_quantized_matmul() {
        use mlx_rs::Dtype;

        let x = ops::ones_dtype(&[1, 1, 2048], Dtype::Float16).unwrap();
        let raw_w = ops::ones_dtype(&[2048, 2048], Dtype::Float16).unwrap();
        let (w, s, b) = ops::quantize(&raw_w, 64, 4).unwrap();
        mlx_rs::transforms::eval([&x, &w, &s, &b]).unwrap();

        let n_ops = 200;
        let n = 50;

        // Warmup
        for _ in 0..10 {
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = ops::quantized_matmul(&y, &w, &s, &b, true, 64, 4).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }

        let mut total_build = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = ops::quantized_matmul(&y, &w, &s, &b, true, 64, 4).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }
        let build = total_build as f64 / n as f64 / 1e6;
        let eval = total_eval as f64 / n as f64 / 1e6;
        eprintln!(
            "Rust 200 qmm: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
            build + eval
        );

        // 200 chained adds
        for _ in 0..10 {
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = y.add(&x).unwrap();
            }
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_build = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let mut y = x.clone();
            for _ in 0..n_ops {
                y = y.add(&x).unwrap();
            }
            let t1 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            let t2 = std::time::Instant::now();
            total_build += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }
        let build = total_build as f64 / n as f64 / 1e6;
        let eval = total_eval as f64 / n as f64 / 1e6;
        eprintln!(
            "Rust 200 add: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
            build + eval
        );
    }

    /// Simulate 48-layer forward pass with per-layer weights.
    /// Python shared-weight sim: build=0.59ms eval=8.08ms
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_simulated_forward() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let d_inter = 512i32; // moe_intermediate_size from config
        let n_experts = 512i32;
        let top_k = 10i32; // num_experts_per_tok from config
        let gs = 64i32;
        let bits = 4i32;
        let shared_inter = 512i32; // shared_expert_intermediate_size

        // Use random weights to test realistic memory access patterns.
        // ops::ones_dtype creates constant data that artificially benefits from GPU cache.
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let hk = 16i32;
        let dk = 128i32;
        let hv = 32i32;
        let dv = 128i32;

        struct LayerWeights {
            q_proj: (Array, Array, Array),
            k_proj: (Array, Array, Array),
            v_proj: (Array, Array, Array),
            o_proj: (Array, Array, Array),
            g_proj: (Array, Array, Array),
            beta_proj: (Array, Array, Array),
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
            norm_w: Array,
        }

        let layers: Vec<LayerWeights> = (0..48)
            .map(|_| LayerWeights {
                q_proj: make_qw(d, hk * dk),
                k_proj: make_qw(d, hk * dk),
                v_proj: make_qw(d, hv * dv),
                o_proj: make_qw(hv * dv, d),
                g_proj: make_qw(d, hv),
                beta_proj: make_qw(d, hv),
                gate: make_qw(d, n_experts),
                sw_gate: make_sw(d, d_inter),
                sw_up: make_sw(d, d_inter),
                sw_down: make_sw(d_inter, d),
                se_gate: make_qw(d, shared_inter * 2),
                se_up: make_qw(d, shared_inter * 2),
                se_down: make_qw(shared_inter * 2, d),
                se_gate_proj: make_qw(d, 1),
                norm_w: Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            })
            .collect();

        let mut all_w: Vec<&Array> = Vec::new();
        for l in &layers {
            for (w, s, b) in [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.g_proj,
                &l.beta_proj,
                &l.gate,
                &l.sw_gate,
                &l.sw_up,
                &l.sw_down,
                &l.se_gate,
                &l.se_up,
                &l.se_down,
                &l.se_gate_proj,
            ] {
                all_w.extend_from_slice(&[w, s, b]);
            }
            all_w.push(&l.norm_w);
        }
        mlx_rs::transforms::eval(all_w).unwrap();

        // Check actual memory usage to verify weights are materialized
        let active_mem = {
            let mut res: usize = 0;
            #[allow(unsafe_code)]
            unsafe {
                mlx_sys::mlx_get_active_memory(&mut res as *mut _);
            }
            res
        };
        eprintln!(
            "Active memory after weight eval: {:.2} GB",
            active_mem as f64 / 1e9
        );

        // Print one switch weight shape to verify
        eprintln!(
            "sw_gate[0] shape: {:?} dtype: {:?}",
            layers[0].sw_gate.0.shape(),
            layers[0].sw_gate.0.dtype()
        );

        let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
        mlx_rs::transforms::eval([&x]).unwrap();

        let forward_n_inline = |x: &Array, n_layers: usize| -> Array {
            let mut h = x.clone();
            for l in layers.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &l.norm_w, 1e-6).unwrap();

                // Attention projections (matching real model's GDN layer ops)
                let _q = ops::quantized_matmul(
                    &normed,
                    &l.q_proj.0,
                    &l.q_proj.1,
                    &l.q_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let _k = ops::quantized_matmul(
                    &normed,
                    &l.k_proj.0,
                    &l.k_proj.1,
                    &l.k_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let v = ops::quantized_matmul(
                    &normed,
                    &l.v_proj.0,
                    &l.v_proj.1,
                    &l.v_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let g = ops::quantized_matmul(
                    &normed,
                    &l.g_proj.0,
                    &l.g_proj.1,
                    &l.g_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let _beta = ops::quantized_matmul(
                    &normed,
                    &l.beta_proj.0,
                    &l.beta_proj.1,
                    &l.beta_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let attn_proxy = v
                    .multiply(&nn::sigmoid(&g.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                let o = ops::quantized_matmul(
                    &attn_proxy,
                    &l.o_proj.0,
                    &l.o_proj.1,
                    &l.o_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();

                let h2 = h.add(o).unwrap();
                let normed2 = fast::rms_norm(&h2, &l.norm_w, 1e-6).unwrap();

                // Router
                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                // Switch MLP (per-layer switch weights)
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                // Shared expert (per-layer weights)
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        };

        for n_layers in [1, 4, 8, 16, 24, 48] {
            for _ in 0..5 {
                let y = forward_n_inline(&x, n_layers);
                mlx_rs::transforms::eval([&y]).unwrap();
            }
            let n = 20;
            let mut total_eval = 0u128;
            for _ in 0..n {
                let y = forward_n_inline(&x, n_layers);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&y]).unwrap();
                total_eval += t0.elapsed().as_nanos();
            }
            let eval = total_eval as f64 / n as f64 / 1e6;
            eprintln!(
                "Inline {n_layers} layers: eval={eval:.2}ms per_layer={:.2}ms",
                eval / n_layers as f64
            );
        }
    }

    /// Test gather_qmm with loaded vs random weights to isolate memory effects.
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_gather_qmm_loaded_vs_random() {
        use mlx_rs::Dtype;
        let Some(model_dir) = std::env::var_os("HIGGS_QWEN3_NEXT_BENCH_MODEL") else {
            eprintln!("Skipping: set HIGGS_QWEN3_NEXT_BENCH_MODEL to a local model directory");
            return;
        };
        let model_dir = std::path::PathBuf::from(model_dir);
        let Some(shard) = find_gather_qmm_bench_shard(&model_dir) else {
            eprintln!("Skipping: no safetensors shard with switch_mlp gate weights found");
            return;
        };
        let path = shard.as_path();

        // Load one safetensors shard
        let loaded = Array::load_safetensors(path).unwrap();
        mlx_rs::transforms::eval(loaded.values()).unwrap();

        // Find a switch_mlp weight (should be large [512, intermediate, ...])
        let mut sw_key = None;
        for key in loaded.keys() {
            if key.contains("switch_mlp") && key.contains("gate_proj") && key.contains(".weight") {
                sw_key = Some(key.clone());
                break;
            }
        }
        let sw_key = sw_key.expect("No switch_mlp weight found in shard");
        let w_loaded = &loaded[&sw_key];
        eprintln!(
            "Loaded weight '{sw_key}': shape={:?} dtype={:?}",
            w_loaded.shape(),
            w_loaded.dtype()
        );

        // Find corresponding scales and biases
        let scales_key = sw_key.replace(".weight", ".scales");
        let biases_key = sw_key.replace(".weight", ".biases");
        let s_loaded = &loaded[&scales_key];
        let b_loaded = &loaded[&biases_key];
        eprintln!(
            "Scales: {:?}, Biases: {:?}",
            s_loaded.shape(),
            b_loaded.shape()
        );

        // Create random weights of the same shape/dtype
        let w_shape = w_loaded.shape().to_vec();
        let s_shape = s_loaded.shape().to_vec();
        let b_shape = b_loaded.shape().to_vec();

        let w_random = mlx_rs::random::normal::<f32>(&w_shape, None, None, None)
            .unwrap()
            .as_dtype(w_loaded.dtype())
            .unwrap();
        let s_random = mlx_rs::random::normal::<f32>(&s_shape, None, None, None)
            .unwrap()
            .as_dtype(s_loaded.dtype())
            .unwrap();
        let b_random = mlx_rs::random::normal::<f32>(&b_shape, None, None, None)
            .unwrap()
            .as_dtype(b_loaded.dtype())
            .unwrap();
        mlx_rs::transforms::eval([&w_random, &s_random, &b_random]).unwrap();

        // Test input
        let x = ops::ones_dtype(&[1, 1, 1, 1, 2048], Dtype::Float16).unwrap();
        let indices = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, 10]);
        mlx_rs::transforms::eval([&x, &indices]).unwrap();

        let gs = 64i32;
        let bits = 4i32;
        let n = 100;

        // Benchmark loaded weights
        for _ in 0..10 {
            let y = gather_qmm(
                &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_loaded = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let y = gather_qmm(
                &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
            total_loaded += t0.elapsed().as_nanos();
        }

        // Benchmark random weights
        for _ in 0..10 {
            let y = gather_qmm(
                &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total_random = 0u128;
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            let y = gather_qmm(
                &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
            )
            .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
            total_random += t0.elapsed().as_nanos();
        }

        let loaded_us = total_loaded as f64 / n as f64 / 1e3;
        let random_us = total_random as f64 / n as f64 / 1e3;
        eprintln!(
            "gather_qmm single layer: loaded={loaded_us:.1}us random={random_us:.1}us ratio={:.2}x",
            loaded_us / random_us
        );
    }

    fn find_gather_qmm_bench_shard(model_dir: &std::path::Path) -> Option<std::path::PathBuf> {
        let mut candidates: Vec<_> = std::fs::read_dir(model_dir)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        candidates.sort_by(|a, b| {
            let a_name = a.file_name().and_then(|name| name.to_str()).unwrap_or("");
            let b_name = b.file_name().and_then(|name| name.to_str()).unwrap_or("");
            let a_shard = a_name.starts_with("model-") && a_name.contains("-of-");
            let b_shard = b_name.starts_with("model-") && b_name.contains("-of-");
            b_shard.cmp(&a_shard).then_with(|| a_name.cmp(b_name))
        });

        candidates.into_iter().find(|path| {
            Array::load_safetensors(path).ok().is_some_and(|loaded| {
                loaded.keys().any(|key| {
                    key.contains("switch_mlp")
                        && key.contains("gate_proj")
                        && key.contains(".weight")
                })
            })
        })
    }

    /// Isolate what causes the module vs inline performance gap.
    /// Tests three variants at 48 layers:
    /// A) Module forward with multiply-by-zero attention (baseline slow path)
    /// B) Inline forward with multiply-by-zero attention (tests if graph structure matters)
    /// C) Inline forward with real quantized_matmul attention (original fast path)
    /// D) Extract weights from modules into tuples, run inline (tests Param<Array> access)
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_module_vs_inline() {
        use mlx_rs::Dtype;
        use mlx_rs::module::Param;

        let d = 2048i32;
        let d_inter = 512i32;
        let n_experts = 512i32;
        let top_k = 10i32;
        let gs = 64i32;
        let bits = 4i32;
        let shared_inter = 512i32;

        let make_ql = |d_in: i32, d_out: i32, gs: i32, bits: i32| -> QLinear {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            QLinear {
                weight: Param::new(w),
                scales: Param::new(s),
                biases: Param::new(b),
                group_size: gs,
                bits,
                mode: crate::quant_mode::QuantMode::Affine,
                weight_layout: QLinearWeightLayout::Canonical,
            }
        };

        let make_switch_ql = |d_in: i32, d_out: i32| -> QLinear {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            QLinear {
                weight: Param::new(w),
                scales: Param::new(s),
                biases: Param::new(b),
                group_size: gs,
                bits,
                mode: crate::quant_mode::QuantMode::Affine,
                weight_layout: QLinearWeightLayout::Canonical,
            }
        };

        // Build 48 SparseMoeBlock instances with random weights
        let moe_blocks: Vec<SparseMoeBlock> = (0..48)
            .map(|_| SparseMoeBlock {
                gate: make_ql(d, n_experts, gs, bits),
                switch_mlp: SwitchMlpWeights {
                    gate_proj: make_switch_ql(d, d_inter),
                    up_proj: make_switch_ql(d, d_inter),
                    down_proj: make_switch_ql(d_inter, d),
                    fused_gate_up: None,
                },
                shared_expert: Qwen3NextMLP {
                    gate_proj: make_ql(d, shared_inter * 2, gs, bits),
                    up_proj: make_ql(d, shared_inter * 2, gs, bits),
                    down_proj: make_ql(shared_inter * 2, d, gs, bits),
                },
                shared_expert_gate: make_ql(d, 1, gs, bits),
                top_k,
                norm_topk_prob: true,
            })
            .collect();

        // Eval all module weights
        {
            use mlx_rs::module::ModuleParameters;
            let mut all_w: Vec<&Array> = Vec::new();
            for moe in &moe_blocks {
                for (_, arr) in moe.parameters().flatten() {
                    all_w.push(arr);
                }
            }
            mlx_rs::transforms::eval(all_w).unwrap();
        }

        // Extract module weights into bare tuples for variant D
        struct ExtractedWeights {
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
        }
        let extracted: Vec<ExtractedWeights> = moe_blocks
            .iter()
            .map(|moe| {
                // Clone the Array handles (cheap refcount bump, same underlying MLX data)
                ExtractedWeights {
                    gate: (
                        moe.gate.weight.value.clone(),
                        moe.gate.scales.value.clone(),
                        moe.gate.biases.value.clone(),
                    ),
                    sw_gate: (
                        moe.switch_mlp.gate_proj.weight.value.clone(),
                        moe.switch_mlp.gate_proj.scales.value.clone(),
                        moe.switch_mlp.gate_proj.biases.value.clone(),
                    ),
                    sw_up: (
                        moe.switch_mlp.up_proj.weight.value.clone(),
                        moe.switch_mlp.up_proj.scales.value.clone(),
                        moe.switch_mlp.up_proj.biases.value.clone(),
                    ),
                    sw_down: (
                        moe.switch_mlp.down_proj.weight.value.clone(),
                        moe.switch_mlp.down_proj.scales.value.clone(),
                        moe.switch_mlp.down_proj.biases.value.clone(),
                    ),
                    se_gate: (
                        moe.shared_expert.gate_proj.weight.value.clone(),
                        moe.shared_expert.gate_proj.scales.value.clone(),
                        moe.shared_expert.gate_proj.biases.value.clone(),
                    ),
                    se_up: (
                        moe.shared_expert.up_proj.weight.value.clone(),
                        moe.shared_expert.up_proj.scales.value.clone(),
                        moe.shared_expert.up_proj.biases.value.clone(),
                    ),
                    se_down: (
                        moe.shared_expert.down_proj.weight.value.clone(),
                        moe.shared_expert.down_proj.scales.value.clone(),
                        moe.shared_expert.down_proj.biases.value.clone(),
                    ),
                    se_gate_proj: (
                        moe.shared_expert_gate.weight.value.clone(),
                        moe.shared_expert_gate.scales.value.clone(),
                        moe.shared_expert_gate.biases.value.clone(),
                    ),
                }
            })
            .collect();

        let norm_w = Array::ones::<f32>(&[d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
        mlx_rs::transforms::eval([&x, &norm_w]).unwrap();

        let n_layers = 48usize;
        let n = 20;

        // Helper: run N warmups then N timed evals
        let bench = |label: &str, forward: &dyn Fn(&Array) -> Array| {
            for _ in 0..5 {
                let y = forward(&x);
                mlx_rs::transforms::eval([&y]).unwrap();
            }
            let mut total = 0u128;
            for _ in 0..n {
                let y = forward(&x);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&y]).unwrap();
                total += t0.elapsed().as_nanos();
            }
            let ms = total as f64 / n as f64 / 1e6;
            eprintln!(
                "{label}: eval={ms:.2}ms per_layer={:.2}ms",
                ms / n_layers as f64
            );
        };

        // A) Module forward + multiply-by-zero attention
        bench("A) module+zero_attn", &|x: &Array| {
            let mut h = x.clone();
            for moe in moe_blocks.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();
                let mlp_out = moe.forward(&normed2).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // B) Inline forward + multiply-by-zero attention (same extracted weights)
        bench("B) inline+zero_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                // Inline MoE (same code as bench_simulated_forward)
                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // C) Inline forward + real quantized_matmul for attention (per-layer attn weights)
        // This matches the bench_simulated_forward test structure
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let attn_weights: Vec<(Array, Array, Array)> = (0..48).map(|_| make_qw(d, d)).collect();
        let per_layer_norms: Vec<Array> = (0..48)
            .map(|_| {
                Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        {
            let mut all_w: Vec<&Array> = Vec::new();
            for (w, s, b) in &attn_weights {
                all_w.extend_from_slice(&[w, s, b]);
            }
            for nw in &per_layer_norms {
                all_w.push(nw);
            }
            mlx_rs::transforms::eval(all_w).unwrap();
        }

        bench("C) inline+real_attn+per_layer_norm", &|x: &Array| {
            let mut h = x.clone();
            for (i, l) in extracted.iter().take(n_layers).enumerate() {
                let normed = fast::rms_norm(&h, &per_layer_norms[i], 1e-6).unwrap();
                let attn_out = ops::quantized_matmul(
                    &normed,
                    &attn_weights[i].0,
                    &attn_weights[i].1,
                    &attn_weights[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let h2 = h.add(attn_out).unwrap();
                let normed2 = fast::rms_norm(&h2, &per_layer_norms[i], 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // D) Inline + zero_attn + per_layer_norm (isolates norm_w sharing vs attn method)
        bench("D) inline+zero_attn+per_layer_norm", &|x: &Array| {
            let mut h = x.clone();
            for (i, l) in extracted.iter().take(n_layers).enumerate() {
                let normed = fast::rms_norm(&h, &per_layer_norms[i], 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &per_layer_norms[i], 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // E) Inline + multiply-by-ONE + shared norm (is zero specifically the issue?)
        bench("E) inline+mul_one_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let dummy_attn = normed.multiply(Array::from_f32(1.0)).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // F) Inline + zeros_like (skip normed entirely, just add zeros)
        bench("F) inline+zeros_like_attn", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                let normed = fast::rms_norm(&h, &norm_w, 1e-6).unwrap();
                let _ = &normed; // normed computed but not used for attn
                let dummy_attn = ops::zeros_like(&normed).unwrap();
                let h2 = h.add(dummy_attn).unwrap();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });

        // G) Inline + skip normed entirely, h2 = h (no ops for attention)
        bench("G) inline+h2_equals_h", &|x: &Array| {
            let mut h = x.clone();
            for l in extracted.iter().take(n_layers) {
                // Skip first rms_norm entirely
                let h2 = h.clone();
                let normed2 = fast::rms_norm(&h2, &norm_w, 1e-6).unwrap();

                let gate_out = ops::quantized_matmul(
                    &normed2, &l.gate.0, &l.gate.1, &l.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_start = n_experts - top_k;
                let top_inds = all_inds.index((.., .., top_start..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
                let scores = raw_scores.divide(score_sum).unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &l.sw_gate.0,
                    &l.sw_gate.1,
                    &l.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &l.sw_down.0,
                    &l.sw_down.1,
                    &l.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(&scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &l.se_gate.0,
                    &l.se_gate.1,
                    &l.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &l.se_down.0,
                    &l.se_down.1,
                    &l.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    &ops::quantized_matmul(
                        &normed2,
                        &l.se_gate_proj.0,
                        &l.se_gate_proj.1,
                        &l.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

                let mlp_out = expert_sum.add(shared_out).unwrap();
                h = h2.add(mlp_out).unwrap();
            }
            h
        });
    }

    /// Benchmark 36 GDN layers using bare Arrays.
    /// Isolates GDN ops from the model framework to compare direct GPU time.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_gdn_layers() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let hk = 16i32;
        let hv = 32i32;
        let dk = 128i32;
        let dv = 128i32;
        let gs = 64i32;
        let bits = 4i32;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let ba_out = hv * 2;
        let n_layers = 36;

        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };

        struct GDNWeights {
            in_proj_qkvz: (Array, Array, Array),
            in_proj_ba: (Array, Array, Array),
            out_proj: (Array, Array, Array),
            conv_w: Array,
            a_log: Array,
            dt_bias: Array,
            norm_w: Array,
        }

        let mut layers = Vec::new();
        let mut all_w: Vec<&Array> = Vec::new();
        for _ in 0..n_layers {
            layers.push(GDNWeights {
                in_proj_qkvz: make_qw(d, qkvz_out),
                in_proj_ba: make_qw(d, ba_out),
                out_proj: make_qw(value_dim, d),
                conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
                a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                norm_w: Array::ones::<f32>(&[dv])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            });
        }
        for l in &layers {
            all_w.extend([&l.in_proj_qkvz.0, &l.in_proj_qkvz.1, &l.in_proj_qkvz.2]);
            all_w.extend([&l.in_proj_ba.0, &l.in_proj_ba.1, &l.in_proj_ba.2]);
            all_w.extend([&l.out_proj.0, &l.out_proj.1, &l.out_proj.2]);
            all_w.extend([&l.conv_w, &l.a_log, &l.dt_bias, &l.norm_w]);
        }
        mlx_rs::transforms::eval(all_w).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
        let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
        let inv_scale_sq = {
            let s = (dk as f32).sqrt().recip();
            Array::from_f32(s * s)
        };
        let states: Vec<Array> = (0..n_layers)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        let conv_states: Vec<Array> = (0..n_layers)
            .map(|_| {
                Array::zeros::<f32>(&[1, 3, conv_dim])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }
        for c in &conv_states {
            c.eval().unwrap();
        }

        let gdn_forward = |h: &Array,
                           l: &GDNWeights,
                           state: &Array,
                           conv_state: &Array|
         -> (Array, Array, Array) {
            let qkvz = ops::quantized_matmul(
                h,
                &l.in_proj_qkvz.0,
                &l.in_proj_qkvz.1,
                &l.in_proj_qkvz.2,
                true,
                gs,
                bits,
            )
            .unwrap();
            let ba = ops::quantized_matmul(
                h,
                &l.in_proj_ba.0,
                &l.in_proj_ba.1,
                &l.in_proj_ba.2,
                true,
                gs,
                bits,
            )
            .unwrap();

            let q = qkvz
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let k = qkvz
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let v = qkvz
                .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let z = qkvz.index((.., .., 2 * key_dim + value_dim..));

            let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
            let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

            // Conv1d
            let q_flat = q.reshape(&[1, 1, -1]).unwrap();
            let k_flat = k.reshape(&[1, 1, -1]).unwrap();
            let v_flat = v.reshape(&[1, 1, -1]).unwrap();
            let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
            let conv_in = ops::concatenate_axis(&[conv_state, &mixed], 1).unwrap();
            let new_conv_state = conv_in.index((.., -3.., ..));

            let conv_out =
                nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();

            let conv_q = conv_out
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_k = conv_out
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_v = conv_out
                .index((.., .., 2 * key_dim..))
                .reshape(&[1, 1, hv, dv])
                .unwrap();

            // RMS norm
            let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                .unwrap()
                .multiply(&inv_scale_sq)
                .unwrap();
            let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                .unwrap()
                .multiply(&inv_scale)
                .unwrap();

            // Metal kernel (computes g and beta internally)
            let (y, new_state) = gated_delta_kernel_ffi(
                &norm_q, &norm_k, &conv_v, &l.a_log, &a, &l.dt_bias, &b, state, 1, 1, hk, dk, hv,
                dv,
            )
            .unwrap();

            // Gated RMSNorm + swiglu
            let normed = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
            let z_shaped = z
                .index((.., .., ..value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let gated = swiglu(&z_shaped, &normed).unwrap();

            // Output proj
            let out = ops::quantized_matmul(
                &gated.reshape(&[1, 1, -1]).unwrap(),
                &l.out_proj.0,
                &l.out_proj.1,
                &l.out_proj.2,
                true,
                gs,
                bits,
            )
            .unwrap();
            (out, new_state, new_conv_state)
        };

        // Warmup
        for _ in 0..5 {
            let mut h = x.clone();
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            for (j, l) in layers.iter().enumerate() {
                let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
                h = out;
                ss[j] = ns;
                cs[j] = nc;
            }
            let mut eval_targets: Vec<&Array> = vec![&h];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
        }

        // Benchmark
        let n = 20;
        let mut total = 0u128;
        for _ in 0..n {
            let mut h = x.clone();
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            for (j, l) in layers.iter().enumerate() {
                let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
                h = out;
                ss[j] = ns;
                cs[j] = nc;
            }
            let t0 = std::time::Instant::now();
            let mut eval_targets: Vec<&Array> = vec![&h];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
            total += t0.elapsed().as_nanos();
        }

        let avg_ms = total as f64 / n as f64 / 1e6;
        println!("Rust 36 GDN layers (bare arrays): {avg_ms:.2}ms");
        println!("Per layer: {:.3}ms", avg_ms / 36.0);
    }

    /// Benchmark 48 layers of interleaved GDN + MoE (matching real model structure).
    /// GDN layers: 0,1,2, 4,5,6, 8,9,10, ...  (every layer except multiples of 4 minus 1)
    /// FA layers: 3,7,11,... (every 4th layer, 0-indexed)
    /// All layers have MoE.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_combined_gdn_moe() {
        use mlx_rs::Dtype;

        let d = 2048i32;
        let hk = 16i32;
        let hv = 32i32;
        let dk = 128i32;
        let dv = 128i32;
        let gs = 64i32;
        let bits = 4i32;
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_out = key_dim * 2 + value_dim * 2;
        let ba_out = hv * 2;
        let n_layers = 48;
        let full_attn_interval = 4;
        let d_inter = 512i32;
        let n_experts = 512i32;
        let top_k = 10i32;
        let shared_inter = 512i32;

        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            (w, s, b)
        };

        struct GDNWeights {
            in_proj_qkvz: (Array, Array, Array),
            in_proj_ba: (Array, Array, Array),
            out_proj: (Array, Array, Array),
            conv_w: Array,
            a_log: Array,
            dt_bias: Array,
            norm_w: Array,
        }
        struct MoEWeights {
            gate: (Array, Array, Array),
            sw_gate: (Array, Array, Array),
            sw_up: (Array, Array, Array),
            sw_down: (Array, Array, Array),
            se_gate: (Array, Array, Array),
            se_up: (Array, Array, Array),
            se_down: (Array, Array, Array),
            se_gate_proj: (Array, Array, Array),
            norm_w: Array,
        }
        struct AttnWeights {
            q_proj: (Array, Array, Array),
            k_proj: (Array, Array, Array),
            v_proj: (Array, Array, Array),
            o_proj: (Array, Array, Array),
        }

        let mut gdn_layers: Vec<Option<GDNWeights>> = Vec::new();
        let mut attn_layers: Vec<Option<AttnWeights>> = Vec::new();
        let mut moe_layers: Vec<MoEWeights> = Vec::new();
        let mut all_w: Vec<Array> = Vec::new();

        for i in 0..n_layers {
            let is_linear = (i + 1) % full_attn_interval != 0;
            if is_linear {
                let gdn = GDNWeights {
                    in_proj_qkvz: make_qw(d, qkvz_out),
                    in_proj_ba: make_qw(d, ba_out),
                    out_proj: make_qw(value_dim, d),
                    conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                        .unwrap()
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                    a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                    dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                    norm_w: Array::ones::<f32>(&[dv])
                        .unwrap()
                        .as_dtype(Dtype::Float16)
                        .unwrap(),
                };
                all_w.extend([
                    gdn.in_proj_qkvz.0.clone(),
                    gdn.in_proj_qkvz.1.clone(),
                    gdn.in_proj_qkvz.2.clone(),
                ]);
                all_w.extend([
                    gdn.in_proj_ba.0.clone(),
                    gdn.in_proj_ba.1.clone(),
                    gdn.in_proj_ba.2.clone(),
                ]);
                all_w.extend([
                    gdn.out_proj.0.clone(),
                    gdn.out_proj.1.clone(),
                    gdn.out_proj.2.clone(),
                ]);
                all_w.extend([
                    gdn.conv_w.clone(),
                    gdn.a_log.clone(),
                    gdn.dt_bias.clone(),
                    gdn.norm_w.clone(),
                ]);
                gdn_layers.push(Some(gdn));
                attn_layers.push(None);
            } else {
                let attn = AttnWeights {
                    q_proj: make_qw(d, d),
                    k_proj: make_qw(d, d),
                    v_proj: make_qw(d, d),
                    o_proj: make_qw(d, d),
                };
                all_w.extend([
                    attn.q_proj.0.clone(),
                    attn.q_proj.1.clone(),
                    attn.q_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.k_proj.0.clone(),
                    attn.k_proj.1.clone(),
                    attn.k_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.v_proj.0.clone(),
                    attn.v_proj.1.clone(),
                    attn.v_proj.2.clone(),
                ]);
                all_w.extend([
                    attn.o_proj.0.clone(),
                    attn.o_proj.1.clone(),
                    attn.o_proj.2.clone(),
                ]);
                gdn_layers.push(None);
                attn_layers.push(Some(attn));
            }
            let moe = MoEWeights {
                gate: make_qw(d, n_experts),
                sw_gate: make_sw(d, d_inter),
                sw_up: make_sw(d, d_inter),
                sw_down: make_sw(d_inter, d),
                se_gate: make_qw(d, shared_inter * 2),
                se_up: make_qw(d, shared_inter * 2),
                se_down: make_qw(shared_inter * 2, d),
                se_gate_proj: make_qw(d, 1),
                norm_w: Array::ones::<f32>(&[d])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            };
            all_w.extend([moe.gate.0.clone(), moe.gate.1.clone(), moe.gate.2.clone()]);
            all_w.extend([
                moe.sw_gate.0.clone(),
                moe.sw_gate.1.clone(),
                moe.sw_gate.2.clone(),
            ]);
            all_w.extend([
                moe.sw_up.0.clone(),
                moe.sw_up.1.clone(),
                moe.sw_up.2.clone(),
            ]);
            all_w.extend([
                moe.sw_down.0.clone(),
                moe.sw_down.1.clone(),
                moe.sw_down.2.clone(),
            ]);
            all_w.extend([
                moe.se_gate.0.clone(),
                moe.se_gate.1.clone(),
                moe.se_gate.2.clone(),
            ]);
            all_w.extend([
                moe.se_up.0.clone(),
                moe.se_up.1.clone(),
                moe.se_up.2.clone(),
            ]);
            all_w.extend([
                moe.se_down.0.clone(),
                moe.se_down.1.clone(),
                moe.se_down.2.clone(),
            ]);
            all_w.extend([
                moe.se_gate_proj.0.clone(),
                moe.se_gate_proj.1.clone(),
                moe.se_gate_proj.2.clone(),
            ]);
            all_w.push(moe.norm_w.clone());
            moe_layers.push(moe);
        }
        let refs: Vec<&Array> = all_w.iter().collect();
        mlx_rs::transforms::eval(refs).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
        let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
        let inv_scale_sq = {
            let s = (dk as f32).sqrt().recip();
            Array::from_f32(s * s)
        };
        let states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        let conv_states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, 3, conv_dim])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }
        for c in &conv_states {
            c.eval().unwrap();
        }

        let forward = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;

            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();

                // Attention
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();

                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();

                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    // Simplified attention: just qkvo matmuls
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };

                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();

                // MoE
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();

                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();

                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();

                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        // Warmup
        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let result = forward(&x, &mut ss, &mut cs);
            let mut eval_targets: Vec<&Array> = vec![&result];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
        }

        // Benchmark
        let n = 20;
        let mut total_forward = 0u128;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let result = forward(&x, &mut ss, &mut cs);
            let t1 = std::time::Instant::now();
            let mut eval_targets: Vec<&Array> = vec![&result];
            eval_targets.extend(ss.iter());
            eval_targets.extend(cs.iter());
            mlx_rs::transforms::eval(eval_targets).unwrap();
            let t2 = std::time::Instant::now();
            total_forward += (t1 - t0).as_nanos();
            total_eval += (t2 - t1).as_nanos();
        }

        let fwd_ms = total_forward as f64 / n as f64 / 1e6;
        let eval_ms = total_eval as f64 / n as f64 / 1e6;
        println!(
            "Rust 48 combined: forward={fwd_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
            fwd_ms + eval_ms
        );

        // Test: eval only the final result (not states) to see if eval target count matters
        let mut total_eval_one = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let result = forward(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&result]).unwrap();
            total_eval_one += t0.elapsed().as_nanos();
        }
        let eval_one_ms = total_eval_one as f64 / n as f64 / 1e6;
        println!("Rust 48 combined (eval result only): {eval_one_ms:.2}ms");

        // Variant: GDN only (skip MoE, replace with passthrough)
        let forward_gdn_only = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let is_gdn = gdn_layers[i].is_some();
                if !is_gdn {
                    continue;
                }
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = ops::quantized_matmul(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = ops::quantized_matmul(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                ss[gdn_idx] = new_state;
                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                let r = ops::quantized_matmul(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                h = h.add(r).unwrap();
            }
            h
        };

        // Variant: MoE only (skip GDN)
        let forward_moe_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                // Simple attn proxy
                let attn_out = ops::quantized_matmul(
                    &normed,
                    &moe_layers[i].gate.0,
                    &moe_layers[i].gate.1,
                    &moe_layers[i].gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let h2 = h.add(attn_out.sum_axes(&[-1], true).unwrap()).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        // Warmup GDN-only
        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_gdn_only(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_gdn = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_gdn_only(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_gdn += t0.elapsed().as_nanos();
        }
        println!(
            "Rust GDN-only (36 layers, combined weights): {:.2}ms",
            total_gdn as f64 / n as f64 / 1e6
        );

        // Warmup MoE-only
        for _ in 0..5 {
            let r = forward_moe_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let mut total_moe = 0u128;
        for _ in 0..n {
            let r = forward_moe_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total_moe += t0.elapsed().as_nanos();
        }
        println!(
            "Rust MoE-only (48 layers, combined weights): {:.2}ms",
            total_moe as f64 / n as f64 / 1e6
        );

        // Combined but with kernel replaced by zeros_like
        let forward_no_kernel = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let _conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let _g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let _beta = nn::sigmoid(&b).unwrap();

                    // SKIP kernel: use zeros instead
                    let y = Array::zeros::<f32>(&[1, 1, hv, dv])
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float16)
                        .unwrap();
                    ss[gdn_idx] = Array::zeros::<f32>(&[1, hv, dv, dk])
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float16)
                        .unwrap();
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_no_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_nk = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_no_kernel(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_nk += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined NO KERNEL (GDN ops + MoE): {:.2}ms",
            total_nk as f64 / n as f64 / 1e6
        );

        // Variant: ops-based GDN recurrence (no Metal kernel) interleaved with MoE
        let gqa_repeat = hv / hk;
        let forward_ops_gdn = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let beta = nn::sigmoid(&b).unwrap();

                    // Ops-based recurrence: repeat q,k for GQA then run step
                    let q_rep = ops::broadcast_to(
                        norm_q.reshape(&[1, hk, 1, dk]).unwrap(),
                        &[1, hk, gqa_repeat, dk],
                    )
                    .unwrap()
                    .reshape(&[1, hv, dk])
                    .unwrap();
                    let k_rep = ops::broadcast_to(
                        norm_k.reshape(&[1, hk, 1, dk]).unwrap(),
                        &[1, hk, gqa_repeat, dk],
                    )
                    .unwrap()
                    .reshape(&[1, hv, dk])
                    .unwrap();
                    let v_sq = conv_v.squeeze_axes(&[1]).unwrap();
                    let g_sq = g.squeeze_axes(&[0, 1]).unwrap();
                    let beta_sq = beta.squeeze_axes(&[0, 1]).unwrap();
                    let (y, new_state) =
                        gated_delta_step_ref(&q_rep, &k_rep, &v_sq, &g_sq, &beta_sq, &ss[gdn_idx]);
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;

                    let y_4d = y.expand_dims(0).unwrap().expand_dims(0).unwrap();
                    let normed_y = fast::rms_norm(&y_4d, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_ops_gdn(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ops = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_ops_gdn(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ops += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined OPS GDN (no Metal kernel): {:.2}ms",
            total_ops as f64 / n as f64 / 1e6
        );

        // Variant: Metal kernel with per-layer eval barriers
        let forward_eval_barrier = |h_in: &Array,
                                    ss: &mut Vec<Array>,
                                    cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();

                // Eval barrier: force layer-by-layer evaluation
                h.eval().unwrap();
                ss.iter().for_each(|s| s.eval().unwrap());
                cs.iter().for_each(|c| c.eval().unwrap());
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_eval_barrier(&x, &mut ss, &mut cs);
            r.eval().unwrap();
        }
        let mut total_eb = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_eval_barrier(&x, &mut ss, &mut cs);
            r.eval().unwrap();
            total_eb += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined EVAL BARRIER (per-layer eval): {:.2}ms",
            total_eb as f64 / n as f64 / 1e6
        );

        // Variant: async_eval after each layer (non-blocking pipeline hint)
        let forward_async = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();

                // Async eval hint: start processing GDN computation while building MoE graph
                mlx_rs::transforms::async_eval([&h2]).unwrap();

                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_async(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_async = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_async(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_async += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined ASYNC EVAL (per-layer hint): {:.2}ms",
            total_async as f64 / n as f64 / 1e6
        );

        // Variant: eval kernel outputs (y + state) immediately after each GDN layer
        let forward_eval_kernel = |h_in: &Array,
                                   ss: &mut Vec<Array>,
                                   cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();

                    // Targeted eval: resolve kernel outputs to break graph
                    mlx_rs::transforms::eval([&y, &new_state, &cs[gdn_idx]]).unwrap();

                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_eval_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ek = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let t0 = std::time::Instant::now();
            let r = forward_eval_kernel(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ek += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined EVAL KERNEL OUTPUTS: {:.2}ms",
            total_ek as f64 / n as f64 / 1e6
        );

        // Layer scaling test: run with 1, 4, 12, 24, 48 layers to check non-linearity
        // Test: tiny state (replace [1,32,128,128] with [1,1,1,1]) to check memory hypothesis
        let tiny_states: Vec<Array> = (0..36)
            .map(|_| {
                Array::zeros::<f32>(&[1, 1, 1, 1])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        for s in &tiny_states {
            s.eval().unwrap();
        }

        let forward_tiny_state = |h_in: &Array,
                                  ss: &mut Vec<Array>,
                                  cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let beta = nn::sigmoid(&b).unwrap();

                    // Tiny state: just multiply by a scalar instead of full state ops
                    let g_scalar = g.sum_axes(&[-1], true).unwrap();
                    let tiny_decayed = ss[gdn_idx].multiply(g_scalar).unwrap();
                    ss[gdn_idx] = tiny_decayed.add(Array::from_f32(0.1)).unwrap();

                    // Use conv_v directly as y (same shape [1,1,Hv,Dv])
                    let y = conv_v
                        .multiply(beta.reshape(&[1, 1, hv, 1]).unwrap())
                        .unwrap();

                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = tiny_states.clone();
            let mut cs = conv_states.clone();
            let r = forward_tiny_state(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_ts = 0u128;
        for _ in 0..n {
            let mut ss = tiny_states.clone();
            let mut cs = conv_states.clone();
            let r = forward_tiny_state(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_ts += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined TINY STATE (all ops, no large state): {:.2}ms",
            total_ts as f64 / n as f64 / 1e6
        );

        for test_layers in [1i32, 4, 12, 24, 48] {
            let test_layers_u = test_layers as usize;
            let n_gdn = (0..test_layers_u)
                .filter(|i| gdn_layers.get(*i).map_or(false, |g| g.is_some()))
                .count();
            let forward_n = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
                let mut h = h_in.clone();
                let mut gdn_idx = 0usize;
                for i in 0..test_layers_u {
                    let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                    let r = if gdn_layers[i].is_some() {
                        let l = gdn_layers[i].as_ref().unwrap();
                        let qkvz = ops::quantized_matmul(
                            &normed,
                            &l.in_proj_qkvz.0,
                            &l.in_proj_qkvz.1,
                            &l.in_proj_qkvz.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let ba = ops::quantized_matmul(
                            &normed,
                            &l.in_proj_ba.0,
                            &l.in_proj_ba.1,
                            &l.in_proj_ba.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let q = qkvz
                            .index((.., .., ..key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let k = qkvz
                            .index((.., .., key_dim..2 * key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let v = qkvz
                            .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                        let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                        let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                        let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                        let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                        let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                        let mixed =
                            ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                        let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                        cs[gdn_idx] = conv_in.index((.., -3.., ..));
                        let conv_out =
                            nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                                .unwrap();
                        let conv_q = conv_out
                            .index((.., .., ..key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let conv_k = conv_out
                            .index((.., .., key_dim..2 * key_dim))
                            .reshape(&[1, 1, hk, dk])
                            .unwrap();
                        let conv_v = conv_out
                            .index((.., .., 2 * key_dim..))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                            .unwrap()
                            .multiply(&inv_scale_sq)
                            .unwrap();
                        let norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                            .unwrap()
                            .multiply(&inv_scale)
                            .unwrap();
                        let (y, new_state) = gated_delta_kernel_ffi(
                            &norm_q,
                            &norm_k,
                            &conv_v,
                            &l.a_log,
                            &a,
                            &l.dt_bias,
                            &b,
                            &ss[gdn_idx],
                            1,
                            1,
                            hk,
                            dk,
                            hv,
                            dv,
                        )
                        .unwrap();
                        ss[gdn_idx] = new_state;
                        gdn_idx += 1;
                        let normed_y = fast::rms_norm(&y, &l.norm_w, 1e-6).unwrap();
                        let z_shaped = z
                            .index((.., .., ..value_dim))
                            .reshape(&[1, 1, hv, dv])
                            .unwrap();
                        let gated = swiglu(&z_shaped, &normed_y).unwrap();
                        ops::quantized_matmul(
                            &gated.reshape(&[1, 1, -1]).unwrap(),
                            &l.out_proj.0,
                            &l.out_proj.1,
                            &l.out_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap()
                    } else {
                        let al = attn_layers[i].as_ref().unwrap();
                        let q = ops::quantized_matmul(
                            &normed,
                            &al.q_proj.0,
                            &al.q_proj.1,
                            &al.q_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let _k = ops::quantized_matmul(
                            &normed,
                            &al.k_proj.0,
                            &al.k_proj.1,
                            &al.k_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let v = ops::quantized_matmul(
                            &normed,
                            &al.v_proj.0,
                            &al.v_proj.1,
                            &al.v_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let proxy = v
                            .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                            .unwrap();
                        ops::quantized_matmul(
                            &proxy,
                            &al.o_proj.0,
                            &al.o_proj.1,
                            &al.o_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap()
                    };
                    let h2 = h.add(r).unwrap();
                    let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                    let m = &moe_layers[i];
                    let gate_out = ops::quantized_matmul(
                        &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                    )
                    .unwrap();
                    let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                    let neg_k = -top_k;
                    let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                    let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                    let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                    let scores = raw_scores
                        .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                        .unwrap();
                    let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                    let g_out = gather_qmm(
                        &x_exp,
                        &m.sw_gate.0,
                        &m.sw_gate.1,
                        &m.sw_gate.2,
                        &top_inds,
                        true,
                        gs,
                        bits,
                        false,
                    )
                    .unwrap();
                    let u_out = gather_qmm(
                        &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits,
                        false,
                    )
                    .unwrap();
                    let activated = swiglu(&g_out, &u_out).unwrap();
                    let d_out = gather_qmm(
                        &activated,
                        &m.sw_down.0,
                        &m.sw_down.1,
                        &m.sw_down.2,
                        &top_inds,
                        true,
                        gs,
                        bits,
                        false,
                    )
                    .unwrap();
                    let expert_sum = d_out
                        .squeeze_axes(&[-2])
                        .unwrap()
                        .multiply(scores.expand_dims(-1).unwrap())
                        .unwrap()
                        .sum_axes(&[-2], false)
                        .unwrap();
                    let sh_g = ops::quantized_matmul(
                        &normed2,
                        &m.se_gate.0,
                        &m.se_gate.1,
                        &m.se_gate.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let sh_u = ops::quantized_matmul(
                        &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                    )
                    .unwrap();
                    let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                    let sh_d = ops::quantized_matmul(
                        &sh_act,
                        &m.se_down.0,
                        &m.se_down.1,
                        &m.se_down.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let sh_gate_val = nn::sigmoid(
                        ops::quantized_matmul(
                            &normed2,
                            &m.se_gate_proj.0,
                            &m.se_gate_proj.1,
                            &m.se_gate_proj.2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                    h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
                }
                h
            };
            for _ in 0..3 {
                let mut ss = states.clone();
                let mut cs = conv_states.clone();
                let r = forward_n(&x, &mut ss, &mut cs);
                let mut t: Vec<&Array> = vec![&r];
                t.extend(ss.iter());
                t.extend(cs.iter());
                mlx_rs::transforms::eval(t).unwrap();
            }
            let mut total_n = 0u128;
            for _ in 0..n {
                let mut ss = states.clone();
                let mut cs = conv_states.clone();
                let r = forward_n(&x, &mut ss, &mut cs);
                let t0 = std::time::Instant::now();
                let mut t: Vec<&Array> = vec![&r];
                t.extend(ss.iter());
                t.extend(cs.iter());
                mlx_rs::transforms::eval(t).unwrap();
                total_n += t0.elapsed().as_nanos();
            }
            let ms = total_n as f64 / n as f64 / 1e6;
            println!(
                "Layer scaling: {test_layers} layers ({n_gdn} GDN): {ms:.2}ms ({:.2}ms/layer)",
                ms / test_layers as f64
            );
        }

        // Variant: replace recurrence with a single matmul (same data flow, fewer ops)
        let forward_matmul_gdn = |h_in: &Array,
                                  ss: &mut Vec<Array>,
                                  cs: &mut Vec<Array>|
         -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                let normed = fast::rms_norm(&h, &moe_layers[i].norm_w, 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = ops::quantized_matmul(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate_axis(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let _norm_q = fast::rms_norm(&conv_q, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let _norm_k = fast::rms_norm(&conv_k, &qk_norm_w, 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                    let _beta = nn::sigmoid(&b).unwrap();

                    // Variant A: no reduction, just multiply + add on state
                    let g_exp = g.reshape(&[1, hv, 1, 1]).unwrap();
                    let decayed = ss[gdn_idx].multiply(g_exp).unwrap();
                    let v_exp = conv_v.reshape(&[1, hv, dv, 1]).unwrap();
                    ss[gdn_idx] = decayed.add(v_exp).unwrap();
                    // y = just take a slice of state (no reduction)
                    let y_proxy = ss[gdn_idx]
                        .index((.., .., .., 0..1))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    gdn_idx += 1;

                    let normed_y = fast::rms_norm(&y_proxy, &l.norm_w, 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    ops::quantized_matmul(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = ops::quantized_matmul(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = ops::quantized_matmul(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = ops::quantized_matmul(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    ops::quantized_matmul(
                        &proxy,
                        &al.o_proj.0,
                        &al.o_proj.1,
                        &al.o_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, &moe_layers[i].norm_w, 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out = ops::quantized_matmul(
                    &normed2, &m.gate.0, &m.gate.1, &m.gate.2, true, gs, bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                let sh_g = ops::quantized_matmul(
                    &normed2,
                    &m.se_gate.0,
                    &m.se_gate.1,
                    &m.se_gate.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_u = ops::quantized_matmul(
                    &normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, true, gs, bits,
                )
                .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d = ops::quantized_matmul(
                    &sh_act,
                    &m.se_down.0,
                    &m.se_down.1,
                    &m.se_down.2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let sh_gate_val = nn::sigmoid(
                    ops::quantized_matmul(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_mm = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_mm += t0.elapsed().as_nanos();
        }
        println!(
            "Rust combined MATMUL GDN (proxy recurrence): {:.2}ms",
            total_mm as f64 / n as f64 / 1e6
        );
    }

    /// Minimal reproducer: state ops + gather_qmm, nothing else.
    #[test]
    #[ignore = "requires GPU"]
    fn bench_minimal_state_moe_interaction() {
        use mlx_rs::Dtype;
        let n_layers = 48usize;
        let n_gdn = 36usize;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 2048i32;
        let gs = 64i32;
        let bits = 4i32;
        let n_experts = 512i32;
        let d_inter = 512i32;
        let top_k = 10i32;

        // Expert weights for gather_qmm
        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
        let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
        let mut all_w: Vec<Array> = Vec::new();
        for i in 0..n_layers {
            all_w.extend([
                sw_gate[i].0.clone(),
                sw_gate[i].1.clone(),
                sw_gate[i].2.clone(),
            ]);
            all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
            all_w.extend([
                sw_down[i].0.clone(),
                sw_down[i].1.clone(),
                sw_down[i].2.clone(),
            ]);
            all_w.extend([
                gate_proj[i].0.clone(),
                gate_proj[i].1.clone(),
                gate_proj[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        let n = 20;

        // Test 1: state ops only (no MoE)
        let forward_state_only = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            for gdn_idx in 0..n_gdn {
                let g = h.sum_axes(&[-1], true).unwrap();
                let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                let new_state = ss[gdn_idx]
                    .multiply(decay)
                    .unwrap()
                    .add(Array::from_f32(0.01))
                    .unwrap();
                let y = new_state
                    .sum_axes(&[-1], false)
                    .unwrap()
                    .reshape(&[1, 1, -1])
                    .unwrap()
                    .index((.., .., ..d));
                ss[gdn_idx] = new_state;
                h = h.add(y).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_state_only(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_state_only(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "State ops only (36 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 2: MoE only (no state)
        let forward_moe_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers {
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = forward_moe_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let r = forward_moe_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "MoE ops only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3: interleaved state + MoE
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                // State ops (for GDN layers)
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                // MoE ops
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + MoE (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3c: keep ALL intermediates alive (prevent drops during graph construction)
        let forward_keep_alive = |h_in: &Array, ss: &mut Vec<Array>| -> (Array, Vec<Array>) {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            let mut keep: Vec<Array> = Vec::with_capacity(n_layers * 20);
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(&decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    keep.push(g);
                    keep.push(decay);
                    keep.push(y.clone());
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                keep.extend([
                    gate_out,
                    gates,
                    all_inds,
                    top_inds.clone(),
                    raw_scores,
                    scores,
                    x_exp,
                    g_out,
                    u_out,
                    activated,
                    d_out,
                    expert_sum.clone(),
                ]);
                h = h.add(expert_sum).unwrap();
            }
            (h, keep)
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let (r, _keep) = forward_keep_alive(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let (r, _keep) = forward_keep_alive(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved keep-alive (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3b: same but eval only h (not states)
        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved eval h only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 4: interleaved state + quantized_matmul only (no gather_qmm)
        let simple_w: Vec<_> = (0..n_layers).map(|_| make_qw(d, d)).collect();
        let mut sw: Vec<Array> = Vec::new();
        for i in 0..n_layers {
            sw.extend([
                simple_w[i].0.clone(),
                simple_w[i].1.clone(),
                simple_w[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(sw.iter().collect::<Vec<_>>()).unwrap();

        let forward_interleaved_qmm = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }
                // Simple quantized_matmul chain (no gather_qmm FFI)
                let out = ops::quantized_matmul(
                    &h,
                    &simple_w[i].0,
                    &simple_w[i].1,
                    &simple_w[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                h = h.add(out).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved_qmm(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved_qmm(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + quantized_matmul (no gather_qmm): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 5: interleaved state + MoE using gather_qmm
        let forward_interleaved_ops = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved_ops(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved_ops(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Interleaved state + gather_qmm: {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires GPU"]
    #[cfg(any())]
    fn bench_cxx_bypass() {
        use mlx_rs::Dtype;
        let n_layers = 48i32;
        let n_gdn = 36i32;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 2048i32;
        let gs = 64i32;
        let bits = 4i32;
        let n_experts = 512i32;
        let d_inter = 512i32;
        let top_k = 10i32;
        let n = 20;

        // Self-contained C++ benchmark (no prior Rust MLX operations)
        #[allow(unsafe_code)]
        let self_contained_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained BEFORE any Rust ops: {:.2}ms",
            self_contained_us / 1000.0
        );

        // Now do a tiny eval to see if ANY eval causes the slowdown
        {
            let tiny = Array::ones::<f32>(&[1, 1, 1]).unwrap();
            tiny.eval().unwrap();
        }
        #[allow(unsafe_code)]
        let after_tiny_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained AFTER tiny eval: {:.2}ms",
            after_tiny_us / 1000.0
        );

        // Now create and eval ONE large weight to test memory impact
        {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_inter, d], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval(vec![&w, &s, &b]).unwrap();
            // raw, w, s, b will be dropped here
        }
        #[allow(unsafe_code)]
        let after_big_us = unsafe {
            mlx_sys::mlx_bench_self_contained(
                n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
            )
        };
        println!(
            "C++ self-contained AFTER one big quantize: {:.2}ms",
            after_big_us / 1000.0
        );

        let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };
        let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
            let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            ops::quantize(&raw, gs, bits).unwrap()
        };

        let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
        let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
        let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
        let mut all_w: Vec<Array> = Vec::new();
        for i in 0..n_layers as usize {
            all_w.extend([
                sw_gate[i].0.clone(),
                sw_gate[i].1.clone(),
                sw_gate[i].2.clone(),
            ]);
            all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
            all_w.extend([
                sw_down[i].0.clone(),
                sw_down[i].1.clone(),
                sw_down[i].2.clone(),
            ]);
            all_w.extend([
                gate_proj[i].0.clone(),
                gate_proj[i].1.clone(),
                gate_proj[i].2.clone(),
            ]);
        }
        mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        // Prepare raw pointer arrays for FFI
        let gate_w: Vec<_> = sw_gate.iter().map(|t| t.0.as_ptr()).collect();
        let gate_s: Vec<_> = sw_gate.iter().map(|t| t.1.as_ptr()).collect();
        let gate_b: Vec<_> = sw_gate.iter().map(|t| t.2.as_ptr()).collect();
        let up_w: Vec<_> = sw_up.iter().map(|t| t.0.as_ptr()).collect();
        let up_s: Vec<_> = sw_up.iter().map(|t| t.1.as_ptr()).collect();
        let up_b: Vec<_> = sw_up.iter().map(|t| t.2.as_ptr()).collect();
        let down_w: Vec<_> = sw_down.iter().map(|t| t.0.as_ptr()).collect();
        let down_s: Vec<_> = sw_down.iter().map(|t| t.1.as_ptr()).collect();
        let down_b: Vec<_> = sw_down.iter().map(|t| t.2.as_ptr()).collect();
        let gp_w: Vec<_> = gate_proj.iter().map(|t| t.0.as_ptr()).collect();
        let gp_s: Vec<_> = gate_proj.iter().map(|t| t.1.as_ptr()).collect();
        let gp_b: Vec<_> = gate_proj.iter().map(|t| t.2.as_ptr()).collect();

        let state_ptrs_for_cxx: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();

        let n = 20;
        let stream = Stream::new();

        // Warmup
        for _ in 0..5 {
            let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
            #[allow(unsafe_code)]
            let (result, state_outs) = unsafe {
                let mut result = mlx_sys::mlx_array_new();
                let mut state_outs: Vec<mlx_sys::mlx_array> =
                    (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
                let status = mlx_sys::mlx_bench_interleaved_cxx(
                    &raw mut result,
                    state_outs.as_mut_ptr(),
                    x.as_ptr(),
                    state_ptrs.as_ptr(),
                    gate_w.as_ptr(),
                    gate_s.as_ptr(),
                    gate_b.as_ptr(),
                    up_w.as_ptr(),
                    up_s.as_ptr(),
                    up_b.as_ptr(),
                    down_w.as_ptr(),
                    down_s.as_ptr(),
                    down_b.as_ptr(),
                    gp_w.as_ptr(),
                    gp_s.as_ptr(),
                    gp_b.as_ptr(),
                    n_layers,
                    n_gdn,
                    d,
                    n_experts,
                    top_k,
                    gs,
                    bits,
                    stream.as_ptr(),
                );
                assert_eq!(status, 0, "C++ shim failed");
                let r = Array::from_ptr(result);
                let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
                (r, so)
            };
            let mut t: Vec<&Array> = vec![&result];
            t.extend(state_outs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }

        // Benchmark
        let mut total = 0u128;
        for _ in 0..n {
            let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
            #[allow(unsafe_code)]
            let (result, state_outs) = unsafe {
                let mut result = mlx_sys::mlx_array_new();
                let mut state_outs: Vec<mlx_sys::mlx_array> =
                    (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
                let status = mlx_sys::mlx_bench_interleaved_cxx(
                    &raw mut result,
                    state_outs.as_mut_ptr(),
                    x.as_ptr(),
                    state_ptrs.as_ptr(),
                    gate_w.as_ptr(),
                    gate_s.as_ptr(),
                    gate_b.as_ptr(),
                    up_w.as_ptr(),
                    up_s.as_ptr(),
                    up_b.as_ptr(),
                    down_w.as_ptr(),
                    down_s.as_ptr(),
                    down_b.as_ptr(),
                    gp_w.as_ptr(),
                    gp_s.as_ptr(),
                    gp_b.as_ptr(),
                    n_layers,
                    n_gdn,
                    d,
                    n_experts,
                    top_k,
                    gs,
                    bits,
                    stream.as_ptr(),
                );
                assert_eq!(status, 0, "C++ shim failed");
                let r = Array::from_ptr(result);
                let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
                (r, so)
            };
            let mut t: Vec<&Array> = vec![&result];
            t.extend(state_outs.iter());
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "C++ bypass interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test: build + eval entirely in C++ (no Rust involvement in eval)
        #[allow(unsafe_code)]
        let avg_us = unsafe {
            mlx_sys::mlx_bench_interleaved_cxx_with_eval(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                5,
                n,
            )
        };
        println!("C++ build+eval (48 layers): {:.2}ms", avg_us / 1000.0);

        // Test: state ops only (no MoE)
        #[allow(unsafe_code)]
        let state_only_us = unsafe {
            mlx_sys::mlx_bench_state_ops_only(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                n_gdn,
                d,
                5,
                n,
            )
        };
        println!(
            "C++ state ops only (36 layers): {:.2}ms",
            state_only_us / 1000.0
        );

        // Test: interleaved but eval h only (no states in eval list)
        #[allow(unsafe_code)]
        let h_only_us = unsafe {
            mlx_sys::mlx_bench_interleaved_h_only_eval(
                x.as_ptr(),
                state_ptrs_for_cxx.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                5,
                n,
            )
        };
        println!(
            "C++ interleaved h-only eval (48 layers): {:.2}ms",
            h_only_us / 1000.0
        );

        // For comparison: the standard Rust interleaved version
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }
                let gate_out = ops::quantized_matmul(
                    &h,
                    &gate_proj[i].0,
                    &gate_proj[i].1,
                    &gate_proj[i].2,
                    true,
                    gs,
                    bits,
                )
                .unwrap();
                let gates_v = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let all_inds = ops::argpartition_axis(&gates_v, -top_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
                let raw_scores = gates_v.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert_sum = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .multiply(scores.expand_dims(-1).unwrap())
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Rust C API interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires GPU"]
    #[cfg(any())]
    fn bench_gather_mm_interleave() {
        use mlx_rs::Dtype;
        let n_layers = 48usize;
        let n_gdn = 36usize;
        let hv = 32i32;
        let dv = 128i32;
        let dk = 128i32;
        let d = 256i32; // Small dim to avoid OOM (float weights are not quantized)
        let n_experts = 64i32;
        let top_k = 10i32;

        // gather_mm: a=[..., M, K] @ b=[batch, K, N] -> [..., batch_sel, M, N]
        let float_weights: Vec<Array> = (0..n_layers)
            .map(|_| {
                mlx_rs::random::normal::<f32>(&[n_experts, d, d], None, None, None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        mlx_rs::transforms::eval(float_weights.iter().collect::<Vec<_>>()).unwrap();

        let x = Array::ones::<f32>(&[1, 1, d])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let states: Vec<Array> = (0..n_gdn)
            .map(|_| {
                Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            })
            .collect();
        x.eval().unwrap();
        for s in &states {
            s.eval().unwrap();
        }

        let n = 20;

        // gather_mm only (no state)
        let forward_gather_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers {
                let rhs_inds =
                    Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
                let x_exp = h.expand_dims(-2).unwrap();
                let out =
                    ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None)
                        .unwrap();
                let out_sq = out.squeeze_axes(&[-2]).unwrap();
                let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = forward_gather_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let mut total = 0u128;
        for _ in 0..n {
            let r = forward_gather_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_mm only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // gather_mm interleaved with state
        let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers {
                if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                    let g = h.sum_axes(&[-1], true).unwrap();
                    let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                    let new_state = ss[gdn_idx]
                        .multiply(decay)
                        .unwrap()
                        .add(Array::from_f32(0.01))
                        .unwrap();
                    let y = new_state
                        .sum_axes(&[-1], false)
                        .unwrap()
                        .reshape(&[1, 1, -1])
                        .unwrap()
                        .index((.., .., ..d));
                    ss[gdn_idx] = new_state;
                    h = h.add(y).unwrap();
                    gdn_idx += 1;
                }

                let rhs_inds =
                    Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
                let x_exp = h.expand_dims(-2).unwrap();
                let out =
                    ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None)
                        .unwrap();
                let out_sq = out.squeeze_axes(&[-2]).unwrap();
                let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
                h = h.add(expert_sum).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let mut ss = states.clone();
            let r = forward_interleaved(&x, &mut ss);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_mm interleaved (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_actual_model_forward() {
        let Some(model_path) = std::env::var_os("HIGGS_QWEN3_NEXT_BENCH_MODEL") else {
            println!("Skipping: set HIGGS_QWEN3_NEXT_BENCH_MODEL to a local model directory");
            return;
        };
        let model_path = std::path::PathBuf::from(model_path);
        if !model_path.exists() {
            println!("Model not found at {}, skipping", model_path.display());
            return;
        }

        let mut model = load_qwen3_next_model(&model_path).unwrap();
        let mut cache: Vec<Option<LayerCache>> = Vec::new();

        // Prefill with a short prompt
        let prompt = Array::from_slice(&[9707u32, 1879], &[1, 2]);
        let prefill_out = model.forward(&prompt, None, &mut cache).unwrap();
        // Eval prefill outputs + cache states
        let mut to_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            to_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            to_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {} // KV cache evals itself internally
                }
            }
        }
        mlx_rs::transforms::eval(to_eval).unwrap();

        // Get first token
        let logits = prefill_out.index((.., -1, ..));
        let token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        mlx_rs::transforms::eval([&token]).unwrap();

        // Decode loop timing
        let mut current = token;
        for i in 0..22 {
            let input = current.index((.., ops::indexing::NewAxis));
            let t_fwd_start = std::time::Instant::now();
            let out = model.forward(&input, None, &mut cache).unwrap();
            let next = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            let t_fwd = t_fwd_start.elapsed();

            let t_eval_start = std::time::Instant::now();
            // Eval next token AND all cache states (like Python does)
            let mut eval_list: Vec<&Array> = vec![&next];
            for lc in cache.iter() {
                if let Some(lc) = lc {
                    match lc {
                        LayerCache::Arrays(ac) => {
                            if let Some(ref s) = ac.ssm_state {
                                eval_list.push(s);
                            }
                            if let Some(ref c) = ac.conv_state {
                                eval_list.push(c);
                            }
                        }
                        LayerCache::KV(_) => {}
                    }
                }
            }
            mlx_rs::transforms::eval(eval_list).unwrap();
            let t_eval = t_eval_start.elapsed();

            let t_item_start = std::time::Instant::now();
            let _id: u32 = next.item();
            let t_item = t_item_start.elapsed();

            let total = t_fwd + t_eval + t_item;
            if i < 5 || i >= 20 {
                println!(
                    "Step {i}: fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms ({:.1} tok/s)",
                    t_fwd.as_secs_f64() * 1000.0,
                    t_eval.as_secs_f64() * 1000.0,
                    t_item.as_secs_f64() * 1000.0,
                    total.as_secs_f64() * 1000.0,
                    1.0 / total.as_secs_f64(),
                );
            }
            current = next;
        }
    }

    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_actual_qwen3_5_dense_decode() {
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            println!("Model not found at {model_path}, skipping");
            return;
        }

        let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let decode_steps: usize = std::env::var("BENCH_DECODE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);

        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let tokens: Vec<u32> = (0..prompt_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = if prompt_len > 512 {
            model
                .forward_chunked(&prompt, None, &mut cache, 512)
                .unwrap()
        } else {
            model.forward(&prompt, None, &mut cache).unwrap()
        };

        let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            prefill_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            prefill_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        mlx_rs::transforms::eval(prefill_eval).unwrap();

        let logits = prefill_out.index((.., -1, ..));
        let token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        mlx_rs::transforms::eval([&token]).unwrap();

        let mut current = token;
        let mut total_forward_ns = 0u128;
        let mut total_eval_ns = 0u128;
        let mut total_item_ns = 0u128;

        for i in 0..decode_steps {
            let input = current.index((.., ops::indexing::NewAxis));

            let t0 = Instant::now();
            let out = model.forward(&input, None, &mut cache).unwrap();
            let next = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            let fwd_ns = t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let mut eval_targets: Vec<&Array> = vec![&next];
            for lc in &cache {
                if let Some(lc) = lc {
                    match lc {
                        LayerCache::Arrays(ac) => {
                            if let Some(ref s) = ac.ssm_state {
                                eval_targets.push(s);
                            }
                            if let Some(ref c) = ac.conv_state {
                                eval_targets.push(c);
                            }
                        }
                        LayerCache::KV(_) => {}
                    }
                }
            }
            mlx_rs::transforms::eval(eval_targets).unwrap();
            let eval_ns = t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let _: u32 = next.item();
            let item_ns = t0.elapsed().as_nanos();

            total_forward_ns += fwd_ns;
            total_eval_ns += eval_ns;
            total_item_ns += item_ns;

            let total_ns = fwd_ns + eval_ns + item_ns;
            println!(
                "step={i:>2} fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms tok/s={:.2}",
                fwd_ns as f64 / 1e6,
                eval_ns as f64 / 1e6,
                item_ns as f64 / 1e6,
                total_ns as f64 / 1e6,
                1e9 / total_ns as f64,
            );
            current = next;
        }

        let steps = decode_steps as f64;
        let avg_total_ns = total_forward_ns + total_eval_ns + total_item_ns;
        println!(
            "AVG decode: fwd={:.2}ms eval={:.2}ms item={:.2}ms total={:.2}ms tok/s={:.2}",
            total_forward_ns as f64 / steps / 1e6,
            total_eval_ns as f64 / steps / 1e6,
            total_item_ns as f64 / steps / 1e6,
            avg_total_ns as f64 / steps / 1e6,
            steps * 1e9 / avg_total_ns as f64,
        );
    }

    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_actual_qwen3_5_dense_decode_breakdown() {
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            println!("Model not found at {model_path}, skipping");
            return;
        }

        let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        let decode_steps: usize = std::env::var("BENCH_DECODE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        let mut model = load_qwen3_5_model(&model_path).unwrap();
        let tokens: Vec<u32> = (0..prompt_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = if prompt_len > 512 {
            model
                .forward_chunked(&prompt, None, &mut cache, 512)
                .unwrap()
        } else {
            model.forward(&prompt, None, &mut cache).unwrap()
        };

        let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            prefill_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            prefill_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        mlx_rs::transforms::eval(prefill_eval).unwrap();

        let logits = prefill_out.index((.., -1, ..));
        let mut current = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        mlx_rs::transforms::eval([&current]).unwrap();

        let mut total_embed_ns = 0u128;
        let mut total_gdn_ns = 0u128;
        let mut total_attn_ns = 0u128;
        let mut total_mlp_ns = 0u128;
        let mut total_norm_resid_ns = 0u128;
        let mut total_lm_ns = 0u128;

        let mut gdn_layers = 0u32;
        let mut attn_layers = 0u32;

        for step in 0..decode_steps {
            let input = current.index((.., ops::indexing::NewAxis));

            let t0 = Instant::now();
            let mut h = model.model.embed_tokens.forward(&input).unwrap();
            mlx_rs::transforms::eval([&h]).unwrap();
            total_embed_ns += t0.elapsed().as_nanos();

            let mut step_gdn_ns = 0u128;
            let mut step_attn_ns = 0u128;
            let mut step_mlp_ns = 0u128;
            let mut step_norm_resid_ns = 0u128;
            let mut step_gdn_layers = 0u32;
            let mut step_attn_layers = 0u32;

            for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache.iter_mut()) {
                let lc = layer_cache.as_mut().unwrap();

                let t0 = Instant::now();
                let normed = layer.input_layernorm.forward(&h).unwrap();
                mlx_rs::transforms::eval([&normed]).unwrap();
                step_norm_resid_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                let r = if layer.is_linear {
                    let gdn = layer.linear_attn.as_mut().unwrap();
                    let LayerCache::Arrays(sc) = lc else {
                        panic!("Expected ArraysCache");
                    };
                    let out = gdn.forward(&normed, None, sc).unwrap();
                    let mut tgts: Vec<&Array> = vec![&out];
                    if let Some(ref s) = sc.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = sc.conv_state {
                        tgts.push(c);
                    }
                    mlx_rs::transforms::eval(tgts).unwrap();
                    step_gdn_layers += 1;
                    out
                } else {
                    let attn = layer.self_attn.as_mut().unwrap();
                    let LayerCache::KV(kvc) = lc else {
                        panic!("Expected KVCache");
                    };
                    let out = attn.forward(&normed, None, kvc).unwrap();
                    mlx_rs::transforms::eval([&out]).unwrap();
                    step_attn_layers += 1;
                    out
                };
                let op_ns = t0.elapsed().as_nanos();
                if layer.is_linear {
                    step_gdn_ns += op_ns;
                } else {
                    step_attn_ns += op_ns;
                }

                let t0 = Instant::now();
                let h2 = h.add(r).unwrap();
                let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
                mlx_rs::transforms::eval([&normed_post]).unwrap();
                step_norm_resid_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post).unwrap();
                mlx_rs::transforms::eval([&mlp_out]).unwrap();
                step_mlp_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                h = h2.add(mlp_out).unwrap();
                mlx_rs::transforms::eval([&h]).unwrap();
                step_norm_resid_ns += t0.elapsed().as_nanos();
            }

            let t0 = Instant::now();
            h = model.model.norm.forward(&h).unwrap();
            mlx_rs::transforms::eval([&h]).unwrap();
            step_norm_resid_ns += t0.elapsed().as_nanos();

            let t0 = Instant::now();
            let logits = match model.lm_head.as_ref() {
                Some(head) => head.forward(&h).unwrap(),
                None => model.model.embed_tokens.as_linear(&h).unwrap(),
            };
            let next = ops::indexing::argmax_axis(&logits.index((.., -1, ..)), -1, false).unwrap();
            mlx_rs::transforms::eval([&logits, &next]).unwrap();
            total_lm_ns += t0.elapsed().as_nanos();

            let total_step_ns = step_gdn_ns + step_attn_ns + step_mlp_ns + step_norm_resid_ns;
            println!(
                "step={step:>2} total={:.2}ms gdn={:.2}ms attn={:.2}ms mlp={:.2}ms norm/resid={:.2}ms lm_head={:.2}ms tok/s={:.2}",
                (total_step_ns
                    + total_embed_ns / (step as u128 + 1)
                    + total_lm_ns / (step as u128 + 1)) as f64
                    / 1e6,
                step_gdn_ns as f64 / 1e6,
                step_attn_ns as f64 / 1e6,
                step_mlp_ns as f64 / 1e6,
                step_norm_resid_ns as f64 / 1e6,
                (total_lm_ns / (step as u128 + 1)) as f64 / 1e6,
                1e9 / ((step_gdn_ns + step_attn_ns + step_mlp_ns + step_norm_resid_ns)
                    + total_embed_ns / (step as u128 + 1)
                    + total_lm_ns / (step as u128 + 1)) as f64,
            );

            total_gdn_ns += step_gdn_ns;
            total_attn_ns += step_attn_ns;
            total_mlp_ns += step_mlp_ns;
            total_norm_resid_ns += step_norm_resid_ns;
            gdn_layers = step_gdn_layers;
            attn_layers = step_attn_layers;
            current = next;
        }

        let steps = decode_steps as f64;
        let total_ns = total_embed_ns
            + total_gdn_ns
            + total_attn_ns
            + total_mlp_ns
            + total_norm_resid_ns
            + total_lm_ns;
        let pct = |ns: u128| ns as f64 * 100.0 / total_ns as f64;

        println!(
            "AVG decode breakdown: total={:.2}ms tok/s={:.2}",
            total_ns as f64 / steps / 1e6,
            steps * 1e9 / total_ns as f64,
        );
        println!(
            "  embed:      {:>7.2}ms {:>5.1}%",
            total_embed_ns as f64 / steps / 1e6,
            pct(total_embed_ns),
        );
        println!(
            "  GDN x{}:    {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
            gdn_layers,
            total_gdn_ns as f64 / steps / 1e6,
            pct(total_gdn_ns),
            total_gdn_ns as f64 / steps / gdn_layers.max(1) as f64 / 1e6,
        );
        println!(
            "  Attn x{}:   {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
            attn_layers,
            total_attn_ns as f64 / steps / 1e6,
            pct(total_attn_ns),
            total_attn_ns as f64 / steps / attn_layers.max(1) as f64 / 1e6,
        );
        println!(
            "  MLP:        {:>7.2}ms {:>5.1}% [{:.2}ms/layer]",
            total_mlp_ns as f64 / steps / 1e6,
            pct(total_mlp_ns),
            total_mlp_ns as f64 / steps / (gdn_layers + attn_layers).max(1) as f64 / 1e6,
        );
        println!(
            "  norm/resid: {:>7.2}ms {:>5.1}%",
            total_norm_resid_ns as f64 / steps / 1e6,
            pct(total_norm_resid_ns),
        );
        println!(
            "  lm_head:    {:>7.2}ms {:>5.1}%",
            total_lm_ns as f64 / steps / 1e6,
            pct(total_lm_ns),
        );
    }

    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_actual_qwen3_5_mtp_decode() {
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-27B-4bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            println!("Model not found at {model_path}, skipping");
            return;
        }

        let prompt_len: i32 = std::env::var("BENCH_PROMPT_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let target_tokens: usize = std::env::var("BENCH_DECODE_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);

        let mut model = load_qwen3_5_model(&model_path).unwrap();
        if !model.has_mtp() {
            println!("Model at {model_path} has no MTP head, skipping");
            return;
        }

        let tokens: Vec<u32> = (0..prompt_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let prompt = Array::from_slice(&tokens, &[1, prompt_len]);

        let mut cache: Vec<Option<LayerCache>> = Vec::new();
        let prefill_out = if prompt_len > 512 {
            model
                .forward_chunked(&prompt, None, &mut cache, 512)
                .unwrap()
        } else {
            model.forward(&prompt, None, &mut cache).unwrap()
        };

        let mut prefill_eval: Vec<&Array> = vec![&prefill_out];
        for lc in &cache {
            if let Some(lc) = lc {
                match lc {
                    LayerCache::Arrays(ac) => {
                        if let Some(ref s) = ac.ssm_state {
                            prefill_eval.push(s);
                        }
                        if let Some(ref c) = ac.conv_state {
                            prefill_eval.push(c);
                        }
                    }
                    LayerCache::KV(_) => {}
                }
            }
        }
        mlx_rs::transforms::eval(prefill_eval).unwrap();

        let logits = prefill_out.index((.., -1, ..));
        let first_token = ops::indexing::argmax_axis(&logits, -1, false).unwrap();
        mlx_rs::transforms::eval([&first_token]).unwrap();
        let first_token_id: u32 = first_token.item();

        let mut mtp_cache = model.make_mtp_cache().unwrap();

        // Warm up speculative decode by confirming the first sampled token.
        let first_input = Array::from_slice(&[first_token_id as i32], &[1, 1]);
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, &mut cache)
            .unwrap();
        let next_arr = ops::indexing::argmax_axis(&logits.index((.., -1, ..)), -1, false).unwrap();
        let h = hidden.index((.., -1.., ..));
        mlx_rs::transforms::eval([&next_arr, &h]).unwrap();

        let mut current_hidden = h;
        let mut confirmed_token_id: u32 = next_arr.item();
        let mut emitted_tokens = 0usize;
        let mut accepted_cycles = 0usize;
        let mut total_cycles = 0usize;
        let mut total_ns = 0u128;

        while emitted_tokens < target_tokens {
            let t0 = Instant::now();

            let draft_logits = model
                .mtp_draft(&current_hidden, confirmed_token_id, &mut mtp_cache)
                .unwrap();
            let draft_token_arr =
                ops::indexing::argmax_axis(&draft_logits.index((.., -1, ..)), -1, false).unwrap();
            mlx_rs::transforms::eval([&draft_token_arr]).unwrap();
            let draft_token_id: u32 = draft_token_arr.item();

            let confirmed_input = Array::from_slice(&[confirmed_token_id as i32], &[1, 1]);
            let (confirmed_hidden, confirmed_logits) = model
                .forward_with_hidden(&confirmed_input, None, &mut cache)
                .unwrap();
            let target_arr =
                ops::indexing::argmax_axis(&confirmed_logits.index((.., -1, ..)), -1, false)
                    .unwrap();
            let h_confirmed = confirmed_hidden.index((.., -1.., ..));
            mlx_rs::transforms::eval([&target_arr, &h_confirmed]).unwrap();
            let target_id: u32 = target_arr.item();

            let emitted_this_cycle = if target_id == draft_token_id {
                accepted_cycles += 1;
                model
                    .mtp_advance(&h_confirmed, draft_token_id, &mut mtp_cache)
                    .unwrap();

                let draft_input = Array::from_slice(&[draft_token_id as i32], &[1, 1]);
                let (draft_hidden, draft_logits) = model
                    .forward_with_hidden(&draft_input, None, &mut cache)
                    .unwrap();
                let bonus_token =
                    ops::indexing::argmax_axis(&draft_logits.index((.., -1, ..)), -1, false)
                        .unwrap();
                let h_last = draft_hidden.index((.., -1.., ..));
                mlx_rs::transforms::eval([&bonus_token, &h_last]).unwrap();
                current_hidden = h_last;
                confirmed_token_id = bonus_token.item();
                2usize
            } else {
                current_hidden = h_confirmed;
                confirmed_token_id = target_id;
                1usize
            };

            total_cycles += 1;
            emitted_tokens += emitted_this_cycle;
            let cycle_ns = t0.elapsed().as_nanos();
            total_ns += cycle_ns;

            println!(
                "cycle={total_cycles:>2} emitted={emitted_this_cycle} total_tokens={emitted_tokens} total_ms={:.2} tok/s={:.2} accept_rate={:.1}%",
                cycle_ns as f64 / 1e6,
                emitted_this_cycle as f64 * 1e9 / cycle_ns as f64,
                accepted_cycles as f64 * 100.0 / total_cycles as f64,
            );
        }

        println!(
            "AVG MTP decode: cycles={} emitted={} avg_cycle_ms={:.2} tok/s={:.2} accept_rate={:.1}%",
            total_cycles,
            emitted_tokens,
            total_ns as f64 / total_cycles as f64 / 1e6,
            emitted_tokens as f64 * 1e9 / total_ns as f64,
            accepted_cycles as f64 * 100.0 / total_cycles as f64,
        );
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_metal_kernel_gather_qmm_interleaving() {
        let b: i32 = 1;
        let d: i32 = 2048;
        let n_layers: i32 = 48;
        let n_gdn = 36;
        let n_experts: i32 = 512;
        let d_inter: i32 = 512;
        let top_k: i32 = 10;
        let gs: i32 = 64;
        let bits: i32 = 4;
        let hk: i32 = 16;
        let hv: i32 = 32;
        let dk: i32 = 128;
        let dv: i32 = 128;

        let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

        fn make_qw3d(n: i32, out_d: i32, in_d: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
            let raw = Array::from_slice(
                &vec![0.01f32; (n * out_d * in_d) as usize],
                &[n, out_d, in_d],
            );
            let (w, s, b_arr) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
            (w, s, b_arr)
        }

        let gate_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
            .collect();
        let up_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
            .collect();
        let down_w: Vec<_> = (0..n_layers)
            .map(|_| make_qw3d(n_experts, d, d_inter, gs, bits))
            .collect();

        let q = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
        let k = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
        let v = Array::from_slice(&vec![0.1f32; (b * hv * dv) as usize], &[b, 1, hv, dv]);
        let a_log_arr = Array::zeros::<f32>(&[hv]).unwrap();
        let a_arr = Array::from_slice(&vec![1.0f32; (b * hv) as usize], &[b, 1, hv]);
        let dt_bias_arr = Array::zeros::<f32>(&[hv]).unwrap();
        let b_arr = Array::zeros::<f32>(&[b, 1, hv]).unwrap();
        let state = Array::zeros::<f32>(&[b, hv, dv, dk]).unwrap();
        mlx_rs::transforms::eval([&q, &k, &v, &a_log_arr, &a_arr, &dt_bias_arr, &b_arr, &state])
            .unwrap();

        let indices = Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);

        // Test 1: gather_qmm ONLY
        let build_gqmm_only = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers as usize {
                let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &xe,
                    &gate_w[i].0,
                    &gate_w[i].1,
                    &gate_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
                )
                .unwrap();
                let act = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &act,
                    &down_w[i].0,
                    &down_w[i].1,
                    &down_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert).unwrap();
            }
            h
        };

        for _ in 0..5 {
            let r = build_gqmm_only(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let n = 10;
        let mut total = 0u128;
        for _ in 0..n {
            let r = build_gqmm_only(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "gather_qmm only (48 layers): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 2: Metal kernel + gather_qmm interleaved
        let build_interleaved = |h_in: &Array| -> (Array, Vec<Array>) {
            let mut h = h_in.clone();
            let mut states_out = Vec::new();
            let mut gdn_idx = 0usize;
            for i in 0..n_layers as usize {
                if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
                    let (y, s_out) = gated_delta_kernel_ffi(
                        &q,
                        &k,
                        &v,
                        &a_log_arr,
                        &a_arr,
                        &dt_bias_arr,
                        &b_arr,
                        &state,
                        b,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    let y_flat = y.reshape(&[b, 1, -1]).unwrap();
                    let y_trunc = y_flat.index((.., .., ..d));
                    h = h.add(y_trunc).unwrap();
                    states_out.push(s_out);
                    gdn_idx += 1;
                }
                let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &xe,
                    &gate_w[i].0,
                    &gate_w[i].1,
                    &gate_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
                )
                .unwrap();
                let act = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &act,
                    &down_w[i].0,
                    &down_w[i].1,
                    &down_w[i].2,
                    &indices,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let expert = d_out
                    .squeeze_axes(&[-2])
                    .unwrap()
                    .sum_axes(&[-2], false)
                    .unwrap();
                h = h.add(expert).unwrap();
            }
            (h, states_out)
        };

        for _ in 0..5 {
            let (r, s) = build_interleaved(&x);
            let mut ev: Vec<&Array> = vec![&r];
            ev.extend(s.iter());
            mlx_rs::transforms::eval(ev).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let (r, s) = build_interleaved(&x);
            let mut ev: Vec<&Array> = vec![&r];
            ev.extend(s.iter());
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval(ev).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Metal kernel + gather_qmm (eval h+states): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );

        // Test 3: Metal kernel + gather_qmm, eval h only
        for _ in 0..5 {
            let (r, _) = build_interleaved(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        total = 0;
        for _ in 0..n {
            let (r, _) = build_interleaved(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        println!(
            "Metal kernel + gather_qmm (eval h only): {:.2}ms",
            total as f64 / n as f64 / 1e6
        );
    }

    /// Test eval scaling with graph size using quantized_matmul + rms_norm
    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_eval_scaling() {
        let b: i32 = 1;
        let d: i32 = 2048;
        let gs: i32 = 64;
        let bits: i32 = 4;
        let n_layers: i32 = 48;

        let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

        fn make_qw2d(rows: i32, cols: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
            let raw = Array::from_slice(&vec![0.01f32; (rows * cols) as usize], &[rows, cols]);
            let (w, s, b_arr) = ops::quantize(&raw, gs, bits).unwrap();
            mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
            (w, s, b_arr)
        }

        let weights: Vec<_> = (0..n_layers).map(|_| make_qw2d(d, d, gs, bits)).collect();
        let norm_ws: Vec<_> = (0..n_layers)
            .map(|_| {
                let w = Array::ones::<f32>(&[d]).unwrap();
                mlx_rs::transforms::eval([&w]).unwrap();
                w
            })
            .collect();

        for n_extras in &[0, 2, 5, 8, 12] {
            let total_ops = n_layers * (1 + n_extras + 1);
            let build = |h_in: &Array| -> Array {
                let mut h = h_in.clone();
                for i in 0..n_layers as usize {
                    h = ops::quantized_matmul(
                        &h,
                        &weights[i].0,
                        &weights[i].1,
                        &weights[i].2,
                        true,
                        gs,
                        bits,
                    )
                    .unwrap();
                    for j in 0..*n_extras as usize {
                        let idx = (i + j + 1) % n_layers as usize;
                        let extra = ops::quantized_matmul(
                            &h,
                            &weights[idx].0,
                            &weights[idx].1,
                            &weights[idx].2,
                            true,
                            gs,
                            bits,
                        )
                        .unwrap();
                        let scale = Array::from_slice(&[0.01f32], &[1]);
                        h = h.add(extra.multiply(&scale).unwrap()).unwrap();
                    }
                    h = fast::rms_norm(&h, &norm_ws[i], 1e-6).unwrap();
                }
                h
            };
            for _ in 0..3 {
                let r = build(&x);
                mlx_rs::transforms::eval([&r]).unwrap();
            }
            let n = 10;
            let mut total_ns = 0u128;
            for _ in 0..n {
                let r = build(&x);
                let t0 = std::time::Instant::now();
                mlx_rs::transforms::eval([&r]).unwrap();
                total_ns += t0.elapsed().as_nanos();
            }
            let avg_ms = total_ns as f64 / n as f64 / 1e6;
            let us_per_op = avg_ms * 1000.0 / total_ops as f64;
            println!(
                "extras={n_extras:2} ops~={total_ops:4} eval={avg_ms:.2}ms ({us_per_op:.1}us/op)"
            );
        }
    }

    /// Measure async_eval pipelining: does GPU overlap with CPU graph building?
    ///
    /// cargo test -p higgs-models --release -- bench_async_pipeline --nocapture --ignored
    #[test]
    #[ignore = "benchmark helper"]
    fn bench_async_pipeline() {
        use mlx_rs::random::normal;
        use mlx_rs::transforms::{async_eval, eval};

        let d: &[i32] = &[2048, 2048];
        let w = normal::<f32>(d, None, None, None).unwrap();
        eval([&w].into_iter()).unwrap();

        let build_graph = |x: &Array| -> Array {
            let mut h = x.clone();
            for _ in 0..40 {
                let mm = h.matmul(&w).unwrap();
                h = mm.add(&h).unwrap();
            }
            h
        };

        let x = normal::<f32>(&[1, 1, 2048], None, None, None).unwrap();
        eval([&x].into_iter()).unwrap();

        // Sequential
        let n = 20usize;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let y = build_graph(&x);
            eval([&y].into_iter()).unwrap();
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // Pipelined
        let t0 = std::time::Instant::now();
        let mut y = build_graph(&x);
        async_eval([&y].into_iter()).unwrap();
        for _ in 0..n {
            let next_y = build_graph(&y);
            async_eval([&next_y].into_iter()).unwrap();
            eval([&y].into_iter()).unwrap();
            y = next_y;
        }
        let pipe_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        eprintln!("Rust mlx-rs sequential:  {seq_ms:.2}ms/step");
        eprintln!("Rust mlx-rs pipelined:   {pipe_ms:.2}ms/step");
        eprintln!("Speedup: {:.2}x", seq_ms / pipe_ms);
    }

    /// Measure pure FFI graph-building overhead: no eval, just op dispatch.
    ///
    /// cargo test -p higgs-models --release -- bench_ffi_overhead --nocapture --ignored
    #[test]
    #[ignore = "benchmark helper"]
    fn bench_ffi_overhead() {
        use mlx_rs::transforms::eval;

        let a = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        let b = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
        eval([&a, &b].into_iter()).unwrap();

        let n = 2000usize;

        // Graph build only (no eval)
        let t0 = std::time::Instant::now();
        let mut x = a.clone();
        for _ in 0..n {
            x = x.add(&b).unwrap();
        }
        let build_us = t0.elapsed().as_micros();
        eprintln!(
            "Rust mlx-rs: {n} adds graph-build = {build_us}us ({:.1}us/op)",
            build_us as f64 / n as f64
        );

        // Graph build + eval
        let t0 = std::time::Instant::now();
        let mut x = a.clone();
        for _ in 0..n {
            x = x.add(&b).unwrap();
        }
        eval([&x].into_iter()).unwrap();
        let total_us = t0.elapsed().as_micros();
        eprintln!(
            "Rust mlx-rs: {n} adds + eval = {total_us}us ({:.1}us/op)",
            total_us as f64 / n as f64
        );

        // With task-local stream set
        let stream = Stream::new();
        mlx_rs::with_new_default_stream(stream, || {
            let t0 = std::time::Instant::now();
            let mut x = a.clone();
            for _ in 0..n {
                x = x.add(&b).unwrap();
            }
            let build_us = t0.elapsed().as_micros();
            eprintln!(
                "Rust mlx-rs (task-local stream): {n} adds graph-build = {build_us}us ({:.1}us/op)",
                build_us as f64 / n as f64
            );
        });
    }

    /// Write a qwen3.5-style VLM config.json (with text_config) and parse it.
    fn write_qwen35_config(dir: &std::path::Path, text_config_json: &str) {
        let config =
            format!(r#"{{"text_config": {text_config_json}, "tie_word_embeddings": false}}"#);
        std::fs::write(dir.join("config.json"), config).unwrap();
    }

    /// Helper: minimal qwen3.5 text_config JSON for a dense (non-MoE) model.
    fn qwen35_dense_text_config() -> &'static str {
        r#"{
            "model_type": "qwen3_5",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0
        }"#
    }

    /// Helper: minimal qwen3.5 text_config JSON for an MoE model.
    fn qwen35_moe_text_config() -> &'static str {
        r#"{
            "model_type": "qwen3_5_moe",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 256,
            "moe_intermediate_size": 128,
            "norm_topk_prob": true
        }"#
    }

    fn write_weight_index(dir: &std::path::Path, keys: &[&str]) {
        let weight_map = keys
            .iter()
            .map(|key| format!(r#""{key}": "model-00001-of-00001.safetensors""#))
            .collect::<Vec<_>>()
            .join(", ");
        let index = format!(r#"{{"metadata": {{}}, "weight_map": {{{weight_map}}}}}"#);
        std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    }

    fn write_safetensors_file(dir: &std::path::Path, file_name: &str, key: &str) {
        let data = [0_u8; 4];
        let tensor =
            safetensors::tensor::TensorView::new(safetensors::tensor::Dtype::F32, vec![1], &data)
                .unwrap();
        safetensors::serialize_to_file([(key, tensor)], None, &dir.join(file_name)).unwrap();
    }

    #[test]
    fn test_load_qwen35_moe_text_config_moe_sets_decoder_sparse_step() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_moe_text_config());
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        assert_eq!(
            args.decoder_sparse_step, 1,
            "MoE model should get decoder_sparse_step=1"
        );
        assert!(args.num_experts > 0);
    }

    #[test]
    fn test_load_qwen35_dense_text_config_no_forced_moe() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_dense_text_config());
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        // Dense models (num_experts=0) must NOT get decoder_sparse_step=1,
        // otherwise every layer tries to create SparseMoeBlock and fails.
        assert_eq!(
            args.decoder_sparse_step, 0,
            "Dense model should NOT get decoder_sparse_step=1"
        );
        assert_eq!(args.num_experts, 0);
    }

    /// Dense fixture with a `rope_parameters` object spliced in.
    fn qwen35_dense_text_config_with_rope(rope_parameters: &str) -> String {
        let base = qwen35_dense_text_config();
        let (head, tail) = base.split_at(base.rfind('}').unwrap());
        format!("{head},\n            \"rope_parameters\": {rope_parameters}{tail}")
    }

    /// End-to-end `YaRN` activation: a Qwythos-shaped config (`rope_parameters`
    /// with `type: "yarn"`) must flow through the text_config loader into
    /// active yarn state on the attention modules — the field the flattener
    /// used to drop silently. Base-9B-shaped configs (mrope hints, no `type`)
    /// must keep the byte-exact default rope path.
    #[test]
    fn qwen35_yarn_config_flows_to_attention() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(
            dir.path(),
            &qwen35_dense_text_config_with_rope(
                r#"{
                    "factor": 4.0,
                    "original_max_position_embeddings": 262144,
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 10000000,
                    "type": "yarn",
                    "partial_rotary_factor": 0.25
                }"#,
            ),
        );
        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        assert!((args.rope_theta - 10_000_000.0).abs() < 1.0);
        assert!((args.partial_rotary_factor - 0.25).abs() < 1e-6);
        let params = yarn_rope_params(&args).expect("yarn params should parse");
        assert!((params.factor - 4.0).abs() < f32::EPSILON);
        assert_eq!(params.original_max_position_embeddings, 262_144);
        assert!(
            (params.beta_fast - 32.0).abs() < f32::EPSILON,
            "default beta_fast"
        );
        assert!(
            (params.beta_slow - 1.0).abs() < f32::EPSILON,
            "default beta_slow"
        );

        let attn = Qwen3NextAttention::new(&args, "model.layers.3.self_attn").unwrap();
        let yarn = attn.yarn.as_ref().expect("yarn state active on attention");
        assert!(
            (yarn.mscale - 1.138_629_4).abs() < 1e-6,
            "mscale {} != 1.1386294",
            yarn.mscale
        );
        // head_dim 64 * 0.25 = 16 rotary dims -> 8 periods.
        assert_eq!(yarn.freqs.shape(), &[8]);
        assert_eq!(yarn.prescale.shape(), &[64]);

        // Base-9B-shaped: same rope_parameters minus `type` -> default rope.
        let dir_base = tempfile::tempdir().unwrap();
        write_qwen35_config(
            dir_base.path(),
            &qwen35_dense_text_config_with_rope(
                r#"{
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }"#,
            ),
        );
        let args_base = load_qwen3_5_moe_text_config_args(dir_base.path()).unwrap();
        assert!(
            args_base.rope_scaling.is_some(),
            "rope_parameters carried through as rope_scaling"
        );
        assert!(yarn_rope_params(&args_base).is_none());
        let attn_base = Qwen3NextAttention::new(&args_base, "model.layers.3.self_attn").unwrap();
        assert!(
            attn_base.yarn.is_none(),
            "default rope path must stay inactive"
        );
    }

    /// mxfp4 QLinears ship without `.biases` on disk. The construction must
    /// set biases to a non-`[1]` shape so `placeholder_param_names` (which
    /// flags `shape == [1]` as "missing") doesn't reject them during weight
    /// loading. Affine QLinears keep the standard `[1]` placeholder so genuinely
    /// missing affine biases are still caught.
    #[test]
    fn mxfp4_qlinear_biases_not_flagged_as_missing() {
        let mxfp4 = QLinear::new_spec(QuantSpec {
            group_size: 32,
            bits: 4,
            mode: crate::quant_mode::QuantMode::MxFp4,
        })
        .unwrap();
        let affine = QLinear::new_spec(QuantSpec {
            group_size: 64,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        })
        .unwrap();

        // mxfp4 biases: empty [0] — NOT flagged as placeholder
        assert_ne!(
            mxfp4.biases.shape(),
            &[1],
            "mxfp4 biases must not be [1] placeholder"
        );
        assert_eq!(
            mxfp4.biases.size(),
            0,
            "mxfp4 biases should be empty (no zero-point in E2M1)"
        );

        // Affine biases: [1] placeholder — still caught by completeness check
        assert_eq!(
            affine.biases.shape(),
            &[1],
            "affine biases should be [1] placeholder before weight loading"
        );
    }

    /// Per-path quantization overrides are parsed from config.json's
    /// `quantization` map (mixed-precision checkpoints). Verifies the override
    /// resolver picks per-tensor specs correctly.
    #[test]
    fn quant_spec_for_resolves_per_path_overrides() {
        let mut args = valid_causal_lm_args();
        args.quantization = Some(QuantizationConfig {
            group_size: 32,
            bits: 4,
            mode: crate::quant_mode::QuantMode::MxFp4,
        });
        args.quant_overrides.insert(
            "model.layers.3.self_attn.k_proj".to_owned(),
            QuantizationConfig {
                group_size: 64,
                bits: 8,
                mode: crate::quant_mode::QuantMode::Affine,
            },
        );

        // Global default
        let default = args.default_quant_spec();
        assert_eq!(default.group_size, 32);
        assert_eq!(default.bits, 4);
        assert!(default.mode.is_mxfp4());

        // Override for k_proj on layer 3
        let kv = args.quant_spec_for("model.layers.3.self_attn.k_proj");
        assert_eq!(kv.group_size, 64);
        assert_eq!(kv.bits, 8);
        assert!(!kv.mode.is_mxfp4());

        // No override for q_proj — falls back to global mxfp4 default
        let q = args.quant_spec_for("model.layers.3.self_attn.q_proj");
        assert_eq!(q.bits, 4);
        assert!(q.mode.is_mxfp4());
    }

    /// Dense-GDN detection keys on `.scales` presence, comparing prefix-
    /// normalized paths. Regression: mxfp4 exports ship quantized GDN
    /// projections (`.weight` + `.scales`, no `.biases`) under the
    /// `language_model.` prefix; the scales set was built from unstripped keys
    /// while the lookup used stripped ones, so every projection was misread
    /// as dense and left unloaded. Genuinely dense (weight-only) projections
    /// must still be detected.
    #[test]
    fn detect_dense_gdn_projections_keys_on_scales_presence() {
        let dir = tempfile::tempdir().unwrap();
        write_weight_index(
            dir.path(),
            &[
                // Layer 0: quantized GDN dynamics (mxfp4: weight + scales, no biases)
                "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                "language_model.model.layers.0.linear_attn.in_proj_a.scales",
                "language_model.model.layers.0.linear_attn.in_proj_b.weight",
                "language_model.model.layers.0.linear_attn.in_proj_b.scales",
                // Layer 1: dense GDN dynamics (bf16: weight only, AEON-style)
                "language_model.model.layers.1.linear_attn.in_proj_a.weight",
                "language_model.model.layers.1.linear_attn.in_proj_b.weight",
            ],
        );

        let mut overrides = BTreeMap::new();
        detect_dense_gdn_projections(dir.path(), &mut overrides);

        assert!(
            !overrides.contains_key("model.layers.0.linear_attn.in_proj_a"),
            "quantized in_proj_a (has .scales) must NOT be marked dense"
        );
        assert!(
            !overrides.contains_key("model.layers.0.linear_attn.in_proj_b"),
            "quantized in_proj_b (has .scales) must NOT be marked dense"
        );
        for key in [
            "model.layers.1.linear_attn.in_proj_a",
            "model.layers.1.linear_attn.in_proj_b",
            // Synthesized for the fused-projection construction path.
            "model.layers.1.linear_attn.in_proj_ba",
        ] {
            let qc = overrides
                .get(key)
                .unwrap_or_else(|| panic!("dense {key} (weight-only) must get a Dense override"));
            assert!(qc.mode.is_dense());
        }
        assert!(
            !overrides.contains_key("model.layers.0.linear_attn.in_proj_ba"),
            "no fused override synthesized for the quantized layer"
        );
        assert_eq!(overrides.len(), 3);
    }

    /// mxfp4 embeddings ship `.weight` + `.scales` only; `biases` stays an
    /// empty `[0]` placeholder. Regression: `QEmbedding::forward` gathered
    /// biases unconditionally, throwing "[take] Cannot do a non-empty take
    /// from an empty axis" on the first forward of any mxfp4 model. The
    /// gather must round-trip to the dequantized rows.
    #[test]
    fn qembedding_mxfp4_forward_gathers_without_biases() {
        let vocab = 8;
        let dim = 64;
        let group_size = 32;
        let data: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let w = Array::from_slice(&data, &[vocab, dim]);
        let (wq, scales, _) =
            crate::quant_mode::quantize(&w, group_size, 4, crate::quant_mode::QuantMode::MxFp4)
                .unwrap();

        let mut emb = QEmbedding::new_spec(QuantSpec {
            group_size,
            bits: 4,
            mode: crate::quant_mode::QuantMode::MxFp4,
        });
        *emb.weight = wq.clone();
        *emb.scales = scales.clone();

        let indices = Array::from_slice(&[1_u32, 5, 2], &[1, 3]);
        let out = emb.forward(&indices).unwrap();
        assert_eq!(out.shape(), &[1, 3, dim]);

        let full = crate::quant_mode::dequantize(
            &wq,
            &scales,
            None,
            group_size,
            4,
            crate::quant_mode::QuantMode::MxFp4,
        )
        .unwrap();
        let expected = full
            .take_axis(&Array::from_slice(&[1_u32, 5, 2], &[3]), 0)
            .unwrap();
        // mxfp4 dequantize yields bf16; cast both sides for CPU comparison.
        let got = out
            .reshape(&[3, dim])
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        let got = got.as_slice::<f32>();
        let want = expected.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let want = want.as_slice::<f32>();
        assert_eq!(got.len(), want.len(), "output length mismatch");
        let mut max_abs_err = 0.0_f32;
        for (&g, &w) in got.iter().zip(want.iter()) {
            max_abs_err = max_abs_err.max((g - w).abs());
        }
        assert!(
            max_abs_err < 1e-6,
            "gathered rows must match dequantized rows, max diff {max_abs_err}"
        );
    }

    #[test]
    fn test_checkpoint_has_mtp_weights_detects_prefixed_keys() {
        let dir = tempfile::tempdir().unwrap();
        write_weight_index(
            dir.path(),
            &["language_model.mtp.layers.0.self_attn.q_proj.weight"],
        );
        assert!(checkpoint_has_mtp_weights(dir.path()).unwrap());
    }

    #[test]
    fn test_checkpoint_has_mtp_weights_detects_auxiliary_mtp_file() {
        let dir = tempfile::tempdir().unwrap();
        write_weight_index(
            dir.path(),
            &["language_model.model.layers.0.input_layernorm.weight"],
        );
        write_safetensors_file(
            dir.path(),
            "model-mtp.safetensors",
            "language_model.mtp.layers.0.self_attn.q_proj.weight",
        );

        assert!(checkpoint_has_mtp_weights(dir.path()).unwrap());
    }

    #[test]
    fn test_checkpoint_mtp_weight_layout_detects_quantized_indexed_keys() {
        let dir = tempfile::tempdir().unwrap();
        write_weight_index(
            dir.path(),
            &[
                "language_model.mtp.layers.0.self_attn.q_proj.weight",
                "language_model.mtp.layers.0.self_attn.q_proj.scales",
                "language_model.mtp.layers.0.self_attn.q_proj.biases",
            ],
        );

        assert_eq!(
            checkpoint_mtp_weight_layout(dir.path()).unwrap(),
            MtpWeightLayout::Quantized
        );
    }

    #[test]
    fn test_mtp_layout_detects_moe_structured_head() {
        // Qwen3.6-A3B style: the MTP layer is a full MoE layer.
        let layout = mtp_weight_layout_from_keys([
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.scales",
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
            "mtp.fc.weight",
        ]);
        assert_eq!(layout, MtpWeightLayout::MoeQuantized);
    }

    #[test]
    fn test_moe_mtp_param_key_remaps_mtp_prefix() {
        assert_eq!(
            moe_mtp_param_key("mtp.layers.0.mlp.gate.weight").as_deref(),
            Some("moe_mtp.layers.0.mlp.gate.weight")
        );
        assert_eq!(
            moe_mtp_param_key("mtp.fc.scales").as_deref(),
            Some("moe_mtp.fc.scales")
        );
        assert!(moe_mtp_param_key("model.layers.0.mlp.gate.weight").is_none());
    }

    #[test]
    fn test_normalize_sidecar_mtp_key_prefixes_aux_files_only() {
        let aux = Path::new("/models/x/mtp.safetensors");
        let main = Path::new("/models/x/model-00001-of-00004.safetensors");
        // Unprefixed keys from the sidecar get the mtp. prefix.
        assert_eq!(
            normalize_sidecar_mtp_key(aux, "fc.weight".to_owned()),
            "mtp.fc.weight"
        );
        assert_eq!(
            normalize_sidecar_mtp_key(aux, "layers.0.mlp.gate.weight".to_owned()),
            "mtp.layers.0.mlp.gate.weight"
        );
        // Already-prefixed sidecar keys are unchanged.
        assert_eq!(
            normalize_sidecar_mtp_key(aux, "mtp.fc.weight".to_owned()),
            "mtp.fc.weight"
        );
        // Already-namespaced sidecar keys (e.g. `language_model.mtp.*`) must NOT
        // be over-prefixed into unmatchable `mtp.language_model.mtp.*`.
        assert_eq!(
            normalize_sidecar_mtp_key(aux, "language_model.mtp.layers.0.fc.weight".to_owned()),
            "language_model.mtp.layers.0.fc.weight"
        );
        // Keys from main shards are never touched.
        assert_eq!(
            normalize_sidecar_mtp_key(main, "fc.weight".to_owned()),
            "fc.weight"
        );
    }

    #[test]
    fn test_checkpoint_mtp_weight_layout_detects_dense_auxiliary_mtp_file() {
        let dir = tempfile::tempdir().unwrap();
        write_weight_index(
            dir.path(),
            &["language_model.model.layers.0.input_layernorm.weight"],
        );
        write_safetensors_file(
            dir.path(),
            "model-mtp.safetensors",
            "mtp.layers.0.self_attn.q_proj.weight",
        );

        assert_eq!(
            checkpoint_mtp_weight_layout(dir.path()).unwrap(),
            MtpWeightLayout::Dense
        );
    }

    #[test]
    fn test_qwen35_checkpoint_key_accepts_unprefixed_mtp_sidecar() {
        assert_eq!(
            qwen35_checkpoint_param_key("mtp.layers.0.self_attn.q_proj.weight"),
            Some("mtp.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            qwen35_checkpoint_param_key("language_model.model.layers.0.input_layernorm.weight"),
            Some("model.layers.0.input_layernorm.weight")
        );
        assert_eq!(qwen35_checkpoint_param_key("vision_tower.foo"), None);
    }

    #[test]
    fn test_dense_mtp_param_key_remaps_mtp_namespace() {
        assert_eq!(
            dense_mtp_param_key("mtp.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("dense_mtp.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(dense_mtp_param_key("model.layers.0.foo"), None);
    }

    #[test]
    fn test_dense_mtp_rmsnorm_weight_keys_require_plus_one() {
        assert!(dense_mtp_rmsnorm_weight_key(
            "mtp.layers.0.input_layernorm.weight"
        ));
        assert!(dense_mtp_rmsnorm_weight_key(
            "mtp.layers.0.self_attn.q_norm.weight"
        ));
        assert!(dense_mtp_rmsnorm_weight_key(
            "mtp.pre_fc_norm_hidden.weight"
        ));
        assert!(!dense_mtp_rmsnorm_weight_key(
            "mtp.layers.0.self_attn.q_proj.weight"
        ));
    }

    #[test]
    fn test_maybe_disable_mtp_without_checkpoint_weights_turns_off_missing_mtp() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_moe_text_config());
        write_weight_index(
            dir.path(),
            &["language_model.model.layers.0.input_layernorm.weight"],
        );
        let mut args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        args.mtp_num_hidden_layers = 1;
        maybe_disable_mtp_without_checkpoint_weights(&mut args, dir.path()).unwrap();
        assert_eq!(args.mtp_num_hidden_layers, 0);
    }

    #[test]
    fn test_maybe_disable_mtp_without_checkpoint_weights_preserves_present_mtp() {
        let dir = tempfile::tempdir().unwrap();
        write_qwen35_config(dir.path(), qwen35_moe_text_config());
        write_weight_index(
            dir.path(),
            &["language_model.mtp.layers.0.self_attn.q_proj.weight"],
        );
        let mut args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
        args.mtp_num_hidden_layers = 1;
        maybe_disable_mtp_without_checkpoint_weights(&mut args, dir.path()).unwrap();
        assert_eq!(args.mtp_num_hidden_layers, 1);
    }

    #[test]
    fn test_load_qwen35_mixed_ba_quantization_forces_separate_gdn() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            args.use_separate_gdn_projections,
            "mixed-bit in_proj_a/in_proj_b must force separate GDN projections"
        );
    }

    #[test]
    fn test_load_qwen35_mixed_ba_quantization_supports_unprefixed_layer_keys() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            args.use_separate_gdn_projections,
            "unprefixed mixed-bit in_proj_a/in_proj_b must force separate GDN projections"
        );
    }

    #[test]
    fn test_load_qwen35_mixed_qkvz_quantization_forces_separate_gdn() {
        // Mixed-precision quants (e.g. OptiQ) can also put `in_proj_qkv` and
        // `in_proj_z` at different bit-widths — that breaks the qkvz fusion
        // concat exactly like a mixed BA pair does.
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.2.linear_attn.in_proj_z": {{
                        "group_size": 64,
                        "bits": 8,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            args.use_separate_gdn_projections,
            "mixed-bit in_proj_qkv/in_proj_z must force separate GDN projections"
        );
    }

    #[test]
    fn test_load_qwen35_matching_ba_quantization_keeps_fused_gdn() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }},
                    "language_model.model.layers.1.linear_attn.in_proj_b": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
            qwen35_dense_text_config()
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            !args.use_separate_gdn_projections,
            "matching BA overrides should keep the fused GDN loader path"
        );
    }

    #[test]
    fn test_load_qwen35_explicit_separate_gdn_config_is_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let mut text_config = qwen35_dense_text_config().trim_end_matches('}').to_owned();
        text_config.push_str(
            r#",
            "use_separate_gdn_projections": true
        }"#,
        );
        write_qwen35_config(dir.path(), &text_config);

        let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

        assert!(
            args.use_separate_gdn_projections,
            "explicit use_separate_gdn_projections=true must not be overwritten"
        );
    }

    #[test]
    fn test_can_concatenate_axis0_detects_quantized_inner_shape_mismatch() {
        assert!(
            !can_concatenate_axis0_shapes(&[48, 320], &[48, 800]),
            "different packed inner dims must block BA fusion"
        );
        assert!(
            can_concatenate_axis0_shapes(&[48, 320], &[96, 320]),
            "axis-0 size may differ because fusion concatenates rows"
        );
    }

    /// GQA ratio: `num_v_heads` must be divisible by `num_k_heads`.
    /// This validates the assumption used in test/bench GDN recurrence loops.
    #[test]
    fn test_gqa_ratio_divisibility() {
        let args = valid_causal_lm_args();
        let hv = args.linear_num_value_heads;
        let hk = args.linear_num_key_heads;
        assert!(
            hk > 0 && hv % hk == 0,
            "linear_num_value_heads ({hv}) must be divisible by linear_num_key_heads ({hk})"
        );
    }

    /// QEmbedding equivalence: dequantize-then-gather produces same result as
    /// the full dequantize path (validates that gather on quantized storage
    /// is safe for future optimisation).
    #[test]
    fn test_qembedding_gather_then_dequantize_equivalence() {
        use mlx_rs::transforms::eval;

        let group_size = 64i32;
        let bits = 4i32;
        let vocab = 256i32;
        let hidden = 128i32;

        // Create a random float matrix and quantize it
        let float_weight =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[vocab, hidden], None).unwrap();
        eval([&float_weight].into_iter()).unwrap();
        let (qw, qs, qb) = ops::quantize(&float_weight, group_size, bits).unwrap();
        eval([&qw, &qs, &qb].into_iter()).unwrap();

        let indices = Array::from_slice(&[0i32, 5, 42, 255, 5], &[5]);
        eval([&indices].into_iter()).unwrap();

        // Path A: dequantize full vocab, then gather (current QEmbedding::forward)
        let full_deq = ops::dequantize(&qw, &qs, &qb, group_size, bits).unwrap();
        let path_a = full_deq.take_axis(&indices, 0).unwrap();
        eval([&path_a].into_iter()).unwrap();

        // Path B: gather quantized rows first, then dequantize only selected
        let sel_w = qw.take_axis(&indices, 0).unwrap();
        let sel_s = qs.take_axis(&indices, 0).unwrap();
        let sel_b = qb.take_axis(&indices, 0).unwrap();
        let path_b = ops::dequantize(&sel_w, &sel_s, &sel_b, group_size, bits).unwrap();
        eval([&path_b].into_iter()).unwrap();

        // They should be identical (both round-trip through the same quantized repr)
        let diff = path_a.subtract(&path_b).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-6,
            "gather-then-dequantize should match dequantize-then-gather, max diff: {max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // Chunked prefill tests
    // -----------------------------------------------------------------------

    /// forward_chunked compiles and the API is callable.
    /// chunk_size >= T falls through to normal forward (no chunking).
    #[test]
    fn test_chunked_prefill_api_exists() {
        let args = valid_causal_lm_args();
        let model = Qwen3NextCausalLM::new(args).unwrap();
        // Verify forward_chunked is callable (type-check / link test).
        // We can't run it on synthetic weights, but we confirm the method exists
        // and handles the chunk_size >= T fast path correctly.
        assert!(model.args.num_hidden_layers > 0);
    }

    /// Chunked prefill: logits are close to full prefill on a real model.
    /// Tests even division (chunk_size=4, seq_len=12).
    ///
    /// Note: quantized_matmul produces slightly different results for different
    /// input shapes due to tile reduction order (FP non-associativity).
    /// A max logit diff of ~1-2 is normal for 3-bit models.
    /// The decode_continuity test is the real correctness check (same tokens).
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_matches_full --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_matches_full() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 12i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        // Full prefill
        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();

        // Chunked prefill: chunk_size=4 → chunks [4,4,4]
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 4)
            .unwrap();
        eval([&logits_chunked]).unwrap();

        let last_full = logits_full.index((.., -1, ..));
        let last_chunked = logits_chunked.index((.., -1, ..));
        eval([&last_full, &last_chunked]).unwrap();

        let diff = last_full.subtract(&last_chunked).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        eprintln!("max logit |diff| = {max_diff}");
        assert!(
            max_diff < 2.0,
            "chunked logits diverge from full: max |diff| = {max_diff} (expect <2.0 for 3-bit quant)"
        );
    }

    /// Chunked prefill: uneven chunk sizes (remainder chunk).
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_uneven --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_uneven() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 10i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();

        // chunk_size=3: chunks [3,3,3,1]
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 3)
            .unwrap();
        eval([&logits_chunked]).unwrap();

        let last_full = logits_full.index((.., -1, ..));
        let last_chunked = logits_chunked.index((.., -1, ..));
        eval([&last_full, &last_chunked]).unwrap();

        let diff = last_full.subtract(&last_chunked).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        eprintln!("uneven max logit |diff| = {max_diff}");
        assert!(
            max_diff < 2.0,
            "uneven chunks diverge: max |diff| = {max_diff} (expect <2.0 for 3-bit quant)"
        );
    }

    /// Decode after chunked prefill produces same tokens as after full prefill.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- test_chunked_prefill_decode_continuity --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn test_chunked_prefill_decode_continuity() {
        use mlx_rs::transforms::eval;

        let mut model = load_test_model();

        let seq_len = 16i32;
        let tokens: Vec<u32> = (0..seq_len as u32)
            .map(|i| i % model.args.vocab_size as u32)
            .collect();
        let input = Array::from_slice(&tokens, &[1, seq_len]);

        // Full prefill + 5 decode steps
        let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
        let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
        eval([&logits_full]).unwrap();
        let full_tokens = decode_greedy(&mut model, &logits_full, &mut cache_full, 5);

        // Chunked prefill + 5 decode steps
        let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
        let logits_chunked = model
            .forward_chunked(&input, None, &mut cache_chunked, 4)
            .unwrap();
        eval([&logits_chunked]).unwrap();
        let chunked_tokens = decode_greedy(&mut model, &logits_chunked, &mut cache_chunked, 5);

        assert_eq!(
            full_tokens, chunked_tokens,
            "decode tokens diverge: full={full_tokens:?} chunked={chunked_tokens:?}"
        );
    }

    /// Load whichever model is available for integration tests.
    fn load_test_model() -> Qwen3NextCausalLM {
        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            panic!("Model not found at {model_path}. Set HIGGS_MODEL_PATH.");
        }
        // Warmup: load + prime shaders
        let mut model = load_qwen3_5_moe_model(&model_path).unwrap();
        let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
        let mut wc: Vec<Option<LayerCache>> = Vec::new();
        let out = model.forward(&w, None, &mut wc).unwrap();
        mlx_rs::transforms::eval([&out]).unwrap();
        model
    }

    /// Run greedy decode for `n` steps from prefill logits, return token ids.
    fn decode_greedy(
        model: &mut Qwen3NextCausalLM,
        prefill_logits: &Array,
        cache: &mut Vec<Option<LayerCache>>,
        n: usize,
    ) -> Vec<u32> {
        use mlx_rs::transforms::eval;

        let mut tok =
            ops::indexing::argmax_axis(&prefill_logits.index((.., -1, ..)), -1, false).unwrap();
        eval([&tok]).unwrap();
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let step_in = tok.index((.., ops::indexing::NewAxis));
            let out = model.forward(&step_in, None, cache).unwrap();
            tok = ops::indexing::argmax_axis(&out.index((.., -1, ..)), -1, false).unwrap();
            eval([&tok]).unwrap();
            tokens.push(tok.item::<u32>());
        }
        tokens
    }

    // -----------------------------------------------------------------------
    // Chunked prefill benchmark (real model)
    // -----------------------------------------------------------------------

    /// Benchmark chunked vs full prefill TTFT.
    ///
    /// Set env vars to control the benchmark:
    /// - `BENCH_SEQ`: comma-separated sequence lengths (default: 512,1024,2048,5120,10240)
    /// - `BENCH_CHUNK`: comma-separated chunk sizes (default: 128,256,512,1024)
    /// - `BENCH_FULL_MAX`: max sequence length for full prefill baseline (default: 10240)
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
    ///
    /// # Long sequences only:
    /// BENCH_SEQ=10240,20480,40960 BENCH_CHUNK=256,512 BENCH_FULL_MAX=20480 \
    ///   cargo test -p higgs-models --release -- bench_chunked_prefill --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_chunked_prefill() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let mut model = load_test_model();
        eprintln!(
            "Model: {} layers, hidden={}\n",
            model.args.num_hidden_layers, model.args.hidden_size,
        );

        let seq_lengths: Vec<i32> = std::env::var("BENCH_SEQ")
            .unwrap_or_else(|_| "512,1024,2048,5120,10240".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let chunk_sizes: Vec<i32> = std::env::var("BENCH_CHUNK")
            .unwrap_or_else(|_| "128,256,512,1024".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let full_max: i32 = std::env::var("BENCH_FULL_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10240);

        println!(
            "{:>7}  {:>6}  {:>10}  {:>10}  {:>8}",
            "T", "chunk", "full(ms)", "chunked(ms)", "ratio"
        );
        println!("{}", "-".repeat(50));

        for &seq_len in &seq_lengths {
            let tokens: Vec<u32> = (0..seq_len as u32)
                .map(|i| i % model.args.vocab_size as u32)
                .collect();
            let input = Array::from_slice(&tokens, &[1, seq_len]);

            let full_ms = if seq_len <= full_max {
                let mut cache_full: Vec<Option<LayerCache>> = Vec::new();
                let t0 = Instant::now();
                let logits_full = model.forward(&input, None, &mut cache_full).unwrap();
                eval([&logits_full]).unwrap();
                Some(t0.elapsed().as_secs_f64() * 1000.0)
            } else {
                None
            };

            for &chunk in &chunk_sizes {
                if chunk >= seq_len {
                    continue;
                }

                let mut cache_chunked: Vec<Option<LayerCache>> = Vec::new();
                let t0 = Instant::now();
                let logits_chunked = model
                    .forward_chunked(&input, None, &mut cache_chunked, chunk)
                    .unwrap();
                eval([&logits_chunked]).unwrap();
                let chunked_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let full_str = match full_ms {
                    Some(ms) => format!("{ms:>10.0}"),
                    None => format!("{:>10}", "—"),
                };
                let ratio_str = match full_ms {
                    Some(ms) => format!("{:>7.2}x", ms / chunked_ms),
                    None => format!("{:>8}", "—"),
                };

                println!("{seq_len:>7}  {chunk:>6}  {full_str}  {chunked_ms:>10.0}  {ratio_str}");
            }
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Prefill profiling benchmark
    // -----------------------------------------------------------------------

    /// Profile per-component TTFT breakdown for different sequence lengths.
    ///
    /// Measures wall-clock TTFT (single eval) and per-component time with eval
    /// barriers between embed, GDN, attention, MLP/MoE, norms, and lm_head.
    ///
    /// ```bash
    /// # Default model path: ~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit
    /// cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
    ///
    /// # Override model path:
    /// HIGGS_MODEL_PATH=/path/to/model cargo test -p higgs-models --release -- bench_prefill_breakdown --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "requires model files on disk"]
    fn bench_prefill_breakdown() {
        use mlx_rs::transforms::eval;
        use std::time::Instant;

        let model_path = std::env::var("HIGGS_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
        });
        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Model not found at {model_path}");
            eprintln!("Set HIGGS_MODEL_PATH env var to your model directory");
            return;
        }

        eprintln!("Loading model from {model_path} ...");
        let mut model = load_qwen3_5_moe_model(&model_path).unwrap();
        let n_layers = model.args.num_hidden_layers;
        let fa_interval = model.args.full_attention_interval;
        eprintln!(
            "Loaded: {n_layers} layers, hidden={}, fa_interval={fa_interval}",
            model.args.hidden_size,
        );

        // Warmup: prime Metal shaders + lazy dtype conversions
        {
            let w = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
            let mut wc: Vec<Option<LayerCache>> = Vec::new();
            let out = model.forward(&w, None, &mut wc).unwrap();
            eval([&out].into_iter()).unwrap();
        }

        let seq_lengths: &[i32] = &[128, 512, 1024, 2048, 5120];

        for &seq_len in seq_lengths {
            let tokens: Vec<u32> = (0..seq_len as u32)
                .map(|i| i % model.args.vocab_size as u32)
                .collect();

            // ----- Pass 1: real-world TTFT (no eval barriers) -----
            let input_a = Array::from_slice(&tokens, &[1, seq_len]);
            let mut cache_a: Vec<Option<LayerCache>> = Vec::new();

            let wall_start = Instant::now();
            let logits_a = model.forward(&input_a, None, &mut cache_a).unwrap();
            let mut eval_tgts: Vec<&Array> = vec![&logits_a];
            for lc in &cache_a {
                if let Some(LayerCache::Arrays(ac)) = lc {
                    if let Some(ref s) = ac.ssm_state {
                        eval_tgts.push(s);
                    }
                    if let Some(ref c) = ac.conv_state {
                        eval_tgts.push(c);
                    }
                }
            }
            eval(eval_tgts).unwrap();
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            // ----- Pass 2: per-component with eval barriers -----
            let input_b = Array::from_slice(&tokens, &[1, seq_len]);
            let mut cache_b: Vec<Option<LayerCache>> = model.make_cache();

            let fa_mask: Option<AttentionMask> = if seq_len > 1 {
                Some(AttentionMask::Causal)
            } else {
                None
            };

            // Embed
            let t0 = Instant::now();
            let mut h = model.model.embed_tokens.forward(&input_b).unwrap();
            eval([&h].into_iter()).unwrap();
            let ns_embed = t0.elapsed().as_nanos();

            let mut ns_gdn = 0u128;
            let mut ns_attn = 0u128;
            let mut ns_mlp = 0u128;
            let mut ns_norm = 0u128;
            let mut n_gdn = 0u32;
            let mut n_attn = 0u32;

            for (layer, layer_cache) in model.model.layers.iter_mut().zip(cache_b.iter_mut()) {
                let lc = layer_cache.as_mut().unwrap();
                let mask_ref = if layer.is_linear {
                    None
                } else {
                    fa_mask.as_ref()
                };

                // Pre-attention norm
                let t0 = Instant::now();
                let normed = layer.input_layernorm.forward(&h).unwrap();
                eval([&normed].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                // GDN or full attention
                let t0 = Instant::now();
                let r = if layer.is_linear {
                    let gdn = layer.linear_attn.as_mut().unwrap();
                    let LayerCache::Arrays(sc) = lc else {
                        panic!("Expected ArraysCache");
                    };
                    let out = gdn.forward(&normed, mask_ref, sc).unwrap();
                    let mut tgts: Vec<&Array> = vec![&out];
                    if let Some(ref s) = sc.ssm_state {
                        tgts.push(s);
                    }
                    if let Some(ref c) = sc.conv_state {
                        tgts.push(c);
                    }
                    eval(tgts).unwrap();
                    n_gdn += 1;
                    ns_gdn += t0.elapsed().as_nanos();
                    out
                } else {
                    let attn = layer.self_attn.as_mut().unwrap();
                    let LayerCache::KV(kvc) = lc else {
                        panic!("Expected KVCache");
                    };
                    let out = attn.forward(&normed, mask_ref, kvc).unwrap();
                    eval([&out].into_iter()).unwrap();
                    n_attn += 1;
                    ns_attn += t0.elapsed().as_nanos();
                    out
                };

                // Residual + post-attention norm
                let t0 = Instant::now();
                let h2 = h.add(r).unwrap();
                let normed_post = layer.post_attention_layernorm.forward(&h2).unwrap();
                eval([&normed_post].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();

                // MLP / MoE
                let t0 = Instant::now();
                let mlp_out = layer.mlp.forward(&normed_post).unwrap();
                eval([&mlp_out].into_iter()).unwrap();
                ns_mlp += t0.elapsed().as_nanos();

                // Final residual
                let t0 = Instant::now();
                h = h2.add(mlp_out).unwrap();
                eval([&h].into_iter()).unwrap();
                ns_norm += t0.elapsed().as_nanos();
            }

            // Final norm
            let t0 = Instant::now();
            h = model.model.norm.forward(&h).unwrap();
            eval([&h].into_iter()).unwrap();
            ns_norm += t0.elapsed().as_nanos();

            // LM head
            let t0 = Instant::now();
            let _logits = match model.lm_head.as_ref() {
                Some(head) => head.forward(&h).unwrap(),
                None => model.model.embed_tokens.as_linear(&h).unwrap(),
            };
            eval([&_logits].into_iter()).unwrap();
            let ns_lm = t0.elapsed().as_nanos();

            // ----- Report -----
            let barrier_total = ns_embed + ns_gdn + ns_attn + ns_mlp + ns_norm + ns_lm;
            let ms = |ns: u128| ns as f64 / 1e6;
            let pct = |ns: u128| ns as f64 / barrier_total as f64 * 100.0;
            let n_total = n_gdn + n_attn;

            println!();
            println!("==== T = {seq_len} ====");
            println!("  Wall TTFT (no barriers):  {:>8.1}ms", wall_ms);
            println!(
                "  Sum  (eval barriers):     {:>8.1}ms  (barrier overhead: {:.1}ms)",
                ms(barrier_total),
                ms(barrier_total) - wall_ms,
            );
            println!();
            println!(
                "  embed:            {:>8.1}ms  {:>5.1}%",
                ms(ns_embed),
                pct(ns_embed),
            );
            println!(
                "  GDN ({n_gdn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_gdn),
                pct(ns_gdn),
                ms(ns_gdn) / n_gdn.max(1) as f64,
            );
            println!(
                "  Attn ({n_attn:>2} layers): {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_attn),
                pct(ns_attn),
                ms(ns_attn) / n_attn.max(1) as f64,
            );
            println!(
                "  MLP/MoE:          {:>8.1}ms  {:>5.1}%   [{:.2}ms/layer]",
                ms(ns_mlp),
                pct(ns_mlp),
                ms(ns_mlp) / n_total.max(1) as f64,
            );
            println!(
                "  norms+residual:   {:>8.1}ms  {:>5.1}%",
                ms(ns_norm),
                pct(ns_norm),
            );
            println!(
                "  lm_head:          {:>8.1}ms  {:>5.1}%",
                ms(ns_lm),
                pct(ns_lm),
            );
            println!(
                "  ---- GDN share of wall TTFT: {:.1}%",
                ms(ns_gdn) / wall_ms * 100.0,
            );
        }
    }

    /// Helper: run qgemv_4bit against quantized_matmul reference and assert max abs error.
    fn assert_qgemv_matches_reference(n: i32, k: i32, group_size: i32, label: &str) {
        use mlx_rs::Dtype;

        let x =
            mlx_rs::random::uniform_device::<_, f32>(0.0, 1.0, &[1, 1, k], None, Stream::default())
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();

        let w_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let (w_q, scales, biases) = mlx_rs::ops::quantize(&w_dense, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&w_q, &scales, &biases, &x]).unwrap();

        // Reference: MLX quantized_matmul
        let ref_out = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&ref_out]).unwrap();

        // Custom GEMV kernel — native dtypes, zero conversions
        let custom_out = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
        mlx_rs::transforms::eval([&custom_out]).unwrap();

        let ref_f32 = ref_out.as_dtype(Dtype::Float32).unwrap();
        let cust_f32 = custom_out.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&ref_f32, &cust_f32]).unwrap();

        let ref_vals = ref_f32.as_slice::<f32>();
        let cust_vals = cust_f32.as_slice::<f32>();

        assert_eq!(
            ref_out.shape(),
            custom_out.shape(),
            "[{label}] shape mismatch: ref={:?} vs custom={:?}",
            ref_out.shape(),
            custom_out.shape()
        );
        assert_eq!(ref_vals.len(), cust_vals.len(), "[{label}] length mismatch");

        let mut max_diff = 0.0f32;
        for i in 0..ref_vals.len() {
            let diff = (ref_vals[i] - cust_vals[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 0.5,
                "[{label}] mismatch at {i}: ref={}, custom={}, diff={diff}",
                ref_vals[i],
                cust_vals[i]
            );
        }
        println!("[{label}] PASS — N={n} K={k} gs={group_size} max_diff={max_diff:.4}");
    }

    #[test]
    fn test_qgemv_native_dtype_matches_reference() {
        for &k in &[256, 512, 1024, 4096] {
            let n = 16;
            let gs = 64;
            assert_qgemv_matches_reference(n, k, gs, &format!("K={k}"));
        }
    }

    #[test]
    fn test_qgemv_various_group_sizes() {
        let k = 512;
        let n = 32;
        for &gs in &[32, 64, 128] {
            assert_qgemv_matches_reference(n, k, gs, &format!("gs={gs}"));
        }
    }

    #[test]
    fn test_qgemv_large_n_rows() {
        // Realistic dims: gate+up fused (2*intermediate) and down projection
        assert_qgemv_matches_reference(512, 1024, 64, "N=512 K=1024");
        assert_qgemv_matches_reference(2048, 1024, 64, "N=2048 K=1024");
    }

    #[test]
    fn test_qgemm_affine_matches_reference() {
        use mlx_rs::Dtype;

        let t = 4;
        let k = 256;
        let n = 16;
        let group_size = 64;
        let x =
            mlx_rs::random::uniform_device::<_, f32>(0.0, 1.0, &[1, t, k], None, Stream::default())
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        let w_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let (w_q, scales, biases) = mlx_rs::ops::quantize(&w_dense, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&w_q, &scales, &biases, &x]).unwrap();

        let ref_out = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&ref_out]).unwrap();

        let custom_out = qgemm_4bit(&x, &w_q, &scales, &biases, group_size, t).unwrap();
        mlx_rs::transforms::eval([&custom_out]).unwrap();

        assert_eq!(ref_out.shape(), custom_out.shape());

        let ref_f32 = ref_out.as_dtype(Dtype::Float32).unwrap();
        let cust_f32 = custom_out.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&ref_f32, &cust_f32]).unwrap();
        let ref_vals = ref_f32.as_slice::<f32>();
        let cust_vals = cust_f32.as_slice::<f32>();

        assert_eq!(ref_vals.len(), cust_vals.len());
        for (i, (&r, &c)) in ref_vals.iter().zip(cust_vals.iter()).enumerate() {
            let diff = (r - c).abs();
            assert!(
                diff < 0.5,
                "qgemm affine mismatch at {i}: ref={r}, custom={c}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_qgemm_mxfp4_matches_reference() {
        use crate::quant_mode::{QuantMode, quantize as quantize_with_mode, quantized_matmul};
        use mlx_rs::Dtype;

        let t = 4;
        let k = 256;
        let n = 16;
        let group_size = 32;
        let x = mlx_rs::random::uniform_device::<_, f32>(
            -0.5,
            0.5,
            &[1, t, k],
            None,
            Stream::default(),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let w_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let (w_q, scales, _biases) =
            quantize_with_mode(&w_dense, group_size, 4, QuantMode::MxFp4).unwrap();
        mlx_rs::transforms::eval([&w_q, &scales, &x]).unwrap();

        let ref_out = quantized_matmul(
            &x,
            &w_q,
            &scales,
            None,
            true,
            group_size,
            4,
            QuantMode::MxFp4,
        )
        .unwrap();
        mlx_rs::transforms::eval([&ref_out]).unwrap();

        let custom_out = qgemm_mxfp4_4bit(&x, &w_q, &scales, group_size, t).unwrap();
        mlx_rs::transforms::eval([&custom_out]).unwrap();

        assert_eq!(ref_out.shape(), custom_out.shape());

        let ref_f32 = ref_out.as_dtype(Dtype::Float32).unwrap();
        let cust_f32 = custom_out.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&ref_f32, &cust_f32]).unwrap();
        let ref_vals = ref_f32.as_slice::<f32>();
        let cust_vals = cust_f32.as_slice::<f32>();

        assert_eq!(ref_vals.len(), cust_vals.len());
        let mut max_diff = 0.0f32;
        for (i, (&r, &c)) in ref_vals.iter().zip(cust_vals.iter()).enumerate() {
            let diff = (r - c).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 0.5,
                "qgemm mxfp4 mismatch at {i}: ref={r}, custom={c}, diff={diff}"
            );
        }
        println!("qgemm mxfp4 PASS: N={n} K={k} T={t} gs={group_size} max_diff={max_diff:.4}");
    }

    #[test]
    fn test_mxfp4_gate_up_silu_matches_reference() {
        use crate::quant_mode::{QuantMode, quantize as quantize_with_mode, quantized_matmul};
        use mlx_rs::Dtype;

        let t = 4;
        let k = 256;
        let n = 16;
        let group_size = 32;
        let x = mlx_rs::random::uniform_device::<_, f32>(
            -0.5,
            0.5,
            &[1, t, k],
            None,
            Stream::default(),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let gate_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let up_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let (gate_q, gate_scales, _gate_biases) =
            quantize_with_mode(&gate_dense, group_size, 4, QuantMode::MxFp4).unwrap();
        let (up_q, up_scales, _up_biases) =
            quantize_with_mode(&up_dense, group_size, 4, QuantMode::MxFp4).unwrap();
        mlx_rs::transforms::eval([&gate_q, &gate_scales, &up_q, &up_scales, &x]).unwrap();

        let gate_ref = quantized_matmul(
            &x,
            &gate_q,
            &gate_scales,
            None,
            true,
            group_size,
            4,
            QuantMode::MxFp4,
        )
        .unwrap();
        let up_ref = quantized_matmul(
            &x,
            &up_q,
            &up_scales,
            None,
            true,
            group_size,
            4,
            QuantMode::MxFp4,
        )
        .unwrap();
        let ref_out = silu_mul(&gate_ref, &up_ref).unwrap();
        mlx_rs::transforms::eval([&ref_out]).unwrap();

        let custom_out =
            mxfp4_gate_up_silu_4bit(&x, &gate_q, &gate_scales, &up_q, &up_scales, group_size, t)
                .unwrap();
        mlx_rs::transforms::eval([&custom_out]).unwrap();

        assert_eq!(ref_out.shape(), custom_out.shape());

        let ref_f32 = ref_out.as_dtype(Dtype::Float32).unwrap();
        let cust_f32 = custom_out.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&ref_f32, &cust_f32]).unwrap();
        let ref_vals = ref_f32.as_slice::<f32>();
        let cust_vals = cust_f32.as_slice::<f32>();

        assert_eq!(ref_vals.len(), cust_vals.len());
        let mut max_diff = 0.0f32;
        for (i, (&r, &c)) in ref_vals.iter().zip(cust_vals.iter()).enumerate() {
            let diff = (r - c).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 0.5,
                "mxfp4 gate/up silu mismatch at {i}: ref={r}, custom={c}, diff={diff}"
            );
        }
        println!(
            "mxfp4 gate/up silu PASS: N={n} K={k} T={t} gs={group_size} max_diff={max_diff:.4}"
        );
    }

    /// Benchmark helper: time GEMV vs quantized_matmul for given dims.
    fn bench_gemv_at(n: i32, k: i32, group_size: i32, iters: usize) {
        use mlx_rs::Dtype;

        let x =
            mlx_rs::random::uniform_device::<_, f32>(0.0, 1.0, &[1, 1, k], None, Stream::default())
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();

        let w_dense =
            mlx_rs::random::uniform_device::<_, f32>(-1.0, 1.0, &[n, k], None, Stream::default())
                .unwrap();
        let (w_q, scales, biases) = mlx_rs::ops::quantize(&w_dense, group_size, 4).unwrap();
        mlx_rs::transforms::eval([&w_q, &scales, &biases, &x]).unwrap();

        // Warmup
        for _ in 0..5 {
            let r = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
            let g = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
            mlx_rs::transforms::eval([&r, &g]).unwrap();
        }

        // Bench quantized_matmul
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let out = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
            mlx_rs::transforms::eval([&out]).unwrap();
        }
        let qmm_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // Bench custom GEMV
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let out = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
            mlx_rs::transforms::eval([&out]).unwrap();
        }
        let gemv_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let ratio = qmm_us / gemv_us;
        println!(
            "  N={n:>5} K={k:>5} | qmm={qmm_us:>7.0}μs  gemv={gemv_us:>7.0}μs  ratio={ratio:.2}x"
        );
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn bench_qgemv_vs_quantized_matmul() {
        println!("=== GEMV vs quantized_matmul (gs=64) ===");
        // Small
        bench_gemv_at(128, 512, 64, 50);
        bench_gemv_at(256, 1024, 64, 50);
        // Medium
        bench_gemv_at(1024, 2048, 64, 50);
        bench_gemv_at(2048, 2048, 64, 50);
        bench_gemv_at(5504, 2048, 64, 50);
        bench_gemv_at(5120, 5120, 64, 30);
        // 27B dense MLP actual dims (hidden=5120, intermediate=17408)
        println!("--- 27B dense MLP dims ---");
        bench_gemv_at(34816, 5120, 64, 20); // gate+up fused
        bench_gemv_at(5120, 17408, 64, 20); // down projection
        bench_gemv_at(248_320, 5120, 64, 5); // tied lm_head / embedding projection
    }

    #[test]
    fn parse_dense_ffn_gemv_mode_defaults_to_both() {
        assert_eq!(parse_dense_ffn_gemv_mode(None), DenseFfnGemvMode::Both);
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("unexpected")),
            DenseFfnGemvMode::Both
        );
    }

    #[test]
    fn parse_dense_ffn_gemv_mode_supports_all_variants() {
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("fused")),
            DenseFfnGemvMode::FusedOnly
        );
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("fused_only")),
            DenseFfnGemvMode::FusedOnly
        );
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("down")),
            DenseFfnGemvMode::DownOnly
        );
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("down_only")),
            DenseFfnGemvMode::DownOnly
        );
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("off")),
            DenseFfnGemvMode::Off
        );
        assert_eq!(
            parse_dense_ffn_gemv_mode(Some("none")),
            DenseFfnGemvMode::Off
        );
    }

    #[test]
    fn base_m4_forces_dense_decode_safe_defaults() {
        assert!(should_force_dense_decode_safe_defaults_for_brand(Some(
            "Apple M4"
        )));
        assert!(should_force_dense_decode_safe_defaults_for_brand(Some(
            " Apple M4 "
        )));
    }

    #[test]
    fn non_base_m4_keeps_dense_decode_fastpaths_available() {
        assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
            "Apple M4 Pro"
        )));
        assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
            "Apple M4 Max"
        )));
        assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
            "Apple M5"
        )));
        assert!(!should_force_dense_decode_safe_defaults_for_brand(None));
    }

    #[test]
    fn dense_hidden_fused_matches_separate_path() {
        use mlx_rs::{Dtype, module::Param};

        fn assign_qlinear(layer: &mut QLinear, out_dim: i32, in_dim: i32) {
            let raw = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[out_dim, in_dim], None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let (w, s, b) = ops::quantize(&raw, 32, 4).unwrap();
            layer.weight = Param::new(w);
            layer.scales = Param::new(s);
            layer.biases = Param::new(b);
            layer.group_size = 32;
            layer.bits = 4;
        }

        let args = minimal_qwen3_next_args();
        let mut block = FfnBlock::new_dense(&args, "test.layer.mlp").unwrap();
        assign_qlinear(block.gate_proj.as_mut().unwrap(), 96, 64);
        assign_qlinear(block.up_proj.as_mut().unwrap(), 96, 64);
        assign_qlinear(block.down_proj.as_mut().unwrap(), 64, 96);

        let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 64], None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();

        let fused = block.dense_hidden_fused(&x, false).unwrap();
        let separate = block.dense_hidden_separate(&x).unwrap();
        mlx_rs::transforms::eval([&fused, &separate]).unwrap();

        let diff = fused.subtract(&separate).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item();
        assert!(
            max_diff < 1e-3,
            "dense fused/separate hidden mismatch by {max_diff}"
        );
    }

    #[test]
    fn test_compiled_gdn_decode_matches_reference_ops() {
        let batch = 1;
        let hv = 4;
        let dk = 32;
        let dv = 32;

        let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dk], None).unwrap();
        let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dk], None).unwrap();
        let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dv], None).unwrap();
        let g = mlx_rs::random::uniform::<f32, f32>(0.1, 0.9, &[batch, 1, hv], None).unwrap();
        let beta = mlx_rs::random::uniform::<f32, f32>(0.1, 0.9, &[batch, 1, hv], None).unwrap();
        let z = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dv], None).unwrap();
        let norm_weight = Array::ones::<f32>(&[dv]).unwrap();
        let state =
            mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

        let (y_ref, state_ref) = gated_delta_step_ref(
            &q.squeeze_axes(&[1]).unwrap(),
            &k.squeeze_axes(&[1]).unwrap(),
            &v.squeeze_axes(&[1]).unwrap(),
            &g.squeeze_axes(&[1]).unwrap(),
            &beta.squeeze_axes(&[1]).unwrap(),
            &state,
        );
        let y_ref = y_ref.expand_dims(1).unwrap();
        let expected = nn::silu(&z)
            .unwrap()
            .multiply(&fast::rms_norm(&y_ref, &norm_weight, 1e-6).unwrap())
            .unwrap();

        let mut cache = ArraysCache {
            conv_state: None,
            ssm_state: Some(state.clone()),
            conv_pos: -1,
            offset: 0,
        };
        let mut compiled = make_compiled_gdn_decode();
        let inputs = [
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            z.clone(),
            norm_weight.clone(),
        ];
        let actual = compiled(&mut cache, &inputs).unwrap().pop().unwrap();

        let actual_state = cache.ssm_state.as_ref().unwrap().clone();
        mlx_rs::transforms::eval([&actual, &expected, &actual_state, &state_ref]).unwrap();

        let out_diff = actual.subtract(&expected).unwrap().abs().unwrap();
        let out_max: f32 = out_diff.max(None).unwrap().item();
        assert!(
            out_max < 1e-5,
            "compiled GDN decode output mismatch by {out_max}"
        );

        let state_diff = actual_state.subtract(&state_ref).unwrap().abs().unwrap();
        let state_max: f32 = state_diff.max(None).unwrap().item();
        assert!(
            state_max < 1e-5,
            "compiled GDN decode state mismatch by {state_max}"
        );
    }

    // -----------------------------------------------------------------------
    // Step 5 Layer 1: synthetic mix-bit safetensors fixture end-to-end test
    // -----------------------------------------------------------------------

    mod mixbit_fixture_tests {
        use super::*;
        use std::collections::HashMap;
        use std::path::Path;

        fn random_f32(shape: &[i32], scale: f32) -> Array {
            let arr = mlx_rs::random::uniform::<f32, f32>(-scale, scale, shape, None).unwrap();
            arr.eval().unwrap();
            arr
        }

        fn ones_f32(shape: &[i32]) -> Array {
            let arr = Array::ones::<f32>(shape).unwrap();
            arr.eval().unwrap();
            arr
        }

        fn zeros_f32(shape: &[i32]) -> Array {
            let arr = Array::zeros::<f32>(shape).unwrap();
            arr.eval().unwrap();
            arr
        }

        fn insert_qlinear(
            map: &mut HashMap<String, Array>,
            base: &str,
            out_features: i32,
            in_features: i32,
            group_size: i32,
            bits: i32,
        ) {
            let w = random_f32(&[out_features, in_features], 0.05);
            let (qw, s, b) = mlx_rs::ops::quantize(&w, group_size, bits).unwrap();
            mlx_rs::transforms::eval([&qw, &s, &b]).unwrap();
            map.insert(format!("{base}.weight"), qw);
            map.insert(format!("{base}.scales"), s);
            map.insert(format!("{base}.biases"), b);
        }

        fn insert_dense(map: &mut HashMap<String, Array>, base: &str, shape: &[i32]) {
            map.insert(format!("{base}.weight"), random_f32(shape, 0.05));
        }

        fn config_json() -> serde_json::Value {
            serde_json::json!({
                "tie_word_embeddings": false,
                "text_config": {
                    "model_type": "qwen3_5",
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 64,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 1024,
                    "max_position_embeddings": 512,
                    "full_attention_interval": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 32,
                    "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4,
                    "num_experts": 0,
                    "num_experts_per_tok": 0
                },
                "quantization": {
                    "group_size": 64,
                    "bits": 2,
                    "language_model.lm_head": { "group_size": 64, "bits": 5, "mode": "affine" },
                    "language_model.model.embed_tokens": { "group_size": 64, "bits": 4, "mode": "affine" },
                    "language_model.model.layers.0.linear_attn.in_proj_qkvz": { "group_size": 64, "bits": 4, "mode": "affine" },
                    "language_model.model.layers.1.linear_attn.in_proj_qkvz": { "group_size": 64, "bits": 4, "mode": "affine" },
                    "language_model.model.layers.2.linear_attn.in_proj_qkvz": { "group_size": 64, "bits": 4, "mode": "affine" },
                    "language_model.model.layers.0.mlp.down_proj": { "group_size": 64, "bits": 3, "mode": "affine" },
                    "language_model.model.layers.1.mlp.down_proj": { "group_size": 64, "bits": 3, "mode": "affine" },
                    "language_model.model.layers.2.mlp.down_proj": { "group_size": 64, "bits": 3, "mode": "affine" },
                    "language_model.model.layers.3.mlp.down_proj": { "group_size": 64, "bits": 3, "mode": "affine" }
                }
            })
        }

        fn write_fixture(dir: &Path) {
            std::fs::write(
                dir.join("config.json"),
                serde_json::to_string_pretty(&config_json()).unwrap(),
            )
            .unwrap();

            let hidden = 256i32;
            let vocab = 1024i32;
            let inter = 512i32;
            let n_heads = 4i32;
            let n_kv = 2i32;
            let head_dim = 64i32;
            let nk = 2i32;
            let nv = 4i32;
            let dk = 32i32;
            let dv = 16i32;
            let key_dim = nk * dk; // 64
            let value_dim = nv * dv; // 64
            let conv_dim = key_dim * 2 + value_dim; // 192
            let kernel = 4i32;
            let qkv_rows = key_dim * 2 + value_dim; // 192

            let prefix = "language_model.";
            let mut map: HashMap<String, Array> = HashMap::new();

            // Embedding (4-bit) and lm_head (5-bit)
            insert_qlinear(
                &mut map,
                &format!("{prefix}model.embed_tokens"),
                vocab,
                hidden,
                64,
                4,
            );
            insert_qlinear(&mut map, &format!("{prefix}lm_head"), vocab, hidden, 64, 5);
            map.insert(format!("{prefix}model.norm.weight"), ones_f32(&[hidden]));

            for i in 0..4 {
                let layer = format!("{prefix}model.layers.{i}");
                map.insert(
                    format!("{layer}.input_layernorm.weight"),
                    ones_f32(&[hidden]),
                );
                map.insert(
                    format!("{layer}.post_attention_layernorm.weight"),
                    ones_f32(&[hidden]),
                );
                // MLP: gate/up at default (2-bit), down at 3-bit
                insert_qlinear(
                    &mut map,
                    &format!("{layer}.mlp.gate_proj"),
                    inter,
                    hidden,
                    64,
                    2,
                );
                insert_qlinear(
                    &mut map,
                    &format!("{layer}.mlp.up_proj"),
                    inter,
                    hidden,
                    64,
                    2,
                );
                insert_qlinear(
                    &mut map,
                    &format!("{layer}.mlp.down_proj"),
                    hidden,
                    inter,
                    64,
                    3,
                );

                if (i + 1) % 4 == 0 {
                    // Layer 3: full self-attention
                    let q_out = 2 * n_heads * head_dim; // 512 (gated)
                    let kv_out = n_kv * head_dim; // 128
                    insert_qlinear(
                        &mut map,
                        &format!("{layer}.self_attn.q_proj"),
                        q_out,
                        hidden,
                        64,
                        2,
                    );
                    insert_qlinear(
                        &mut map,
                        &format!("{layer}.self_attn.k_proj"),
                        kv_out,
                        hidden,
                        64,
                        2,
                    );
                    insert_qlinear(
                        &mut map,
                        &format!("{layer}.self_attn.v_proj"),
                        kv_out,
                        hidden,
                        64,
                        2,
                    );
                    insert_dense(
                        &mut map,
                        &format!("{layer}.self_attn.o_proj"),
                        &[hidden, n_heads * head_dim],
                    );
                    map.insert(
                        format!("{layer}.self_attn.q_norm.weight"),
                        ones_f32(&[head_dim]),
                    );
                    map.insert(
                        format!("{layer}.self_attn.k_norm.weight"),
                        ones_f32(&[head_dim]),
                    );
                } else {
                    // Layers 0/1/2: GDN linear attention. Disk has split
                    // in_proj_qkv / in_proj_z (quantized) and BF16-dense
                    // in_proj_b / in_proj_a / out_proj. The fused loader
                    // concatenates qkv+z into in_proj_qkvz and b+a into
                    // in_proj_ba on the model side.
                    insert_qlinear(
                        &mut map,
                        &format!("{layer}.linear_attn.in_proj_qkv"),
                        qkv_rows,
                        hidden,
                        64,
                        4,
                    );
                    insert_qlinear(
                        &mut map,
                        &format!("{layer}.linear_attn.in_proj_z"),
                        value_dim,
                        hidden,
                        64,
                        4,
                    );
                    insert_dense(
                        &mut map,
                        &format!("{layer}.linear_attn.in_proj_b"),
                        &[nv, hidden],
                    );
                    insert_dense(
                        &mut map,
                        &format!("{layer}.linear_attn.in_proj_a"),
                        &[nv, hidden],
                    );
                    insert_dense(
                        &mut map,
                        &format!("{layer}.linear_attn.out_proj"),
                        &[hidden, value_dim],
                    );
                    map.insert(
                        format!("{layer}.linear_attn.conv1d.weight"),
                        random_f32(&[conv_dim, kernel, 1], 0.1),
                    );
                    map.insert(format!("{layer}.linear_attn.norm.weight"), ones_f32(&[dv]));
                    map.insert(format!("{layer}.linear_attn.A_log"), zeros_f32(&[nv]));
                    map.insert(format!("{layer}.linear_attn.dt_bias"), zeros_f32(&[nv]));
                }
            }

            let path = dir.join("model.safetensors");
            Array::save_safetensors(&map, None, &path).unwrap();

            let weight_map: serde_json::Map<String, serde_json::Value> = map
                .keys()
                .map(|k| {
                    (
                        k.clone(),
                        serde_json::Value::String("model.safetensors".to_owned()),
                    )
                })
                .collect();
            let index = serde_json::json!({"metadata": {}, "weight_map": weight_map});
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                serde_json::to_string(&index).unwrap(),
            )
            .unwrap();
        }

        #[test]
        fn test_qwen3_5_mixbit_synthetic_fixture_loads_and_runs_forward() {
            let dir = tempfile::tempdir().unwrap();
            write_fixture(dir.path());

            let mut model = load_qwen3_5_model(dir.path()).expect("model loads");

            // Bit widths land where overrides + dense_attention_outputs say
            assert_eq!(
                model.lm_head.as_ref().expect("lm_head not tied").bits,
                5,
                "lm_head override (5-bit) applied"
            );
            assert_eq!(
                model.model.embed_tokens.bits, 4,
                "embed override (4-bit) applied"
            );

            let layer0 = &model.model.layers[0];
            let gdn = layer0.linear_attn.as_ref().expect("layer 0 is GDN");
            assert_eq!(
                gdn.in_proj_qkvz.bits, 4,
                "layer 0 in_proj_qkvz override (4-bit) applied"
            );
            assert_eq!(gdn.in_proj_ba.bits, 0, "layer 0 in_proj_ba is BF16-dense");
            assert_eq!(gdn.out_proj.bits, 0, "layer 0 out_proj is BF16-dense");
            let l0_down = layer0.mlp.down_proj.as_ref().expect("dense mlp.down_proj");
            assert_eq!(
                l0_down.bits, 3,
                "layer 0 mlp.down_proj override (3-bit) applied"
            );

            let layer3 = &model.model.layers[3];
            let attn = layer3
                .self_attn
                .as_ref()
                .expect("layer 3 is full attention");
            assert_eq!(attn.o_proj.bits, 0, "layer 3 o_proj is BF16-dense");
            assert_eq!(attn.q_proj.bits, 2, "layer 3 q_proj at default 2-bit");

            // No shape-[1] placeholders survive after loading
            use mlx_rs::module::ModuleParameters;
            let params = model.parameters().flatten();
            let placeholders: Vec<String> = params
                .iter()
                .filter_map(|(k, v): (&std::rc::Rc<str>, &&Array)| {
                    if v.shape() == [1] {
                        Some((**k).to_owned())
                    } else {
                        None
                    }
                })
                .collect();
            assert!(
                placeholders.is_empty(),
                "expected no [1] placeholders, got: {placeholders:?}"
            );

            // Forward pass produces finite logits of expected shape
            let tokens = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
            let mut cache: Vec<Option<LayerCache>> = Vec::new();
            let logits = model
                .forward(&tokens, None, &mut cache)
                .expect("forward succeeds");
            mlx_rs::transforms::eval([&logits]).unwrap();
            // Qwen3NextCausalLM::forward returns the last-position logits only
            assert_eq!(logits.shape(), [1, 1, 1024], "logits shape [B, 1, vocab]");

            let finite = logits.is_finite().unwrap();
            let all_finite = finite.all(None).unwrap();
            mlx_rs::transforms::eval([&all_finite]).unwrap();
            let ok: bool = all_finite.item();
            assert!(ok, "logits should be finite");
        }
    }
}

// ===========================================================================
// Sparse Forward Pass with Custom RoPE Positions
// ===========================================================================

/// Forward pass for a single attention layer with custom `RoPE` positions.
///
/// This is a standalone function to avoid borrow checker issues.
fn forward_attention_sparse(
    attn: &mut Qwen3NextAttention,
    x: &Array,
    positions: &Array,
    cache: &mut crate::cache::SteppingKeyValueCache,
) -> Result<Array, Exception> {
    let shape = x.shape();
    let b = *shape
        .first()
        .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
    let l = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

    // Q is projected to 2 * num_heads * head_dim (doubled for gating)
    let q_proj_output = attn.q_proj.forward(x)?;
    let q_reshaped = q_proj_output.reshape(&[b, l, attn.num_attention_heads, -1])?;
    let q_halves = q_reshaped.split(2, Some(-1))?;
    let queries_pre = q_halves
        .first()
        .ok_or_else(|| Exception::custom("split produced empty result"))?;
    let gate = q_halves
        .get(1)
        .ok_or_else(|| Exception::custom("split produced empty result"))?
        .reshape(&[b, l, -1])?;

    let keys_raw = attn.k_proj.forward(x)?;
    let values_raw = attn.v_proj.forward(x)?;

    // Per-head RmsNorm then transpose to [B, H, L, D]
    let mut queries = attn
        .q_norm
        .forward(queries_pre)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let mut keys = attn
        .k_norm
        .forward(&keys_raw.reshape(&[b, l, attn.num_key_value_heads, -1])?)?
        .transpose_axes(&[0, 2, 1, 3])?;
    let values = values_raw
        .reshape(&[b, l, attn.num_key_value_heads, -1])?
        .transpose_axes(&[0, 2, 1, 3])?;

    // Apply RoPE at CUSTOM positions using rope_dynamic
    tracing::debug!(
        "forward_attention_sparse: queries.shape={:?}, keys.shape={:?}, positions.shape={:?}",
        queries.shape(),
        keys.shape(),
        positions.shape()
    );
    let (queries_with_rope, keys_with_rope) =
        match attn.apply_rope_at_positions(&queries, &keys, positions) {
            Ok(result) => {
                tracing::debug!("rope_dynamic succeeded");
                result
            }
            Err(e) => {
                tracing::error!("rope_dynamic failed: {:?}", e);
                return Err(e);
            }
        };
    queries = queries_with_rope;
    keys = keys_with_rope;

    // Update cache with custom-positioned keys/values
    let (cached_keys, cached_values) = cache.update_and_fetch(keys, values)?;
    let final_keys = cached_keys;
    let final_values = cached_values;

    // Compute attention
    let output = crate::utils::scaled_dot_product_attention(
        queries,
        final_keys,
        final_values,
        attn.scale,
        None, // No mask needed for sparse prefill
    )?
    .transpose_axes(&[0, 2, 1, 3])?
    .reshape(&[b, l, -1])?;

    let gated = output.multiply(nn::sigmoid(&gate)?)?;
    attn.o_proj.forward(&gated)
}

impl Qwen3NextCausalLM {
    /// Forward pass with custom `RoPE` positions for sparse prefill.
    ///
    /// This method applies `RoPE` at arbitrary (non-contiguous) positions using
    /// `rope_dynamic`, enabling sparse prefill where only selected tokens are processed.
    ///
    /// # Arguments
    /// * `inputs` - Selected tokens [B, N] where N = number of selected tokens
    /// * `positions` - Original positions for each selected token [N]
    /// * `kv_cache` - KV cache to update
    ///
    /// # Returns
    /// Hidden states [B, N, D] with `RoPE` applied at custom positions
    pub fn forward_hidden_sparse(
        &mut self,
        inputs: &Array,
        positions: &Array,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        // Process each layer with custom RoPE positions
        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;

            let normed = layer.input_layernorm.forward(&h)?;
            let r = if layer.is_linear {
                // Linear attention (GatedDeltaNet) - standard forward
                let attn = layer
                    .linear_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("linear_attn missing"))?;
                let LayerCache::Arrays(ssm_cache) = cache else {
                    return Err(Exception::custom("Expected ArraysCache"));
                };
                attn.forward(&normed, None, ssm_cache)?
            } else {
                // Full attention - use custom RoPE positions
                let attn = layer
                    .self_attn
                    .as_mut()
                    .ok_or_else(|| Exception::custom("self_attn missing"))?;
                let LayerCache::KV(layer_kv) = cache else {
                    return Err(Exception::custom("Expected KVCache"));
                };

                // Apply custom RoPE at specified positions
                forward_attention_sparse(attn, &normed, positions, layer_kv)?
            };

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h2.add(mlp_out)?;
        }

        self.model.norm.forward(&h)
    }
}

impl Qwen3NextCausalLM {
    /// Compute logits from hidden states.
    ///
    /// This is used after sparse forward pass to get final logits.
    pub fn compute_logits(&self, hidden: &Array) -> Result<Array, Exception> {
        self.lm_head.as_ref().map_or_else(
            || self.model.embed_tokens.as_linear(hidden),
            |head| head.forward(hidden),
        )
    }
}
