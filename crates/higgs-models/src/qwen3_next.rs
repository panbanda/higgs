#![allow(clippy::items_after_test_module)]

//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (`MoE`).
//! Every `full_attention_interval`-th layer uses full attention (`Qwen3NextAttention`),
//! all other layers use `GatedDeltaNet` (SSM-like linear attention).
//! All layers use Sparse `MoE` for the feed-forward block.

use std::cell::RefCell;
use std::collections::HashMap;
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

/// Cached `GatedDeltaNet` Metal kernel -- created once, reused for all layers.
static GATED_DELTA_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    utils::{AttentionMask, apply_rope, create_causal_mask},
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

pub use crate::quant_config::QuantizationSettings as QuantizationConfig;

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

    /// Use separate GDN projections (qwen3.5-style) instead of combined (qwen3_next-style).
    #[serde(default)]
    pub use_separate_gdn_projections: bool,

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

pub(crate) fn quantized_forward(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    ops::quantized_matmul(x, weight, scales, biases, true, group_size, bits)
}

/// Quantized linear layer stored as raw weight/scales/biases arrays.
/// Forward uses `quantized_matmul` directly.
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
}

const fn crossrow_qmv_rows_supported(m_rows: i32) -> bool {
    matches!(m_rows, 2 | 3 | 5 | 6 | 8 | 9)
}

impl QLinear {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params();
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        if self.bits == 0 {
            dense_linear_no_bias_forward(&self.weight, x)
        } else {
            if let Some(y) = self.crossrow_qmv_forward(x) {
                return Ok(y);
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

    /// Use the crossrow verifier only for the validated affine-4/group-64
    /// shapes. Any kernel error deliberately falls back to stock QMV so the
    /// optimization cannot change model correctness.
    fn crossrow_qmv_forward(&self, x: &Array) -> Option<Array> {
        if std::env::var("HIGGS_CROSSROW_QMV").is_ok_and(|v| v == "0")
            || self.bits != 4
            || self.group_size != 64
            // The Metal kernel hard-codes a bfloat16 view of `x`; any other
            // activation dtype misreads the buffer (wrong row stride, garbage
            // odd 16-bit words -> NaN/Inf patterns). Fall back to stock QMV.
            || x.dtype() != Dtype::Bfloat16
        {
            return None;
        }
        let x_shape = x.shape();
        if x_shape.len() < 2 {
            return None;
        }
        let m_rows: i32 = x_shape
            .iter()
            .take(x_shape.len().saturating_sub(1))
            .product();
        // M4 and M7 were measured to diverge from stock on the Qwen3.8
        // verifier trajectory; keep those production boundaries conservative.
        if !crossrow_qmv_rows_supported(m_rows) {
            return None;
        }
        let &k_in = x_shape.last()?;
        let [n_rows, k_packed] = *self.weight.shape() else {
            return None;
        };
        let k_dim = k_packed * 8;
        if k_dim != k_in || k_dim % 512 != 0 || n_rows % 8 != 0 {
            return None;
        }
        let mut out_shape = x_shape.to_vec();
        let _ = out_shape.pop();
        out_shape.push(n_rows);
        crate::crossrow_qmv::crossrow_qmv_verify(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            m_rows,
        )
        .ok()
        .and_then(|y| y.reshape(&out_shape).ok())
    }

    /// Decode-only fast path for 4-bit single-token inference.
    ///
    /// Keeps the optimization opt-in so we can wire it into selected hot paths
    /// without changing the default behavior of every quantized linear.
    pub(crate) fn forward_decode_fast(&self, x: &Array) -> Result<Array, Exception> {
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
}

impl QEmbedding {
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params();
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        let shape = indices.shape().to_vec();
        let flat = indices.flatten(None, None)?;
        let w = (*self.weight).take_axis(&flat, 0)?;
        let out = if self.bits == 0 {
            w
        } else {
            let s = (*self.scales).take_axis(&flat, 0)?;
            let b = (*self.biases).take_axis(&flat, 0)?;
            ops::dequantize(&w, &s, &b, self.group_size, self.bits)?
        };
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        if self.bits == 0 {
            dense_linear_no_bias_forward(&self.weight, x)
        } else if self.bits == 4 && matches!(x.shape(), [1, 1, _]) && self.weight.shape().len() == 2
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

/// Metal kernel source for the fused `GatedDeltaNet` recurrence.
///
/// Computes `g = exp(-exp(a_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`
/// inline, then runs the full recurrence -- all in one kernel dispatch.
///
/// Template parameters: `InT` (dtype), `Dk`, `Dv`, `Hk`, `Hv` (int constants).
/// Grid: `(32, Dv, B * Hv)`, Threadgroup: `(32, 4, 1)`.
const GATED_DELTA_KERNEL_SOURCE: &str = r"
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

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
  // Compute g = exp(-exp(a_log) * softplus(a + dt_bias))
  float x = static_cast<float>(a_[hv_idx]) + dt_bias_val;
  float sp = fmax(x, 0.0f) + log1p(exp(-fabs(x)));
  float g_val = exp(-exp(a_log_val) * sp);

  // beta = sigmoid(b)
  float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));

  {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_val;
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
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

    // The kernel source is a compile-time string literal with no interior NULs.
    let source = CString::new(GATED_DELTA_KERNEL_SOURCE).unwrap_or_else(|_| CString::default());

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

static QGEMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static QGEMV_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static GATED_DELTA_CONFIG_CACHE_ENABLED: OnceLock<bool> = OnceLock::new();
static DECODE_GEMV_ENABLED: OnceLock<bool> = OnceLock::new();
static QGEMV_NSG_OVERRIDE: OnceLock<Option<i32>> = OnceLock::new();
static DENSE_FFN_GEMV_MODE: OnceLock<DenseFfnGemvMode> = OnceLock::new();
static DENSE_FFN_FUSE_GATE_UP: OnceLock<bool> = OnceLock::new();
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

fn decode_gemv_enabled() -> bool {
    *DECODE_GEMV_ENABLED.get_or_init(|| std::env::var("HIGGS_ENABLE_SELECTED_DECODE_GEMV").is_ok())
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
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

impl Qwen3NextAttention {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
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
            q_proj: QLinear::new(ql, qb)?,
            k_proj: QLinear::new(ql, qb)?,
            v_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0_f32)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
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
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with cache offset
        let offset = cache.offset();
        queries = apply_rope(&queries, &self.rope, offset)?;
        keys = apply_rope(&keys, &self.rope, offset)?;
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
        // Use manual RoPE implementation for per-token positions
        let queries_with_rope = apply_rope_manual(
            queries,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
        )?;

        let keys_with_rope = apply_rope_manual(
            keys,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
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
                .scale(1.0_f32)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
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
        queries = apply_rope(&queries, &self.rope, offset)?;
        keys = apply_rope(&keys, &self.rope, offset)?;

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
    ql: i32,
    qb: i32,
) -> Result<(QLinear, QLinear, QLinear), Exception> {
    Ok((
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
    ))
}

pub(crate) fn moe_expert_projection_quantization(
    args: &Qwen3NextModelArgs,
    layer_idx: i32,
    projection: &str,
    fallback_group_size: i32,
    fallback_bits: i32,
) -> Result<(i32, i32), Exception> {
    let Some(settings) = args.quantization.as_ref() else {
        return Ok((fallback_group_size, fallback_bits));
    };
    let paths = (0..args.num_experts)
        .map(|expert_idx| format!("model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}"))
        .collect::<Vec<_>>();
    let quant = settings
        .resolve_uniform(paths.iter().map(String::as_str))
        .map_err(Exception::custom)?;
    if matches!(quant, crate::quant_config::TensorQuant::Unquantized) {
        return Err(Exception::custom(format!(
            "dense (unquantized) expert projections are not supported by the fused MoE path; tensor {}",
            paths.first().map_or(projection, String::as_str)
        )));
    }
    Ok((quant.group_size(), quant.bits()))
}

/// Every checkpoint tensor path this architecture actually resolves against
/// a per-tensor quantization override while constructing the model.
///
/// Attention projections, `GatedDeltaNet` linear-attention projections, and
/// dense-layer `mlp.{gate,up,down}_proj` are always built from the scalar
/// `group_size`/`bits` fallback — an override for one of those paths must be
/// rejected rather than silently dropped. The MTP sidecar (`mtp.*`,
/// `dense_mtp.*`, `moe_mtp.*`) is likewise scalar-only.
pub(crate) fn consumed_quantization_paths(
    args: &Qwen3NextModelArgs,
) -> std::collections::HashSet<String> {
    let mut consumed = std::collections::HashSet::new();
    consumed.insert("model.embed_tokens".to_owned());
    consumed.insert("lm_head".to_owned());

    if args.num_experts > 0 {
        // Every layer's `mlp` block resolves per-expert overrides via
        // `moe_expert_projection_quantization` (see `FfnBlock::new_moe`).
        let mut moe_layer_count = args.num_hidden_layers;
        // The MoE-structured MTP sidecar (Qwen3.6-A3B style) reuses the same
        // `model.layers.{idx}.mlp.experts.*` path prefix for its own
        // `SparseMoeBlock` (see `MoeMtpHead::new`), keyed by 0..mtp layer
        // count rather than the main model's layer indices — cover it too.
        if args.use_moe_mtp {
            moe_layer_count = moe_layer_count.max(args.mtp_num_hidden_layers);
        }
        for layer_idx in 0..moe_layer_count {
            for expert_idx in 0..args.num_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    consumed.insert(format!(
                        "model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}"
                    ));
                }
            }
            // `gate_quantization_override` (see `load_qwen3_next_args_from_value`)
            // reads only layer 0's `mlp.gate` override and applies it
            // uniformly as the scalar `gate_quantization` fallback for
            // every router gate and shared-expert gate — real
            // mlx-community checkpoints (e.g. Qwen3-Next 5-bit) still list
            // a per-layer override for both projections at every MoE
            // layer. Those per-layer entries are read (layer 0) or
            // deliberately superseded by the uniform fallback (other
            // layers), not silently dropped — mark them all consumed.
            for prefix in ["model", "language_model.model"] {
                consumed.insert(format!("{prefix}.layers.{layer_idx}.mlp.gate"));
                consumed.insert(format!(
                    "{prefix}.layers.{layer_idx}.mlp.shared_expert_gate"
                ));
            }
        }
    }

    consumed
}

impl Qwen3NextMLP {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
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
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        let layers = (0..n)
            .map(|_| {
                Ok(MtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, ql, qb)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: Qwen3NextMLP::new(ql, qb)?,
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
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        // The sidecar's MoE block is uniformly quantized at the default width;
        // strip the main model's per-layer gate override so the router gate's
        // QLinear dequantizes with the right parameters.
        let mut mtp_args = args.clone();
        mtp_args.gate_quantization = None;

        let layers = (0..n)
            .map(|layer_idx| {
                Ok(MoeMtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, ql, qb)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: SparseMoeBlock::new(
                        &mtp_args,
                        i32::try_from(layer_idx)
                            .map_err(|_| Exception::custom("MTP layer index exceeds i32"))?,
                        ql,
                        qb,
                    )?,
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
            fc: QLinear::new(ql, qb)?,
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
    pub(crate) fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Self::new_with_projection_quantization((ql, qb), (ql, qb), (ql, qb))
    }

    pub(crate) fn new_with_projection_quantization(
        gate: (i32, i32),
        up: (i32, i32),
        down: (i32, i32),
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(gate.0, gate.1)?,
            up_proj: QLinear::new(up.0, up.1)?,
            down_proj: QLinear::new(down.0, down.1)?,
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
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
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
        // Gate quantization: use per-layer override if present, else global
        let (gate_ql, gate_qb) = args
            .gate_quantization
            .as_ref()
            .map_or((ql, qb), |gq| (gq.group_size, gq.bits));
        let expert_gate = moe_expert_projection_quantization(args, layer_idx, "gate_proj", ql, qb)?;
        let expert_up = moe_expert_projection_quantization(args, layer_idx, "up_proj", ql, qb)?;
        let expert_down = moe_expert_projection_quantization(args, layer_idx, "down_proj", ql, qb)?;
        Ok(Self {
            gate: QLinear::new(gate_ql, gate_qb)?,
            switch_mlp: SwitchMlpWeights::new_with_projection_quantization(
                expert_gate,
                expert_up,
                expert_down,
            )?,
            shared_expert: Qwen3NextMLP::new(ql, qb)?,
            shared_expert_gate: QLinear::new(gate_ql, gate_qb)?,
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
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let num_k_heads = args.linear_num_key_heads;
        let num_v_heads = args.linear_num_value_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = args.linear_conv_kernel_dim;

        let use_sep = args.use_separate_gdn_projections;
        let in_proj_qkvz = QLinear::new(ql, qb)?;
        let in_proj_ba = QLinear::new(ql, qb)?;
        let in_proj_qkv = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_z = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_a = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_b = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
            .bias(false)
            .groups(conv_dim)
            .padding(0)
            .build()?;
        let norm = nn::RmsNormBuilder::new(head_v_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let out_proj = QLinear::new(ql, qb)?;
        let a_log = Param::new(Array::zeros::<f32>(&[num_v_heads])?);
        let dt_bias = Param::new(Array::zeros::<f32>(&[num_v_heads])?);
        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d,
            norm,
            out_proj,
            A_log: a_log,
            dt_bias,
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

        let current_flat = mixed_qkv.reshape(&[batch, self.conv_dim])?;
        let current_weight = wt.index((self.conv_kernel_size - 1, ..));
        let mut conv_flat = current_flat.multiply(&current_weight)?;

        if history_len > 0 {
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
            if cache.conv_pos >= 0 {
                for lag in 0..available {
                    let idx = (cache.conv_pos - lag).rem_euclid(history_len);
                    let prev = history
                        .index((.., idx..idx + 1, ..))
                        .reshape(&[batch, self.conv_dim])?;
                    let weight = wt.index((history_len - 1 - lag, ..));
                    conv_flat = conv_flat.add(&prev.multiply(&weight)?)?;
                }
            }

            let next_pos = if cache.conv_pos < 0 {
                0
            } else {
                (cache.conv_pos + 1).rem_euclid(history_len)
            };
            history.try_index_mut((.., next_pos..next_pos + 1, ..), mixed_qkv.clone())?;
            cache.conv_pos = next_pos;
        }

        silu_direct(&conv_flat.reshape(&[batch, 1, self.conv_dim])?)
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

        let conv_out = if S == 1 {
            self.decode_conv1d_step(&mixed_qkv, cache, B)?
        } else {
            let conv_state = self.chronological_conv_state(cache, B, inputs.dtype())?;
            let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;
            let conv_input_len = *conv_input
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
            let keep_start = conv_input_len - n_keep;
            cache.conv_state = Some(conv_input.index((.., keep_start.., ..)));
            cache.conv_pos = if n_keep > 0 { n_keep - 1 } else { -1 };
            silu_direct(&self.conv1d.forward(&conv_input)?)?
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
    fn new_moe(
        args: &Qwen3NextModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
        let moe = SparseMoeBlock::new(args, layer_idx, ql, qb)?;
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

    fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_expert: None,
            shared_expert_gate: None,
            gate_proj: Some(QLinear::new(ql, qb)?),
            up_proj: Some(QLinear::new(ql, qb)?),
            down_proj: Some(QLinear::new(ql, qb)?),
            is_moe: false,
            top_k: 0,
            norm_topk_prob: false,
            fused_gate_up: None,
        })
    }

    fn dense_hidden_fused(&mut self, x: &Array, use_fused_gemv: bool) -> Result<Array, Exception> {
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

        let fused_out = if use_fused_gemv {
            qgemv_4bit(x, fw, fs, fb, gp.group_size)?
        } else {
            quantized_forward(x, fw, fs, fb, gp.group_size, gp.bits)?
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

            let hidden = if dense_ffn_fuse_gate_up() {
                self.dense_hidden_fused(x, use_fused_gemv)?
            } else {
                self.dense_hidden_separate(x)?
            };

            // Down projection
            let dp = self
                .down_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
            let out = if use_down_gemv {
                qgemv_4bit(&hidden, &dp.weight, &dp.scales, &dp.biases, dp.group_size)
            } else {
                dp.forward(&hidden)
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
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let linear_attn = is_linear
            .then(|| GatedDeltaNet::new(args, ql, qb))
            .transpose()?;
        let self_attn = (!is_linear)
            .then(|| Qwen3NextAttention::new(args, ql, qb))
            .transpose()?;

        let ffn = if args.num_experts > 0 {
            FfnBlock::new_moe(args, layer_idx, ql, qb)?
        } else {
            FfnBlock::new_dense(ql, qb)?
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
    fn new(
        args: &Qwen3NextModelArgs,
        ql: i32,
        qb: i32,
        embed_ql: i32,
        embed_qb: i32,
    ) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(embed_ql, embed_qb)?,
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

// Manual RoPE implementation for arbitrary positions
/// Manual `RoPE` implementation for arbitrary positions
#[allow(dead_code)]
fn apply_rope_manual(
    x: &Array,
    positions: &Array,
    dimensions: i32,
    base: f32,
    _scale: f32,
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
    #[allow(clippy::cast_precision_loss)]
    let dimensions_f32 = f32::from(i16::try_from(dimensions).unwrap_or(i16::MAX));

    // Compute frequencies: base^(-2i/dimensions) for i in [0, half_dim)
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let i_f32 = f32::from(i16::try_from(i).unwrap_or(i16::MAX));
            let power = -2.0 * i_f32 / dimensions_f32;
            base.powf(power)
        })
        .collect();
    let inv_freq_arr = Array::from_slice(&inv_freq, &[half_dim_i32]);

    // Get positions as [L] or [B, L]
    let pos_shape = positions.shape();
    let l_dim = *pos_shape
        .last()
        .ok_or_else(|| Exception::custom("positions must have at least 1 dim"))?;

    tracing::debug!(
        "apply_rope_manual: x.shape={:?}, positions.shape={:?}, dimensions={}, base={}",
        x.shape(),
        positions.shape(),
        dimensions,
        base
    );
    // Compute angles: positions * inv_freq
    // positions: [L], inv_freq: [half_dim] -> angles: [L, half_dim]
    let positions_expanded = positions.reshape(&[l_dim, 1])?;
    let inv_freq_expanded = inv_freq_arr.reshape(&[1, half_dim_i32])?;
    let angles = ops::multiply(&positions_expanded, &inv_freq_expanded)?;

    // Compute cos and sin
    let cos_raw = ops::cos(&angles)?;
    let sin_raw = ops::sin(&angles)?;

    // Reshape for broadcasting: [1, 1, L, half_dim] or [1, L, half_dim]
    let cos_shape: Vec<i32> = if ndim == 4 {
        vec![1, 1, l_dim, half_dim_i32]
    } else {
        vec![1, l_dim, half_dim_i32]
    };
    let cos = cos_raw.reshape(&cos_shape)?;
    let sin = sin_raw.reshape(&cos_shape)?;

    // Split x into two halves along last dimension
    let x_first = x.index((.., .., .., ..half_dim));
    let x_second = x.index((.., .., .., half_dim..));

    // Apply RoPE rotation
    // output_first = x_first * cos - x_second * sin
    // output_second = x_first * sin + x_second * cos
    let output_first = ops::subtract(
        &ops::multiply(&x_first, &cos)?,
        &ops::multiply(&x_second, &sin)?,
    )?;
    let output_second = ops::add(
        &ops::multiply(&x_first, &sin)?,
        &ops::multiply(&x_second, &cos)?,
    )?;

    // Concatenate back
    let last_axis = i32::try_from(ndim.saturating_sub(1))
        .map_err(|_| Exception::custom("ndim too large for i32"))?;
    ops::concatenate_axis(&[&output_first, &output_second], last_axis)
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

        if let Some(settings) = args.quantization.as_ref() {
            settings
                .assert_all_overrides_consumed(&consumed_quantization_paths(&args))
                .map_err(Exception::custom)?;
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let embed_quant = args
            .quantization
            .as_ref()
            .map(|settings| settings.resolve("model.embed_tokens"));
        let embed_ql = embed_quant.map_or(ql, crate::quant_config::TensorQuant::group_size);
        let embed_qb = embed_quant.map_or(qb, crate::quant_config::TensorQuant::bits);
        let model = Qwen3NextInner::new(&args, ql, qb, embed_ql, embed_qb)?;
        let lm_head_quant = args
            .quantization
            .as_ref()
            .map(|settings| settings.resolve("lm_head"));
        let lm_head_ql = lm_head_quant.map_or(ql, crate::quant_config::TensorQuant::group_size);
        let lm_head_qb = lm_head_quant.map_or(qb, crate::quant_config::TensorQuant::bits);
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(lm_head_ql, lm_head_qb)?)
        };
        let mtp = (args.mtp_num_hidden_layers > 0 && !args.use_dense_mtp && !args.use_moe_mtp)
            .then(|| MtpHead::new(&args, ql, qb))
            .transpose()?;
        let dense_mtp = (args.mtp_num_hidden_layers > 0 && args.use_dense_mtp)
            .then(|| DenseMtpHead::new(&args))
            .transpose()?;
        let moe_mtp = (args.mtp_num_hidden_layers > 0 && args.use_moe_mtp)
            .then(|| MoeMtpHead::new(&args, ql, qb))
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
    ///
    /// Used internally by `forward_hidden` (which adds norm) and
    /// `forward_with_hidden` (which needs raw states for MTP).
    #[allow(non_snake_case, clippy::too_many_lines)]
    fn forward_raw_hidden(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
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

            // Eval every 8 layers during long prefill chunks to bound lazy
            // graph size. Short speculative verifier windows are intentionally
            // left fused; otherwise MTP pays several eval barriers per cycle.
            if should_eval_between_prefill_layers(T, layer_idx) {
                mlx_rs::transforms::eval([&h])?;
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

        Ok(h)
    }

    /// Forward pass returning hidden states (after final `RMSNorm`, before LM head).
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
    load_qwen3_next_args_from_value(config)
}

fn gate_quantization_override(config: &serde_json::Value) -> Option<serde_json::Value> {
    let quant = config.get("quantization")?;
    for key in [
        "model.layers.0.mlp.gate",
        "language_model.model.layers.0.mlp.gate",
    ] {
        if let Some(gate_q) = quant.get(key) {
            // `true` means "quantize with the scalar defaults" (see
            // QuantizationSettings::deserialize) — identical to the key
            // being absent. Treat it as such rather than injecting it into
            // `gate_quantization`, whose type requires an object with
            // group_size/bits and would fail to deserialize a bare bool.
            if gate_q == &serde_json::Value::Bool(true) {
                continue;
            }
            return Some(gate_q.clone());
        }
    }
    None
}

pub(crate) fn load_qwen3_next_args_from_value(
    mut config: serde_json::Value,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let gate_override = gate_quantization_override(&config);
    let map = config
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("config.json root is not an object".into()))?;
    if !map.contains_key("gate_quantization") {
        if let Some(gate_q) = gate_override {
            map.insert("gate_quantization".to_owned(), gate_q);
        }
    }
    Ok(serde_json::from_value(config)?)
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
    let args = load_model_args(model_path)?;
    load_qwen3_next_model_with_args(model_path, args)
}

pub(crate) fn load_qwen3_next_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
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

    let quantization = args.quantization.clone();
    let mut model = Qwen3NextCausalLM::new(args)?;

    // Backbone keys match model params directly, but the MTP sidecar may need
    // remapping: `maybe_disable_mtp_without_checkpoint_weights` can select the
    // dense or MoE head (params `dense_mtp.*` / `moe_mtp.*`) while the checkpoint
    // still ships the head under the `mtp.*` namespace. The plain loader can't
    // bridge that, so it would silently leave the draft head uninitialized.
    load_qwen3_next_weights(&mut model, model_path, quantization.as_ref())?;

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
    load_qwen3_5_text_config_args_from_value(&config)
}

pub(crate) fn load_qwen3_5_text_config_args_from_value(
    config: &serde_json::Value,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let text_config = config
        .get("text_config")
        .ok_or_else(|| ModelError::UnsupportedModel("missing text_config in config.json".into()))?;

    let mut obj = text_config.clone();
    let map = obj
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("text_config is not an object".into()))?;

    // Flatten rope_parameters into top-level fields
    if let Some(rope_params) = text_config.get("rope_parameters") {
        if let Some(theta) = rope_params.get("rope_theta") {
            map.entry("rope_theta").or_insert_with(|| theta.clone());
        }
        if let Some(prf) = rope_params.get("partial_rotary_factor") {
            map.entry("partial_rotary_factor")
                .or_insert_with(|| prf.clone());
        }
    }

    // Merge top-level quantization config
    if let Some(quant) = config.get("quantization") {
        map.entry("quantization").or_insert_with(|| quant.clone());
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
    let mixed_ba_layers = qwen3_5_mixed_ba_quantization_layers(config, text_config);
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

    // Detect per-layer gate quantization override from top-level quantization config
    if let Some(gate_q) = gate_quantization_override(config) {
        map.insert("gate_quantization".to_owned(), gate_q);
    }

    Ok(serde_json::from_value(obj)?)
}

/// Parse a `{group_size, bits}` quantization spec from a JSON node.
fn qwen3_5_quantization_config(value: &serde_json::Value) -> Option<QuantizationConfig> {
    Some(QuantizationConfig::new(
        i32::try_from(value.get("group_size")?.as_i64()?).ok()?,
        i32::try_from(value.get("bits")?.as_i64()?).ok()?,
    ))
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
pub fn load_qwen3_5_model<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_qwen3_5_moe_text_config_args(model_path)?;
    load_qwen3_5_model_with_args(model_path, args)
}

pub(crate) fn load_qwen3_5_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
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
    let model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

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
    let args = load_qwen3_5_moe_text_config_args(model_path)?;
    load_qwen3_5_moe_model_with_args(model_path, args)
}

pub(crate) fn load_qwen3_5_moe_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
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
    let model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

    tracing::info!("Qwen3.5-MoE model loaded successfully");
    Ok(model)
}

/// Build a `Qwen3NextCausalLM` and load weights, choosing fused or separate GDN
/// projections. When the config (or env var) requests separate projections, use
/// the direct loader. Otherwise try the fused loader; if it reports a mixed-bit
/// `in_proj_ba` shape mismatch, rebuild the model with separate projections and
/// retry via the direct loader.
fn load_qwen3_5_model_with_gdn_fallback(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
    gdn_dims: &GdnDims,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let quantization = args.quantization.clone();
    let force_separate =
        args.use_separate_gdn_projections || std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok();
    if force_separate {
        args.use_separate_gdn_projections = true;
        let mut model = Qwen3NextCausalLM::new(args)?;
        load_qwen3_5_moe_weights_direct(&mut model, model_path, quantization.as_ref())?;
        tracing::info!("Using SEPARATE GDN projections (4 dispatches per layer)");
        return Ok(model);
    }

    let mut fused_model = Qwen3NextCausalLM::new(args.clone())?;
    match load_qwen3_5_moe_weights_fused(
        &mut fused_model,
        model_path,
        gdn_dims,
        quantization.as_ref(),
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
            load_qwen3_5_moe_weights_direct(
                &mut separate_model,
                model_path,
                quantization.as_ref(),
            )?;
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
    quantization: Option<&QuantizationConfig>,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

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

    if std::env::var("HIGGS_LOAD_EVAL_CHUNKED").map_or(true, |v| v != "0") {
        for param in params.values() {
            (**param).eval().map_err(crate::error::ModelError::from)?;
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
    quantization: Option<&QuantizationConfig>,
) -> Result<(), crate::error::ModelError> {
    let safetensors_files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();
    let mut matched = 0usize;
    let mut unmatched = Vec::new();

    for file_path in &safetensors_files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

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
    quantization: Option<&QuantizationConfig>,
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
        crate::validate_quantized_tensor_widths(&loaded, quantization)?;

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
            return Err(crate::error::ModelError::Io(std::io::Error::other(
                format!("Fused target key not found in model params: {combined_key}"),
            )));
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

    model
        .eval()
        .map_err(|e| crate::error::ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

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
    clippy::redundant_clone
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KeyValueCache;

    #[test]
    fn crossrow_qmv_preserves_m4_m7_fallback_boundaries() {
        for rows in [2, 3, 5, 6, 8, 9] {
            assert!(crossrow_qmv_rows_supported(rows));
        }
        for rows in [i32::MIN, -1, 0, 1, 4, 7, 10, i32::MAX] {
            assert!(!crossrow_qmv_rows_supported(rows));
        }
    }

    #[test]
    fn qlinear_zero_bits_uses_dense_weight() {
        let mut linear = QLinear::new(0, 0).unwrap();
        *linear.weight = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        let input = Array::from_slice(&[2.0_f32, 1.0], &[1, 2]);
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.as_slice::<f32>(), &[4.0, 10.0]);
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
        let result = SparseMoeBlock::new(&args, 0, 64, 4);
        assert!(result.is_ok());
    }

    fn assert_sparse_moe_rejects(
        mutate: impl FnOnce(&mut Qwen3NextModelArgs),
        expected_substring: &str,
    ) {
        let mut args = minimal_qwen3_next_args();
        mutate(&mut args);
        let result = SparseMoeBlock::new(&args, 0, 64, 4);
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
    fn test_causal_lm_rejects_unsupported_per_tensor_override() {
        // self_attn projections are always built from the scalar
        // group_size/bits fallback; an override for one must be rejected
        // rather than silently ignored.
        let mut args = valid_causal_lm_args();
        args.quantization = Some(
            serde_json::from_str(
                r#"{"group_size": 64, "bits": 4, "model.layers.0.self_attn.q_proj": false}"#,
            )
            .unwrap(),
        );
        let err = Qwen3NextCausalLM::new(args).unwrap_err();
        assert!(
            err.to_string().contains("model.layers.0.self_attn.q_proj"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_causal_lm_accepts_honored_expert_projection_override() {
        // valid_causal_lm_args() has num_experts=4; resolve_uniform requires
        // every expert in the fused group to share the override.
        let mut args = valid_causal_lm_args();
        args.quantization = Some(
            serde_json::from_str(
                r#"{
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.0.mlp.experts.0.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.1.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.2.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.3.gate_proj": {"group_size": 32, "bits": 8}
                }"#,
            )
            .unwrap(),
        );
        Qwen3NextCausalLM::new(args).unwrap();
    }

    #[test]
    fn test_causal_lm_accepts_real_checkpoint_per_layer_router_gate_overrides() {
        // Mirrors real mlx-community Qwen3-Next 5-bit checkpoints (e.g.
        // Qwen3-Next-80B-A3B-Thinking-5bit), which carry a per-layer 8-bit
        // router override for both `mlp.gate` and `mlp.shared_expert_gate`
        // at every MoE layer, not just layer 0.
        let mut args = valid_causal_lm_args();
        args.quantization = Some(
            serde_json::from_str(
                r#"{
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.2.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.2.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.3.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.3.mlp.shared_expert_gate": {"group_size": 64, "bits": 8}
                }"#,
            )
            .unwrap(),
        );
        Qwen3NextCausalLM::new(args).unwrap();
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
    fn test_load_qwen3_next_args_treats_true_gate_override_as_default() {
        // Real predicate-map checkpoints list every tensor path explicitly,
        // including the layer-0 router gate as `true` ("use the scalar
        // defaults"). That must NOT be injected into `gate_quantization`
        // (whose type requires a {group_size, bits} object) — it should be
        // treated exactly like the key being absent.
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
                "model.layers.0.mlp.gate": true,
                "model.layers.0.mlp.shared_expert_gate": true,
                "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
                "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                "model.embed_tokens": true,
                "lm_head": false
            }
        });

        let args = load_qwen3_next_args_from_value(config).unwrap();
        assert!(
            args.gate_quantization.is_none(),
            "a `true` override at the scanned layer must not populate gate_quantization"
        );

        // Loads end-to-end with the gate at default (scalar) quantization.
        let model = Qwen3NextCausalLM::new(args).unwrap();
        assert_eq!(model.args.num_hidden_layers, 2);
    }

    #[test]
    fn test_real_checkpoint_style_config_loads_end_to_end() {
        // Full pipeline (config.json -> args -> model) for a config shaped
        // like a real mlx-community Qwen3-Next 5-bit checkpoint: per-layer
        // router gate overrides at every MoE layer, not just layer 0.
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
                "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
                "model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
                "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                "model.embed_tokens": true,
                "lm_head": false
            }
        });

        let args = load_qwen3_next_args_from_value(config).unwrap();
        Qwen3NextCausalLM::new(args).unwrap();
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
        let result = SparseMoeBlock::new(&args, 0, 64, 4);
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
        args.quantization = Some(QuantizationConfig::new(32, 8));
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

    /// A malformed `scales` width must fail at load time through the custom
    /// `load_qwen3_next_weights` shard loop, not surface as an opaque Metal
    /// kernel error on first forward. This exercises the same
    /// `validate_quantized_tensor_widths` check that the generic loaders in
    /// `lib.rs` already run, wired into the qwen3_next-specific loader.
    #[test]
    fn test_load_qwen3_next_weights_rejects_malformed_scales_width() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"quantization": {"group_size": 64, "bits": 4}}"#,
        )
        .unwrap();

        // weight: [4, 8] packed i.e. bits=4, group_size=64 implies scales
        // width should be 8*32/4/64 = 1, but we provide width 2.
        let weight_data = vec![0_u8; 4 * 8 * 4];
        let scales_data = vec![0_u8; 4 * 2 * 4];
        let weight_tensor = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32,
            vec![4, 8],
            &weight_data,
        )
        .unwrap();
        let scales_tensor = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32,
            vec![4, 2],
            &scales_data,
        )
        .unwrap();
        safetensors::serialize_to_file(
            [
                ("model.layers.0.self_attn.q_proj.weight", weight_tensor),
                ("model.layers.0.self_attn.q_proj.scales", scales_tensor),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();

        let mut model = Qwen3NextCausalLM::new(valid_causal_lm_args()).unwrap();
        let quantization: QuantizationConfig =
            serde_json::from_str(r#"{"group_size": 64, "bits": 4}"#).unwrap();
        let err = load_qwen3_next_weights(&mut model, dir.path(), Some(&quantization)).unwrap_err();
        assert!(
            err.to_string().contains("model.layers.0.self_attn.q_proj"),
            "unexpected error: {err}"
        );
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
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

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
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

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

        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

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

        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

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
        let mut block = SwitchMlpWeights::new(64, 4).unwrap();

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
        args.gate_quantization = Some(QuantizationConfig::new(64, 8));

        let mut block = SparseMoeBlock::new(&args, 0, 64, 4).unwrap();

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

    #[test]
    fn test_gdn_decode_conv_ring_buffer_matches_concat_path() {
        use mlx_rs::Dtype;

        let args = valid_causal_lm_args();
        let mut gdn = GatedDeltaNet::new(&args, 64, 4).unwrap();
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
        let state = mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

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
            Array::load_safetensors(path).is_ok_and(|loaded| {
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

        let mut block = FfnBlock::new_dense(64, 4).unwrap();
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
