//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (`MoE`).
//! Every `full_attention_interval`-th layer uses full attention (`Qwen3NextAttention`),
//! all other layers use `GatedDeltaNet` (SSM-like linear attention).
//! All layers use Sparse `MoE` for the feed-forward block.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use mlx_rs::{
    Array, Stream,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// FFI error capture for gather_qmm
// ---------------------------------------------------------------------------

/// Captures the most recent MLX error message from our FFI calls.
static FFI_LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Error handler registered once with MLX to capture error messages.
#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    if let Ok(mut guard) = FFI_LAST_ERROR.lock() {
        *guard = Some(s);
    }
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

// The kernel object is immutable after creation and used read-only in apply.
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
    utils::{apply_rope, scaled_dot_product_attention},
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

/// Quantization parameters from config.json (top-level defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

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
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

// ---------------------------------------------------------------------------
// Quantized weight containers
// ---------------------------------------------------------------------------

type QuantizedParams = (Param<Array>, Param<Array>, Param<Array>);

pub(crate) fn init_quantized_params() -> Result<QuantizedParams, Exception> {
    Ok((
        Param::new(Array::zeros::<f32>(&[1])?),
        Param::new(Array::zeros::<f32>(&[1])?),
        Param::new(Array::zeros::<f32>(&[1])?),
    ))
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

impl QLinear {
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params()?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array, Exception> {
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
    pub(crate) fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        let (weight, scales, biases) = init_quantized_params()?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    pub(crate) fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        let full = ops::dequantize(
            &*self.weight,
            &*self.scales,
            &*self.biases,
            self.group_size,
            self.bits,
        )?;
        full.take_axis(indices, 0)
    }

    pub(crate) fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
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

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

pub(crate) fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    gate.multiply(nn::sigmoid(gate)?)?.multiply(x)
}

fn silu_direct(x: &Array) -> Result<Array, Exception> {
    x.multiply(nn::sigmoid(x)?)
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
            .lock()
            .ok()
            .and_then(|mut g| g.take())
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

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

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
  o_state[s_idx] = static_cast<InT>(state[i]);
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
            in_dtype,
        );

        config
    }
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
    let config = configure_gated_delta_kernel(
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
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
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
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(t_scalar);
    }

    result
}

// ---------------------------------------------------------------------------
// Qwen3NextAttention (full attention with gated Q and partial RoPE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextAttention {
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
                .scale(1.0)
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
        mask: Option<&Array>,
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

        // Per-head RmsNorm then transpose to [B, H, L, D]
        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with cache offset
        let offset = cache.offset();
        queries = apply_rope(&queries, &self.rope, offset)?;
        keys = apply_rope(&keys, &self.rope, offset)?;

        let (cached_keys, cached_values) = cache.update_and_fetch(keys, values)?;
        keys = cached_keys;
        values = cached_values;

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        let gated = output.multiply(nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextMLP (standard SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextMLP {
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
}

impl SwitchMlpWeights {
    pub(crate) fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Apply the full `SwiGLU` `MoE` block for all selected experts in one shot
    /// using `gather_qmm` (fused expert-indexed quantized matmul).
    ///
    /// `x`: `[..., D]` input
    /// `indices`: `[..., top_k]` expert indices
    /// Returns: `[..., top_k, D]`
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
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
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
        // Router gate and shared_expert_gate use 8-bit quantization
        Ok(Self {
            gate: QLinear::new(64, 8)?,
            switch_mlp: SwitchMlpWeights::new(ql, qb)?,
            shared_expert: Qwen3NextMLP::new(ql, qb)?,
            shared_expert_gate: QLinear::new(64, 8)?,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

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

        // Expert computation via fused gather_qmm
        let y = self.switch_mlp.forward_gather(x, &top_inds, false)?;

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
    pub offset: i32,
}

impl ArraysCache {
    pub const fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            offset: 0,
        }
    }
}

impl Default for ArraysCache {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
struct GatedDeltaNet {
    #[param]
    in_proj_qkvz: QLinear,
    #[param]
    in_proj_ba: QLinear,
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
    qk_norm_weight_q: Array,
    qk_norm_weight_k: Array,
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

        Ok(Self {
            in_proj_qkvz: QLinear::new(ql, qb)?,
            in_proj_ba: QLinear::new(ql, qb)?,
            conv1d: nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
                .bias(false)
                .groups(conv_dim)
                .padding(0)
                .build()?,
            norm: nn::RmsNormBuilder::new(head_v_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            out_proj: QLinear::new(ql, qb)?,
            A_log: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            dt_bias: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
            qk_norm_weight_q: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let s = dim_f32.sqrt().recip();
                let w = Array::ones::<f32>(&[head_k_dim])?.multiply(Array::from_f32(s * s))?;
                w.eval()?;
                w
            },
            qk_norm_weight_k: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let s = dim_f32.sqrt().recip();
                let w = Array::ones::<f32>(&[head_k_dim])?.multiply(Array::from_f32(s))?;
                w.eval()?;
                w
            },
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs
        let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
        let mixed_ba = self.in_proj_ba.forward(inputs)?;

        // Split into q, k, v, z, b, a
        let (q, k, v, z, b, a) = self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?;

        // Conv1d with state management
        let conv_state = match cache.conv_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.conv_kernel_size - 1, self.conv_dim],
                inputs.dtype(),
            )?,
        };

        // Concatenate q, k, v for conv input
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Prepend conv state
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // Update conv state cache (keep last kernel-1 timesteps)
        let n_keep = self.conv_kernel_size - 1;
        let conv_input_len = *conv_input
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
        let keep_start = conv_input_len - n_keep;
        cache.conv_state = Some(conv_input.index((.., keep_start.., ..)));

        let conv_out = silu_direct(&self.conv1d.forward(&conv_input)?)?;

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

        // Get or initialize SSM state: [B, Hv, Dv, Dk]
        let state = match cache.ssm_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                inputs.dtype(),
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

        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;

        // Output projection
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        self.out_proj.forward(&out_flat)
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
    mlp: SparseMoeBlock,
    is_linear: bool,
}

impl DecoderLayer {
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let linear_attn = if is_linear {
            Some(GatedDeltaNet::new(args, ql, qb)?)
        } else {
            None
        };
        let self_attn = if is_linear {
            None
        } else {
            Some(Qwen3NextAttention::new(args, ql, qb)?)
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
            mlp: SparseMoeBlock::new(args, ql, qb)?,
            is_linear,
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
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
    full_attention_interval: i32,
}

impl Qwen3NextInner {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            full_attention_interval: args.full_attention_interval,
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

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = Qwen3NextInner::new(&args, ql, qb)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(ql, qb)?)
        };

        Ok(Self {
            args,
            model,
            lm_head,
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

    /// Forward pass returning hidden states before the LM head.
    #[allow(non_snake_case)]
    pub fn forward_hidden(
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

        let fa_mask = if T > 1 {
            // Find the first full-attention layer's cache to get offset
            let fa_idx = self.model.full_attention_interval - 1;
            let fa_idx_usize =
                usize::try_from(fa_idx).map_err(|_| Exception::custom("fa_idx overflow"))?;
            let offset =
                kv_cache
                    .get(fa_idx_usize)
                    .and_then(|c| c.as_ref())
                    .map_or(0, |c| match c {
                        LayerCache::KV(kv) => kv.offset(),
                        LayerCache::Arrays(a) => a.offset,
                    });
            Some(crate::utils::create_causal_mask(T, Some(offset))?)
        } else {
            None
        };

        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask = if layer.is_linear {
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

            let h2 = h.add(r)?;
            let normed_post = layer.post_attention_layernorm.forward(&h2)?;

            let mlp_out = layer.mlp.forward(&normed_post)?;

            h = h2.add(mlp_out)?;
        }

        self.model.norm.forward(&h)
    }

    /// Forward pass producing logits.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h),
            None => self.model.embed_tokens.as_linear(&h),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load model args from config.json.
pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load a `Qwen3Next` model from a directory containing safetensors + config.json.
pub fn load_qwen3_next_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

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

    // Load weights directly from safetensors (no key remapping needed
    // since our param names match the safetensors keys exactly)
    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("Qwen3Next model loaded successfully");
    Ok(model)
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
    clippy::ref_as_ptr
)]
mod tests {
    use super::*;

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
        let result = SparseMoeBlock::new(&args, 64, 4);
        assert!(result.is_ok());
    }

    fn assert_sparse_moe_rejects(
        mutate: impl FnOnce(&mut Qwen3NextModelArgs),
        expected_substring: &str,
    ) {
        let mut args = minimal_qwen3_next_args();
        mutate(&mut args);
        let result = SparseMoeBlock::new(&args, 64, 4);
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
        assert!(!args.norm_topk_prob);
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
        let result = SparseMoeBlock::new(&args, 64, 4);
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

        let mut block = SparseMoeBlock::new(&args, 64, 4).unwrap();

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

        // Test with mlx-rs built-in ops::gather_qmm
        let n4 = 50;
        let mut total_builtin_build = 0u128;
        let mut total_builtin_eval = 0u128;
        for _ in 0..n4 {
            let t0 = std::time::Instant::now();
            let mut y3 = x.clone();
            for _ in 0..48 {
                let g = ops::gather_qmm(
                    &y3,
                    &gate_w,
                    &gate_s,
                    Some(&gate_b),
                    None::<&Array>,
                    Some(&indices),
                    true,
                    64,
                    4,
                    false,
                )
                .unwrap();
                let u = ops::gather_qmm(
                    &y3,
                    &up_w,
                    &up_s,
                    Some(&up_b),
                    None::<&Array>,
                    Some(&indices),
                    true,
                    64,
                    4,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g, &u).unwrap();
                y3 = ops::gather_qmm(
                    &activated,
                    &down_w,
                    &down_s,
                    Some(&down_b),
                    None::<&Array>,
                    Some(&indices),
                    true,
                    64,
                    4,
                    false,
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
        let model_dir = "/Users/panbanda/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        let shard = format!("{}/model-00001-of-00009.safetensors", model_dir);
        let path = std::path::Path::new(&shard);
        if !path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

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

    /// Benchmark 36 GDN layers using bare Arrays (matching Python bench_gdn_real_python.py).
    /// Isolates GDN ops from the model framework to compare GPU time vs Python.
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

        // Test 5: interleaved state + MoE using ops::gather_qmm (library version)
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
                let g_out = ops::gather_qmm(
                    &x_exp,
                    &sw_gate[i].0,
                    &sw_gate[i].1,
                    &sw_gate[i].2,
                    None::<&Array>,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = ops::gather_qmm(
                    &x_exp,
                    &sw_up[i].0,
                    &sw_up[i].1,
                    &sw_up[i].2,
                    None::<&Array>,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = ops::gather_qmm(
                    &activated,
                    &sw_down[i].0,
                    &sw_down[i].1,
                    &sw_down[i].2,
                    None::<&Array>,
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
            "Interleaved state + ops::gather_qmm (library version): {:.2}ms",
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
        let model_path = "/Users/panbanda/.cache/huggingface/hub/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/7b9321eabb85ce79625cac3f61ea691e4ea984b5";
        if !std::path::Path::new(model_path).exists() {
            println!("Model not found at {model_path}, skipping");
            return;
        }

        let mut model = load_qwen3_next_model(model_path).unwrap();
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
    #[ignore = "benchmark, requires GPU"]
    fn bench_metal_kernel_gather_qmm_interleaving() {
        let b: i32 = 1;
        let d: i32 = 2048;
        let n_layers: i32 = 48;
        let n_gdn: i32 = 36;
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
}
