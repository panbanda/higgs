//! Custom Metal kernel for the `GatedDeltaNet` recurrence.

use super::{
    cached_kernel::CachedMetalKernel,
    ffi::{FFI_LAST_ERROR, ensure_ffi_error_handler},
    flags::gated_delta_config_cache_enabled,
};
use mlx_rs::{Array, Stream, error::Exception};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CString, c_char},
    sync::OnceLock,
};

/// Cached `GatedDeltaNet` Metal kernel -- created once, reused for all layers.
static GATED_DELTA_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

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

thread_local! {
    static GATED_DELTA_CONFIG_CACHE: RefCell<HashMap<GatedDeltaKernelConfigKey, mlx_sys::mlx_fast_metal_kernel_config>> =
        RefCell::new(HashMap::new());
}

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
pub(super) fn gated_delta_kernel_ffi(
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

    let stream = Stream::thread_local_or_default();
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
