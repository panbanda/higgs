//! Custom 4-bit affine quantized GEMV Metal kernel for decode.

use super::{
    cached_kernel::CachedMetalKernel,
    ffi::{FFI_LAST_ERROR, ensure_ffi_error_handler},
    flags::{qgemv_config_cache_enabled, qgemv_nsg_override},
};
use mlx_rs::{Array, Stream, error::Exception};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CString, c_char},
    sync::OnceLock,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QgemvKernelConfigKey {
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    nsg: i32,
}

thread_local! {
    static QGEMV_CONFIG_CACHE: RefCell<HashMap<QgemvKernelConfigKey, mlx_sys::mlx_fast_metal_kernel_config>> =
        RefCell::new(HashMap::new());
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
pub fn qgemv_4bit(
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

    let stream = Stream::thread_local_or_default();
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
