//! Runtime JIT Metal kernels for Bonsai-Q1 (1-bit affine quantization).
//!
//! Upstream `oxideai/mlx-rs` ships no `bits=1` affine kernels (MLX gates affine
//! quant to `bits >= 2`), so `ops::quantized_matmul`/`ops::dequantize` with
//! `bits=1` fail at runtime with `Unable to load kernel affine_dequantize_*_b_1`.
//!
//! Rather than fork mlx-rs (which forces a full from-source mlx-c rebuild), we
//! add the missing kernels *from this crate* using the runtime JIT facility that
//! mlx-c already exposes (`mlx_fast_metal_kernel_*`) and that `mlx-sys` compiles
//! in. The kernels below are JIT-compiled by Metal at first use and cached by
//! MLX internally per template instantiation. This keeps us on the stock
//! `oxideai/mlx-rs` pin with no extra native recompile.
//!
//! The FFI plumbing (kernel handle wrapper, `Array` <-> `mlx_array`, vector
//! construction, error capture) mirrors the proven `qgemv_4bit` path in
//! [`crate::qwen3_next`]; the kernel math mirrors
//! [`crate::bonsai_q1::PackedQ1Linear::dequant_row_to_fp32`]:
//! `W[r,c] = scale[r, c/G] * bit + bias[r, c/G]`, `bit = (w[r, c/32] >> (c%32)) & 1`.
//! Checkpoints whose affine metadata is symmetric use an empty bias sentinel;
//! their kernels derive `bias = -scale / 2` and never read a bias buffer.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, error::Exception};

// ---------------------------------------------------------------------------
// FFI error capture (per-thread, mirrors qwen3_next).
// ---------------------------------------------------------------------------

thread_local! {
    static FFI_LAST_ERROR: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Error handler registered once with MLX to capture error messages on the
/// calling thread.
#[allow(unsafe_code)]
unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    FFI_LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(s));
}

fn ensure_ffi_error_handler() {
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| {
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
        }
    });
}

fn take_last_error() -> String {
    FFI_LAST_ERROR
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_else(|| "(no MLX error message captured)".to_owned())
}

// ---------------------------------------------------------------------------
// Cached kernel handle.
// ---------------------------------------------------------------------------

/// Wraps a compiled `mlx_fast_metal_kernel`, freed on drop.
struct CachedMetalKernel(mlx_sys::mlx_fast_metal_kernel);

// SAFETY: the handle is created once and only ever read (passed by value to
// `mlx_fast_metal_kernel_apply`); no interior mutability is shared across threads.
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

/// Number of simdgroups per threadgroup for the fused matvec. More simdgroups
/// help large-K layers (fewer chunk barriers). Overridable for tuning.
fn qmv_nsg(k_dim: i32) -> i32 {
    static OVERRIDE: OnceLock<Option<i32>> = OnceLock::new();
    let ovr = *OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_QMV_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|n| matches!(n, 4 | 8 | 16 | 32))
    });
    ovr.unwrap_or(if k_dim > 8192 { 16 } else { 8 })
}

/// Build the vector-of-strings that names kernel inputs/outputs.
#[allow(unsafe_code)]
fn cstr_vec(names: &[&CStr]) -> mlx_sys::mlx_vector_string {
    let ptrs: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
    unsafe { mlx_sys::mlx_vector_string_new_data(ptrs.as_ptr().cast_mut(), ptrs.len()) }
}

// ---------------------------------------------------------------------------
// Materializing device copy.
//
// MLX 0.30.6's `mlx_copy` is a copy-on-write alias: evaluation calls
// `copy_shared_buffer`, so a small view can keep a much larger backing
// allocation alive. A FastMetal custom-kernel output has a stronger backend
// contract: `CustomKernel::eval_gpu` unconditionally allocates
// `allocator::malloc(out.nbytes())` before dispatch and has no donation path.
// This per-element identity assignment therefore preserves dtype bits while
// guaranteeing fresh, logical-size device storage.
// ---------------------------------------------------------------------------

const MATERIALIZED_COPY_KERNEL_SOURCE: &str = r"
uint elem = thread_position_in_grid.x;
dst[elem] = src[elem];
";

fn create_materialized_copy_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"src"]);
    let out_vec = cstr_vec(&[c"dst"]);
    let source = CString::new(MATERIALIZED_COPY_KERNEL_SOURCE).unwrap_or_default();
    #[allow(unsafe_code)]
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_materialized_device_copy".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // linear indexing requires row-contiguous input
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

static MATERIALIZED_COPY_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Copy `src` into independent, compact device storage without a host roundtrip.
///
/// Unlike `mlx_copy` under the pinned MLX 0.30.6 backend, FastMetal custom
/// outputs cannot alias or donate an input buffer: the backend allocates
/// exactly `out.nbytes()` before running this bit-preserving identity kernel.
#[allow(unsafe_code)]
pub(crate) fn materialized_device_copy(src: &Array) -> Result<Array, Exception> {
    if src.size() == 0 {
        let empty = mlx_rs::ops::zeros_dtype(src.shape(), src.dtype())?;
        let _token = (!crate::mlx_exec::held()).then(crate::mlx_exec::acquire);
        crate::mlx_exec::eval([&empty])?;
        return Ok(empty);
    }

    let element_count = i32::try_from(src.size()).map_err(|_| {
        Exception::custom("materialized device copy exceeds Metal's i32 grid limit")
    })?;
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let result = unsafe {
        let dtype = mlx_sys::mlx_array_dtype(src.as_ptr());
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let thread_count = element_count.min(256);
        let config_status =
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                config,
                c"T".as_ptr(),
                dtype,
            ) | mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, element_count, 1, 1)
                | mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    config,
                    thread_count,
                    1,
                    1,
                )
                | mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    config,
                    src.shape().as_ptr(),
                    src.shape().len(),
                    dtype,
                );

        if config_status != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            return Err(Exception::custom(format!(
                "configure materialized device copy failed: {}",
                take_last_error()
            )));
        }

        let cached = MATERIALIZED_COPY_KERNEL
            .get_or_init(|| CachedMetalKernel(create_materialized_copy_kernel()));
        let input_ptrs = [src.as_ptr()];
        let inputs_vec = mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len());
        let mut outputs_vec = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        );

        let result = if status != 0 {
            Err(Exception::custom(format!(
                "materialized device copy failed: {}",
                take_last_error()
            )))
        } else {
            let mut output_ptr = mlx_sys::mlx_array_new();
            let get_status = mlx_sys::mlx_vector_array_get(&raw mut output_ptr, outputs_vec, 0);
            if get_status == 0 {
                Ok(Array::from_ptr(output_ptr))
            } else {
                mlx_sys::mlx_array_free(output_ptr);
                Err(Exception::custom(format!(
                    "read materialized device copy output failed: {}",
                    take_last_error()
                )))
            }
        };

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        result
    };

    let result = result?;
    let _token = (!crate::mlx_exec::held()).then(crate::mlx_exec::acquire);
    crate::mlx_exec::eval([&result])?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Fused 1-bit quantized matvec (decode hot path).
//
// y = x @ dequant(W).T  for a single token (M = 1).
// Mirrors qgemv_4bit but unpacks 32 1-bit weights per uint32 word.
// One simdgroup per output row; x staged in threadgroup memory; simd_sum reduce.
// ---------------------------------------------------------------------------

const QMV_KERNEL_SOURCE: &str = r"
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
        int wp_off = k_off / 32;
        int wp_end = k_end / 32;
        auto w_row = w + row * KPacked;

        for (int idx = wp_off + int(lane); idx < wp_end; idx += 32) {
            uint packed = w_row[idx];
            int kl = (idx - wp_off) * 32;

            float dot_val = 0.0f;
            float sum_x = 0.0f;
            for (uint j = 0u; j < 32u; ++j) {
                float xv = float(x_sh[kl + int(j)]);
                float bit = float((packed >> j) & 1u);
                dot_val += bit * xv;
                sum_x += xv;
            }

            int g = idx * 32 / GroupSize;
            float s_val = float(sc[row * NumGroups + g]);
            float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
            acc += s_val * dot_val + b_val * sum_x;
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

#[allow(unsafe_code)]
fn create_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_qmv".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // raw pointer arithmetic requires row-contiguous inputs
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_qmv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
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
            k_dim / 32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        let nsg = qmv_nsg(k_dim);
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

/// Original per-row 1-bit matvec: one simdgroup computes one output row, with
/// `x` staged in threadgroup memory. Kept as the A/B baseline (selected when
/// `HIGGS_BONSAI_QMV_KERNEL=legacy`). See [`bonsai_q1_qmv`] for the dispatcher.
#[allow(unsafe_code)]
pub fn bonsai_q1_qmv_legacy(
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
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv: weight has no columns"))?;
    let k_dim = k_packed * 32; // 32 one-bit weights per uint32 word

    let x_flat = x.reshape(&[k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    // FastMetal still binds the affine input signature. Reuse the scale array
    // as a harmless dummy; the `Symmetric` template constant removes the bias
    // load from the compiled kernel.
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_qmv_kernel()));
    let config = configure_qmv_kernel(out_dtype, n_rows, k_dim, group_size, symmetric);

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
        Err(Exception::custom(format!(
            "bonsai_q1_qmv failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_qmv: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

static QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static FAST_QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_CONTRACT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_CONTRACT_M5_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_GATE_UP_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static TG_LUT4_GATE_UP_M5_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Simdgroups per threadgroup for the `qmv_fast`-class kernel. Each simdgroup
/// computes `RESULTS_PER_SIMDGROUP` (= 4) output rows. Tunable via
/// `HIGGS_BONSAI_FAST_NSG` (Phase-2 sweep); MLX's reference uses 2.
fn fast_qmv_nsg() -> i32 {
    static OVERRIDE: OnceLock<i32> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_FAST_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|n| matches!(n, 1 | 2 | 4 | 8))
            .unwrap_or(2)
    })
}

fn fast_q2_qmv_nsg() -> i32 {
    static OVERRIDE: OnceLock<i32> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_FAST_NSG")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|n| matches!(n, 1 | 2 | 4 | 8))
            .unwrap_or(8)
    })
}

/// Whether the fast QMV kernel may specialize away output-row bounds checks.
///
/// This is deliberately opt-in while the specialization is benchmarked on
/// real Bonsai verifier shapes. The unaligned kernel remains the fallback for
/// shapes that do not fill a complete threadgroup's output-row tile.
fn use_aligned_fast_qmv() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_ALIGNED_FAST_QMV").is_ok_and(|value| value == "1")
    })
}

const fn fast_qmv_has_aligned_rows(n_rows: i32, nsg: i32, prefer_aligned: bool) -> bool {
    const RESULTS_PER_SIMDGROUP: i32 = 4;
    prefer_aligned && n_rows > 0 && n_rows % (nsg * RESULTS_PER_SIMDGROUP) == 0
}

/// Whether to route the decode matvec through the `qmv_fast`-class kernel.
/// It is the **default** (measured 2.3× faster on Bonsai-8B decode and bit-exact
/// vs the CPU reference); opt back to the original per-row kernel with
/// `HIGGS_BONSAI_QMV_KERNEL=legacy`.
fn use_fast_qmv() -> bool {
    static FAST: OnceLock<bool> = OnceLock::new();
    *FAST.get_or_init(|| {
        !std::env::var("HIGGS_BONSAI_QMV_KERNEL").is_ok_and(|v| v.eq_ignore_ascii_case("legacy"))
    })
}

/// Fused 1-bit quantized matvec: `y = x @ dequant(weight).T` for a single token.
///
/// `x` must hold exactly `in_features` elements (M = 1). `weight` is the packed
/// `[out_features, in_features/32]` uint32 matrix; `scales`/`biases` are
/// `[out_features, in_features/group_size]`. Output dtype matches `x`.
///
/// Dispatches to the `qmv_fast`-class kernel ([`bonsai_q1_qmv_fast`]) by
/// default; set `HIGGS_BONSAI_QMV_KERNEL=legacy` to force the per-row kernel.
pub fn bonsai_q1_qmv(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    if use_fast_qmv() {
        bonsai_q1_qmv_fast(x, weight, scales, biases, group_size)
    } else {
        bonsai_q1_qmv_legacy(x, weight, scales, biases, group_size)
    }
}

// ---------------------------------------------------------------------------
// `qmv_fast`-class 1-bit narrow matrix multiply (decode / verify hot path).
//
// Ports MLX/PrismML `qmv_fast` tiling onto our uint32 packing: each simdgroup
// computes RESULTS_PER_SIMDGROUP (4) output rows for one input row; the grid's
// z dimension covers narrow M > 1 verifier batches without materializing the
// dense weight matrix. Each lane holds VPT (32) input values in registers and
// reuses them across all 4 output rows. Keeping one packed word per lane
// reduces register pressure and raises occupancy for 1-bit weights. The bits=1
// affine math is identical to the legacy kernel —
// `scale * sum(bit*x) + bias * sum(x)` — only the data movement differs.
// Group scales/biases are per-lane (a lane's 32 values lie in one 128-wide
// group); per-row partials are simd_sum-reduced.
// ---------------------------------------------------------------------------

const FAST_QMV_KERNEL_SOURCE: &str = r"
constexpr int VPT = 32;          // values_per_thread (one packed word per lane)
constexpr int RPS = 4;           // results_per_simdgroup
constexpr int WPT = VPT / 32;    // packed uint32 words per thread (1)
constexpr int BLK = VPT * 32;    // block_size = 1024

uint tgx = threadgroup_position_in_grid.x;
uint sg  = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;
uint batch = threadgroup_position_in_grid.z;

int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;
auto x_row = x + int(batch) * K;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

int aligned_end = (K / BLK) * BLK;

// Main loop: full 1024-element blocks (covers every real Bonsai layer, since
// all K are multiples of 2048).
for (int k = 0; k < aligned_end; k += BLK) {
    int xbase = k + int(lid) * VPT;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) { float v = float(x_row[xbase + i]); xt[i] = v; sum += v; }

    int wcol = (k / 32) + int(lid) * WPT;
    int g = xbase / GroupSize;   // all VPT values fall in one group

    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (!AlignedN) {
            if (row >= n_param) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            uint packed = w[row * KPacked + wcol + wp];
            int xo = wp * 32;
            for (int bk = 0; bk < 4; ++bk) {
                uint wb = (packed >> (uint(bk) * 8u)) & 0xFFu;
                int b = xo + bk * 8;
                accum += select(0.0f, xt[b + 0], (wb & 0x01u) != 0u);
                accum += select(0.0f, xt[b + 1], (wb & 0x02u) != 0u);
                accum += select(0.0f, xt[b + 2], (wb & 0x04u) != 0u);
                accum += select(0.0f, xt[b + 3], (wb & 0x08u) != 0u);
                accum += select(0.0f, xt[b + 4], (wb & 0x10u) != 0u);
                accum += select(0.0f, xt[b + 5], (wb & 0x20u) != 0u);
                accum += select(0.0f, xt[b + 6], (wb & 0x40u) != 0u);
                accum += select(0.0f, xt[b + 7], (wb & 0x80u) != 0u);
            }
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

// Tail: only exercised by tests with K < 2048 or K % 2048 != 0.
if (aligned_end < K) {
    int xbase = aligned_end + int(lid) * VPT;
    bool in_bounds = xbase < K;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) {
        float v = (in_bounds && (xbase + i) < K) ? float(x_row[xbase + i]) : 0.0f;
        xt[i] = v;
        sum += v;
    }
    int wcol = (aligned_end / 32) + int(lid) * WPT;
    int g = in_bounds ? (xbase / GroupSize) : 0;
    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            if (!in_bounds) { continue; }
        } else {
            if (row >= n_param || !in_bounds) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            int widx = wcol + wp;
            if (widx >= KPacked) { continue; }
            uint packed = w[row * KPacked + widx];
            int xo = wp * 32;
            for (int bk = 0; bk < 4; ++bk) {
                uint wb = (packed >> (uint(bk) * 8u)) & 0xFFu;
                int b = xo + bk * 8;
                accum += select(0.0f, xt[b + 0], (wb & 0x01u) != 0u);
                accum += select(0.0f, xt[b + 1], (wb & 0x02u) != 0u);
                accum += select(0.0f, xt[b + 2], (wb & 0x04u) != 0u);
                accum += select(0.0f, xt[b + 3], (wb & 0x08u) != 0u);
                accum += select(0.0f, xt[b + 4], (wb & 0x10u) != 0u);
                accum += select(0.0f, xt[b + 5], (wb & 0x20u) != 0u);
                accum += select(0.0f, xt[b + 6], (wb & 0x40u) != 0u);
                accum += select(0.0f, xt[b + 7], (wb & 0x80u) != 0u);
            }
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = Symmetric ? (-0.5f * s_val) : float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

for (int r = 0; r < RPS; ++r) {
    int row = out_row + r;
    float v = simd_sum(result[r]);
    if (lid == 0u) {
        if constexpr (AlignedN) {
            y[int(batch) * n_param + row] = OutT(v);
        } else if (row < n_param) {
            y[int(batch) * n_param + row] = OutT(v);
        }
    }
}
";

#[allow(unsafe_code)]
fn create_fast_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(FAST_QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_qmv_fast".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // raw pointer arithmetic requires row-contiguous inputs
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_fast_qmv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    m_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
    prefer_aligned: bool,
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
            k_dim / 32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        // Each simdgroup computes 4 rows; nsg simdgroups per threadgroup.
        let nsg = fast_q2_qmv_nsg();
        let rows_per_tg = nsg * 4;
        let aligned_n = fast_qmv_has_aligned_rows(n_rows, nsg, prefer_aligned || nsg == 8);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"AlignedN".as_ptr(),
            i32::from(aligned_n),
        );
        let n_tgs = (n_rows + rows_per_tg - 1) / rows_per_tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, m_rows);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);

        let y_shape = [m_rows, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

/// `qmv_fast`-class variant of [`bonsai_q1_qmv_legacy`]. Same inputs/outputs and
/// bit-exact result; faster tiling. Set `HIGGS_BONSAI_ALIGNED_FAST_QMV=1` to
/// specialize away row bounds checks when N fills complete threadgroup tiles.
/// Unaligned shapes retain the guarded kernel. See [`bonsai_q1_qmv`] for
/// dispatch.
#[allow(unsafe_code)]
pub fn bonsai_q1_qmv_fast(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q1_qmv_fast_impl(
        x,
        weight,
        scales,
        biases,
        group_size,
        use_aligned_fast_qmv(),
    )
}

#[allow(unsafe_code)]
fn bonsai_q1_qmv_fast_impl(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    prefer_aligned: bool,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: weight has no columns"))?;
    let k_dim = k_packed * 32;
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();

    let x_flat = x.reshape(&[m_rows, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = FAST_QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_fast_qmv_kernel()));
    let config = configure_fast_qmv_kernel(
        out_dtype,
        n_rows,
        m_rows,
        k_dim,
        group_size,
        symmetric,
        prefer_aligned,
    );

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
        Err(Exception::custom(format!(
            "bonsai_q1_qmv_fast failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_qmv_fast: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

/// Packed affine Q1 matrix multiply for narrow verifier batches.
///
/// This shares the decode-optimized kernel with [`bonsai_q1_qmv_fast`] but
/// dispatches one grid slice per flattened input row. It intentionally targets
/// small sequence lengths: weights stay packed and resident, avoiding the very
/// large temporary produced by full dequantization.
pub fn bonsai_q1_qmm(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q1_qmv_fast(x, weight, scales, biases, group_size)
}

// ---------------------------------------------------------------------------
// `qmv_fast`-class 2-bit narrow matrix multiply (decode / verify hot path).
//
// Direct port of the 1-bit `qmv_fast` tiling for Q2 affine packed weights:
// each simdgroup computes RESULTS_PER_SIMDGROUP (4) output rows for one input
// row; the grid's z dimension covers narrow M > 1 verifier batches without
// materializing the dense weight matrix. Each lane holds VPT (16) input
// values in registers (= one packed u32 word, which holds 16 Q2 weights) and
// reuses them across all 4 output rows.
//
// Q2 math: per weight, `w = scale*q + bias` where `q = (word >> 2*col) & 0b11`.
// Per 16-weight tile per row: `result = scale * sum(q_i * x_i) + bias * sum(x_i)`.
// Biases are always retained (no symmetric compaction for Q2 v1).
//
// No LUT identity here -- at M=1, direct `q*x` fma is faster than building a
// LUT. The LUT identity becomes useful at M=5 in the TG-LUT4 M=5 kernel
// (Phase 3D), where one packed-word load is shared across 5 verifier rows.
// ---------------------------------------------------------------------------

const FAST_Q2_QMV_KERNEL_SOURCE: &str = r"
constexpr int VPT = 32;          // values_per_thread (TWO packed u32 words per lane, matches Q1's register footprint)
constexpr int RPS = 4;           // results_per_simdgroup
constexpr int WPT = VPT / 16;    // packed uint32 words per thread (2)
constexpr int BLK = VPT * 32;    // block_size = 1024 (same iteration count as Q1)

uint tgx = threadgroup_position_in_grid.x;
uint sg  = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;
uint batch = threadgroup_position_in_grid.z;

int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;
auto x_row = x + int(batch) * K;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

int aligned_end = (K / BLK) * BLK;

// Main loop: full 1024-element blocks (same iteration count as Q1).
for (int k = 0; k < aligned_end; k += BLK) {
    int xbase = k + int(lid) * VPT;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) { float v = float(x_row[xbase + i]); xt[i] = v; sum += v; }

    int wcol = (k / 16) + int(lid) * WPT;
    int g = xbase / GroupSize;   // all VPT=32 values still fall in one 128-wide group

    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (!AlignedN) {
            if (row >= n_param) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            uint packed = w[row * KPacked + wcol + wp];
            int xo = wp * 16;
            // Unrolled 16-way 2-bit unpack + fma per word. Metal compiler unrolls fully.
            accum += float((packed >>  0u) & 0x3u) * xt[xo +  0];
            accum += float((packed >>  2u) & 0x3u) * xt[xo +  1];
            accum += float((packed >>  4u) & 0x3u) * xt[xo +  2];
            accum += float((packed >>  6u) & 0x3u) * xt[xo +  3];
            accum += float((packed >>  8u) & 0x3u) * xt[xo +  4];
            accum += float((packed >> 10u) & 0x3u) * xt[xo +  5];
            accum += float((packed >> 12u) & 0x3u) * xt[xo +  6];
            accum += float((packed >> 14u) & 0x3u) * xt[xo +  7];
            accum += float((packed >> 16u) & 0x3u) * xt[xo +  8];
            accum += float((packed >> 18u) & 0x3u) * xt[xo +  9];
            accum += float((packed >> 20u) & 0x3u) * xt[xo + 10];
            accum += float((packed >> 22u) & 0x3u) * xt[xo + 11];
            accum += float((packed >> 24u) & 0x3u) * xt[xo + 12];
            accum += float((packed >> 26u) & 0x3u) * xt[xo + 13];
            accum += float((packed >> 28u) & 0x3u) * xt[xo + 14];
            accum += float((packed >> 30u) & 0x3u) * xt[xo + 15];
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

// Tail: only exercised by tests with K < 1024 or K % 1024 != 0.
if (aligned_end < K) {
    int xbase = aligned_end + int(lid) * VPT;
    bool in_bounds = xbase < K;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) {
        float v = (in_bounds && (xbase + i) < K) ? float(x_row[xbase + i]) : 0.0f;
        xt[i] = v;
        sum += v;
    }
    int wcol = (aligned_end / 16) + int(lid) * WPT;
    int g = in_bounds ? (xbase / GroupSize) : 0;
    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            if (!in_bounds) { continue; }
        } else {
            if (row >= n_param || !in_bounds) { continue; }
        }
        float accum = 0.0f;
        for (int wp = 0; wp < WPT; ++wp) {
            int widx = wcol + wp;
            if (widx >= KPacked) { continue; }
            uint packed = w[row * KPacked + widx];
            int xo = wp * 16;
            // Tail bounds-check: each code may be past K.
            int codes_in_bounds = (xo + 16 <= K - xbase) ? 16 : max(0, K - xbase - xo);
            if (codes_in_bounds >  0) { accum += float((packed >>  0u) & 0x3u) * xt[xo +  0]; }
            if (codes_in_bounds >  1) { accum += float((packed >>  2u) & 0x3u) * xt[xo +  1]; }
            if (codes_in_bounds >  2) { accum += float((packed >>  4u) & 0x3u) * xt[xo +  2]; }
            if (codes_in_bounds >  3) { accum += float((packed >>  6u) & 0x3u) * xt[xo +  3]; }
            if (codes_in_bounds >  4) { accum += float((packed >>  8u) & 0x3u) * xt[xo +  4]; }
            if (codes_in_bounds >  5) { accum += float((packed >> 10u) & 0x3u) * xt[xo +  5]; }
            if (codes_in_bounds >  6) { accum += float((packed >> 12u) & 0x3u) * xt[xo +  6]; }
            if (codes_in_bounds >  7) { accum += float((packed >> 14u) & 0x3u) * xt[xo +  7]; }
            if (codes_in_bounds >  8) { accum += float((packed >> 16u) & 0x3u) * xt[xo +  8]; }
            if (codes_in_bounds >  9) { accum += float((packed >> 18u) & 0x3u) * xt[xo +  9]; }
            if (codes_in_bounds > 10) { accum += float((packed >> 20u) & 0x3u) * xt[xo + 10]; }
            if (codes_in_bounds > 11) { accum += float((packed >> 22u) & 0x3u) * xt[xo + 11]; }
            if (codes_in_bounds > 12) { accum += float((packed >> 24u) & 0x3u) * xt[xo + 12]; }
            if (codes_in_bounds > 13) { accum += float((packed >> 26u) & 0x3u) * xt[xo + 13]; }
            if (codes_in_bounds > 14) { accum += float((packed >> 28u) & 0x3u) * xt[xo + 14]; }
            if (codes_in_bounds > 15) { accum += float((packed >> 30u) & 0x3u) * xt[xo + 15]; }
        }
        float s_val = float(sc[row * NumGroups + g]);
        float b_val = float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

for (int r = 0; r < RPS; ++r) {
    int row = out_row + r;
    float v = simd_sum(result[r]);
    if (lid == 0u) {
        if constexpr (AlignedN) {
            y[int(batch) * n_param + row] = OutT(v);
        } else if (row < n_param) {
            y[int(batch) * n_param + row] = OutT(v);
        }
    }
}
";

static FAST_Q2_QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

#[allow(unsafe_code)]
fn create_fast_q2_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(FAST_Q2_QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_qmv_fast".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // raw pointer arithmetic requires row-contiguous inputs
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_fast_q2_qmv_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    m_rows: i32,
    k_dim: i32,
    group_size: i32,
    prefer_aligned: bool,
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
        // Q2 packs 16 weights per u32 (vs Q1's 32).
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 16,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );

        // Each simdgroup computes 4 rows; nsg simdgroups per threadgroup.
        let nsg = fast_qmv_nsg();
        let rows_per_tg = nsg * 4;
        let aligned_n = fast_qmv_has_aligned_rows(n_rows, nsg, prefer_aligned);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"AlignedN".as_ptr(),
            i32::from(aligned_n),
        );
        let n_tgs = (n_rows + rows_per_tg - 1) / rows_per_tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, m_rows);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);

        let y_shape = [m_rows, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Fused 2-bit quantized matvec: `y = x @ dequant(weight).T` for narrow M.
///
/// `weight` is the packed `[out_features, in_features/16]` uint32 matrix (each
/// u32 word holds 16 Q2 codes). `scales`/`biases` are
/// `[out_features, in_features/group_size]` fp16/bf16/fp32. Output dtype
/// matches `x`.
///
/// Dispatches one grid slice per flattened input row (M can be 1 for decode
/// or up to `bonsai_q2_qmm_max_rows()` for verifier batches). The kernel
/// computes `result = scale * sum(q_i * x_i) + bias * sum(x_i)` per 16-weight
/// tile per row, with biases always retained (Q2 v1 has no symmetric
/// compaction).
pub fn bonsai_q2_qmv(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q2_qmv_fast_impl(
        x,
        weight,
        scales,
        biases,
        group_size,
        use_aligned_fast_qmv(),
    )
}

/// Packed affine Q2 matrix multiply for narrow verifier batches. Wraps
/// [`bonsai_q2_qmv`] (same dispatch pattern as Q1's `bonsai_q1_qmm`).
pub fn bonsai_q2_qmm(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    bonsai_q2_qmv(x, weight, scales, biases, group_size)
}

#[allow(unsafe_code)]
fn bonsai_q2_qmv_fast_impl(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    prefer_aligned: bool,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_qmv_fast: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_qmv_fast: weight has no columns"))?;
    // Q2 packs 16 weights per u32.
    let k_dim = k_packed * 16;
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();

    let x_flat = x.reshape(&[m_rows, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    // Q2 v1 always retains biases; empty biases array is a caller bug.
    if biases.size() == 0 {
        return Err(Exception::custom(
            "bonsai_q2_qmv_fast: Q2 v1 requires a nonempty affine bias (no symmetric compaction)",
        ));
    }
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = FAST_Q2_QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_fast_q2_qmv_kernel()));
    let config =
        configure_fast_q2_qmv_kernel(out_dtype, n_rows, m_rows, k_dim, group_size, prefer_aligned);

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
        Err(Exception::custom(format!(
            "bonsai_q2_qmv_fast failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q2_qmv_fast: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

// ---------------------------------------------------------------------------
// Q2 simdgroup-cooperating decode kernel (ports MLX qmv_fast for bits=2).
//
// Unlike the scalar FAST_Q2_QMV_KERNEL (one thread per row), this kernel
// distributes the K dimension across 32 threads in a simdgroup. Each thread
// handles VPT=16 Q2 values (one packed u32 word); 32 threads cooperatively
// cover 512 K values per block. Partial results are simd_sum-reduced.
//
// This is the pattern MLX's own qmv_fast uses (quantized.h:844) and is the
// only way to match MLX stock performance — scalar per-row dispatch starves
// GPU occupancy.
// ---------------------------------------------------------------------------

const FAST_Q2_QMV_SIMD_HEADER: &str = r"
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}
template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 1) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i]+x[i+1]+x[i+2]+x[i+3]+x[i+4]+x[i+5]+x[i+6]+x[i+7];
      x_thread[i]=x[i];x_thread[i+1]=x[i+1];x_thread[i+2]=x[i+2];x_thread[i+3]=x[i+3];
      x_thread[i+4]=x[i+4];x_thread[i+5]=x[i+5];x_thread[i+6]=x[i+6];x_thread[i+7]=x[i+7];
    }
  } else if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i]+x[i+1]+x[i+2]+x[i+3];
      x_thread[i]=x[i]; x_thread[i+1]=x[i+1]/4.0f; x_thread[i+2]=x[i+2]/16.0f; x_thread[i+3]=x[i+3]/64.0f;
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)x;
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i]+x[i+1]+x[i+2]+x[i+3];
      x_thread[i]=x[i]; x_thread[i+1]=x[i+1]/16.0f; x_thread[i+2]=x[i+2]/256.0f; x_thread[i+3]=x[i+3]/4096.0f;
    }
  } else { for (int i = 0; i < values_per_thread; i++) { sum += x[i]; x_thread[i] = x[i]; } }
  return sum;
}
template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  if (bits == 1) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t wb = w[i];
      accum += select(U(0), x_thread[8*i], bool(wb&0x01));
      accum += select(U(0), x_thread[8*i+1], bool(wb&0x02));
      accum += select(U(0), x_thread[8*i+2], bool(wb&0x04));
      accum += select(U(0), x_thread[8*i+3], bool(wb&0x08));
      accum += select(U(0), x_thread[8*i+4], bool(wb&0x10));
      accum += select(U(0), x_thread[8*i+5], bool(wb&0x20));
      accum += select(U(0), x_thread[8*i+6], bool(wb&0x40));
      accum += select(U(0), x_thread[8*i+7], bool(wb&0x80));
    }
  } else if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4*i]*(w[i]&0x03) + x_thread[4*i+1]*(w[i]&0x0c) + x_thread[4*i+2]*(w[i]&0x30) + x_thread[4*i+3]*(w[i]&0xc0));
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4*i]*(ws[i]&0x000f) + x_thread[4*i+1]*(ws[i]&0x00f0) + x_thread[4*i+2]*(ws[i]&0x0f00) + x_thread[4*i+3]*(ws[i]&0xf000));
    }
  } else { for (int i = 0; i < values_per_thread; i++) { accum += x_thread[i] * w[i]; } }
  return scale * accum + bias * sum;
}
";

const FAST_Q2_QMV_SIMD_SOURCE: &str = r"
typedef half T;
typedef float U;

constexpr int bits = 2;
constexpr int packs_per_thread = bits == 2 ? 1 : 2;
constexpr int RPS = 4;                             // results per simdgroup
constexpr int pack_factor = get_pack_factor<bits, 32>(); // values packed per byte-slice
constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
constexpr int VPT = pack_factor * packs_per_thread; // values-per-thread
constexpr int BLK = VPT * 32;                      // block size
constexpr int scale_step_per_thread = GroupSize / VPT;

uint tgx = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;
uint batch = threadgroup_position_in_grid.z;

const device uint8_t* ws = (const device uint8_t*)w;
int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;

auto x_row = x + int(batch) * K;
int in_vec_size_w = K * bytes_per_pack / pack_factor;
int in_vec_size_g = K / GroupSize;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

ws += out_row * in_vec_size_w + int(lid) * packs_per_thread * bytes_per_pack;
sc += out_row * in_vec_size_g + int(lid) / scale_step_per_thread;
bi += out_row * in_vec_size_g + int(lid) / scale_step_per_thread;
x_row += int(lid) * VPT;

int aligned_end = (K / BLK) * BLK;

for (int k = 0; k < aligned_end; k += BLK) {
    U sum = load_vector<T, U, VPT, bits>(x_row, xt);
    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (!AlignedN) {
            if (row >= n_param) { continue; }
        }
        auto wl = ws + r * in_vec_size_w;
        U s_val = U(sc[r * in_vec_size_g]);
        U b_val = U(bi[r * in_vec_size_g]);
        result[r] += qdot<U, VPT, bits>(wl, xt, s_val, b_val, sum);
    }

    ws += BLK * bytes_per_pack / pack_factor;
    sc += BLK / GroupSize;
    bi += BLK / GroupSize;
    x_row += BLK;
}

if (aligned_end < K) {
    bool in_bounds = (aligned_end + int(lid) * VPT) < K;
    U sum = 0;
    if (in_bounds) {
        sum = load_vector<T, U, VPT, bits>(x_row, xt);
    } else {
        for (int i = 0; i < VPT; ++i) { xt[i] = 0.0f; }
    }

    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            if (!in_bounds) { continue; }
        } else {
            if (row >= n_param || !in_bounds) { continue; }
        }
        U s_val = in_bounds ? U(sc[r * in_vec_size_g]) : U(0);
        U b_val = in_bounds ? U(bi[r * in_vec_size_g]) : U(0);
        auto wl = ws + r * in_vec_size_w;
        result[r] += qdot<U, VPT, bits>(wl, xt, s_val, b_val, sum);
    }
}

for (int r = 0; r < RPS; ++r) {
    float v = simd_sum(result[r]);
    if (lid == 0u) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            y[int(batch) * n_param + row] = OutT(v);
        } else if (row < n_param) {
            y[int(batch) * n_param + row] = OutT(v);
        }
    }
}
";

static FAST_Q2_QMV_SIMD_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static FAST_Q2_TERNARY_QMV_SIMD_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_M5_ARGMAX_CANDIDATES_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_M5_TERNARY_ARGMAX_CANDIDATES_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_M5_ARGMAX_REDUCE_IDS_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

#[allow(unsafe_code)]
fn create_fast_q2_qmv_simd() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    // Split: helper functions go in the header (file scope), kernel body in source.
    let header = CString::new(FAST_Q2_QMV_SIMD_HEADER).unwrap_or_default();
    let source = CString::new(FAST_Q2_QMV_SIMD_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_qmv_simd_v3".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn create_fast_q2_ternary_qmv_simd() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let header = CString::new(FAST_Q2_QMV_SIMD_HEADER).unwrap_or_default();
    let source = CString::new(FAST_Q2_TERNARY_QMV_SIMD_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_ternary_qmv_simd_v2".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_fast_q2_qmv_simd(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    m_rows: i32,
    k_dim: i32,
    group_size: i32,
    prefer_aligned: bool,
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
            k_dim / 16,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        let nsg = fast_qmv_nsg();
        let rows_per_tg = nsg * 4;
        let aligned_n = fast_qmv_has_aligned_rows(n_rows, nsg, prefer_aligned);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"AlignedN".as_ptr(),
            i32::from(aligned_n),
        );
        let n_tgs = (n_rows + rows_per_tg - 1) / rows_per_tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * 32, nsg, m_rows);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1);
        let y_shape = [m_rows, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Q2 simdgroup-cooperating decode kernel. Ports MLX's qmv_fast for bits=2:
/// 32 threads per simdgroup, 4 output rows per simdgroup, simd_sum reduction.
/// This is the only pattern that matches MLX stock performance.
#[allow(unsafe_code, dead_code)]
pub fn bonsai_q2_qmv_simd(
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
        .ok_or_else(|| Exception::custom("bonsai_q2_qmv_simd: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_qmv_simd: weight has no columns"))?;
    let k_dim = k_packed * 16;
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();

    let x_flat = x.reshape(&[m_rows, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached =
        FAST_Q2_QMV_SIMD_KERNEL.get_or_init(|| CachedMetalKernel(create_fast_q2_qmv_simd()));
    let config = configure_fast_q2_qmv_simd(
        out_dtype,
        n_rows,
        m_rows,
        k_dim,
        group_size,
        use_aligned_fast_qmv(),
    );

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
        Err(Exception::custom(format!(
            "bonsai_q2_qmv_simd failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q2_qmv_simd: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

const FAST_Q2_TERNARY_QMV_SIMD_SOURCE: &str = r"
typedef half T;
typedef float U;

constexpr int bits = 2;
constexpr int packs_per_thread = 1;
constexpr int RPS = 4;
constexpr int pack_factor = get_pack_factor<bits, 32>();
constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
constexpr int VPT = pack_factor * packs_per_thread;
constexpr int BLK = VPT * 32;
constexpr int scale_step_per_thread = GroupSize / VPT;

uint tgx = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;
uint batch = threadgroup_position_in_grid.z;

const device uint8_t* ws = (const device uint8_t*)w;
int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;

auto x_row = x + int(batch) * K;
int in_vec_size_w = K * bytes_per_pack / pack_factor;
int in_vec_size_g = K / GroupSize;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

ws += out_row * in_vec_size_w + int(lid) * packs_per_thread * bytes_per_pack;
sc += out_row * in_vec_size_g + int(lid) / scale_step_per_thread;
x_row += int(lid) * VPT;

int aligned_end = (K / BLK) * BLK;

for (int k = 0; k < aligned_end; k += BLK) {
    for (int i = 0; i < VPT; ++i) { xt[i] = U(x_row[i]); }
    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (!AlignedN) {
            if (row >= n_param) { continue; }
        }
        auto wl = ws + r * in_vec_size_w;
        U s_val = U(sc[r * in_vec_size_g]);
        U accum = 0;
        for (int i = 0; i < (VPT / 4); ++i) {
            uint8_t wb = wl[i];
            accum += (U(wb & 0x03) - U(1)) * xt[4*i];
            accum += (U((wb >> 2) & 0x03) - U(1)) * xt[4*i+1];
            accum += (U((wb >> 4) & 0x03) - U(1)) * xt[4*i+2];
            accum += (U((wb >> 6) & 0x03) - U(1)) * xt[4*i+3];
        }
        result[r] += s_val * accum;
    }

    ws += BLK * bytes_per_pack / pack_factor;
    sc += BLK / GroupSize;
    x_row += BLK;
}

if (aligned_end < K) {
    bool in_bounds = (aligned_end + int(lid) * VPT) < K;
    if (in_bounds) {
        for (int i = 0; i < VPT; ++i) { xt[i] = U(x_row[i]); }
    } else {
        for (int i = 0; i < VPT; ++i) { xt[i] = 0.0f; }
    }

    for (int r = 0; r < RPS; ++r) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            if (!in_bounds) { continue; }
        } else {
            if (row >= n_param || !in_bounds) { continue; }
        }
        U s_val = in_bounds ? U(sc[r * in_vec_size_g]) : U(0);
        auto wl = ws + r * in_vec_size_w;
        U accum = 0;
        for (int i = 0; i < (VPT / 4); ++i) {
            uint8_t wb = wl[i];
            accum += (U(wb & 0x03) - U(1)) * xt[4*i];
            accum += (U((wb >> 2) & 0x03) - U(1)) * xt[4*i+1];
            accum += (U((wb >> 4) & 0x03) - U(1)) * xt[4*i+2];
            accum += (U((wb >> 6) & 0x03) - U(1)) * xt[4*i+3];
        }
        result[r] += s_val * accum;
    }
}

for (int r = 0; r < RPS; ++r) {
    float v = simd_sum(result[r]);
    if (lid == 0u) {
        int row = out_row + r;
        if constexpr (AlignedN) {
            y[int(batch) * n_param + row] = OutT(v);
        } else if (row < n_param) {
            y[int(batch) * n_param + row] = OutT(v);
        }
    }
}
";

#[allow(unsafe_code, dead_code)]
pub fn bonsai_q2_ternary_qmv_simd(
    x: &Array,
    weight: &Array,
    scales: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    let x_shape = x.shape();
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_ternary_qmv_simd: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_ternary_qmv_simd: weight has no columns"))?;
    let k_dim = k_packed * 16;
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();

    let x_flat = x.reshape(&[m_rows, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached = FAST_Q2_TERNARY_QMV_SIMD_KERNEL
        .get_or_init(|| CachedMetalKernel(create_fast_q2_ternary_qmv_simd()));
    let config = configure_fast_q2_qmv_simd(
        out_dtype,
        n_rows,
        m_rows,
        k_dim,
        group_size,
        use_aligned_fast_qmv(),
    );

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
        Err(Exception::custom(format!(
            "bonsai_q2_ternary_qmv_simd failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q2_ternary_qmv_simd: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape)
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

const Q2_M5_ARGMAX_CANDIDATES_SOURCE: &str = r"
constexpr int WG = 256;
threadgroup float v0[WG]; threadgroup float v1[WG]; threadgroup float v2[WG];
threadgroup float v3[WG]; threadgroup float v4[WG];
threadgroup float i0[WG]; threadgroup float i1[WG]; threadgroup float i2[WG];
threadgroup float i3[WG]; threadgroup float i4[WG];

uint tid = thread_index_in_threadgroup;
uint block = threadgroup_position_in_grid.x;
int n = int(block) * WG + int(tid);

float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f; float acc4 = 0.0f;

if (n < n_param) {
    for (int g = 0; g < NumGroups; ++g) {
        float qx0 = 0.0f; float qx1 = 0.0f; float qx2 = 0.0f; float qx3 = 0.0f; float qx4 = 0.0f;
        float sx0 = 0.0f; float sx1 = 0.0f; float sx2 = 0.0f; float sx3 = 0.0f; float sx4 = 0.0f;
        int word_base = n * KPacked + g * (GroupSize / 16);
        int x_base = g * GroupSize;
        #pragma clang loop unroll(full)
        for (int word = 0; word < (GroupSize / 16); ++word) {
            uint packed = w[word_base + word];
            int xb = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) {
                float q = float((packed >> uint(2 * j)) & 0x3u);
                float x0 = float(x[0 * K + xb + j]);
                float x1 = float(x[1 * K + xb + j]);
                float x2 = float(x[2 * K + xb + j]);
                float x3 = float(x[3 * K + xb + j]);
                float x4 = float(x[4 * K + xb + j]);
                qx0 += q * x0; qx1 += q * x1; qx2 += q * x2; qx3 += q * x3; qx4 += q * x4;
                sx0 += x0; sx1 += x1; sx2 += x2; sx3 += x3; sx4 += x4;
            }
        }
        float s = float(sc[n * NumGroups + g]);
        float b = float(bi[n * NumGroups + g]);
        acc0 += s * qx0 + b * sx0;
        acc1 += s * qx1 + b * sx1;
        acc2 += s * qx2 + b * sx2;
        acc3 += s * qx3 + b * sx3;
        acc4 += s * qx4 + b * sx4;
    }
    v0[tid] = acc0; v1[tid] = acc1; v2[tid] = acc2; v3[tid] = acc3; v4[tid] = acc4;
    float fid = float(n);
    i0[tid] = fid; i1[tid] = fid; i2[tid] = fid; i3[tid] = fid; i4[tid] = fid;
} else {
    v0[tid] = -INFINITY; v1[tid] = -INFINITY; v2[tid] = -INFINITY; v3[tid] = -INFINITY; v4[tid] = -INFINITY;
    i0[tid] = 0.0f; i1[tid] = 0.0f; i2[tid] = 0.0f; i3[tid] = 0.0f; i4[tid] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        uint rhs = tid + stride;
        if (v0[rhs] > v0[tid] || (v0[rhs] == v0[tid] && i0[rhs] < i0[tid])) { v0[tid] = v0[rhs]; i0[tid] = i0[rhs]; }
        if (v1[rhs] > v1[tid] || (v1[rhs] == v1[tid] && i1[rhs] < i1[tid])) { v1[tid] = v1[rhs]; i1[tid] = i1[rhs]; }
        if (v2[rhs] > v2[tid] || (v2[rhs] == v2[tid] && i2[rhs] < i2[tid])) { v2[tid] = v2[rhs]; i2[tid] = i2[rhs]; }
        if (v3[rhs] > v3[tid] || (v3[rhs] == v3[tid] && i3[rhs] < i3[tid])) { v3[tid] = v3[rhs]; i3[tid] = i3[rhs]; }
        if (v4[rhs] > v4[tid] || (v4[rhs] == v4[tid] && i4[rhs] < i4[tid])) { v4[tid] = v4[rhs]; i4[tid] = i4[rhs]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0u) {
    int out = int(block);
    maxv[0 * Blocks + out] = v0[0]; maxv[1 * Blocks + out] = v1[0]; maxv[2 * Blocks + out] = v2[0];
    maxv[3 * Blocks + out] = v3[0]; maxv[4 * Blocks + out] = v4[0];
    maxid[0 * Blocks + out] = i0[0]; maxid[1 * Blocks + out] = i1[0]; maxid[2 * Blocks + out] = i2[0];
    maxid[3 * Blocks + out] = i3[0]; maxid[4 * Blocks + out] = i4[0];
}
";

const Q2_M5_TERNARY_ARGMAX_CANDIDATES_SOURCE: &str = r"
constexpr int WG = 256;
threadgroup float v0[WG]; threadgroup float v1[WG]; threadgroup float v2[WG];
threadgroup float v3[WG]; threadgroup float v4[WG];
threadgroup float i0[WG]; threadgroup float i1[WG]; threadgroup float i2[WG];
threadgroup float i3[WG]; threadgroup float i4[WG];

uint tid = thread_index_in_threadgroup;
uint block = threadgroup_position_in_grid.x;
int n = int(block) * WG + int(tid);

float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f; float acc4 = 0.0f;

if (n < n_param) {
    for (int g = 0; g < NumGroups; ++g) {
        float a0 = 0.0f; float a1 = 0.0f; float a2 = 0.0f; float a3 = 0.0f; float a4 = 0.0f;
        int word_base = n * KPacked + g * (GroupSize / 16);
        int x_base = g * GroupSize;
        #pragma clang loop unroll(full)
        for (int word = 0; word < (GroupSize / 16); ++word) {
            uint packed = w[word_base + word];
            int xb = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) {
                float t = float((packed >> uint(2 * j)) & 0x3u) - 1.0f;
                a0 += t * float(x[0 * K + xb + j]);
                a1 += t * float(x[1 * K + xb + j]);
                a2 += t * float(x[2 * K + xb + j]);
                a3 += t * float(x[3 * K + xb + j]);
                a4 += t * float(x[4 * K + xb + j]);
            }
        }
        float s = float(sc[n * NumGroups + g]);
        acc0 += s * a0;
        acc1 += s * a1;
        acc2 += s * a2;
        acc3 += s * a3;
        acc4 += s * a4;
    }
    v0[tid] = acc0; v1[tid] = acc1; v2[tid] = acc2; v3[tid] = acc3; v4[tid] = acc4;
    float fid = float(n);
    i0[tid] = fid; i1[tid] = fid; i2[tid] = fid; i3[tid] = fid; i4[tid] = fid;
} else {
    v0[tid] = -INFINITY; v1[tid] = -INFINITY; v2[tid] = -INFINITY; v3[tid] = -INFINITY; v4[tid] = -INFINITY;
    i0[tid] = 0.0f; i1[tid] = 0.0f; i2[tid] = 0.0f; i3[tid] = 0.0f; i4[tid] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        uint rhs = tid + stride;
        if (v0[rhs] > v0[tid] || (v0[rhs] == v0[tid] && i0[rhs] < i0[tid])) { v0[tid] = v0[rhs]; i0[tid] = i0[rhs]; }
        if (v1[rhs] > v1[tid] || (v1[rhs] == v1[tid] && i1[rhs] < i1[tid])) { v1[tid] = v1[rhs]; i1[tid] = i1[rhs]; }
        if (v2[rhs] > v2[tid] || (v2[rhs] == v2[tid] && i2[rhs] < i2[tid])) { v2[tid] = v2[rhs]; i2[tid] = i2[rhs]; }
        if (v3[rhs] > v3[tid] || (v3[rhs] == v3[tid] && i3[rhs] < i3[tid])) { v3[tid] = v3[rhs]; i3[tid] = i3[rhs]; }
        if (v4[rhs] > v4[tid] || (v4[rhs] == v4[tid] && i4[rhs] < i4[tid])) { v4[tid] = v4[rhs]; i4[tid] = i4[rhs]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0u) {
    int out = int(block);
    maxv[0 * Blocks + out] = v0[0]; maxv[1 * Blocks + out] = v1[0]; maxv[2 * Blocks + out] = v2[0];
    maxv[3 * Blocks + out] = v3[0]; maxv[4 * Blocks + out] = v4[0];
    maxid[0 * Blocks + out] = i0[0]; maxid[1 * Blocks + out] = i1[0]; maxid[2 * Blocks + out] = i2[0];
    maxid[3 * Blocks + out] = i3[0]; maxid[4 * Blocks + out] = i4[0];
}
";

#[allow(unsafe_code)]
fn create_q2_m5_argmax_candidates_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"maxv", c"maxid"]);
    let source = CString::new(Q2_M5_ARGMAX_CANDIDATES_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_m5_argmax_candidates_v1".as_ptr(),
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
fn create_q2_m5_ternary_argmax_candidates_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x", c"n_param"]);
    let out_vec = cstr_vec(&[c"maxv", c"maxid"]);
    let source = CString::new(Q2_M5_TERNARY_ARGMAX_CANDIDATES_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_m5_ternary_argmax_candidates_v1".as_ptr(),
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

const Q2_M5_ARGMAX_REDUCE_IDS_SOURCE: &str = r"
constexpr int WG = 1024;
threadgroup float vals[WG];
threadgroup uint ids_tg[WG];

uint tid = thread_index_in_threadgroup;
uint row = threadgroup_position_in_grid.x;
int idx = int(row) * Blocks + int(tid);

if (tid < uint(Blocks)) {
    vals[tid] = maxv[idx];
    ids_tg[tid] = uint(maxid[idx]);
} else {
    vals[tid] = -INFINITY;
    ids_tg[tid] = 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        uint rhs = tid + stride;
        if (vals[rhs] > vals[tid] || (vals[rhs] == vals[tid] && ids_tg[rhs] < ids_tg[tid])) {
            vals[tid] = vals[rhs];
            ids_tg[tid] = ids_tg[rhs];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0u) {
    ids[row] = ids_tg[0];
}
";

#[allow(unsafe_code)]
fn create_q2_m5_argmax_reduce_ids_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"maxv", c"maxid"]);
    let out_vec = cstr_vec(&[c"ids"]);
    let source = CString::new(Q2_M5_ARGMAX_REDUCE_IDS_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_m5_argmax_reduce_ids_v1".as_ptr(),
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

#[allow(unsafe_code, dead_code)]
pub fn bonsai_q2_m5_argmax_reduce_ids(maxv: &Array, maxid: &Array) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    let shape = maxv.shape();
    if shape.len() != 2 || shape[0] != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_m5_argmax_reduce_ids: expected [5, blocks], got {shape:?}"
        )));
    }
    if maxid.shape() != shape {
        return Err(Exception::custom(
            "bonsai_q2_m5_argmax_reduce_ids: maxv/maxid shape mismatch",
        ));
    }
    let blocks = shape[1];
    if blocks > 1024 {
        return Err(Exception::custom(format!(
            "bonsai_q2_m5_argmax_reduce_ids: blocks={blocks} exceeds WG=1024"
        )));
    }

    let stream = Stream::task_local_or_default();
    let cached = Q2_M5_ARGMAX_REDUCE_IDS_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_m5_argmax_reduce_ids_kernel()));
    let config = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Blocks".as_ptr(),
            blocks,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 5 * 1024, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 1024, 1, 1);
        let out_shape = [1, 5];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_UINT32,
        );
        config
    };

    let input_ptrs = [maxv.as_ptr(), maxid.as_ptr()];
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
        Err(Exception::custom(format!(
            "bonsai_q2_m5_argmax_reduce_ids failed: {}",
            take_last_error()
        )))
    } else {
        let mut ids_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut ids_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(ids_ptr) })
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

#[allow(unsafe_code, dead_code)]
pub fn bonsai_q2_m5_argmax_candidates(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<(Array, Array), Exception> {
    ensure_ffi_error_handler();
    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_m5_argmax_candidates: expected M=5, got {m_rows}"
        )));
    }
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_m5_argmax_candidates: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q2_m5_argmax_candidates: weight has no columns"))?;
    let k_dim = k_packed * 16;
    let blocks = (n_rows + 255) / 256;
    let x_flat = x.reshape(&[5, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let cached = Q2_M5_ARGMAX_CANDIDATES_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_m5_argmax_candidates_kernel()));
    let config = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 16,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Blocks".as_ptr(),
            blocks,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, blocks * 256, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);
        let out_shape = [5, blocks];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    };

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
        Err(Exception::custom(format!(
            "bonsai_q2_m5_argmax_candidates failed: {}",
            take_last_error()
        )))
    } else {
        let mut v_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut i_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut v_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut i_ptr, outputs_vec, 1);
        }
        Ok((unsafe { Array::from_ptr(v_ptr) }, unsafe {
            Array::from_ptr(i_ptr)
        }))
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

#[allow(unsafe_code, dead_code)]
pub fn bonsai_q2_m5_ternary_argmax_candidates(
    x: &Array,
    weight: &Array,
    scales: &Array,
    group_size: i32,
) -> Result<(Array, Array), Exception> {
    ensure_ffi_error_handler();
    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_m5_ternary_argmax_candidates: expected M=5, got {m_rows}"
        )));
    }
    let weight_shape = weight.shape();
    let n_rows = weight_shape.first().copied().ok_or_else(|| {
        Exception::custom("bonsai_q2_m5_ternary_argmax_candidates: weight has no rows")
    })?;
    let k_packed = weight_shape.get(1).copied().ok_or_else(|| {
        Exception::custom("bonsai_q2_m5_ternary_argmax_candidates: weight has no columns")
    })?;
    let k_dim = k_packed * 16;
    let blocks = (n_rows + 255) / 256;
    let x_flat = x.reshape(&[5, k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let cached = Q2_M5_TERNARY_ARGMAX_CANDIDATES_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_m5_ternary_argmax_candidates_kernel()));
    let config = unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"KPacked".as_ptr(),
            k_dim / 16,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Blocks".as_ptr(),
            blocks,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, blocks * 256, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);
        let out_shape = [5, blocks];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    };

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
        Err(Exception::custom(format!(
            "bonsai_q2_m5_ternary_argmax_candidates failed: {}",
            take_last_error()
        )))
    } else {
        let mut v_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut i_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut v_ptr, outputs_vec, 0);
            mlx_sys::mlx_vector_array_get(&raw mut i_ptr, outputs_vec, 1);
        }
        Ok((unsafe { Array::from_ptr(v_ptr) }, unsafe {
            Array::from_ptr(i_ptr)
        }))
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
    }
    result
}

/// Upper bound on M for the narrow Q2 verifier path. Same value as Q1
/// (`HIGGS_BONSAI_QMM_MAX_ROWS`, default 8) since both share the z-batched
/// dispatch shape. Wider M goes through `bonsai_q2_wide_qmm` (Phase 3E).
#[allow(dead_code)]
pub fn bonsai_q2_qmm_max_rows() -> i32 {
    crate::qwen3_next::bonsai_q1_qmm_max_rows()
}

// ---------------------------------------------------------------------------
// Experimental symmetric-Q1 threadgroup-local LUT4 path.
//
// This is deliberately exposed through a typed row4 container instead of raw
// arrays. The kernel's pointer arithmetic requires the physical order encoded
// by the shapes below; accepting a flat array merely because its byte count
// matches would silently reinterpret ordinary row-major checkpoint weights.
// ---------------------------------------------------------------------------

/// One-time row4 materialization consumed by the TG-LUT4 kernels.
///
/// Physical shapes are `[N/4, K/128, 4 words, 4 output lanes]` for packed Q1
/// bits and `[N/4, K/128, 4 output lanes]` for scales. Fields stay private so
/// callers cannot construct this contract from ambiguous flat buffers.
#[derive(Debug, Clone)]
pub(super) struct BonsaiQ1Row4 {
    weights: Array,
    scales: Array,
    n_rows: i32,
    k_dim: i32,
    cached_bytes: usize,
}

/// Borrowed, validated view of primary row4 model parameters.
///
/// Unlike [`BonsaiQ1Row4`], this does not own or clone MLX handles. It lets the
/// model's `Param<Array>` fields remain the single authoritative owners while
/// preserving the typed physical-layout contract at every kernel boundary.
#[derive(Debug, Clone, Copy)]
pub(super) struct BonsaiQ1Row4Ref<'a> {
    weights: &'a Array,
    scales: &'a Array,
    n_rows: i32,
    k_dim: i32,
}

impl<'a> BonsaiQ1Row4Ref<'a> {
    /// Validate primary row4 arrays before exposing them to raw Metal pointer
    /// arithmetic. `n_rows` and `k_dim` are logical matrix dimensions; the
    /// physical shapes must be `[N/4,K/128,4,4]` and `[N/4,K/128,4]`.
    pub(crate) fn from_primary_parts(
        weights: &'a Array,
        scales: &'a Array,
        n_rows: i32,
        k_dim: i32,
    ) -> Result<Self, Exception> {
        let expected_weights = [n_rows / 4, k_dim / 128, 4, 4];
        let expected_scales = [n_rows / 4, k_dim / 128, 4];
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 4 != 0
            || k_dim % 128 != 0
            || weights.shape() != expected_weights
            || scales.shape() != expected_scales
            || weights.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: invalid packed contract bits={:?}/{:?} scales={:?}/{:?}; expected {:?} Uint32 and {:?} Float16/Bfloat16",
                weights.shape(),
                weights.dtype(),
                scales.shape(),
                scales.dtype(),
                expected_weights,
                expected_scales
            )));
        }
        if !array_is_row_contiguous(weights)? || !array_is_row_contiguous(scales)? {
            return Err(Exception::custom(
                "BonsaiQ1Row4: packed arrays must be physically row-contiguous",
            ));
        }
        Ok(Self {
            weights,
            scales,
            n_rows,
            k_dim,
        })
    }

    pub(crate) fn accepts_input(self, input: &Array) -> bool {
        self.accepts_input_rows(input, 8)
    }

    pub(crate) fn accepts_fused_gate_up(self, input: &Array) -> bool {
        self.accepts_input_rows(input, 5)
    }

    fn accepts_input_rows(self, input: &Array, max_rows: i32) -> bool {
        if !matches!(input.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || input.shape().last().copied() != Some(self.k_dim)
        {
            return false;
        }
        let rows: i32 = input
            .shape()
            .iter()
            .take(input.shape().len().saturating_sub(1))
            .product();
        (1..=max_rows).contains(&rows)
    }
}

impl BonsaiQ1Row4 {
    /// Transform canonical checkpoint arrays `[N,K/32]` and `[N,K/128]` into
    /// the row4 layout entirely through MLX. `mlx_contiguous(..., false)` is
    /// essential: allowing column-major storage could preserve the transposed
    /// view and make the kernel's flattened indexing incorrect.
    pub(crate) fn from_row_major(weight: &Array, scales: &Array) -> Result<Self, Exception> {
        let [n_rows, k_packed] = *weight.shape() else {
            return Err(Exception::custom(
                "BonsaiQ1Row4: canonical weight must have shape [N,K/32]",
            ));
        };
        let [scale_rows, groups] = *scales.shape() else {
            return Err(Exception::custom(
                "BonsaiQ1Row4: canonical scales must have shape [N,K/128]",
            ));
        };
        if weight.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: expected Uint32 bits and Float16/Bfloat16 scales, got {:?}/{:?}",
                weight.dtype(),
                scales.dtype()
            )));
        }
        let k_dim = k_packed
            .checked_mul(32)
            .ok_or_else(|| Exception::custom("BonsaiQ1Row4: K overflow"))?;
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 4 != 0
            || k_dim % 128 != 0
            || scale_rows != n_rows
            || groups != k_dim / 128
        {
            return Err(Exception::custom(format!(
                "BonsaiQ1Row4: incompatible canonical shapes {:?}/{:?}; require N%4=0 and K%128=0",
                weight.shape(),
                scales.shape()
            )));
        }

        let weights_reshaped = weight.reshape(&[n_rows / 4, 4, groups, 4])?;
        let weights_view = weights_reshaped.transpose_axes(&[0, 2, 3, 1])?;
        let scales_reshaped = scales.reshape(&[n_rows / 4, 4, groups])?;
        let scales_view = scales_reshaped.transpose_axes(&[0, 2, 1])?;
        let weights = row_contiguous_copy(&weights_view)?;
        let packed_scales = row_contiguous_copy(&scales_view)?;
        // Force the two device copies now. Keeping only lazy transpose graphs
        // would repeat or defer the multi-GiB model-wide materialization into
        // timed decode.
        crate::mlx_exec::eval([&weights, &packed_scales])?;
        Self::from_packed_parts(weights, packed_scales, n_rows, k_dim)
    }

    /// Reconstruct canonical checkpoint arrays `[N,K/32]` and `[N,K/128]`.
    ///
    /// This test-only inverse verifies that the primary row4 transform does
    /// not change any packed bits or scales.
    #[cfg(test)]
    pub(crate) fn to_row_major(&self) -> Result<(Array, Array), Exception> {
        let weight_view = self.weights.transpose_axes(&[0, 3, 1, 2])?;
        let scale_view = self.scales.transpose_axes(&[0, 2, 1])?;
        let weight_storage = row_contiguous_copy(&weight_view)?;
        let scale_storage = row_contiguous_copy(&scale_view)?;
        crate::mlx_exec::eval([&weight_storage, &scale_storage])?;
        let weights = weight_storage.reshape(&[self.n_rows, self.k_dim / 32])?;
        let scales = scale_storage.reshape(&[self.n_rows, self.k_dim / 128])?;
        Ok((weights, scales))
    }

    fn from_packed_parts(
        weights: Array,
        scales: Array,
        n_rows: i32,
        k_dim: i32,
    ) -> Result<Self, Exception> {
        BonsaiQ1Row4Ref::from_primary_parts(&weights, &scales, n_rows, k_dim)?;
        let cached_bytes = weights.nbytes().saturating_add(scales.nbytes());
        Ok(Self {
            weights,
            scales,
            n_rows,
            k_dim,
            cached_bytes,
        })
    }

    pub(crate) const fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    #[cfg(test)]
    pub(crate) const fn as_ref(&self) -> BonsaiQ1Row4Ref<'_> {
        BonsaiQ1Row4Ref {
            weights: &self.weights,
            scales: &self.scales,
            n_rows: self.n_rows,
            k_dim: self.k_dim,
        }
    }

    pub(crate) fn into_primary_parts(self) -> (Array, Array, i32, i32) {
        (self.weights, self.scales, self.n_rows, self.k_dim)
    }
}

/// One-time row2 materialization consumed by the Q2 M=5 verifier kernels.
///
/// Mirror of [`BonsaiQ1Row4`] for 2-bit affine packed weights. Physical shapes
/// are `[N/2, K/128, 8 words, 2 output lanes]` for packed Q2 bits and
/// `[N/2, K/128, 2 output lanes]` for scales. The 8-words-per-group dimension
/// reflects Q2's packing: 128 cols / 16 cols-per-word = 8 packed u32s per
/// affine group, vs Q1's 4 (128 cols / 32 cols-per-word).
///
/// Two adjacent output rows share contiguous memory (coalesced loads) but
/// each output row still reads its own packed words. The dominant weight-reuse
/// benefit comes from sharing each packed-word read across 5 verifier rows
/// in the M=5 native spec kernel (`bonsai_q2_row2_m5_contract`).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Phase 3D integration lands in a follow-up commit
pub(crate) struct BonsaiQ2Row2 {
    weights: Array,
    scales: Array,
    n_rows: i32,
    k_dim: i32,
    cached_bytes: usize,
}

/// Borrowed, validated view of primary row2 model parameters.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct BonsaiQ2Row2Ref<'a> {
    weights: &'a Array,
    scales: &'a Array,
    n_rows: i32,
    k_dim: i32,
}

impl<'a> BonsaiQ2Row2Ref<'a> {
    pub(crate) fn n_rows(&self) -> i32 {
        self.n_rows
    }

    pub(crate) fn k_dim(&self) -> i32 {
        self.k_dim
    }

    pub(crate) fn from_primary_parts(
        weights: &'a Array,
        scales: &'a Array,
        n_rows: i32,
        k_dim: i32,
    ) -> Result<Self, Exception> {
        let expected_weights = [n_rows / 2, k_dim / 128, 8, 2];
        let expected_scales = [n_rows / 2, k_dim / 128, 2];
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 2 != 0
            || k_dim % 128 != 0
            || weights.shape() != expected_weights
            || scales.shape() != expected_scales
            || weights.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ2Row2: invalid packed contract bits={:?}/{:?} scales={:?}/{:?}; expected {:?} Uint32 and {:?} Float16/Bfloat16",
                weights.shape(),
                weights.dtype(),
                scales.shape(),
                scales.dtype(),
                expected_weights,
                expected_scales
            )));
        }
        if !array_is_row_contiguous(weights)? || !array_is_row_contiguous(scales)? {
            return Err(Exception::custom(
                "BonsaiQ2Row2: packed arrays must be physically row-contiguous",
            ));
        }
        Ok(Self {
            weights,
            scales,
            n_rows,
            k_dim,
        })
    }
}

#[allow(dead_code)]
impl BonsaiQ2Row2 {
    /// Transform canonical checkpoint arrays `[N,K/16]` and `[N,K/128]` into
    /// the row2 layout entirely through MLX.
    pub(crate) fn from_row_major(weight: &Array, scales: &Array) -> Result<Self, Exception> {
        let [n_rows, k_packed] = *weight.shape() else {
            return Err(Exception::custom(
                "BonsaiQ2Row2: canonical weight must have shape [N,K/16]",
            ));
        };
        let [scale_rows, groups] = *scales.shape() else {
            return Err(Exception::custom(
                "BonsaiQ2Row2: canonical scales must have shape [N,K/128]",
            ));
        };
        if weight.dtype() != Dtype::Uint32
            || !matches!(scales.dtype(), Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err(Exception::custom(format!(
                "BonsaiQ2Row2: expected Uint32 bits and Float16/Bfloat16 scales, got {:?}/{:?}",
                weight.dtype(),
                scales.dtype()
            )));
        }
        let k_dim = k_packed
            .checked_mul(16)
            .ok_or_else(|| Exception::custom("BonsaiQ2Row2: K overflow"))?;
        // 8 packed words per group (128 cols / 16 cols per word).
        let words_per_group = k_dim / 128 * 8;
        if k_packed != words_per_group {
            return Err(Exception::custom(format!(
                "BonsaiQ2Row2: k_packed={k_packed} does not match expected {words_per_group} (8 per group * {} groups)",
                k_dim / 128
            )));
        }
        if n_rows <= 0
            || k_dim <= 0
            || n_rows % 2 != 0
            || k_dim % 128 != 0
            || scale_rows != n_rows
            || groups != k_dim / 128
        {
            return Err(Exception::custom(format!(
                "BonsaiQ2Row2: incompatible canonical shapes {:?}/{:?}; require N%2=0 and K%128=0",
                weight.shape(),
                scales.shape()
            )));
        }

        // weight: [N, K/16] -> [N/2, 2, NumGroups, 8] -> [N/2, NumGroups, 8, 2]
        let weights_reshaped = weight.reshape(&[n_rows / 2, 2, groups, 8])?;
        let weights_view = weights_reshaped.transpose_axes(&[0, 2, 3, 1])?;
        // scales: [N, NumGroups] -> [N/2, 2, NumGroups] -> [N/2, NumGroups, 2]
        let scales_reshaped = scales.reshape(&[n_rows / 2, 2, groups])?;
        let scales_view = scales_reshaped.transpose_axes(&[0, 2, 1])?;
        let weights = row_contiguous_copy(&weights_view)?;
        let packed_scales = row_contiguous_copy(&scales_view)?;
        crate::mlx_exec::eval([&weights, &packed_scales])?;
        Self::from_packed_parts(weights, packed_scales, n_rows, k_dim)
    }

    /// Reconstruct canonical checkpoint arrays `[N,K/16]` and `[N,K/128]`.
    #[cfg(test)]
    pub(crate) fn to_row_major(&self) -> Result<(Array, Array), Exception> {
        let weight_view = self.weights.transpose_axes(&[0, 3, 1, 2])?;
        let scale_view = self.scales.transpose_axes(&[0, 2, 1])?;
        let weight_storage = row_contiguous_copy(&weight_view)?;
        let scale_storage = row_contiguous_copy(&scale_view)?;
        crate::mlx_exec::eval([&weight_storage, &scale_storage])?;
        let weights = weight_storage.reshape(&[self.n_rows, self.k_dim / 16])?;
        let scales = scale_storage.reshape(&[self.n_rows, self.k_dim / 128])?;
        Ok((weights, scales))
    }

    fn from_packed_parts(
        weights: Array,
        scales: Array,
        n_rows: i32,
        k_dim: i32,
    ) -> Result<Self, Exception> {
        BonsaiQ2Row2Ref::from_primary_parts(&weights, &scales, n_rows, k_dim)?;
        let cached_bytes = weights.nbytes().saturating_add(scales.nbytes());
        Ok(Self {
            weights,
            scales,
            n_rows,
            k_dim,
            cached_bytes,
        })
    }

    pub(crate) const fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }

    pub(crate) const fn as_ref(&self) -> BonsaiQ2Row2Ref<'_> {
        BonsaiQ2Row2Ref {
            weights: &self.weights,
            scales: &self.scales,
            n_rows: self.n_rows,
            k_dim: self.k_dim,
        }
    }

    pub(crate) fn into_primary_parts(self) -> (Array, Array, i32, i32) {
        (self.weights, self.scales, self.n_rows, self.k_dim)
    }
}

// ---------------------------------------------------------------------------
// Q2 M=5 native spec verifier kernel.
//
// Decomposes Q2 2-bit code as q=2*q_h + q_l and applies the 16-entry TG-LUT4
// identity on both bit-planes. This keeps one packed-word read shared across all
// five verifier rows per group.
//
// Layout contract: BonsaiQ2Row2 (`[N/2, NumGroups, 8 words, 2 lanes]` packed
// bits, `[N/2, NumGroups, 2 lanes]` scales). Adjacent output rows read
// contiguous memory (coalesced loads).
//
// Per-thread work: one output row n. row_tile = n/2, row_lane = n%2.
//   For each group g:
//     1) Build the row-local LUT (cooperative once per tile, one barrier).
//     2) Read each 2-bit nibble, decompose into q_h/q_l bit-planes.
//     3) sum(qa_h), sum(qa_l), sum(x) and combine:
//        acc_m += scale * (2*sum(qa_h) + sum(qa_l)) + bias * sum_x.
// ---------------------------------------------------------------------------

const Q2_ROW2_M5_KERNEL_SOURCE: &str = r"
constexpr int WG = 256;           // threads per threadgroup
constexpr int ROWS_PER_TG = 256;  // output rows handled by one threadgroup
threadgroup half lut[2560];

uint tid = thread_index_in_threadgroup;
uint tgx = threadgroup_position_in_grid.x;
int n = int(tgx) * ROWS_PER_TG + int(tid);

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
float acc4 = 0.0f;

if (n < NRows) {
    int row_tile = n / 2;
    int row_lane = n & 1;

    for (int g = 0; g < NumGroups; ++g) {
        for (uint build = tid; build < 160u; build += uint(WG)) {
            int mlocal = int(build) / 32;
            int nibble = int(build) & 31;
            int kbase = mlocal * K + g * 128 + nibble * 4;
            float x0 = float(x[kbase + 0]);
            float x1 = float(x[kbase + 1]);
            float x2 = float(x[kbase + 2]);
            float x3 = float(x[kbase + 3]);
            float xy = x0 + x1;
            float xz = x0 + x2;
            float yz = x1 + x2;
            float xyz = xy + x2;
            float c = 0.5f * (x0 + x1 + x2 + x3);
            int base = (mlocal * 32 + nibble) * 16;
            lut[base + 0] = half(-c);
            lut[base + 1] = half(x0 - c);
            lut[base + 2] = half(x1 - c);
            lut[base + 3] = half(xy - c);
            lut[base + 4] = half(x2 - c);
            lut[base + 5] = half(xz - c);
            lut[base + 6] = half(yz - c);
            lut[base + 7] = half(xyz - c);
            lut[base + 8] = half(x3 - c);
            lut[base + 9] = half(x0 + x3 - c);
            lut[base + 10] = half(x1 + x3 - c);
            lut[base + 11] = half(xy + x3 - c);
            lut[base + 12] = half(x2 + x3 - c);
            lut[base + 13] = half(xz + x3 - c);
            lut[base + 14] = half(yz + x3 - c);
            lut[base + 15] = half(xyz + x3 - c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qa0h = 0.0f; float qa1h = 0.0f; float qa2h = 0.0f;
        float qa3h = 0.0f; float qa4h = 0.0f;
        float qa0l = 0.0f; float qa1l = 0.0f; float qa2l = 0.0f;
        float qa3l = 0.0f; float qa4l = 0.0f;
        float sx0 = 0.0f;   float sx1 = 0.0f;   float sx2 = 0.0f;
        float sx3 = 0.0f;   float sx4 = 0.0f;

        int group_base = (row_tile * NumGroups + g) * 8;  // 8 packed words per group
        int x_base = g * 128;

        #pragma clang loop unroll(full)
        for (int word = 0; word < 8; ++word) {
            uint packed = w[(group_base + word) * 2 + row_lane];
            int xo = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int chunk = 0; chunk < 4; ++chunk) {
                uint block = (packed >> (uint(chunk) * 8u)) & 0xFFu;
                uint q_hi = ((block & 0x02u) >> 1u) | ((block & 0x08u) >> 2u) |
                            ((block & 0x20u) >> 3u) | ((block & 0x80u) >> 4u);
                uint q_lo = (block & 0x01u) | ((block & 0x04u) >> 1u) |
                            ((block & 0x10u) >> 2u) | ((block & 0x40u) >> 3u);
                int li = (word * 4 + chunk) * 16 + int(q_lo);
                int hi = (word * 4 + chunk) * 16 + int(q_hi);
                qa0l += float(lut[li]);
                qa1l += float(lut[512 + li]);
                qa2l += float(lut[1024 + li]);
                qa3l += float(lut[1536 + li]);
                qa4l += float(lut[2048 + li]);
                qa0h += float(lut[hi]);
                qa1h += float(lut[512 + hi]);
                qa2h += float(lut[1024 + hi]);
                qa3h += float(lut[1536 + hi]);
                qa4h += float(lut[2048 + hi]);

                int xchunk = xo + chunk * 4;
                float xv00 = float(x[0 * K + xchunk + 0]);
                float xv01 = float(x[0 * K + xchunk + 1]);
                float xv02 = float(x[0 * K + xchunk + 2]);
                float xv03 = float(x[0 * K + xchunk + 3]);
                float xv10 = float(x[1 * K + xchunk + 0]);
                float xv11 = float(x[1 * K + xchunk + 1]);
                float xv12 = float(x[1 * K + xchunk + 2]);
                float xv13 = float(x[1 * K + xchunk + 3]);
                float xv20 = float(x[2 * K + xchunk + 0]);
                float xv21 = float(x[2 * K + xchunk + 1]);
                float xv22 = float(x[2 * K + xchunk + 2]);
                float xv23 = float(x[2 * K + xchunk + 3]);
                float xv30 = float(x[3 * K + xchunk + 0]);
                float xv31 = float(x[3 * K + xchunk + 1]);
                float xv32 = float(x[3 * K + xchunk + 2]);
                float xv33 = float(x[3 * K + xchunk + 3]);
                float xv40 = float(x[4 * K + xchunk + 0]);
                float xv41 = float(x[4 * K + xchunk + 1]);
                float xv42 = float(x[4 * K + xchunk + 2]);
                float xv43 = float(x[4 * K + xchunk + 3]);
                sx0 += xv00 + xv01 + xv02 + xv03;
                sx1 += xv10 + xv11 + xv12 + xv13;
                sx2 += xv20 + xv21 + xv22 + xv23;
                sx3 += xv30 + xv31 + xv32 + xv33;
                sx4 += xv40 + xv41 + xv42 + xv43;
            }
        }

        // Scales are in row2 layout: sc_transposed[t, g, l] = scales[t*2+l, g].
        // Biases are in canonical [N, NumGroups] layout: bi[n, g] = biases[n*NumGroups+g].
        int sb_row2_idx = (row_tile * NumGroups + g) * 2 + row_lane;
        int b_canon_idx = (row_tile * 2 + row_lane) * NumGroups + g;
        float s_val = float(sc[sb_row2_idx]);
        float b_val = float(bi[b_canon_idx]);
        acc0 += s_val * (2.0f * qa0h + qa0l + 1.5f * sx0) + b_val * sx0;
        acc1 += s_val * (2.0f * qa1h + qa1l + 1.5f * sx1) + b_val * sx1;
        acc2 += s_val * (2.0f * qa2h + qa2l + 1.5f * sx2) + b_val * sx2;
        acc3 += s_val * (2.0f * qa3h + qa3l + 1.5f * sx3) + b_val * sx3;
        acc4 += s_val * (2.0f * qa4h + qa4l + 1.5f * sx4) + b_val * sx4;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Output layout: [M, NRows] row-major (matches Q1 TG-LUT4 M=5 kernel).
if (n < NRows) {
    y[0 * NRows + n] = OutT(acc0);
    y[1 * NRows + n] = OutT(acc1);
    y[2 * NRows + n] = OutT(acc2);
    y[3 * NRows + n] = OutT(acc3);
    y[4 * NRows + n] = OutT(acc4);
}
";

static Q2_ROW2_M5_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_ROW2_M5_TERNARY_DIRECT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_ROW2_M5_TERNARY_FUSED_GATE_UP_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_ROW2_M5_TERNARY_SPLITK_PARTIAL_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static Q2_ROW2_M5_TERNARY_SPLITK_REDUCE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

#[allow(unsafe_code)]
fn create_q2_row2_m5_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(Q2_ROW2_M5_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_row2_m5_contract_v3".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true, // raw pointer arithmetic requires row-contiguous inputs
            false,
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(unsafe_code)]
fn configure_q2_row2_m5_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    const WG: i32 = 256;
    const ROWS_PER_TG: i32 = 256;
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
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NRows".as_ptr(),
            n_rows,
        );

        let n_tgs = (n_rows + ROWS_PER_TG - 1) / ROWS_PER_TG;
        let grid_x = n_tgs * ROWS_PER_TG;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid_x, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, WG, 1, 1);

        // Output: [M=5, NRows] row-major.
        let y_shape = [5, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        let _ = (WG, ROWS_PER_TG); // suppress unused-constant warning if any
        config
    }
}

/// Phase 3D Q2 M=5 native spec verifier kernel.
///
/// Computes `y[m, n] = sum_k dequant(w[n,k]) * x[m,k]` for m=0..4 and all
/// output rows n, where `w` is in [`BonsaiQ2Row2`] layout and `x` is the
/// `[5, K]` activation tile (anchor + 4 draft rows). Each thread handles one
/// output row across all 5 verifier rows, sharing the 8 packed-word reads per
/// group across all 5 verifier-row accumulators.
///
/// Bit-exact vs CPU oracle (verified in `bonsai_q2::tests`). Performance:
/// microbench vs z-batched `bonsai_q2_qmm` is the kill gate for the 1.45×
/// end-to-end target.
#[allow(dead_code, unsafe_code)]
pub fn bonsai_q2_row2_m5_contract(
    x: &Array,
    packed: BonsaiQ2Row2Ref<'_>,
    biases: &Array,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5: expected M=5 verifier tile, got M={m_rows}"
        )));
    }
    let x_flat = x.reshape(&[5, packed.k_dim])?;
    let w_flat = packed.weights.reshape(&[-1])?;
    let s_flat = packed.scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = Q2_ROW2_M5_KERNEL.get_or_init(|| CachedMetalKernel(create_q2_row2_m5_kernel()));
    let config = configure_q2_row2_m5_kernel(out_dtype, packed.n_rows, packed.k_dim, 128);

    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
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
        Err(Exception::custom(format!(
            "bonsai_q2_row2_m5 failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(y_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

const Q2_ROW2_M5_TERNARY_DIRECT_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int ROWS_PER_TG = 256;

uint tid = thread_index_in_threadgroup;
uint tgx = threadgroup_position_in_grid.x;
int n = int(tgx) * ROWS_PER_TG + int(tid);

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
float acc4 = 0.0f;

if (n < NRows) {
    int row_tile = n / 2;
    int row_lane = n & 1;

    for (int g = 0; g < NumGroups; ++g) {
        float a0 = 0.0f; float a1 = 0.0f; float a2 = 0.0f; float a3 = 0.0f; float a4 = 0.0f;
        int group_base = (row_tile * NumGroups + g) * 8;
        int x_base = g * GroupSize;
        #pragma clang loop unroll(full)
        for (int word = 0; word < 8; ++word) {
            uint packed = w[(group_base + word) * 2 + row_lane];
            int xb = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) {
                float t = float((packed >> uint(2 * j)) & 0x3u) - 1.0f;
                a0 += t * float(x[0 * K + xb + j]);
                a1 += t * float(x[1 * K + xb + j]);
                a2 += t * float(x[2 * K + xb + j]);
                a3 += t * float(x[3 * K + xb + j]);
                a4 += t * float(x[4 * K + xb + j]);
            }
        }
        float s = float(sc[(row_tile * NumGroups + g) * 2 + row_lane]);
        acc0 += s * a0;
        acc1 += s * a1;
        acc2 += s * a2;
        acc3 += s * a3;
        acc4 += s * a4;
    }
}

if (n < NRows) {
    y[0 * NRows + n] = OutT(acc0);
    y[1 * NRows + n] = OutT(acc1);
    y[2 * NRows + n] = OutT(acc2);
    y[3 * NRows + n] = OutT(acc3);
    y[4 * NRows + n] = OutT(acc4);
}
";

const Q2_ROW2_M5_TERNARY_FUSED_GATE_UP_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int ROWS_PER_TG = 256;

uint tid = thread_index_in_threadgroup;
uint tgx = threadgroup_position_in_grid.x;
int n = int(tgx) * ROWS_PER_TG + int(tid);

float g0 = 0.0f; float g1 = 0.0f; float g2 = 0.0f; float g3 = 0.0f; float g4 = 0.0f;
float u0 = 0.0f; float u1 = 0.0f; float u2 = 0.0f; float u3 = 0.0f; float u4 = 0.0f;

if (n < NRows) {
    int row_tile = n / 2;
    int row_lane = n & 1;

    for (int g = 0; g < NumGroups; ++g) {
        float ga0 = 0.0f; float ga1 = 0.0f; float ga2 = 0.0f; float ga3 = 0.0f; float ga4 = 0.0f;
        float ua0 = 0.0f; float ua1 = 0.0f; float ua2 = 0.0f; float ua3 = 0.0f; float ua4 = 0.0f;
        int group_base = (row_tile * NumGroups + g) * 8;
        int x_base = g * GroupSize;
        #pragma clang loop unroll(full)
        for (int word = 0; word < 8; ++word) {
            uint gpacked = wg[(group_base + word) * 2 + row_lane];
            uint upacked = wu[(group_base + word) * 2 + row_lane];
            int xb = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) {
                float gt = float((gpacked >> uint(2 * j)) & 0x3u) - 1.0f;
                float ut = float((upacked >> uint(2 * j)) & 0x3u) - 1.0f;
                float x0 = float(x[0 * K + xb + j]);
                float x1 = float(x[1 * K + xb + j]);
                float x2 = float(x[2 * K + xb + j]);
                float x3 = float(x[3 * K + xb + j]);
                float x4 = float(x[4 * K + xb + j]);
                ga0 += gt * x0; ga1 += gt * x1; ga2 += gt * x2; ga3 += gt * x3; ga4 += gt * x4;
                ua0 += ut * x0; ua1 += ut * x1; ua2 += ut * x2; ua3 += ut * x3; ua4 += ut * x4;
            }
        }
        float gs = float(scg[(row_tile * NumGroups + g) * 2 + row_lane]);
        float us = float(scu[(row_tile * NumGroups + g) * 2 + row_lane]);
        g0 += gs * ga0; g1 += gs * ga1; g2 += gs * ga2; g3 += gs * ga3; g4 += gs * ga4;
        u0 += us * ua0; u1 += us * ua1; u2 += us * ua2; u3 += us * ua3; u4 += us * ua4;
    }
}

if (n < NRows) {
    constexpr int OutRows = 2 * NRows;
    y[0 * OutRows + n] = OutT(g0);
    y[1 * OutRows + n] = OutT(g1);
    y[2 * OutRows + n] = OutT(g2);
    y[3 * OutRows + n] = OutT(g3);
    y[4 * OutRows + n] = OutT(g4);
    y[0 * OutRows + NRows + n] = OutT(u0);
    y[1 * OutRows + NRows + n] = OutT(u1);
    y[2 * OutRows + NRows + n] = OutT(u2);
    y[3 * OutRows + NRows + n] = OutT(u3);
    y[4 * OutRows + NRows + n] = OutT(u4);
}
";

const Q2_ROW2_M5_TERNARY_SPLITK_PARTIAL_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int ROWS_PER_TG = 256;

uint tid = thread_index_in_threadgroup;
uint tgx = threadgroup_position_in_grid.x;
uint chunk = threadgroup_position_in_grid.y;
int n = int(tgx) * ROWS_PER_TG + int(tid);

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
float acc4 = 0.0f;

if (n < NRows) {
    int row_tile = n / 2;
    int row_lane = n & 1;
    int g_start = (NumGroups * int(chunk)) / Chunks;
    int g_end = (NumGroups * (int(chunk) + 1)) / Chunks;

    for (int g = g_start; g < g_end; ++g) {
        float a0 = 0.0f; float a1 = 0.0f; float a2 = 0.0f; float a3 = 0.0f; float a4 = 0.0f;
        int group_base = (row_tile * NumGroups + g) * 8;
        int x_base = g * GroupSize;
        #pragma clang loop unroll(full)
        for (int word = 0; word < 8; ++word) {
            uint packed = w[(group_base + word) * 2 + row_lane];
            int xb = x_base + word * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 16; ++j) {
                float t = float((packed >> uint(2 * j)) & 0x3u) - 1.0f;
                a0 += t * float(x[0 * K + xb + j]);
                a1 += t * float(x[1 * K + xb + j]);
                a2 += t * float(x[2 * K + xb + j]);
                a3 += t * float(x[3 * K + xb + j]);
                a4 += t * float(x[4 * K + xb + j]);
            }
        }
        float s = float(sc[(row_tile * NumGroups + g) * 2 + row_lane]);
        acc0 += s * a0;
        acc1 += s * a1;
        acc2 += s * a2;
        acc3 += s * a3;
        acc4 += s * a4;
    }
}

if (n < NRows) {
    int base = int(chunk) * 5 * NRows;
    partial[base + 0 * NRows + n] = acc0;
    partial[base + 1 * NRows + n] = acc1;
    partial[base + 2 * NRows + n] = acc2;
    partial[base + 3 * NRows + n] = acc3;
    partial[base + 4 * NRows + n] = acc4;
}
";

const Q2_ROW2_M5_TERNARY_SPLITK_REDUCE_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int ROWS_PER_TG = 256;

uint tid = thread_index_in_threadgroup;
uint tgx = threadgroup_position_in_grid.x;
int n = int(tgx) * ROWS_PER_TG + int(tid);

if (n < NRows) {
    float acc0 = 0.0f; float acc1 = 0.0f; float acc2 = 0.0f; float acc3 = 0.0f; float acc4 = 0.0f;
    for (int c = 0; c < Chunks; ++c) {
        int base = c * 5 * NRows;
        acc0 += partial[base + 0 * NRows + n];
        acc1 += partial[base + 1 * NRows + n];
        acc2 += partial[base + 2 * NRows + n];
        acc3 += partial[base + 3 * NRows + n];
        acc4 += partial[base + 4 * NRows + n];
    }
    y[0 * NRows + n] = OutT(acc0);
    y[1 * NRows + n] = OutT(acc1);
    y[2 * NRows + n] = OutT(acc2);
    y[3 * NRows + n] = OutT(acc3);
    y[4 * NRows + n] = OutT(acc4);
}
";

#[allow(unsafe_code)]
fn create_q2_row2_m5_ternary_direct_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(Q2_ROW2_M5_TERNARY_DIRECT_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_row2_m5_ternary_direct_v1".as_ptr(),
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
fn create_q2_row2_m5_ternary_fused_gate_up_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"wg", c"wu", c"scg", c"scu", c"x"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(Q2_ROW2_M5_TERNARY_FUSED_GATE_UP_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_row2_m5_ternary_fused_gate_up_v1".as_ptr(),
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
fn configure_q2_row2_m5_fused_gate_up_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    const WG: i32 = 256;
    const ROWS_PER_TG: i32 = 256;
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
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NRows".as_ptr(),
            n_rows,
        );

        let n_tgs = (n_rows + ROWS_PER_TG - 1) / ROWS_PER_TG;
        let grid_x = n_tgs * ROWS_PER_TG;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid_x, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, WG, 1, 1);

        let y_shape = [5, n_rows * 2];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

#[allow(unsafe_code)]
fn create_q2_row2_m5_ternary_splitk_partial_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x"]);
    let out_vec = cstr_vec(&[c"partial"]);
    let source = CString::new(Q2_ROW2_M5_TERNARY_SPLITK_PARTIAL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_row2_m5_ternary_splitk_partial_v1".as_ptr(),
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
fn create_q2_row2_m5_ternary_splitk_reduce_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"partial"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(Q2_ROW2_M5_TERNARY_SPLITK_REDUCE_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q2_row2_m5_ternary_splitk_reduce_v1".as_ptr(),
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
fn configure_q2_row2_m5_splitk_partial_kernel(
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    chunks: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    const WG: i32 = 256;
    const ROWS_PER_TG: i32 = 256;
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k_dim);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NRows".as_ptr(),
            n_rows,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Chunks".as_ptr(),
            chunks,
        );
        let n_tgs = (n_rows + ROWS_PER_TG - 1) / ROWS_PER_TG;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * ROWS_PER_TG, chunks, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, WG, 1, 1);
        let partial_shape = [chunks, 5, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            partial_shape.as_ptr(),
            partial_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );
        config
    }
}

#[allow(unsafe_code)]
fn configure_q2_row2_m5_splitk_reduce_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    chunks: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    const WG: i32 = 256;
    const ROWS_PER_TG: i32 = 256;
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NRows".as_ptr(),
            n_rows,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Chunks".as_ptr(),
            chunks,
        );
        let n_tgs = (n_rows + ROWS_PER_TG - 1) / ROWS_PER_TG;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tgs * ROWS_PER_TG, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, WG, 1, 1);
        let y_shape = [5, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            out_dtype,
        );
        config
    }
}

#[allow(dead_code, unsafe_code)]
pub fn bonsai_q2_row2_m5_ternary_direct(
    x: &Array,
    packed: BonsaiQ2Row2Ref<'_>,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_direct: expected M=5 verifier tile, got M={m_rows}"
        )));
    }
    let x_flat = x.reshape(&[5, packed.k_dim])?;
    let w_flat = packed.weights.reshape(&[-1])?;
    let s_flat = packed.scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached = Q2_ROW2_M5_TERNARY_DIRECT_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_row2_m5_ternary_direct_kernel()));
    let config = configure_q2_row2_m5_kernel(out_dtype, packed.n_rows, packed.k_dim, 128);

    let input_ptrs = [w_flat.as_ptr(), s_flat.as_ptr(), x_flat.as_ptr()];
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
        Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_direct failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(y_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

#[allow(dead_code, unsafe_code)]
pub fn bonsai_q2_row2_m5_ternary_fused_gate_up(
    x: &Array,
    gate: BonsaiQ2Row2Ref<'_>,
    up: BonsaiQ2Row2Ref<'_>,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    if gate.n_rows != up.n_rows || gate.k_dim != up.k_dim {
        return Err(Exception::custom(
            "bonsai_q2_row2_m5_ternary_fused_gate_up: gate/up shape mismatch",
        ));
    }

    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_fused_gate_up: expected M=5 verifier tile, got M={m_rows}"
        )));
    }
    let x_flat = x.reshape(&[5, gate.k_dim])?;
    let wg_flat = gate.weights.reshape(&[-1])?;
    let wu_flat = up.weights.reshape(&[-1])?;
    let sg_flat = gate.scales.flatten(None, None)?;
    let su_flat = up.scales.flatten(None, None)?;

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached = Q2_ROW2_M5_TERNARY_FUSED_GATE_UP_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_row2_m5_ternary_fused_gate_up_kernel()));
    let config = configure_q2_row2_m5_fused_gate_up_kernel(out_dtype, gate.n_rows, gate.k_dim, 128);

    let input_ptrs = [
        wg_flat.as_ptr(),
        wu_flat.as_ptr(),
        sg_flat.as_ptr(),
        su_flat.as_ptr(),
        x_flat.as_ptr(),
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
        Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_fused_gate_up failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(y_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

#[allow(dead_code, unsafe_code)]
pub fn bonsai_q2_row2_m5_ternary_splitk(
    x: &Array,
    packed: BonsaiQ2Row2Ref<'_>,
    chunks: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    if chunks < 2 || chunks > 8 {
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_splitk: expected chunks in 2..=8, got {chunks}"
        )));
    }
    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows != 5 {
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_splitk: expected M=5 verifier tile, got M={m_rows}"
        )));
    }

    let x_flat = x.reshape(&[5, packed.k_dim])?;
    let w_flat = packed.weights.reshape(&[-1])?;
    let s_flat = packed.scales.flatten(None, None)?;
    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let partial_cached = Q2_ROW2_M5_TERNARY_SPLITK_PARTIAL_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_row2_m5_ternary_splitk_partial_kernel()));
    let partial_config =
        configure_q2_row2_m5_splitk_partial_kernel(packed.n_rows, packed.k_dim, 128, chunks);
    let partial_inputs = [w_flat.as_ptr(), s_flat.as_ptr(), x_flat.as_ptr()];
    let partial_inputs_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(partial_inputs.as_ptr(), partial_inputs.len())
    };
    let mut partial_outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let partial_status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut partial_outputs_vec,
            partial_cached.0,
            partial_inputs_vec,
            partial_config,
            stream.as_ptr(),
        )
    };
    let partial = if partial_status != 0 {
        let err = take_last_error();
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(partial_config);
            mlx_sys::mlx_vector_array_free(partial_inputs_vec);
            mlx_sys::mlx_vector_array_free(partial_outputs_vec);
        }
        return Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_splitk partial failed: {err}"
        )));
    } else {
        let mut partial_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut partial_ptr, partial_outputs_vec, 0) };
        unsafe { Array::from_ptr(partial_ptr) }
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(partial_config);
        mlx_sys::mlx_vector_array_free(partial_inputs_vec);
        mlx_sys::mlx_vector_array_free(partial_outputs_vec);
    }

    let reduce_cached = Q2_ROW2_M5_TERNARY_SPLITK_REDUCE_KERNEL
        .get_or_init(|| CachedMetalKernel(create_q2_row2_m5_ternary_splitk_reduce_kernel()));
    let reduce_config = configure_q2_row2_m5_splitk_reduce_kernel(out_dtype, packed.n_rows, chunks);
    let reduce_inputs = [partial.as_ptr()];
    let reduce_inputs_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(reduce_inputs.as_ptr(), reduce_inputs.len())
    };
    let mut reduce_outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let reduce_status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut reduce_outputs_vec,
            reduce_cached.0,
            reduce_inputs_vec,
            reduce_config,
            stream.as_ptr(),
        )
    };
    let result = if reduce_status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q2_row2_m5_ternary_splitk reduce failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, reduce_outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(y_ptr) })
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(reduce_config);
        mlx_sys::mlx_vector_array_free(reduce_inputs_vec);
        mlx_sys::mlx_vector_array_free(reduce_outputs_vec);
    }
    result
}

#[allow(unsafe_code)]
fn row_contiguous_copy(array: &Array) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    let stream = Stream::task_local_or_default();
    let mut output = unsafe { mlx_sys::mlx_array_new() };
    let status = unsafe {
        mlx_sys::mlx_contiguous(
            &raw mut output,
            array.as_ptr(),
            false, // never preserve a column-major transposed view
            stream.as_ptr(),
        )
    };
    if status != 0 {
        unsafe { mlx_sys::mlx_array_free(output) };
        Err(Exception::custom(format!(
            "mlx_contiguous row-major copy failed: {}",
            take_last_error()
        )))
    } else {
        Ok(unsafe { Array::from_ptr(output) })
    }
}

#[allow(unsafe_code)]
fn array_is_row_contiguous(array: &Array) -> Result<bool, Exception> {
    ensure_ffi_error_handler();
    let mut result = false;
    let status = unsafe { mlx_sys::_mlx_array_is_row_contiguous(&raw mut result, array.as_ptr()) };
    if status == 0 {
        Ok(result)
    } else {
        Err(Exception::custom(format!(
            "MLX row-contiguous query failed: {}",
            take_last_error()
        )))
    }
}

const TG_LUT4_CONTRACT_KERNEL_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int NTILE = 256;
constexpr int MTILE = 4;
threadgroup half lut[2048];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
uint mbase = threadgroup_position_in_grid.z * uint(MTILE);
float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    if (tid < 128u) {
        int mlocal = int(tid) / 32;
        int nibble = int(tid) & 31;
        int m = int(mbase) + mlocal;
        int kbase = g * 128 + nibble * 4;
        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        float x3 = 0.0f;
        if (m < MRows) {
            int xb = m * K + kbase;
            x0 = float(x[xb + 0]); x1 = float(x[xb + 1]);
            x2 = float(x[xb + 2]); x3 = float(x[xb + 3]);
        }
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float qa0 = 0.0f;
        float qa1 = 0.0f;
        float qa2 = 0.0f;
        float qa3 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint packed = w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint mask = (packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16 + int(mask);
                qa0 += float(lut[li]);
                qa1 += float(lut[512 + li]);
                qa2 += float(lut[1024 + li]);
                qa3 += float(lut[1536 + li]);
            }
        }
        float scale = float(sc[group_base + row_lane]);
        acc0 += scale * qa0;
        acc1 += scale * qa1;
        acc2 += scale * qa2;
        acc3 += scale * qa3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    if (mbase + 0u < uint(MRows)) { y[int(mbase + 0u) * NRows + int(n)] = OutT(acc0); }
    if (mbase + 1u < uint(MRows)) { y[int(mbase + 1u) * NRows + int(n)] = OutT(acc1); }
    if (mbase + 2u < uint(MRows)) { y[int(mbase + 2u) * NRows + int(n)] = OutT(acc2); }
    if (mbase + 3u < uint(MRows)) { y[int(mbase + 3u) * NRows + int(n)] = OutT(acc3); }
}
";

// The five activation rows are independent of the four adjacent output rows
// encoded by row4 packing. One scale and one packed-weight pass feed all five
// accumulators.
const TG_LUT4_CONTRACT_M5_KERNEL_SOURCE: &str = r"
threadgroup half lut[2560];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
float acc4 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    for (uint build = tid; build < 160u; build += uint(WG)) {
        int mlocal = int(build) / 32;
        int nibble = int(build) & 31;
        int kbase = g * 128 + nibble * 4;
        int xb = mlocal * K + kbase;
        float x0 = float(x[xb + 0]);
        float x1 = float(x[xb + 1]);
        float x2 = float(x[xb + 2]);
        float x3 = float(x[xb + 3]);
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float qa0 = 0.0f;
        float qa1 = 0.0f;
        float qa2 = 0.0f;
        float qa3 = 0.0f;
        float qa4 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint packed = w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint mask = (packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16 + int(mask);
                qa0 += float(lut[li]);
                qa1 += float(lut[512 + li]);
                qa2 += float(lut[1024 + li]);
                qa3 += float(lut[1536 + li]);
                qa4 += float(lut[2048 + li]);
            }
        }
        float scale = float(sc[group_base + row_lane]);
        acc0 += scale * qa0;
        acc1 += scale * qa1;
        acc2 += scale * qa2;
        acc3 += scale * qa3;
        acc4 += scale * qa4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    y[int(n)] = OutT(acc0);
    y[NRows + int(n)] = OutT(acc1);
    y[2 * NRows + int(n)] = OutT(acc2);
    y[3 * NRows + int(n)] = OutT(acc3);
    y[4 * NRows + int(n)] = OutT(acc4);
}
";

// Gate and up use independent accumulator streams whose update order matches
// two TG-LUT4 projection dispatches. Only the activation LUT is shared. The
// kernel deliberately returns the two OutT-rounded projections: MLX's compiled
// SiLU remains authoritative because an in-kernel transcription is not bit
// exact for every F16/BF16 value.
const TG_LUT4_GATE_UP_KERNEL_SOURCE: &str = r"
constexpr int WG = 256;
constexpr int NTILE = 256;
constexpr int MTILE = 4;
threadgroup half lut[2048];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
uint mbase = threadgroup_position_in_grid.z * uint(MTILE);
float gate_acc0 = 0.0f;
float gate_acc1 = 0.0f;
float gate_acc2 = 0.0f;
float gate_acc3 = 0.0f;
float up_acc0 = 0.0f;
float up_acc1 = 0.0f;
float up_acc2 = 0.0f;
float up_acc3 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    if (tid < 128u) {
        int mlocal = int(tid) / 32;
        int nibble = int(tid) & 31;
        int m = int(mbase) + mlocal;
        int kbase = g * 128 + nibble * 4;
        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        float x3 = 0.0f;
        if (m < MRows) {
            int xb = m * K + kbase;
            x0 = float(x[xb + 0]); x1 = float(x[xb + 1]);
            x2 = float(x[xb + 2]); x3 = float(x[xb + 3]);
        }
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float gate_qa0 = 0.0f;
        float gate_qa1 = 0.0f;
        float gate_qa2 = 0.0f;
        float gate_qa3 = 0.0f;
        float up_qa0 = 0.0f;
        float up_qa1 = 0.0f;
        float up_qa2 = 0.0f;
        float up_qa3 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint gate_packed = gate_w[(group_base + word) * 4 + row_lane];
            uint up_packed = up_w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint gate_mask = (gate_packed >> (uint(ni) * 4u)) & 0xFu;
                uint up_mask = (up_packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16;
                int gate_li = li + int(gate_mask);
                int up_li = li + int(up_mask);
                gate_qa0 += float(lut[gate_li]);
                gate_qa1 += float(lut[512 + gate_li]);
                gate_qa2 += float(lut[1024 + gate_li]);
                gate_qa3 += float(lut[1536 + gate_li]);
                up_qa0 += float(lut[up_li]);
                up_qa1 += float(lut[512 + up_li]);
                up_qa2 += float(lut[1024 + up_li]);
                up_qa3 += float(lut[1536 + up_li]);
            }
        }
        float gate_scale = float(gate_sc[group_base + row_lane]);
        float up_scale = float(up_sc[group_base + row_lane]);
        gate_acc0 += gate_scale * gate_qa0;
        gate_acc1 += gate_scale * gate_qa1;
        gate_acc2 += gate_scale * gate_qa2;
        gate_acc3 += gate_scale * gate_qa3;
        up_acc0 += up_scale * up_qa0;
        up_acc1 += up_scale * up_qa1;
        up_acc2 += up_scale * up_qa2;
        up_acc3 += up_scale * up_qa3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    if (mbase + 0u < uint(MRows)) {
        gate_y[int(mbase + 0u) * NRows + int(n)] = OutT(gate_acc0);
        up_y[int(mbase + 0u) * NRows + int(n)] = OutT(up_acc0);
    }
    if (mbase + 1u < uint(MRows)) {
        gate_y[int(mbase + 1u) * NRows + int(n)] = OutT(gate_acc1);
        up_y[int(mbase + 1u) * NRows + int(n)] = OutT(up_acc1);
    }
    if (mbase + 2u < uint(MRows)) {
        gate_y[int(mbase + 2u) * NRows + int(n)] = OutT(gate_acc2);
        up_y[int(mbase + 2u) * NRows + int(n)] = OutT(up_acc2);
    }
    if (mbase + 3u < uint(MRows)) {
        gate_y[int(mbase + 3u) * NRows + int(n)] = OutT(gate_acc3);
        up_y[int(mbase + 3u) * NRows + int(n)] = OutT(up_acc3);
    }
}
";

const TG_LUT4_GATE_UP_M5_KERNEL_SOURCE: &str = r"
threadgroup half lut[2560];

uint tid = thread_index_in_threadgroup;
uint n = threadgroup_position_in_grid.x * uint(NTILE) + tid;
float gate_acc0 = 0.0f;
float gate_acc1 = 0.0f;
float gate_acc2 = 0.0f;
float gate_acc3 = 0.0f;
float gate_acc4 = 0.0f;
float up_acc0 = 0.0f;
float up_acc1 = 0.0f;
float up_acc2 = 0.0f;
float up_acc3 = 0.0f;
float up_acc4 = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    for (uint build = tid; build < 160u; build += uint(WG)) {
        int mlocal = int(build) / 32;
        int nibble = int(build) & 31;
        int kbase = g * 128 + nibble * 4;
        int xb = mlocal * K + kbase;
        float x0 = float(x[xb + 0]);
        float x1 = float(x[xb + 1]);
        float x2 = float(x[xb + 2]);
        float x3 = float(x[xb + 3]);
        float xy = x0 + x1;
        float xz = x0 + x2;
        float yz = x1 + x2;
        float xyz = xy + x2;
        float c = 0.5f * (x0 + x1 + x2 + x3);
        int base = (mlocal * 32 + nibble) * 16;
        lut[base + 0] = half(-c);
        lut[base + 1] = half(x0 - c);
        lut[base + 2] = half(x1 - c);
        lut[base + 3] = half(xy - c);
        lut[base + 4] = half(x2 - c);
        lut[base + 5] = half(xz - c);
        lut[base + 6] = half(yz - c);
        lut[base + 7] = half(xyz - c);
        lut[base + 8] = half(x3 - c);
        lut[base + 9] = half(x0 + x3 - c);
        lut[base + 10] = half(x1 + x3 - c);
        lut[base + 11] = half(xy + x3 - c);
        lut[base + 12] = half(x2 + x3 - c);
        lut[base + 13] = half(xz + x3 - c);
        lut[base + 14] = half(yz + x3 - c);
        lut[base + 15] = half(xyz + x3 - c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (n < uint(NRows)) {
        int row_tile = int(n) / 4;
        int row_lane = int(n) & 3;
        int group_base = (row_tile * NumGroups + g) * 4;
        float gate_qa0 = 0.0f;
        float gate_qa1 = 0.0f;
        float gate_qa2 = 0.0f;
        float gate_qa3 = 0.0f;
        float gate_qa4 = 0.0f;
        float up_qa0 = 0.0f;
        float up_qa1 = 0.0f;
        float up_qa2 = 0.0f;
        float up_qa3 = 0.0f;
        float up_qa4 = 0.0f;
#pragma clang loop unroll(full)
        for (int word = 0; word < 4; ++word) {
            uint gate_packed = gate_w[(group_base + word) * 4 + row_lane];
            uint up_packed = up_w[(group_base + word) * 4 + row_lane];
#pragma clang loop unroll(full)
            for (int ni = 0; ni < 8; ++ni) {
                uint gate_mask = (gate_packed >> (uint(ni) * 4u)) & 0xFu;
                uint up_mask = (up_packed >> (uint(ni) * 4u)) & 0xFu;
                int li = (word * 8 + ni) * 16;
                int gate_li = li + int(gate_mask);
                int up_li = li + int(up_mask);
                gate_qa0 += float(lut[gate_li]);
                gate_qa1 += float(lut[512 + gate_li]);
                gate_qa2 += float(lut[1024 + gate_li]);
                gate_qa3 += float(lut[1536 + gate_li]);
                gate_qa4 += float(lut[2048 + gate_li]);
                up_qa0 += float(lut[up_li]);
                up_qa1 += float(lut[512 + up_li]);
                up_qa2 += float(lut[1024 + up_li]);
                up_qa3 += float(lut[1536 + up_li]);
                up_qa4 += float(lut[2048 + up_li]);
            }
        }
        float gate_scale = float(gate_sc[group_base + row_lane]);
        float up_scale = float(up_sc[group_base + row_lane]);
        gate_acc0 += gate_scale * gate_qa0;
        gate_acc1 += gate_scale * gate_qa1;
        gate_acc2 += gate_scale * gate_qa2;
        gate_acc3 += gate_scale * gate_qa3;
        gate_acc4 += gate_scale * gate_qa4;
        up_acc0 += up_scale * up_qa0;
        up_acc1 += up_scale * up_qa1;
        up_acc2 += up_scale * up_qa2;
        up_acc3 += up_scale * up_qa3;
        up_acc4 += up_scale * up_qa4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (n < uint(NRows)) {
    gate_y[int(n)] = OutT(gate_acc0); up_y[int(n)] = OutT(up_acc0);
    gate_y[NRows + int(n)] = OutT(gate_acc1); up_y[NRows + int(n)] = OutT(up_acc1);
    gate_y[2 * NRows + int(n)] = OutT(gate_acc2); up_y[2 * NRows + int(n)] = OutT(up_acc2);
    gate_y[3 * NRows + int(n)] = OutT(gate_acc3); up_y[3 * NRows + int(n)] = OutT(up_acc3);
    gate_y[4 * NRows + int(n)] = OutT(gate_acc4); up_y[4 * NRows + int(n)] = OutT(up_acc4);
}
";

#[allow(unsafe_code)]
fn create_tg_lut4_kernel(native_m5: bool) -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"x"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(if native_m5 {
        TG_LUT4_CONTRACT_M5_KERNEL_SOURCE
    } else {
        TG_LUT4_CONTRACT_KERNEL_SOURCE
    })
    .unwrap_or_default();
    let name = if native_m5 {
        c"higgs_bonsai_q1_tg_lut4_contract_m5"
    } else {
        c"higgs_bonsai_q1_tg_lut4_contract"
    };
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
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
fn create_tg_lut4_gate_up_kernel(native_m5: bool) -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"gate_w", c"gate_sc", c"up_w", c"up_sc", c"x"]);
    let out_vec = cstr_vec(&[c"gate_y", c"up_y"]);
    let source = CString::new(if native_m5 {
        TG_LUT4_GATE_UP_M5_KERNEL_SOURCE
    } else {
        TG_LUT4_GATE_UP_KERNEL_SOURCE
    })
    .unwrap_or_default();
    let name = if native_m5 {
        c"higgs_bonsai_q1_tg_lut4_gate_up_m5"
    } else {
        c"higgs_bonsai_q1_tg_lut4_gate_up"
    };
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
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

const TG_LUT4_NATIVE_M5_WGS: [i32; 5] = [128, 160, 192, 224, 256];

fn tg_lut4_native_m5_wg() -> i32 {
    static WG: OnceLock<i32> = OnceLock::new();
    *WG.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_TG_LUT4_M5_WG")
            .ok()
            .and_then(|raw| raw.parse::<i32>().ok())
            .filter(|wg| TG_LUT4_NATIVE_M5_WGS.contains(wg))
            .unwrap_or(256)
    })
}

#[allow(unsafe_code)]
fn configure_tg_lut4_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    packed: BonsaiQ1Row4Ref<'_>,
    m_rows: i32,
    native_m5_wg: Option<i32>,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        for (name, value) in [
            (c"NRows", packed.n_rows),
            (c"MRows", m_rows),
            (c"K", packed.k_dim),
            (c"NumGroups", packed.k_dim / 128),
        ] {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                name.as_ptr(),
                value,
            );
        }
        let n_tile = native_m5_wg.unwrap_or(256);
        if let Some(wg) = native_m5_wg {
            for name in [c"WG", c"NTILE"] {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config,
                    name.as_ptr(),
                    wg,
                );
            }
        }
        let n_tiles = (packed.n_rows + n_tile - 1) / n_tile;
        let m_tiles = if native_m5_wg.is_some() {
            1
        } else {
            (m_rows + 3) / 4
        };
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tiles * n_tile, 1, m_tiles);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, n_tile, 1, 1);
        let output_shape = [m_rows, packed.n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Apply the faithful F16-LUT/F32-accumulation plan to primary row4 arrays.
/// M=1..4 and M=6..8 use the scalar contract kernel; exactly M=5 uses the
/// native five-accumulator kernel.
#[allow(unsafe_code)]
pub(super) fn bonsai_q1_tg_lut4_qmm_view(
    x: &Array,
    packed: BonsaiQ1Row4Ref<'_>,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();
    if !matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: input must be Float16/Bfloat16, got {:?}",
            x.dtype()
        )));
    }
    let input_shape = x.shape();
    if input_shape.last().copied() != Some(packed.k_dim) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: input last dim {:?}, expected {}",
            input_shape.last(),
            packed.k_dim
        )));
    }
    let m_rows: i32 = input_shape
        .iter()
        .take(input_shape.len().saturating_sub(1))
        .product();
    if !(1..=8).contains(&m_rows) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm: requires 1..=8 flattened rows, got {m_rows}"
        )));
    }
    let Some((_, leading_shape)) = input_shape.split_last() else {
        return Err(Exception::custom(
            "bonsai_q1_tg_lut4_qmm: invalid input shape",
        ));
    };
    let mut output_shape = leading_shape.to_vec();
    output_shape.push(packed.n_rows);
    let x_flat = x.reshape(&[m_rows * packed.k_dim])?;
    let native_m5_wg = (m_rows == 5).then(tg_lut4_native_m5_wg);
    let cached = if native_m5_wg.is_some() {
        TG_LUT4_CONTRACT_M5_KERNEL.get_or_init(|| CachedMetalKernel(create_tg_lut4_kernel(true)))
    } else {
        TG_LUT4_CONTRACT_KERNEL.get_or_init(|| CachedMetalKernel(create_tg_lut4_kernel(false)))
    };
    let config = configure_tg_lut4_kernel(
        unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) },
        packed,
        m_rows,
        native_m5_wg,
    );
    let input_ptrs = [
        packed.weights.as_ptr(),
        packed.scales.as_ptr(),
        x_flat.as_ptr(),
    ];
    let inputs =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };
    let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
    let stream = Stream::task_local_or_default();
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs,
            cached.0,
            inputs,
            config,
            stream.as_ptr(),
        )
    };
    let raw_result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_qmm failed: {}",
            take_last_error()
        )))
    } else {
        let mut output = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut output, outputs, 0) };
        Ok(unsafe { Array::from_ptr(output) })
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
    }
    raw_result?.reshape(&output_shape)
}

/// Owned-container convenience wrapper retained for transforms and tests.
#[cfg(test)]
pub(crate) fn bonsai_q1_tg_lut4_qmm(x: &Array, packed: &BonsaiQ1Row4) -> Result<Array, Exception> {
    bonsai_q1_tg_lut4_qmm_view(x, packed.as_ref())
}

/// Compute exact-order symmetric-Q1 gate/up projections in one dispatch.
///
/// Gate and up must use the same `[N,K]` contract. Both outputs preserve every
/// leading input dimension and replace `K` by `N`. `SwiGLU` intentionally stays
/// in the caller's compiled MLX graph so its numerics remain authoritative.
#[allow(unsafe_code, clippy::too_many_lines)]
pub(super) fn bonsai_q1_tg_lut4_gate_up_view(
    x: &Array,
    gate: BonsaiQ1Row4Ref<'_>,
    up: BonsaiQ1Row4Ref<'_>,
) -> Result<(Array, Array), Exception> {
    ensure_ffi_error_handler();
    if !matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_gate_up: input must be Float16/Bfloat16, got {:?}",
            x.dtype()
        )));
    }
    if gate.n_rows != up.n_rows || gate.k_dim != up.k_dim {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_gate_up: gate/up dimensions differ [{},{}] vs [{},{}]",
            gate.n_rows, gate.k_dim, up.n_rows, up.k_dim
        )));
    }
    let input_shape = x.shape();
    if input_shape.last().copied() != Some(gate.k_dim) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_gate_up: input last dim {:?}, expected {}",
            input_shape.last(),
            gate.k_dim
        )));
    }
    let m_rows: i32 = input_shape
        .iter()
        .take(input_shape.len().saturating_sub(1))
        .product();
    if !(1..=5).contains(&m_rows) {
        return Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_gate_up: requires 1..=5 flattened rows, got {m_rows}"
        )));
    }
    let Some((_, leading_shape)) = input_shape.split_last() else {
        return Err(Exception::custom(
            "bonsai_q1_tg_lut4_gate_up: invalid input shape",
        ));
    };
    let mut output_shape = leading_shape.to_vec();
    output_shape.push(gate.n_rows);
    // Complete every fallible operation before allocating raw FFI resources.
    // No `?` below this point may bypass config/vector cleanup.
    let x_flat = x.reshape(&[m_rows * gate.k_dim])?;

    let native_m5_wg = (m_rows == 5).then(tg_lut4_native_m5_wg);
    let cached = if native_m5_wg.is_some() {
        TG_LUT4_GATE_UP_M5_KERNEL
            .get_or_init(|| CachedMetalKernel(create_tg_lut4_gate_up_kernel(true)))
    } else {
        TG_LUT4_GATE_UP_KERNEL
            .get_or_init(|| CachedMetalKernel(create_tg_lut4_gate_up_kernel(false)))
    };
    let config = configure_tg_lut4_kernel(
        unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) },
        gate,
        m_rows,
        native_m5_wg,
    );
    let flat_output_shape = [m_rows, gate.n_rows];
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            flat_output_shape.as_ptr(),
            flat_output_shape.len(),
            mlx_sys::mlx_array_dtype(x.as_ptr()),
        );
    }
    let input_ptrs = [
        gate.weights.as_ptr(),
        gate.scales.as_ptr(),
        up.weights.as_ptr(),
        up.scales.as_ptr(),
        x_flat.as_ptr(),
    ];
    let inputs =
        unsafe { mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len()) };
    let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
    let stream = Stream::task_local_or_default();
    let status = unsafe {
        mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs,
            cached.0,
            inputs,
            config,
            stream.as_ptr(),
        )
    };
    let raw_result = if status != 0 {
        Err(Exception::custom(format!(
            "bonsai_q1_tg_lut4_gate_up failed: {}",
            take_last_error()
        )))
    } else {
        let mut gate_output_ptr = unsafe { mlx_sys::mlx_array_new() };
        let mut up_output_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe {
            mlx_sys::mlx_vector_array_get(&raw mut gate_output_ptr, outputs, 0);
            mlx_sys::mlx_vector_array_get(&raw mut up_output_ptr, outputs, 1);
        }
        let gate_output = unsafe { Array::from_ptr(gate_output_ptr) };
        let up_output = unsafe { Array::from_ptr(up_output_ptr) };
        Ok((gate_output, up_output))
    };
    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
    }
    let (gate_output, up_output) = raw_result?;
    Ok((
        gate_output.reshape(&output_shape)?,
        up_output.reshape(&output_shape)?,
    ))
}

/// Owned-container convenience wrapper retained for transforms and tests.
#[cfg(test)]
pub(crate) fn bonsai_q1_tg_lut4_gate_up(
    x: &Array,
    gate: &BonsaiQ1Row4,
    up: &BonsaiQ1Row4,
) -> Result<(Array, Array), Exception> {
    bonsai_q1_tg_lut4_gate_up_view(x, gate.as_ref(), up.as_ref())
}

// ---------------------------------------------------------------------------
// 1-bit dequantize to dense (embedding gather + prefill matmul path).
//
// wd[n, c] = scales[n, c/G] * bit(w[n, c/32], c%32) + biases[n, c/G].
// One thread per packed uint32 word (writes 32 dense outputs).
// ---------------------------------------------------------------------------

const DEQUANT_KERNEL_SOURCE: &str = r"
uint gid = thread_position_in_grid.x;
if (gid >= uint(NWords)) { return; }

uint n = gid / uint(KPacked);
uint idx = gid % uint(KPacked);
uint packed = w[gid];

int g = int(idx) * 32 / GroupSize;
float s_val = float(sc[n * uint(NumGroups) + uint(g)]);
float b_val = Symmetric ? (-0.5f * s_val) : float(bi[n * uint(NumGroups) + uint(g)]);

uint base = n * uint(K) + idx * 32u;
for (uint j = 0u; j < 32u; ++j) {
    float bit = float((packed >> j) & 1u);
    wd[base + j] = OutT(s_val * bit + b_val);
}
";

// Symmetric-Q1 dequantization directly from the primary row4 representation.
//
// A canonical packed word `(n, idx)` belongs to `group = idx / 4` and
// `word = idx % 4`. Row4 stores that word at
// `[n/4, group, word, n%4]`, while its scale is at
// `[n/4, group, n%4]`. Keeping this remap in the kernel avoids reconstructing
// a second, canonical packed matrix solely for wide-prefill dequantization.
const ROW4_DEQUANT_KERNEL_SOURCE: &str = r"
uint gid = thread_position_in_grid.x;
if (gid >= uint(NWords)) { return; }

uint n = gid / uint(KPacked);
uint idx = gid % uint(KPacked);
uint tile = n / 4u;
uint lane = n & 3u;
uint group = idx / 4u;
uint word = idx & 3u;

uint row4_group = tile * uint(NumGroups) + group;
uint packed = w[(row4_group * 4u + word) * 4u + lane];
float s_val = float(sc[row4_group * 4u + lane]);
float b_val = -0.5f * s_val;

uint base = n * uint(K) + idx * 32u;
for (uint j = 0u; j < 32u; ++j) {
    float bit = float((packed >> j) & 1u);
    wd[base + j] = OutT(s_val * bit + b_val);
}
";

#[allow(unsafe_code)]
fn create_dequant_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi"]);
    let out_vec = cstr_vec(&[c"wd"]);
    let source = CString::new(DEQUANT_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_dequant".as_ptr(),
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

#[allow(unsafe_code)]
fn create_row4_dequant_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc"]);
    let out_vec = cstr_vec(&[c"wd"]);
    let source = CString::new(ROW4_DEQUANT_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_row4_dequant".as_ptr(),
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

#[allow(unsafe_code)]
fn configure_dequant_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let k_packed = k_dim / 32;
    let n_words = n_rows * k_packed;
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
            c"KPacked".as_ptr(),
            k_packed,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"GroupSize".as_ptr(),
            group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NWords".as_ptr(),
            n_words,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"Symmetric".as_ptr(),
            i32::from(symmetric),
        );

        let tg: i32 = 256;
        let grid = ((n_words + tg - 1) / tg) * tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1);

        let wd_shape = [n_rows, k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            wd_shape.as_ptr(),
            wd_shape.len(),
            out_dtype,
        );
        config
    }
}

#[allow(unsafe_code)]
fn configure_row4_dequant_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    k_dim: i32,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let k_packed = k_dim / 32;
    let n_words = n_rows * k_packed;
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
            c"KPacked".as_ptr(),
            k_packed,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / 128,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NWords".as_ptr(),
            n_words,
        );

        let tg: i32 = 256;
        let grid = ((n_words + tg - 1) / tg) * tg;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, grid, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1);

        let wd_shape = [n_rows, k_dim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            wd_shape.as_ptr(),
            wd_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Dequantize a packed 1-bit matrix to a dense `[out_features, in_features]`
/// array (dtype matches `scales`). Used for embedding gather and the prefill
/// (M > 1) matmul path.
#[allow(unsafe_code)]
pub fn bonsai_q1_dequant(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_dequant: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_dequant: weight has no columns"))?;
    let k_dim = k_packed * 32;

    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(scales.as_ptr()) };

    let cached = DEQUANT_KERNEL.get_or_init(|| CachedMetalKernel(create_dequant_kernel()));
    let config = configure_dequant_kernel(out_dtype, n_rows, k_dim, group_size, symmetric);

    let input_ptrs = [w_flat.as_ptr(), s_flat.as_ptr(), b_flat.as_ptr()];
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
        Err(Exception::custom(format!(
            "bonsai_q1_dequant failed: {}",
            take_last_error()
        )))
    } else {
        let mut wd_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut wd_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(wd_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

/// Dequantize a symmetric-Q1 row4 matrix directly to dense `[N,K]` storage.
///
/// The output dtype and arithmetic match [`bonsai_q1_dequant`] with an empty
/// bias sentinel and `group_size = 128`, without first materializing canonical
/// `[N,K/32]` weights and `[N,K/128]` scales.
#[allow(unsafe_code)]
pub(super) fn bonsai_q1_row4_dequant_view(packed: BonsaiQ1Row4Ref<'_>) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(packed.scales.as_ptr()) };
    let cached =
        ROW4_DEQUANT_KERNEL.get_or_init(|| CachedMetalKernel(create_row4_dequant_kernel()));
    let config = configure_row4_dequant_kernel(out_dtype, packed.n_rows, packed.k_dim);
    let input_ptrs = [packed.weights.as_ptr(), packed.scales.as_ptr()];
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
        Err(Exception::custom(format!(
            "bonsai_q1_row4_dequant failed: {}",
            take_last_error()
        )))
    } else {
        let mut wd_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut wd_ptr, outputs_vec, 0) };
        Ok(unsafe { Array::from_ptr(wd_ptr) })
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
    }
    result
}

/// Owned-container convenience wrapper retained for transforms and tests.
#[cfg(test)]
pub(crate) fn bonsai_q1_row4_dequant(packed: &BonsaiQ1Row4) -> Result<Array, Exception> {
    bonsai_q1_row4_dequant_view(packed.as_ref())
}

static DEQUANT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static ROW4_DEQUANT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
static WIDE_QMM_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

// ---------------------------------------------------------------------------
// Wide packed-Q1 matrix multiply (`bonsai_q1_wide_qmm`).
//
// Replaces the fp16 dequant + dense matmul fallback used by wide prefill
// (M > `bonsai_q1_qmm_max_rows`). The dequant path materializes a ~170 MiB
// fp16 projection per layer and pays ~352 MiB of weight traffic; this kernel
// keeps weights packed and reads each packed word roughly once per m-tile.
//
// Tiling: grid `(ceil(N/BN)*BM, ceil(M/BM), 1)` with `thread_group (BM,1,1)`.
// Each thread owns one input row `m_local = tid` and accumulates `BN` output
// columns. The `BN` packed words and scales for the current group are loaded
// cooperatively into threadgroup shared memory, so a single weight read feeds
// all `BM` input rows of the tile. The affine contract is identical to
// [`bonsai_q1_qmv_fast`]: per group of `GroupSize` (128),
//   `acc += scale * sum(bit * x) + bias * sum(x)`,
// with `bias = -scale/2` for the symmetric case. The narrow verify path
// (`bonsai_q1_qmm`, M <= 8) is deliberately untouched.
//
// `Row4` reuses the authoritative primary row4 buffers (`[N/4,K/128,4,4]`
// weights, `[N/4,K/128,4]` scales) without reconstructing canonical packing.
// ---------------------------------------------------------------------------

const WIDE_QMM_BN: i32 = 8;
const WIDE_QMM_BM: i32 = 32;

const WIDE_QMM_KERNEL_SOURCE: &str = r"
constexpr int WORDS_PER_GROUP = GroupSize / 32;
constexpr int BK = 128;                         // one affine group per K-tile
constexpr int VPT = 32;
constexpr int CELLS = BM * BN;                  // threadgroup size (one cell per thread)

uint tgx = threadgroup_position_in_grid.x;
uint tgy = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
int n_local = int(tid) % BN;
int m_local = int(tid) / BN;
int n_global = int(tgx) * BN + n_local;
int m_global = int(tgy) * BM + m_local;

threadgroup float x_tile[BM * BK];
threadgroup uint w_tile[BN * WORDS_PER_GROUP];
threadgroup float sc_tile[BN];
threadgroup float bi_tile[BN];

float acc = 0.0f;

for (int g = 0; g < NumGroups; ++g) {
    // Cooperative coalesced load of x_tile[BM, BK]. Consecutive threads map to
    // consecutive K within a row, so each wave of BK threads loads one m-row
    // contiguously from global memory. Widened to float so the tile dtype is
    // independent of whether the input is fp16 or bf16.
    int x_row_base = int(tgy) * BM * K + g * BK;
    for (uint i = tid; i < uint(BM * BK); i += uint(CELLS)) {
        int xm = int(i) / BK;
        int xk = int(i) % BK;
        int xm_global = int(tgy) * BM + xm;
        // Tail m-tiles must not read past the input's row count. Out-of-range
        // rows are zeroed; their results are discarded by the `active` guard.
        x_tile[i] = (xm_global < m_param) ? float(x[x_row_base + xm * K + xk]) : 0.0f;
    }
    // Cooperative load of this group's packed words + scales (+ biases) for the
    // BN output rows of the tile. Out-of-range rows are zeroed so the compute
    // path reads deterministic values it will discard.
    for (uint i = tid; i < uint(BN * WORDS_PER_GROUP); i += uint(CELLS)) {
        int wn = int(i) / WORDS_PER_GROUP;
        int ww = int(i) % WORDS_PER_GROUP;
        int wn_global = int(tgx) * BN + wn;
        uint packed = 0u;
        if (wn_global < n_param) {
            uint tile = uint(wn_global) / 4u;
            uint lane = uint(wn_global) & 3u;
            if constexpr (Row4) {
                uint row4_group = tile * uint(NumGroups) + uint(g);
                packed = w[(row4_group * 4u + uint(ww)) * 4u + lane];
            } else {
                packed = w[uint(wn_global) * uint(KPacked) + uint(g) * uint(WORDS_PER_GROUP) + uint(ww)];
            }
        }
        w_tile[wn * WORDS_PER_GROUP + ww] = packed;
    }
    for (uint i = tid; i < uint(BN); i += uint(CELLS)) {
        int sn_global = int(tgx) * BN + int(i);
        float s_val = 0.0f;
        float b_val = 0.0f;
        if (sn_global < n_param) {
            uint tile = uint(sn_global) / 4u;
            uint lane = uint(sn_global) & 3u;
            if constexpr (Row4) {
                s_val = float(sc[(tile * uint(NumGroups) + uint(g)) * 4u + lane]);
            } else {
                s_val = float(sc[uint(sn_global) * uint(NumGroups) + uint(g)]);
                if constexpr (!Symmetric) {
                    b_val = float(bi[uint(sn_global) * uint(NumGroups) + uint(g)]);
                }
            }
        }
        sc_tile[i] = s_val;
        bi_tile[i] = b_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (m_global < m_param && n_global < n_param) {
        float sum_x = 0.0f;
        float dot = 0.0f;
        for (int word = 0; word < WORDS_PER_GROUP; ++word) {
            uint packed = w_tile[n_local * WORDS_PER_GROUP + word];
            int xb = word * VPT;
            float d = 0.0f;
            for (int bk = 0; bk < 4; ++bk) {
                uint wb = (packed >> (uint(bk) * 8u)) & 0xFFu;
                int b = xb + bk * 8;
                d += select(0.0f, float(x_tile[m_local * BK + b + 0]), (wb & 0x01u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 1]), (wb & 0x02u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 2]), (wb & 0x04u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 3]), (wb & 0x08u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 4]), (wb & 0x10u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 5]), (wb & 0x20u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 6]), (wb & 0x40u) != 0u);
                d += select(0.0f, float(x_tile[m_local * BK + b + 7]), (wb & 0x80u) != 0u);
            }
            dot += d;
            for (int i = 0; i < VPT; ++i) {
                sum_x += float(x_tile[m_local * BK + xb + i]);
            }
        }
        float s_val = float(sc_tile[n_local]);
        float b_val;
        if constexpr (Symmetric) {
            b_val = -0.5f * s_val;
        } else {
            b_val = float(bi_tile[n_local]);
        }
        acc += s_val * dot + b_val * sum_x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (m_global < m_param && n_global < n_param) {
    y[m_global * n_param + n_global] = OutT(acc);
}
";

#[allow(unsafe_code)]
fn create_wide_qmm_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"w", c"sc", c"bi", c"x", c"n_param", c"m_param"]);
    let out_vec = cstr_vec(&[c"y"]);
    let source = CString::new(WIDE_QMM_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_bonsai_q1_wide_qmm".as_ptr(),
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
fn configure_wide_qmm_kernel(
    out_dtype: mlx_sys::mlx_dtype,
    n_rows: i32,
    m_rows: i32,
    k_dim: i32,
    group_size: i32,
    symmetric: bool,
    row4: bool,
) -> mlx_sys::mlx_fast_metal_kernel_config {
    let k_packed = k_dim / 32;
    let num_groups = k_dim / group_size;
    let bn = WIDE_QMM_BN;
    let bm = WIDE_QMM_BM;
    let cells = bm * bn;
    let n_tiles = (n_rows + bn - 1) / bn;
    let m_tiles = (m_rows + bm - 1) / bm;
    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            c"OutT".as_ptr(),
            out_dtype,
        );
        for (name, value) in [
            (c"K", k_dim),
            (c"GroupSize", group_size),
            (c"KPacked", k_packed),
            (c"NumGroups", num_groups),
            (c"Symmetric", i32::from(symmetric)),
            (c"Row4", i32::from(row4)),
            (c"BN", bn),
            (c"BM", bm),
        ] {
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                name.as_ptr(),
                value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n_tiles * cells, m_tiles, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, cells, 1, 1);
        let output_shape = [m_rows, n_rows];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            output_shape.as_ptr(),
            output_shape.len(),
            out_dtype,
        );
        config
    }
}

/// Whether the wide packed-Q1 QMM may replace the fp16-dequant prefill path.
/// Off by default until benchmarked on the real Bonsai-27B checkpoint; the
/// dequant fallback remains exact and is always available.
fn wide_qmm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HIGGS_BONSAI_QMM_WIDE").is_ok_and(|v| v == "1"))
}

/// Wide packed-Q1 multiply against canonical `[N, K/32]` packed weights.
/// Falls back to `Ok(None)` when the wide path is disabled or outside its
/// validated shape domain, leaving the caller on the existing dequant path.
#[allow(unsafe_code)]
pub(super) fn bonsai_q1_wide_qmm(
    x: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Option<Array>, Exception> {
    if !wide_qmm_enabled() {
        return Ok(None);
    }
    ensure_ffi_error_handler();
    let x_shape = x.shape();
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows <= bonsai_q1_narrow_qmm_cap() {
        return Ok(None);
    }
    let weight_shape = weight.shape();
    let n_rows = weight_shape
        .first()
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_wide_qmm: weight has no rows"))?;
    let k_packed = weight_shape
        .get(1)
        .copied()
        .ok_or_else(|| Exception::custom("bonsai_q1_wide_qmm: weight has no columns"))?;
    let k_dim = k_packed * 32;
    // The wide kernel hardcodes BK = 128 (one affine group per K-tile) and
    // WORDS_PER_GROUP = GroupSize/32. Only group_size = 128 is sound here;
    // any other valid group size would overrun x_tile. Fall back to dequant.
    if group_size != 128 || k_dim % group_size != 0 || k_dim % 32 != 0 {
        return Ok(None);
    }

    let x_flat = row_contiguous_copy(&x.reshape(&[m_rows, k_dim])?)?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let symmetric = biases.size() == 0;
    let b_flat = if symmetric {
        s_flat.clone()
    } else {
        biases.flatten(None, None)?
    };

    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached = WIDE_QMM_KERNEL.get_or_init(|| CachedMetalKernel(create_wide_qmm_kernel()));
    let config = configure_wide_qmm_kernel(
        out_dtype, n_rows, m_rows, k_dim, group_size, symmetric, false,
    );

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(n_rows) };
    let m_scalar = unsafe { mlx_sys::mlx_array_new_int(m_rows) };
    let input_ptrs = [
        w_flat.as_ptr(),
        s_flat.as_ptr(),
        b_flat.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
        m_scalar,
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
        Err(Exception::custom(format!(
            "bonsai_q1_wide_qmm failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_wide_qmm: x_shape too small"))?
            .to_vec();
        out_shape.push(n_rows);
        y.reshape(&out_shape).map(Some)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
        mlx_sys::mlx_array_free(m_scalar);
    }
    result
}

/// Wide packed-Q1 multiply against authoritative primary row4 buffers.
/// Symmetric affine only (the row4 promotion domain). Returns `Ok(None)` when
/// the wide path is disabled or the input is inside the narrow TG-LUT4 range.
#[allow(unsafe_code)]
pub(super) fn bonsai_q1_row4_wide_qmm_view(
    x: &Array,
    packed: BonsaiQ1Row4Ref<'_>,
) -> Result<Option<Array>, Exception> {
    if !wide_qmm_enabled() {
        return Ok(None);
    }
    ensure_ffi_error_handler();
    if !matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16) {
        return Ok(None);
    }
    let x_shape = x.shape();
    if x_shape.last().copied() != Some(packed.k_dim) {
        return Ok(None);
    }
    let m_rows: i32 = x_shape
        .iter()
        .take(x_shape.len().saturating_sub(1))
        .product();
    if m_rows <= bonsai_q1_narrow_qmm_cap() {
        return Ok(None);
    }

    let x_flat = row_contiguous_copy(&x.reshape(&[m_rows, packed.k_dim])?)?;
    let stream = Stream::task_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };
    let cached = WIDE_QMM_KERNEL.get_or_init(|| CachedMetalKernel(create_wide_qmm_kernel()));
    let config = configure_wide_qmm_kernel(
        out_dtype,
        packed.n_rows,
        m_rows,
        packed.k_dim,
        128,
        true,
        true,
    );

    let n_scalar = unsafe { mlx_sys::mlx_array_new_int(packed.n_rows) };
    let m_scalar = unsafe { mlx_sys::mlx_array_new_int(m_rows) };
    // Row4 promotion is symmetric-Q1 only; the kernel never indexes `bi` under
    // `Symmetric`, so the scales buffer stands in as a valid placeholder input.
    let input_ptrs = [
        packed.weights.as_ptr(),
        packed.scales.as_ptr(),
        packed.scales.as_ptr(),
        x_flat.as_ptr(),
        n_scalar,
        m_scalar,
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
        Err(Exception::custom(format!(
            "bonsai_q1_row4_wide_qmm failed: {}",
            take_last_error()
        )))
    } else {
        let mut y_ptr = unsafe { mlx_sys::mlx_array_new() };
        unsafe { mlx_sys::mlx_vector_array_get(&raw mut y_ptr, outputs_vec, 0) };
        let y = unsafe { Array::from_ptr(y_ptr) };
        let trim_to = x_shape.len().saturating_sub(1);
        let mut out_shape = x_shape
            .get(..trim_to)
            .ok_or_else(|| Exception::custom("bonsai_q1_row4_wide_qmm: x_shape too small"))?
            .to_vec();
        out_shape.push(packed.n_rows);
        y.reshape(&out_shape).map(Some)
    };

    unsafe {
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_array_free(n_scalar);
        mlx_sys::mlx_array_free(m_scalar);
    }
    result
}

/// Upper bound on M for which the narrow packed path (`bonsai_q1_qmm`,
/// [`bonsai_q1_tg_lut4_qmm_view`]) is authoritative. The wide kernel only
/// engages above this to avoid duplicating the proven narrow domain.
fn bonsai_q1_narrow_qmm_cap() -> i32 {
    static CAP: OnceLock<i32> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("HIGGS_BONSAI_QMM_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|n| (0..=64).contains(n))
            .unwrap_or(8)
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::shadow_reuse,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::mlx_exec::eval;
    use mlx_rs::Dtype;

    const GROUP_SIZE: i32 = 128;

    fn patterned_weights(n: i32, k: i32, dtype: Dtype, symmetric: bool) -> (Array, Array, Array) {
        assert_eq!(k % GROUP_SIZE, 0);
        let packed = (0..n * k / 32)
            .map(|index| {
                let shift = u32::try_from((index * 7 + 3).rem_euclid(31)).unwrap();
                0x963C_A5F0_u32.rotate_left(shift)
            })
            .collect::<Vec<_>>();
        let scale_values = (0..n * k / GROUP_SIZE)
            .map(|index| ((index % 7) as f32).mul_add(0.031_25, 0.125))
            .collect::<Vec<_>>();
        let bias_values = (0..n * k / GROUP_SIZE)
            .map(|index| ((index % 5) as f32).mul_add(0.015_625, -0.093_75))
            .collect::<Vec<_>>();

        let weight = Array::from_slice(&packed, &[n, k / 32]);
        let scales = Array::from_slice(&scale_values, &[n, k / GROUP_SIZE])
            .as_dtype(dtype)
            .unwrap();
        let biases = if symmetric {
            let empty = Vec::<f32>::new();
            Array::from_slice(&empty, &[0])
        } else {
            Array::from_slice(&bias_values, &[n, k / GROUP_SIZE])
                .as_dtype(dtype)
                .unwrap()
        };
        (weight, scales, biases)
    }

    fn patterned_input(m: i32, k: i32, dtype: Dtype) -> Array {
        Array::from_slice(
            &(0..m * k)
                .map(|index| ((index * 11 + 9).rem_euclid(53) - 26) as f32 * 0.007_812_5)
                .collect::<Vec<_>>(),
            &[m, k],
        )
        .as_dtype(dtype)
        .unwrap()
    }

    fn assert_array_exact(label: &str, actual: &Array, expected: &Array) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        assert_eq!(actual.dtype(), expected.dtype(), "{label} dtype");
        let actual_f32 = actual.as_dtype(Dtype::Float32).unwrap();
        let expected_f32 = expected.as_dtype(Dtype::Float32).unwrap();
        eval([&actual_f32, &expected_f32]).unwrap();
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

    #[test]
    fn aligned_fast_qmv_matches_guarded_kernel_for_aligned_and_unaligned_n() {
        const M: i32 = 5;
        const K: i32 = 1024;
        let _exec = crate::mlx_exec::acquire();
        let nsg = fast_qmv_nsg();

        for &(n, symmetric) in &[(64_i32, true), (65_i32, false)] {
            assert_eq!(
                fast_qmv_has_aligned_rows(n, nsg, true),
                n == 64,
                "test shape must select the intended specialization"
            );
            let (weight, scales, biases) = patterned_weights(n, K, Dtype::Bfloat16, symmetric);
            let x = patterned_input(M, K, Dtype::Bfloat16);
            let guarded =
                bonsai_q1_qmv_fast_impl(&x, &weight, &scales, &biases, GROUP_SIZE, false).unwrap();
            let candidate =
                bonsai_q1_qmv_fast_impl(&x, &weight, &scales, &biases, GROUP_SIZE, true).unwrap();
            let public = bonsai_q1_qmv_fast(&x, &weight, &scales, &biases, GROUP_SIZE).unwrap();
            assert_array_exact(
                if n == 64 {
                    "aligned-N specialized QMV"
                } else {
                    "unaligned-N guarded fallback"
                },
                &candidate,
                &guarded,
            );
            assert_array_exact("public aligned-QMV dispatch", &public, &guarded);
        }
    }

    #[test]
    fn tg_lut4_row4_transform_matches_index_oracle_and_rejects_flat_layout() {
        const N: i32 = 8;
        const K: i32 = 256;
        let _exec = crate::mlx_exec::acquire();
        let groups = K / GROUP_SIZE;
        let k_packed = K / 32;
        let bits = (0..N * k_packed)
            .map(|index| u32::try_from(index).unwrap())
            .collect::<Vec<_>>();
        let scale_values = (0..N * groups)
            .map(|index| index as f32 + 0.25)
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&bits, &[N, k_packed]);
        let scales = Array::from_slice(&scale_values, &[N, groups])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let packed = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
        eval([&packed.weights, &packed.scales]).unwrap();

        let packed_bits = packed.weights.as_slice::<u32>();
        let packed_scales_f32 = packed.scales.as_dtype(Dtype::Float32).unwrap();
        eval([&packed_scales_f32]).unwrap();
        let packed_scales = packed_scales_f32.as_slice::<f32>();
        for tile in 0..N / 4 {
            for group in 0..groups {
                for word in 0..4 {
                    for lane in 0..4 {
                        let src_row = tile * 4 + lane;
                        let src = (src_row * k_packed + group * 4 + word) as usize;
                        let dst = ((((tile * groups + group) * 4 + word) * 4) + lane) as usize;
                        assert_eq!(packed_bits[dst], bits[src]);
                    }
                }
                for lane in 0..4 {
                    let src = ((tile * 4 + lane) * groups + group) as usize;
                    let dst = ((tile * groups + group) * 4 + lane) as usize;
                    let expected = half::bf16::from_f32(scale_values[src]).to_f32();
                    assert_eq!(packed_scales[dst].to_bits(), expected.to_bits());
                }
            }
        }

        let (roundtrip_weight, roundtrip_scales) = packed.to_row_major().unwrap();
        assert_eq!(roundtrip_weight.shape(), weight.shape());
        assert_eq!(roundtrip_scales.shape(), scales.shape());
        assert!(array_is_row_contiguous(&roundtrip_weight).unwrap());
        assert!(array_is_row_contiguous(&roundtrip_scales).unwrap());
        eval([&roundtrip_weight]).unwrap();
        assert_eq!(roundtrip_weight.as_slice::<u32>(), bits.as_slice());
        let roundtrip_scales_f32 = roundtrip_scales.as_dtype(Dtype::Float32).unwrap();
        eval([&roundtrip_scales_f32]).unwrap();
        for (index, &actual) in roundtrip_scales_f32.as_slice::<f32>().iter().enumerate() {
            let expected = half::bf16::from_f32(scale_values[index]).to_f32();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        let flat_bits = Array::from_slice(&bits, &[bits.len() as i32]);
        let flat_scales = scales.reshape(&[-1]).unwrap();
        let error = BonsaiQ1Row4::from_packed_parts(flat_bits, flat_scales, N, K).unwrap_err();
        assert!(error.to_string().contains("invalid packed contract"));
    }

    #[test]
    fn row4_symmetric_dequant_matches_canonical_for_fp16_and_bf16() {
        const N: i32 = 12;
        const K: i32 = 384;
        let _exec = crate::mlx_exec::acquire();
        let bits = (0..N * K / 32)
            .map(|index| {
                let index = u32::try_from(index).unwrap();
                0xA53C_96F0_u32.rotate_left(index % 31) ^ index.wrapping_mul(0x9E37_79B9)
            })
            .collect::<Vec<_>>();
        let scale_values = (0..N * K / GROUP_SIZE)
            .map(|index| (index as f32).mul_add(0.003_906_25, 0.062_5))
            .collect::<Vec<_>>();
        let weight = Array::from_slice(&bits, &[N, K / 32]);
        let empty_bias = Array::from_slice(&Vec::<f32>::new(), &[0]);

        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let scales = Array::from_slice(&scale_values, &[N, K / GROUP_SIZE])
                .as_dtype(dtype)
                .unwrap();
            let row4 = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
            let canonical = bonsai_q1_dequant(&weight, &scales, &empty_bias, GROUP_SIZE).unwrap();
            let direct = bonsai_q1_row4_dequant(&row4).unwrap();
            assert_eq!(direct.shape(), &[N, K]);
            assert_eq!(direct.dtype(), dtype);
            assert_array_exact(
                &format!("row4 symmetric dequant dtype={dtype:?}"),
                &direct,
                &canonical,
            );
        }
    }

    #[test]
    fn tg_lut4_fp16_bf16_scales_preserve_shape_and_m1_plan_through_m8() {
        const N: i32 = 64;
        const K: i32 = 1024;
        const MAX_M: i32 = 8;
        let _exec = crate::mlx_exec::acquire();
        let values = (0..MAX_M * K)
            .map(|index| {
                let row = index / K;
                let col = index % K;
                ((row * 29 + col * 11 + 7).rem_euclid(61) - 30) as f32 * 0.007_812_5
            })
            .collect::<Vec<_>>();

        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let (weight, scales, _) = patterned_weights(N, K, dtype, true);
            let packed = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
            assert_eq!(packed.scales.dtype(), dtype);

            for m in 1..=MAX_M {
                let input = Array::from_slice(&values[..(m * K) as usize], &[1, m, K])
                    .as_dtype(dtype)
                    .unwrap();
                let stacked = bonsai_q1_tg_lut4_qmm(&input, &packed).unwrap();
                assert_eq!(stacked.shape(), &[1, m, N]);
                let stacked_f32 = stacked.as_dtype(Dtype::Float32).unwrap();
                eval([&stacked_f32]).unwrap();

                for row in 0..m {
                    let start = (row * K) as usize;
                    let single = Array::from_slice(&values[start..start + K as usize], &[1, 1, K])
                        .as_dtype(dtype)
                        .unwrap();
                    let single = bonsai_q1_tg_lut4_qmm(&single, &packed)
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap();
                    eval([&single]).unwrap();
                    for column in 0..N as usize {
                        let got = stacked_f32.as_slice::<f32>()[(row * N) as usize + column];
                        let expected = single.as_slice::<f32>()[column];
                        assert_eq!(
                            got.to_bits(),
                            expected.to_bits(),
                            "dtype={dtype:?} M={m} row={row} column={column}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tg_lut4_fused_gate_up_matches_separate_projections_for_m1_through_m5() {
        const N: i32 = 64;
        const K: i32 = 1024;
        const MAX_M: i32 = 5;
        let _exec = crate::mlx_exec::acquire();

        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let (gate_weight, gate_scales, _) = patterned_weights(N, K, dtype, true);
            let up_bits = (0..N * K / 32)
                .map(|index| {
                    let shift = u32::try_from((index * 13 + 17).rem_euclid(31)).unwrap();
                    0x5A36_C90F_u32.rotate_left(shift)
                })
                .collect::<Vec<_>>();
            let up_scale_values = (0..N * K / GROUP_SIZE)
                .map(|index| ((index % 11) as f32).mul_add(0.023_437_5, 0.093_75))
                .collect::<Vec<_>>();
            let up_weight = Array::from_slice(&up_bits, &[N, K / 32]);
            let up_scales = Array::from_slice(&up_scale_values, &[N, K / GROUP_SIZE])
                .as_dtype(dtype)
                .unwrap();
            let gate = BonsaiQ1Row4::from_row_major(&gate_weight, &gate_scales).unwrap();
            let up = BonsaiQ1Row4::from_row_major(&up_weight, &up_scales).unwrap();

            for m in 1..=MAX_M {
                let input = patterned_input(m, K, dtype).reshape(&[1, m, K]).unwrap();
                let gate_out = bonsai_q1_tg_lut4_qmm(&input, &gate).unwrap();
                let up_out = bonsai_q1_tg_lut4_qmm(&input, &up).unwrap();
                let (fused_gate, fused_up) = bonsai_q1_tg_lut4_gate_up(&input, &gate, &up).unwrap();
                assert_eq!(fused_gate.shape(), &[1, m, N]);
                assert_eq!(fused_up.shape(), &[1, m, N]);
                assert_array_exact(
                    &format!("fused gate dtype={dtype:?} M={m}"),
                    &fused_gate,
                    &gate_out,
                );
                assert_array_exact(
                    &format!("fused up dtype={dtype:?} M={m}"),
                    &fused_up,
                    &up_out,
                );
            }
        }
    }

    #[test]
    fn tg_lut4_fused_gate_up_rejects_mismatched_projection_contracts() {
        const K: i32 = 1024;
        let _exec = crate::mlx_exec::acquire();
        let (gate_weight, gate_scales, _) = patterned_weights(64, K, Dtype::Bfloat16, true);
        let (up_weight, up_scales, _) = patterned_weights(68, K, Dtype::Bfloat16, true);
        let gate = BonsaiQ1Row4::from_row_major(&gate_weight, &gate_scales).unwrap();
        let up = BonsaiQ1Row4::from_row_major(&up_weight, &up_scales).unwrap();
        let input = patterned_input(1, K, Dtype::Bfloat16);
        let error = bonsai_q1_tg_lut4_gate_up(&input, &gate, &up).unwrap_err();
        assert!(error.to_string().contains("gate/up dimensions differ"));
    }

    /// Paired microbenchmark for the shared-LUT gate/up projection on the
    /// dominant Bonsai-27B dense-MLP shape. Samples alternate A/B and B/A
    /// ordering so slow thermal drift does not consistently favor one path.
    ///
    /// ```bash
    /// HIGGS_BONSAI_GATE_UP_BENCH_SAMPLES=31 \
    /// cargo test -p higgs-models --release --lib -- \
    ///   metal_kernel::tests::bench_tg_lut4_gate_up_bonsai_27b \
    ///   --ignored --nocapture --exact \
    ///   --test-threads=1
    /// ```
    #[test]
    #[ignore = "microbenchmark, requires Apple Metal GPU"]
    fn bench_tg_lut4_gate_up_bonsai_27b() {
        use std::time::Instant;

        const N: i32 = 17_408;
        const K: i32 = 5_120;
        const M_VALUES: [i32; 2] = [1, 5];
        const DEFAULT_WARMUP: usize = 8;
        const DEFAULT_SAMPLES: usize = 31;

        fn elapsed_separate(input: &Array, gate: &BonsaiQ1Row4, up: &BonsaiQ1Row4) -> f64 {
            let start = Instant::now();
            let gate_output = bonsai_q1_tg_lut4_qmm(input, gate).unwrap();
            let up_output = bonsai_q1_tg_lut4_qmm(input, up).unwrap();
            eval([&gate_output, &up_output]).unwrap();
            start.elapsed().as_secs_f64() * 1_000_000.0
        }

        fn elapsed_fused(input: &Array, gate: &BonsaiQ1Row4, up: &BonsaiQ1Row4) -> f64 {
            let start = Instant::now();
            let (gate_output, up_output) = bonsai_q1_tg_lut4_gate_up(input, gate, up).unwrap();
            eval([&gate_output, &up_output]).unwrap();
            start.elapsed().as_secs_f64() * 1_000_000.0
        }

        fn median(values: &[f64]) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[sorted.len() / 2]
        }

        fn mean(values: &[f64]) -> f64 {
            values.iter().sum::<f64>() / values.len() as f64
        }

        let _exec = crate::mlx_exec::acquire();
        let warmup = std::env::var("HIGGS_BONSAI_GATE_UP_BENCH_WARMUP")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(DEFAULT_WARMUP);
        let samples = std::env::var("HIGGS_BONSAI_GATE_UP_BENCH_SAMPLES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(DEFAULT_SAMPLES);

        let (gate_weight, gate_scales, _) = patterned_weights(N, K, Dtype::Bfloat16, true);
        let up_bits = (0..N * K / 32)
            .map(|index| {
                let shift = u32::try_from((index * 13 + 17).rem_euclid(31)).unwrap();
                0x5A36_C90F_u32.rotate_left(shift)
            })
            .collect::<Vec<_>>();
        let up_scale_values = (0..N * K / GROUP_SIZE)
            .map(|index| ((index % 11) as f32).mul_add(0.023_437_5, 0.093_75))
            .collect::<Vec<_>>();
        let up_weight = Array::from_slice(&up_bits, &[N, K / 32]);
        let up_scales = Array::from_slice(&up_scale_values, &[N, K / GROUP_SIZE])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let gate = BonsaiQ1Row4::from_row_major(&gate_weight, &gate_scales).unwrap();
        let up = BonsaiQ1Row4::from_row_major(&up_weight, &up_scales).unwrap();

        for m in M_VALUES {
            let input = patterned_input(m, K, Dtype::Bfloat16)
                .reshape(&[1, m, K])
                .unwrap();
            eval([&input]).unwrap();

            for iteration in 0..warmup {
                if iteration % 2 == 0 {
                    let _ = elapsed_separate(&input, &gate, &up);
                    let _ = elapsed_fused(&input, &gate, &up);
                } else {
                    let _ = elapsed_fused(&input, &gate, &up);
                    let _ = elapsed_separate(&input, &gate, &up);
                }
            }

            let mut separate_us = Vec::with_capacity(samples);
            let mut fused_us = Vec::with_capacity(samples);
            for sample in 0..samples {
                if sample % 2 == 0 {
                    separate_us.push(elapsed_separate(&input, &gate, &up));
                    fused_us.push(elapsed_fused(&input, &gate, &up));
                } else {
                    fused_us.push(elapsed_fused(&input, &gate, &up));
                    separate_us.push(elapsed_separate(&input, &gate, &up));
                }
            }

            let separate_median = median(&separate_us);
            let fused_median = median(&fused_us);
            let separate_mean = mean(&separate_us);
            let fused_mean = mean(&fused_us);
            println!(
                "Bonsai-27B TG-LUT4 gate/up BF16 M={m} N={N} K={K} samples={samples}: \
                 separate median={separate_median:.2}us mean={separate_mean:.2}us; \
                 fused median={fused_median:.2}us mean={fused_mean:.2}us; \
                 speedup median={:.4}x mean={:.4}x",
                separate_median / fused_median,
                separate_mean / fused_mean,
            );
        }
    }

    /// Compare the stock guarded QMV curve with the opt-in aligned-N
    /// specialization on Bonsai-27B's dominant gate/up projection shape.
    ///
    /// ```bash
    /// cargo test -p higgs-models --release -- \
    ///   bench_aligned_fast_qmv_m_sweep --ignored --nocapture --exact
    /// ```
    #[test]
    #[ignore = "microbenchmark, requires Apple Metal GPU"]
    fn bench_aligned_fast_qmv_m_sweep() {
        use std::time::Instant;

        const N: i32 = 17_408;
        const K: i32 = 5_120;
        const M_VALUES: [i32; 6] = [1, 2, 3, 4, 5, 8];
        const WARMUP_ITERS: usize = 8;
        const DEFAULT_SAMPLES: usize = 51;

        let _exec = crate::mlx_exec::acquire();
        assert!(fast_qmv_has_aligned_rows(N, fast_qmv_nsg(), true));
        let (weight, scales, biases) = patterned_weights(N, K, Dtype::Bfloat16, true);
        let inputs = M_VALUES
            .iter()
            .map(|&m| (m, patterned_input(m, K, Dtype::Bfloat16)))
            .collect::<Vec<_>>();
        let mut resident = vec![&weight, &scales, &biases];
        resident.extend(inputs.iter().map(|(_, input)| input));
        eval(resident).unwrap();

        let samples = std::env::var("HIGGS_BONSAI_ALIGNED_QMV_BENCH_SAMPLES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(DEFAULT_SAMPLES);

        let measure = |input: &Array, prefer_aligned: bool| -> f64 {
            let start = Instant::now();
            let output = bonsai_q1_qmv_fast_impl(
                input,
                &weight,
                &scales,
                &biases,
                GROUP_SIZE,
                prefer_aligned,
            )
            .unwrap();
            eval([&output]).unwrap();
            let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
            std::hint::black_box(output);
            elapsed_us
        };

        let summarize = |values: &mut [f64]| -> (f64, f64) {
            values.sort_by(f64::total_cmp);
            let median = values[values.len() / 2];
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (median, mean)
        };

        let mut rows = Vec::with_capacity(M_VALUES.len());
        for (m, input) in &inputs {
            let guarded_check =
                bonsai_q1_qmv_fast_impl(input, &weight, &scales, &biases, GROUP_SIZE, false)
                    .unwrap();
            let aligned_check =
                bonsai_q1_qmv_fast_impl(input, &weight, &scales, &biases, GROUP_SIZE, true)
                    .unwrap();
            assert_array_exact(
                &format!("aligned-N benchmark M={m}"),
                &aligned_check,
                &guarded_check,
            );

            for iteration in 0..WARMUP_ITERS {
                if iteration % 2 == 0 {
                    std::hint::black_box(measure(input, false));
                    std::hint::black_box(measure(input, true));
                } else {
                    std::hint::black_box(measure(input, true));
                    std::hint::black_box(measure(input, false));
                }
            }

            let mut guarded_us = Vec::with_capacity(samples);
            let mut aligned_us = Vec::with_capacity(samples);
            for sample in 0..samples {
                if sample % 2 == 0 {
                    guarded_us.push(measure(input, false));
                    aligned_us.push(measure(input, true));
                } else {
                    aligned_us.push(measure(input, true));
                    guarded_us.push(measure(input, false));
                }
            }
            let (guarded_median, guarded_mean) = summarize(&mut guarded_us);
            let (aligned_median, aligned_mean) = summarize(&mut aligned_us);
            rows.push((
                *m,
                guarded_median,
                guarded_mean,
                aligned_median,
                aligned_mean,
            ));
        }

        let (_, guarded_m1_median, guarded_m1_mean, aligned_m1_median, aligned_m1_mean) = rows[0];
        println!("Bonsai Q1 BF16 N={N} K={K}, samples={samples}");
        println!(
            " M | OFF median  mean  med/M1 mean/M1 | ON median  mean  med/M1 mean/M1 | ON speedup"
        );
        for (m, guarded_median, guarded_mean, aligned_median, aligned_mean) in rows {
            println!(
                "{m:>2} | {guarded_median:>9.1} {guarded_mean:>7.1} {guarded_median_norm:>6.2}x {guarded_mean_norm:>7.2}x | \
                 {aligned_median:>9.1} {aligned_mean:>7.1} {aligned_median_norm:>6.2}x {aligned_mean_norm:>7.2}x | \
                 {speedup:>8.3}x",
                guarded_median_norm = guarded_median / guarded_m1_median,
                guarded_mean_norm = guarded_mean / guarded_m1_mean,
                aligned_median_norm = aligned_median / aligned_m1_median,
                aligned_mean_norm = aligned_mean / aligned_m1_mean,
                speedup = guarded_median / aligned_median,
            );
        }
    }

    // CPU reference for the affine 1-bit contract: per group of `GroupSize`,
    //   acc += scale * sum(bit * x) + bias * sum(x),
    // with `bias = -scale / 2` when biases are empty (symmetric). Scales and
    // x are read back at their stored dtype so the reference matches what the
    // kernel observes after its `float(...)` widening.
    fn cpu_wide_qmm_reference(
        weight: &Array,
        scales: &Array,
        biases: &Array,
        x: &Array,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
    ) -> Vec<f32> {
        let k_packed = (k / 32) as usize;
        let num_groups = (k / group_size) as usize;
        let group_size = group_size as usize;
        let k = k as usize;
        let m = m as usize;
        let n = n as usize;

        let w = weight.as_slice::<u32>().to_vec();
        let sc_f32 = scales.as_dtype(Dtype::Float32).unwrap();
        eval([&sc_f32]).unwrap();
        let sc = sc_f32.as_slice::<f32>().to_vec();
        let symmetric = biases.size() == 0;
        let bi = if symmetric {
            Vec::new()
        } else {
            let bi_f32 = biases.as_dtype(Dtype::Float32).unwrap();
            eval([&bi_f32]).unwrap();
            bi_f32.as_slice::<f32>().to_vec()
        };
        let x_f32_arr = x.as_dtype(Dtype::Float32).unwrap();
        eval([&x_f32_arr]).unwrap();
        let x_f32 = x_f32_arr.as_slice::<f32>().to_vec();

        let mut y = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for g in 0..num_groups {
                    let mut dot = 0.0f32;
                    let mut sum_x = 0.0f32;
                    for kk in 0..group_size {
                        let k_abs = g * group_size + kk;
                        let word = w[ni * k_packed + k_abs / 32];
                        let bit = ((word >> (k_abs % 32)) & 1) as f32;
                        let x_val = x_f32[mi * k + k_abs];
                        dot += bit * x_val;
                        sum_x += x_val;
                    }
                    let s_val = sc[ni * num_groups + g];
                    let b_val = if symmetric {
                        -0.5 * s_val
                    } else {
                        bi[ni * num_groups + g]
                    };
                    acc += s_val * dot + b_val * sum_x;
                }
                y[mi * n + ni] = acc;
            }
        }
        y
    }

    fn assert_wide_qmm_matches_reference(
        label: &str,
        got: &Array,
        reference: &[f32],
        m: i32,
        n: i32,
    ) {
        let got_f32 = got.as_dtype(Dtype::Float32).unwrap();
        eval([&got_f32]).unwrap();
        let got_slice = got_f32.as_slice::<f32>();
        assert_eq!(
            got_slice.len(),
            (m * n) as usize,
            "{label}: output length {} expected {}",
            got_slice.len(),
            m * n
        );
        let mut worst = 0.0f32;
        for (index, (g, w)) in got_slice.iter().zip(reference.iter()).enumerate() {
            let tol = 1e-2 * w.abs().max(1.0);
            let err = (g - w).abs();
            worst = worst.max(err / w.abs().max(1.0).max(1e-6));
            assert!(
                err <= tol,
                "{label}[{index}] (m={},n={}): got {g} want {w} err {err} > tol {tol}",
                index / n as usize,
                index % n as usize
            );
        }
        println!(
            "{label}: worst relative error = {worst:.4e} over {} cells",
            m * n
        );
    }

    /// The wide packed-Q1 QMM must reproduce the affine contract for canonical
    /// `[N, K/32]` weights, across M values that exercise a single tile, an
    /// m-tail, and the asymmetric-bias path. Bit-comparable to the CPU oracle
    /// within the same tolerance the narrow QMV kernel uses.
    #[test]
    #[allow(unsafe_code)]
    fn wide_qmm_canonical_matches_cpu_reference() {
        unsafe {
            std::env::set_var("HIGGS_BONSAI_QMM_WIDE", "1");
        }
        let _exec = crate::mlx_exec::acquire();

        // (N, K, M, symmetric): N=64 fills BN=8 tiles exactly; N=50 exercises
        // the N%BN != 0 tail; K=256 spans two groups; K=512 spans four.
        for &(n, k, m, symmetric) in &[
            (64_i32, 256_i32, 16_i32, true),
            (64, 256, 600, true),
            (50, 512, 300, false),
            (64, 512, 1024, true),
        ] {
            let (weight, scales, biases) = patterned_weights(n, k, Dtype::Bfloat16, symmetric);
            let x = patterned_input(m, k, Dtype::Bfloat16);

            let got = bonsai_q1_wide_qmm(&x, &weight, &scales, &biases, GROUP_SIZE)
                .unwrap()
                .unwrap_or_else(|| panic!("wide QMM did not engage for n={n} k={k} m={m}"));
            eval([&got]).unwrap();

            let reference =
                cpu_wide_qmm_reference(&weight, &scales, &biases, &x, m, n, k, GROUP_SIZE);
            assert_wide_qmm_matches_reference(
                &format!("canonical wide QMM (n={n},k={k},m={m},sym={symmetric})"),
                &got,
                &reference,
                m,
                n,
            );
        }
    }

    /// The row4 wide QMM must match the same affine contract while reading the
    /// authoritative primary row4 buffers. Covers N divisible by 4 (the row4
    /// promotion domain) and m-tail tiling.
    #[test]
    #[allow(unsafe_code)]
    fn wide_qmm_row4_matches_cpu_reference() {
        unsafe {
            std::env::set_var("HIGGS_BONSAI_QMM_WIDE", "1");
        }
        let _exec = crate::mlx_exec::acquire();

        for &(n, k, m) in &[(64_i32, 256_i32, 16_i32), (48, 256, 300), (64, 512, 600)] {
            let (weight, scales, _biases) = patterned_weights(n, k, Dtype::Bfloat16, true);
            let packed = BonsaiQ1Row4::from_row_major(&weight, &scales).unwrap();
            eval([&packed.weights, &packed.scales]).unwrap();
            let x = patterned_input(m, k, Dtype::Bfloat16);

            let got = bonsai_q1_row4_wide_qmm_view(&x, packed.as_ref())
                .unwrap()
                .unwrap_or_else(|| panic!("row4 wide QMM did not engage for n={n} k={k} m={m}"));
            eval([&got]).unwrap();

            // The reference uses the canonical packed arrays that produced the
            // row4 transform; row4 promotion preserves bits and scales exactly.
            let reference =
                cpu_wide_qmm_reference(&weight, &scales, &_biases, &x, m, n, k, GROUP_SIZE);
            assert_wide_qmm_matches_reference(
                &format!("row4 wide QMM (n={n},k={k},m={m})"),
                &got,
                &reference,
                m,
                n,
            );
        }
    }
}
