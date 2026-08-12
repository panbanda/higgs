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

use mlx_rs::{Array, Stream, error::Exception};

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
            false, // ensure_row_contiguous
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
        if (row >= n_param) { continue; }
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
        if (row >= n_param || !in_bounds) { continue; }
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
    if (lid == 0u && row < n_param) {
        y[int(batch) * n_param + row] = OutT(v);
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
            false, // ensure_row_contiguous
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
        let nsg = fast_qmv_nsg();
        let rows_per_tg = nsg * 4;
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
/// bit-exact result; faster tiling. See [`bonsai_q1_qmv`] for dispatch.
#[allow(unsafe_code)]
pub fn bonsai_q1_qmv_fast(
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
    let config = configure_fast_qmv_kernel(out_dtype, n_rows, m_rows, k_dim, group_size, symmetric);

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

static DEQUANT_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();
