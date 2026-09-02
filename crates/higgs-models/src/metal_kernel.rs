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

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, error::Exception};

use crate::eschamoe::EschaSpec;

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
// Eschamoe trellis tile decode (Phase 2: decode only, no matvec).
//
// One threadgroup decodes one tile of 16 by 16. Thread t of 128 gives the
// codes 2t and 2t + 1. The bit math copies `eschamoe::unpack_tile`. The hash
// copies `eschamoe::decode_code`. The element order comes from the closed
// form of `eschamoe::tile_perm`. The MSL keeps the closed form because the
// form is eight lines and needs no table input.
// ---------------------------------------------------------------------------

// Phase 3 connects this kernel to the forward path. Until then, only the
// tests use this chain. The allow attributes keep the lib target quiet.
#[allow(dead_code)]
const ESCHA_TILE_KERNEL_SOURCE: &str = r"
// The packed tile holds 8 * K words of 32 bits.
constexpr int WORDS = 8 * K;

threadgroup uint w_sh[WORDS];

uint tn = threadgroup_position_in_grid.y;
uint tk = threadgroup_position_in_grid.z;
uint t = thread_index_in_threadgroup;

// Stage the tile words. Two 16-bit words make one 32-bit word.
const device short* tile = code + (tk * uint(TN) + tn) * uint(16 * K);
if (t < uint(WORDS)) {
    uint lo = uint(ushort(tile[2 * t]));
    uint hi = uint(ushort(tile[2 * t + 1]));
    w_sh[t] = lo | (hi << 16);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// The bit offsets copy unpack_tile. The wrap term 256 * K comes before
// the term -16. Thus the unsigned value stays 0 or more.
uint b0 = 2u * t * uint(K) + uint(K) + 256u * uint(K) - 16u;
uint b2 = b0 + uint(K) + 16u;
uint j0 = b0 / 32u;
uint j1 = (b2 - 1u) / 32u;
uint s1 = (j1 + 1u) * 32u - b2;

// The 64-bit funnel makes the shift safe when s1 is 0.
ulong pair = (ulong(w_sh[j0 % uint(WORDS)]) << 32) | ulong(w_sh[j1 % uint(WORDS)]);
uint w1 = uint(pair >> s1);

uint pair_codes[2];
pair_codes[0] = (w1 >> uint(K)) & 0xFFFFu;
pair_codes[1] = w1 & 0xFFFFu;

uint row_len = uint(TN) * 16u;
for (uint e = 0u; e < 2u; ++e) {
    uint i = 2u * t + e;

    // The closed form of tile_perm. Stored element i goes to the
    // row-major slot (r, c) inside the tile.
    uint g = i >> 3;
    uint ii = i & 3u;
    uint r = (g % 4u) * 2u + (ii & 1u) + 8u * (ii >> 1);
    uint c = g / 4u + 8u * ((i >> 2) & 1u);

    // The codebook hash. The reduce step adds the two f16 halves. The
    // four constants come from the cb input: multiply, add, mask, XOR.
    uint x = pair_codes[e] * cb[0] + cb[1];
    x = (x & cb[2]) ^ cb[3];
    half2 h = as_type<half2>(x);
    float v = float(h.x) + float(h.y);

    dst[(tk * 16u + r) * row_len + tn * 16u + c] = half(v);
}
";

#[allow(unsafe_code, dead_code)]
fn create_escha_tile_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"code", c"cb"]);
    let out_vec = cstr_vec(&[c"dst"]);
    let source = CString::new(ESCHA_TILE_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_eschamoe_dequant_tiles".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // The tile pointer math needs a row-contiguous input.
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(dead_code)]
static ESCHA_TILE_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Decode the packed trellis tiles of one expert on the GPU.
///
/// The input is the `[tiles_k, tiles_n, 16 * K]` int16 code slice of one
/// expert. The result is the `[in, out]` f16 matrix `Ŵ` in row-major order.
/// The values equal the CPU decode in [`crate::eschamoe`]. This function
/// does not apply the Hadamard or the channel scales.
#[allow(unsafe_code, dead_code)]
pub fn eschamoe_dequant_tiles(code: &Array, spec: &EschaSpec) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let [mul, add, mask, xor] = crate::eschamoe::gpu_codebook(spec).ok_or_else(|| {
        Exception::custom("eschamoe_dequant_tiles: the codebook has no verified GPU decode")
    })?;
    if code.dtype() != Dtype::Int16 {
        return Err(Exception::custom(format!(
            "eschamoe_dequant_tiles: code dtype {:?} is not int16",
            code.dtype()
        )));
    }
    let k = i32::try_from(spec.k).unwrap_or(0);
    if !(1..=8).contains(&k) {
        return Err(Exception::custom(format!(
            "eschamoe_dequant_tiles: K={k} out of range 1..=8"
        )));
    }
    let (tiles_k, tiles_n) = spec.tiles();
    let expected = [
        i32::try_from(tiles_k).unwrap_or(i32::MAX),
        i32::try_from(tiles_n).unwrap_or(i32::MAX),
        i32::try_from(spec.words_per_tile()).unwrap_or(i32::MAX),
    ];
    if code.shape() != expected {
        return Err(Exception::custom(format!(
            "eschamoe_dequant_tiles: code shape {:?} does not match {expected:?}",
            code.shape()
        )));
    }

    let stream = Stream::task_local_or_default();
    let cached = ESCHA_TILE_KERNEL.get_or_init(|| CachedMetalKernel(create_escha_tile_kernel()));
    let out_shape = [spec.in_features, spec.out_features];
    // A template value goes into the generated MSL function name. A negative
    // value makes that name invalid. Thus the codebook constants travel in a
    // small uint32 input, and only K and TN are template values.
    let cb_arr = Array::from_slice(&[mul, add, mask, xor], &[4]);

    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"TN".as_ptr(),
            expected[1],
        );
        // One threadgroup of 128 threads decodes one tile. The grid gives
        // the total thread count on each axis.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 128, expected[1], expected[0]);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT16,
        );

        let input_ptrs = [code.as_ptr(), cb_arr.as_ptr()];
        let inputs_vec = mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len());
        let mut outputs_vec = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        );

        let result = if status == 0 {
            let mut out_ptr = mlx_sys::mlx_array_new();
            let get_status = mlx_sys::mlx_vector_array_get(&raw mut out_ptr, outputs_vec, 0);
            if get_status == 0 {
                Ok(Array::from_ptr(out_ptr))
            } else {
                mlx_sys::mlx_array_free(out_ptr);
                Err(Exception::custom(format!(
                    "eschamoe_dequant_tiles: output read failed: {}",
                    take_last_error()
                )))
            }
        } else {
            Err(Exception::custom(format!(
                "eschamoe_dequant_tiles failed: {}",
                take_last_error()
            )))
        };

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        result
    }
}

const ESCHA_QMV_KERNEL_SOURCE: &str = r"
// One simdgroup computes one output element for one selected expert.
constexpr int WORDS = 8 * K;
constexpr int IN = TK * 16;
constexpr int OUT = TN * 16;

threadgroup float x_sh[TK * 16];

uint row = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

// Stage the transformed activation row of this expert.
for (uint i = tid; i < uint(IN); i += 128u) {
    x_sh[i] = xh[row * uint(IN) + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint o = threadgroup_position_in_grid.x * 4u + sg;
if (o >= uint(OUT)) {
    return;
}

// Split the output index into a tile column and a slot column.
uint tn = o >> 4;
uint c = o & 15u;
uint cb2 = (c >> 3) & 1u;
uint c7 = c & 7u;

const device short* base =
    code + ulong(eids[row]) * ulong(TK) * ulong(TN) * ulong(16 * K);

// Each lane owns one code pair. The pair index inverts the closed form
// of tile_perm for a fixed column. One pair gives two adjacent rows.
uint q = lane & 3u;
uint rh = (lane >> 2) & 1u;
uint t = 4u * (4u * c7 + q) + 2u * cb2 + rh;
uint r0 = 8u * rh + 2u * q;

// The bit offsets copy unpack_tile. The wrap term 256 * K comes before
// the term -16. Thus the unsigned value stays 0 or more.
uint b0 = 2u * t * uint(K) + uint(K) + 256u * uint(K) - 16u;
uint b2 = b0 + uint(K) + 16u;
uint i0 = (b0 / 32u) % uint(WORDS);
uint i1w = (b2 - 1u) / 32u;
uint s1 = (i1w + 1u) * 32u - b2;
uint i1 = i1w % uint(WORDS);

float acc = 0.0f;
for (uint tk = lane >> 3; tk < uint(TK); tk += 4u) {
    const device short* tile = base + (tk * uint(TN) + tn) * uint(16 * K);
    uint w0 = uint(ushort(tile[2u * i0])) | (uint(ushort(tile[2u * i0 + 1u])) << 16);
    uint wb = uint(ushort(tile[2u * i1])) | (uint(ushort(tile[2u * i1 + 1u])) << 16);

    // The 64-bit funnel makes the shift safe when s1 is 0.
    ulong pair = (ulong(w0) << 32) | ulong(wb);
    uint w1 = uint(pair >> s1);

    // The codebook hash. Refer to the tile decode kernel. The half cast
    // repeats the f16 round of the CPU decode.
    uint x0 = ((w1 >> uint(K)) & 0xFFFFu) * cb[0] + cb[1];
    x0 = (x0 & cb[2]) ^ cb[3];
    uint x1 = (w1 & 0xFFFFu) * cb[0] + cb[1];
    x1 = (x1 & cb[2]) ^ cb[3];
    half2 h0 = as_type<half2>(x0);
    half2 h1 = as_type<half2>(x1);
    float v0 = float(half(float(h0.x) + float(h0.y)));
    float v1 = float(half(float(h1.x) + float(h1.y)));

    acc = fma(x_sh[tk * 16u + r0], v0, acc);
    acc = fma(x_sh[tk * 16u + r0 + 1u], v1, acc);
}

acc = simd_sum(acc);
if (lane == 0u) {
    dst[row * uint(OUT) + o] = acc;
}
";

#[allow(unsafe_code, dead_code)]
fn create_escha_qmv_kernel() -> mlx_sys::mlx_fast_metal_kernel {
    let in_vec = cstr_vec(&[c"xh", c"code", c"eids", c"cb"]);
    let out_vec = cstr_vec(&[c"dst"]);
    let source = CString::new(ESCHA_QMV_KERNEL_SOURCE).unwrap_or_default();
    unsafe {
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            c"higgs_eschamoe_gather_qmv".as_ptr(),
            in_vec,
            out_vec,
            source.as_ptr(),
            c"".as_ptr(),
            true,  // The tile pointer math needs row-contiguous inputs.
            false, // atomic_outputs
        );
        mlx_sys::mlx_vector_string_free(in_vec);
        mlx_sys::mlx_vector_string_free(out_vec);
        kernel
    }
}

#[allow(dead_code)]
static ESCHA_QMV_KERNEL: OnceLock<CachedMetalKernel> = OnceLock::new();

/// Check the gather inputs. Give the row count and the tile dims.
#[allow(dead_code)]
fn check_gather_inputs(
    xh: &Array,
    code: &Array,
    expert_ids: &Array,
    spec: &EschaSpec,
) -> Result<(i32, [i32; 3]), Exception> {
    let (tiles_k, tiles_n) = spec.tiles();
    let tile_dims = [
        i32::try_from(tiles_k).unwrap_or(i32::MAX),
        i32::try_from(tiles_n).unwrap_or(i32::MAX),
        i32::try_from(spec.words_per_tile()).unwrap_or(i32::MAX),
    ];
    // Bind the expert axis: the kernels index `code` by expert id with no
    // bound check, so a short checkpoint would read out of bounds.
    let code_ok =
        matches!(code.shape(), &[e, a, b, c] if e == spec.num_experts && [a, b, c] == tile_dims);
    if code.dtype() != Dtype::Int16 || !code_ok {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: code {:?} {:?} does not match [{}, {tile_dims:?}] int16",
            code.dtype(),
            code.shape(),
            spec.num_experts
        )));
    }
    let &[rows, cols] = xh.shape() else {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: xh has shape {:?}, not two dims",
            xh.shape()
        )));
    };
    if xh.dtype() != Dtype::Float32 || cols != spec.in_features {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: xh {:?} {:?} does not match [rows, {}] float32",
            xh.dtype(),
            xh.shape(),
            spec.in_features
        )));
    }
    if expert_ids.dtype() != Dtype::Uint32 || expert_ids.shape() != [rows] {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: expert_ids {:?} {:?} does not match [{rows}] uint32",
            expert_ids.dtype(),
            expert_ids.shape()
        )));
    }
    Ok((rows, tile_dims))
}

/// Multiply activation rows with selected expert trellis weights.
///
/// The input `xh` is the `[rows, in]` float32 matrix. Each row already
/// carries the input scales and the input Hadamard of its expert. The
/// input `code` is the full `[experts, tiles_k, tiles_n, 16 * K]` int16
/// trellis tensor. The input `expert_ids` maps each row to one expert.
/// The kernel stages the whole activation row in `x_sh[TK * 16]`, which is
/// static threadgroup memory: Apple GPUs cap it at 32 KiB, so past 8192
/// floats the kernel stops compiling with an error that names nothing.
const MAX_THREADGROUP_FLOATS: i32 = 32 * 1024 / 4;

/// The result is `y_pre = xh @ Ŵ` as a `[rows, out]` float32 matrix. The
/// caller applies the output Hadamard and the output scales.
#[allow(unsafe_code, dead_code)]
pub fn eschamoe_gather_qmv(
    xh: &Array,
    code: &Array,
    expert_ids: &Array,
    spec: &EschaSpec,
) -> Result<Array, Exception> {
    ensure_ffi_error_handler();

    let [mul, add, mask, xor] = crate::eschamoe::gpu_codebook(spec).ok_or_else(|| {
        Exception::custom("eschamoe_gather_qmv: the codebook has no verified GPU decode")
    })?;
    let k = i32::try_from(spec.k).unwrap_or(0);
    if !(1..=8).contains(&k) {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: K={k} out of range 1..=8"
        )));
    }
    let (rows, tile_dims) = check_gather_inputs(xh, code, expert_ids, spec)?;
    if spec.in_features > MAX_THREADGROUP_FLOATS {
        return Err(Exception::custom(format!(
            "eschamoe_gather_qmv: in_features {} exceeds the {}-float threadgroup staging limit",
            spec.in_features, MAX_THREADGROUP_FLOATS
        )));
    }

    let stream = Stream::task_local_or_default();
    let cached = ESCHA_QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_escha_qmv_kernel()));
    let out_shape = [rows, spec.out_features];
    // The codebook constants travel in a small uint32 input. Refer to
    // the note on template values in eschamoe_dequant_tiles.
    let cb_arr = Array::from_slice(&[mul, add, mask, xor], &[4]);

    unsafe {
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c"K".as_ptr(), k);
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"TK".as_ptr(),
            tile_dims[0],
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"TN".as_ptr(),
            tile_dims[1],
        );
        // One threadgroup holds 4 simdgroups. Each simdgroup computes one
        // output element. The grid gives total thread counts.
        let groups_x = (spec.out_features + 3) / 4;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, groups_x * 128, rows, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            mlx_sys::mlx_dtype__MLX_FLOAT32,
        );

        let input_ptrs = [
            xh.as_ptr(),
            code.as_ptr(),
            expert_ids.as_ptr(),
            cb_arr.as_ptr(),
        ];
        let inputs_vec = mlx_sys::mlx_vector_array_new_data(input_ptrs.as_ptr(), input_ptrs.len());
        let mut outputs_vec = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_fast_metal_kernel_apply(
            &raw mut outputs_vec,
            cached.0,
            inputs_vec,
            config,
            stream.as_ptr(),
        );

        let result = if status == 0 {
            let mut out_ptr = mlx_sys::mlx_array_new();
            let get_status = mlx_sys::mlx_vector_array_get(&raw mut out_ptr, outputs_vec, 0);
            if get_status == 0 {
                Ok(Array::from_ptr(out_ptr))
            } else {
                mlx_sys::mlx_array_free(out_ptr);
                Err(Exception::custom(format!(
                    "eschamoe_gather_qmv: output read failed: {}",
                    take_last_error()
                )))
            }
        } else {
            Err(Exception::custom(format!(
                "eschamoe_gather_qmv failed: {}",
                take_last_error()
            )))
        };

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_vector_array_free(outputs_vec);
        result
    }
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
            float b_val = float(bi[row * NumGroups + g]);
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
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::thread_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_qmv_kernel()));
    let config = configure_qmv_kernel(out_dtype, n_rows, k_dim, group_size);

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
// `qmv_fast`-class 1-bit matvec (decode hot path).
//
// Ports MLX/PrismML `qmv_fast` tiling onto our uint32 packing: each simdgroup
// computes RESULTS_PER_SIMDGROUP (4) output rows; each of its 32 lanes holds
// VPT (64) input values in registers (no threadgroup memory, no barriers) and
// reuses them across all 4 rows. block_size = 64 * 32 = 2048. The bits=1 affine
// math is identical to the legacy kernel — `scale * sum(bit*x) + bias * sum(x)`
// — only the data movement differs. Group scales/biases are per-lane (a lane's
// 64 values lie in one 128-wide group); per-row partials are simd_sum-reduced.
// ---------------------------------------------------------------------------

const FAST_QMV_KERNEL_SOURCE: &str = r"
constexpr int VPT = 64;          // values_per_thread
constexpr int RPS = 4;           // results_per_simdgroup
constexpr int WPT = VPT / 32;    // packed uint32 words per thread (2)
constexpr int BLK = VPT * 32;    // block_size = 2048

uint tgx = threadgroup_position_in_grid.x;
uint sg  = simdgroup_index_in_threadgroup;
uint lid = thread_index_in_simdgroup;
uint nsg = simdgroups_per_threadgroup;

int out_row = int(tgx) * (int(nsg) * RPS) + int(sg) * RPS;

float xt[VPT];
float result[RPS];
for (int r = 0; r < RPS; ++r) { result[r] = 0.0f; }

int aligned_end = (K / BLK) * BLK;

// Main loop: full 2048-element blocks (covers every real Bonsai layer, since
// all K are multiples of 2048).
for (int k = 0; k < aligned_end; k += BLK) {
    int xbase = k + int(lid) * VPT;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) { float v = float(x[xbase + i]); xt[i] = v; sum += v; }

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
        float b_val = float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

// Tail: only exercised by tests with K < 2048 or K % 2048 != 0.
if (aligned_end < K) {
    int xbase = aligned_end + int(lid) * VPT;
    bool in_bounds = xbase < K;
    float sum = 0.0f;
    for (int i = 0; i < VPT; ++i) {
        float v = (in_bounds && (xbase + i) < K) ? float(x[xbase + i]) : 0.0f;
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
        float b_val = float(bi[row * NumGroups + g]);
        result[r] += s_val * accum + b_val * sum;
    }
}

for (int r = 0; r < RPS; ++r) {
    int row = out_row + r;
    float v = simd_sum(result[r]);
    if (lid == 0u && row < n_param) {
        y[row] = OutT(v);
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
            k_dim / 32,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config,
            c"NumGroups".as_ptr(),
            k_dim / group_size,
        );

        // Each simdgroup computes 4 rows; nsg simdgroups per threadgroup.
        let nsg = fast_qmv_nsg();
        let rows_per_tg = nsg * 4;
        let n_tgs = (n_rows + rows_per_tg - 1) / rows_per_tg;
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

    let x_flat = x.reshape(&[k_dim])?;
    let w_flat = weight.reshape(&[-1])?;
    let s_flat = scales.flatten(None, None)?;
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::thread_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(x.as_ptr()) };

    let cached = FAST_QMV_KERNEL.get_or_init(|| CachedMetalKernel(create_fast_qmv_kernel()));
    let config = configure_fast_qmv_kernel(out_dtype, n_rows, k_dim, group_size);

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
float b_val = float(bi[n * uint(NumGroups) + uint(g)]);

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
    let b_flat = biases.flatten(None, None)?;

    let stream = Stream::thread_local_or_default();
    let out_dtype = unsafe { mlx_sys::mlx_array_dtype(scales.as_ptr()) };

    let cached = DEQUANT_KERNEL.get_or_init(|| CachedMetalKernel(create_dequant_kernel()));
    let config = configure_dequant_kernel(out_dtype, n_rows, k_dim, group_size);

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
