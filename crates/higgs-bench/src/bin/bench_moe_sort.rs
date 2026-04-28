#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::useless_vec,
    clippy::too_many_arguments,
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::doc_markdown,
    unsafe_code
)]
//! `bench_moe_sort` -- port of `bench_moe_sort.py`.
//!
//! Times MoE dispatch via `gather_qmm` under three schedules:
//!   1. Higgs no-sort (per-token indices, `sorted_indices=false`)
//!   2. Higgs per-token-sort (sort along `top_k` axis)
//!   3. mlx-lm global-sort (flatten + argsort across all tokens)
//!
//! All three call the same `gather_qmm` MLX kernel; the difference is what
//! the indices look like. Output: median wall-clock per call across
//! `--iters` iterations after `--warmup` warmups.
//!
//! The MLX `gather_qmm` symbol is not exposed publicly from `higgs-models`,
//! so this binary reaches into `mlx-sys` directly for an FFI wrapper. Same
//! pattern as `crates/higgs-models/src/qwen3_next.rs`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::stats::median;
use higgs_bench::{
    BenchOutput, OutputFormat, RunMetadata, format_json, format_markdown, persist_result,
};
use mlx_rs::{Array, Stream, error::Exception, nn, ops, random, transforms::eval};
use serde::Serialize;

// DeepSeek-V2-Lite MoE dimensions (match Python).
const HIDDEN: i32 = 2048;
const INTERMEDIATE: i32 = 1408;
const NUM_EXPERTS: i32 = 64;
const TOP_K: i32 = 6;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

const DEFAULT_SEQ_LENS: &[i32] = &[1, 32, 128, 512, 1024, 2048];
const SORT_OVERHEAD_LENS: &[i32] = &[128, 512, 2048];

#[derive(Debug, Parser)]
#[command(
    name = "bench_moe_sort",
    about = "Time MoE dispatch (no-sort vs per-token-sort vs global-sort)",
    version
)]
struct Args {
    /// Sequence lengths to sweep (comma-separated).
    #[arg(long, value_delimiter = ',')]
    seq_lens: Option<Vec<i32>>,

    /// Warmup iterations per measurement.
    #[arg(long, default_value_t = 3)]
    warmup: u32,

    /// Timed iterations per measurement.
    #[arg(long, default_value_t = 10)]
    iters: u32,

    /// Output format (json, markdown).
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    seq_lens: Vec<i32>,
    warmup: u32,
    iters: u32,
    hidden: i32,
    intermediate: i32,
    num_experts: i32,
    top_k: i32,
    group_size: i32,
    bits: i32,
}

#[derive(Debug, Serialize)]
struct PerLength {
    seq_len: i32,
    nosort_ms: f64,
    ptsort_ms: f64,
    global_ms: f64,
    fastest_higgs_speedup_over_global: f64,
}

#[derive(Debug, Serialize)]
struct SortOverhead {
    seq_len: i32,
    global_sort_ms: f64,
}

#[derive(Debug, Serialize)]
struct Results {
    per_length: Vec<PerLength>,
    sort_overhead: Vec<SortOverhead>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(args: Args) -> Result<()> {
    let mut metadata = RunMetadata::capture("bench_moe_sort");
    let started = Instant::now();

    let seq_lens = args
        .seq_lens
        .clone()
        .unwrap_or_else(|| DEFAULT_SEQ_LENS.to_vec());

    eprintln!("DeepSeek-V2-Lite MoE dispatch profiling");
    eprintln!(
        "  {NUM_EXPERTS} experts, top_k={TOP_K}, {BITS}-bit, hidden={HIDDEN}, intermediate={INTERMEDIATE}"
    );
    eprintln!(
        "  {} warmup, {} iters, reporting median",
        args.warmup, args.iters
    );
    eprintln!();

    let weights = MoeWeights::new()?;

    eprintln!(
        "{:>8} | {:>14} | {:>14} | {:>14} | {:>10}",
        "SeqLen", "Higgs(nosort)", "Higgs(ptsort)", "mlx-lm(global)", "Speedup"
    );
    eprintln!("{}", "-".repeat(75));

    let mut per_length = Vec::new();
    for &l in &seq_lens {
        let (x, indices) = make_inputs(1, l)?;

        let t_nosort = bench(args.warmup, args.iters, || {
            let r = forward_gather_higgs(&x, &indices, &weights, false)?;
            eval([&r])?;
            Ok(())
        })?;
        let t_ptsort = bench(args.warmup, args.iters, || {
            let r = forward_gather_higgs(&x, &indices, &weights, true)?;
            eval([&r])?;
            Ok(())
        })?;
        let t_global = bench(args.warmup, args.iters, || {
            let r = forward_gather_mlxlm(&x, &indices, &weights)?;
            eval([&r])?;
            Ok(())
        })?;

        let fastest_higgs = t_nosort.min(t_ptsort);
        let speedup = if t_global > 0.0 {
            fastest_higgs / t_global
        } else {
            f64::INFINITY
        };

        eprintln!(
            "{:>8} | {:>11.2} ms | {:>11.2} ms | {:>11.2} ms | {:>8.2}x",
            l,
            t_nosort * 1000.0,
            t_ptsort * 1000.0,
            t_global * 1000.0,
            speedup
        );

        per_length.push(PerLength {
            seq_len: l,
            nosort_ms: t_nosort * 1000.0,
            ptsort_ms: t_ptsort * 1000.0,
            global_ms: t_global * 1000.0,
            fastest_higgs_speedup_over_global: speedup,
        });
    }

    eprintln!("\n--- Sort + gather overhead ---");
    let mut sort_overhead = Vec::new();
    for &l in SORT_OVERHEAD_LENS {
        let (x, indices) = make_inputs(1, l)?;
        let x_exp = x.reshape(&[1, l, 1, 1, HIDDEN])?;

        let t = bench(args.warmup, args.iters, || {
            let (a, b, c) = gather_sort(&x_exp, &indices)?;
            eval([&a, &b, &c])?;
            Ok(())
        })?;
        eprintln!("  L={l:>5}: global sort = {:.2} ms", t * 1000.0);
        sort_overhead.push(SortOverhead {
            seq_len: l,
            global_sort_ms: t * 1000.0,
        });
    }

    metadata.duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let params = Params {
        seq_lens: seq_lens.clone(),
        warmup: args.warmup,
        iters: args.iters,
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        group_size: GROUP_SIZE,
        bits: BITS,
    };
    let results = Results {
        per_length,
        sort_overhead,
    };
    let output = BenchOutput {
        metadata,
        params,
        results,
    };
    let path = persist_result(&output)?;
    eprintln!("[persisted] {}", path.display());
    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");
    Ok(())
}

struct QuantWeights {
    w: Array,
    scales: Array,
    biases: Array,
}

fn quantize_random(out_dim: i32, in_dim: i32) -> Result<QuantWeights, Exception> {
    let w_full = random::normal::<f32>(&[NUM_EXPERTS, out_dim, in_dim], None, None, None)?;
    let (w, scales, biases) = ops::quantize(&w_full, GROUP_SIZE, BITS)?;
    eval([&w, &scales, &biases])?;
    Ok(QuantWeights { w, scales, biases })
}

struct MoeWeights {
    gate: QuantWeights,
    up: QuantWeights,
    down: QuantWeights,
}

impl MoeWeights {
    fn new() -> Result<Self, Exception> {
        Ok(Self {
            gate: quantize_random(INTERMEDIATE, HIDDEN)?,
            up: quantize_random(INTERMEDIATE, HIDDEN)?,
            down: quantize_random(HIDDEN, INTERMEDIATE)?,
        })
    }
}

fn make_inputs(b: i32, l: i32) -> Result<(Array, Array), Exception> {
    let x = random::normal::<f32>(&[b, l, HIDDEN], None, None, None)?;
    let indices =
        random::randint::<_, i32>(0, NUM_EXPERTS, &[b, l, TOP_K], None)?.as_type::<u32>()?;
    eval([&x, &indices])?;
    Ok((x, indices))
}

/// Higgs path: `gather_qmm` with per-token indices, optionally sorted along the
/// `top_k` axis.
fn forward_gather_higgs(
    x: &Array,
    indices: &Array,
    w: &MoeWeights,
    do_sort: bool,
) -> Result<Array, Exception> {
    let shape = x.shape();
    let b = shape[0];
    let l = shape[1];
    let d = shape[2];

    let indices = if do_sort {
        ops::sort_axis(indices, -1)?
    } else {
        indices.clone()
    };

    let x_exp = x.reshape(&[b, l, 1, 1, d])?;

    let gate_out = gather_qmm(
        &x_exp,
        &w.gate.w,
        &w.gate.scales,
        &w.gate.biases,
        &indices,
        true,
        GROUP_SIZE,
        BITS,
        do_sort,
    )?;
    let up_out = gather_qmm(
        &x_exp,
        &w.up.w,
        &w.up.scales,
        &w.up.biases,
        &indices,
        true,
        GROUP_SIZE,
        BITS,
        do_sort,
    )?;
    let activated = nn::silu(&gate_out)?.multiply(up_out)?;

    let down_out = gather_qmm(
        &activated,
        &w.down.w,
        &w.down.scales,
        &w.down.biases,
        &indices,
        true,
        GROUP_SIZE,
        BITS,
        do_sort,
    )?;
    down_out.squeeze_axes(&[-2])
}

/// mlx-lm path: global flatten + argsort, dispatch with `sorted_indices=true`,
/// then unsort.
fn forward_gather_mlxlm(x: &Array, indices: &Array, w: &MoeWeights) -> Result<Array, Exception> {
    let orig_shape = indices.shape().to_vec();
    let shape = x.shape();
    let b = shape[0];
    let l = shape[1];
    let d = shape[2];
    let x_exp = x.reshape(&[b, l, 1, 1, d])?;

    let (x_sorted, idx_sorted, inv_order) = gather_sort(&x_exp, indices)?;

    let gate_out = gather_qmm(
        &x_sorted,
        &w.gate.w,
        &w.gate.scales,
        &w.gate.biases,
        &idx_sorted,
        true,
        GROUP_SIZE,
        BITS,
        true,
    )?;
    let up_out = gather_qmm(
        &x_sorted,
        &w.up.w,
        &w.up.scales,
        &w.up.biases,
        &idx_sorted,
        true,
        GROUP_SIZE,
        BITS,
        true,
    )?;
    let activated = nn::silu(&gate_out)?.multiply(up_out)?;
    let down_out = gather_qmm(
        &activated,
        &w.down.w,
        &w.down.scales,
        &w.down.biases,
        &idx_sorted,
        true,
        GROUP_SIZE,
        BITS,
        true,
    )?;

    scatter_unsort(&down_out, &inv_order, &orig_shape)?.squeeze_axes(&[-2])
}

/// mlx-lm `_gather_sort`: flatten indices, argsort, reorder `x` and indices.
/// Returns `(x_sorted, idx_sorted, inv_order)`.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let m = *indices
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("indices needs last dim"))?;
    let idx_flat = indices.flatten(None, None)?;
    let order = ops::argsort_axis(&idx_flat, 0)?;
    let inv_order = ops::argsort_axis(&order, 0)?;

    let m_u32 = u32::try_from(m).map_err(|_| Exception::custom("m too large"))?;
    let m_arr = Array::from_slice(&[m_u32], &[1]);
    let token_idx = order.floor_divide(&m_arr)?;

    // x: [..., 1, 1, D] -> reshape last-three-axes-flat: [-1, 1, D]
    let d = *x
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("x needs last dim"))?;
    let x_flat = x.reshape(&[-1, 1, d])?;
    let x_sorted = x_flat.take_axis(&token_idx, 0)?;
    let idx_sorted = idx_flat.take_axis(&order, 0)?;
    Ok((x_sorted, idx_sorted, inv_order))
}

fn scatter_unsort(x: &Array, inv_order: &Array, orig_shape: &[i32]) -> Result<Array, Exception> {
    let unsorted = x.take_axis(inv_order, 0)?;
    // Reshape to orig_shape + [1, D] (keep the M=1 axis from gather_qmm output).
    let d = *unsorted
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("unsorted needs last dim"))?;
    let mut out_shape: Vec<i32> = orig_shape.to_vec();
    out_shape.push(1);
    out_shape.push(d);
    unsorted.reshape(&out_shape)
}

// ---------------------------------------------------------------------------
// gather_qmm FFI wrapper
// ---------------------------------------------------------------------------
//
// `higgs_models::qwen3_next::gather_qmm` is `pub(crate)` so we can't reuse it.
// The wrapper below mirrors that one byte-for-byte; same MLX call.

thread_local! {
    static FFI_LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

unsafe extern "C" fn ffi_error_handler(msg: *const c_char, _data: *mut c_void) {
    let s = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    FFI_LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(s);
    });
}

fn ensure_ffi_error_handler() {
    use std::sync::OnceLock;
    static REGISTERED: OnceLock<()> = OnceLock::new();
    REGISTERED.get_or_init(|| unsafe {
        mlx_sys::mlx_set_error_handler(Some(ffi_error_handler), std::ptr::null_mut(), None);
    });
}

fn gather_qmm(
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
    let mode = CString::new("affine").expect("static");
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
            mode.as_ptr(),
            sorted_indices,
            stream.as_ptr(),
        )
    };
    unsafe { mlx_sys::mlx_array_free(null_lhs) };
    if status != 0 {
        unsafe { mlx_sys::mlx_array_free(result) };
        let mlx_msg = FFI_LAST_ERROR
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_default();
        return Err(Exception::custom(format!("gather_qmm failed: {mlx_msg}")));
    }
    Ok(unsafe { Array::from_ptr(result) })
}

fn bench<F>(warmup: u32, iters: u32, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<(), Exception>,
{
    for _ in 0..warmup {
        f().context("warmup iteration")?;
    }
    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = Instant::now();
        f().context("timed iteration")?;
        times.push(t0.elapsed().as_secs_f64());
    }
    Ok(median(&times))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_weights_construct() {
        let _ = MoeWeights::new().expect("weights build");
    }

    #[test]
    fn forward_paths_have_matching_output_shape() {
        let w = MoeWeights::new().unwrap();
        let (x, indices) = make_inputs(1, 4).unwrap();
        let a = forward_gather_higgs(&x, &indices, &w, false).unwrap();
        let b = forward_gather_higgs(&x, &indices, &w, true).unwrap();
        let c = forward_gather_mlxlm(&x, &indices, &w).unwrap();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.shape(), c.shape());
    }

    #[test]
    fn sort_overhead_runs() {
        let (x, indices) = make_inputs(1, 16).unwrap();
        let x_exp = x.reshape(&[1, 16, 1, 1, HIDDEN]).unwrap();
        let (xs, is, inv) = gather_sort(&x_exp, &indices).unwrap();
        eval([&xs, &is, &inv]).unwrap();
    }

    #[test]
    fn bench_returns_positive() {
        let (x, _) = make_inputs(1, 1).unwrap();
        let t = bench(1, 2, || {
            let _r = x.add(&x)?;
            Ok(())
        })
        .unwrap();
        assert!(t > 0.0);
    }
}
