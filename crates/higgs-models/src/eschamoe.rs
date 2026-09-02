//! `EschaLabs` `eschamoe` weights. The format uses a trellis code.
//!
//! `eschamoe` is the EXL3 format of exllamav3. EXL3 is a variant of QTIP.
//! `eschamoe` applies EXL3 to each expert. The checkpoint holds these tensors
//! for each expert projection:
//!
//! | tensor        | dtype | shape                    |
//! |---------------|-------|--------------------------|
//! | `escha_code`  | I16   | `[E, in/16, out/16, 16K]` |
//! | `escha_rin`   | F16   | `[E, in]`                |
//! | `escha_rout`  | F16   | `[E, out]`               |
//! | `escha_s_in`  | F32   | `[E, in]`                |
//! | `escha_s_out` | F32   | `[E, out]`               |
//! | `escha_config`| I32   | `[9]`                    |
//!
//! A dense checkpoint stores the same tensors without the `E` axis. The code
//! tensor then has rank 3, and each scale vector has rank 1. Refer to
//! [`ExpertAxis`].
//!
//! This equation gives the weight:
//!
//! ```text
//! W[out, in] = (H128 . Ŵ . H128 * rin[:, None] * rout[None, :]).T
//! ```
//!
//! `Ŵ` is the `[in, out]` matrix from the trellis code. The two scale vectors
//! apply outside the Hadamard blocks. The tool `tools/escha_ref.py` shows the
//! test data for this equation.
//!
//! This module gives correct results, but it is not the fast path. The CPU
//! decodes the bits and the codebook. MLX does the Hadamard and the scale
//! operations. The Metal kernel in [`crate::metal_kernel`] uses this module as
//! its reference. The bit format and the tile order come from these MIT files
//! of exllamav3: `exllamav3_ext/quant/pack.cu` and `exl3_lib/quantize.py`.

// This module copies the reference kernels bit for bit. The mask operations
// and the short casts are part of the algorithm. Tile indexes stay in range
// because of the tile size. The tests `tile_perm_is_a_permutation` and
// `decode_expert_codes_*` show this. The modules `turboquant` and `bonsai_q1`
// use the same permission for the same reason.
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]

use half::f16;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, ops};

use crate::error::ModelError;

/// The tile edge as an `i32` for the shape checks.
const TILE_I32: i32 = 16;
/// The number of elements on each edge of a tile.
const TILE: usize = TILE_I32 as usize;
/// The number of elements in one tile.
const TILE_ELEMS: usize = TILE * TILE;
/// The width of one Hadamard block. This is the only supported size. The
/// function [`had_size`] selects the size for each axis.
pub const HAD_BLOCK: i32 = 128;
/// The Hadamard block sizes that this module supports.
const HAD_SIZES: [i32; 1] = [HAD_BLOCK];
/// The permitted distance of the `|rout|` mean from 1.0.
const ROUT_MEAN_TOL: f32 = 0.05;
/// The multiply constant of codebook 1 (`decode_3inst<1>`).
const MCG_MULT: u32 = 0xCBAC_1FED;
/// The instruction `lop3.b32 ..., 0x8fff8fff, 0x3b603b60, 0x6a` is equal to
/// `(x & AND) ^ XOR`.
const LOP3_AND: u32 = 0x8FFF_8FFF;
const LOP3_XOR: u32 = 0x3B60_3B60;

// ---------------------------------------------------------------------------
// Codebooks
// ---------------------------------------------------------------------------

/// The reduce step of a codebook. It changes the 32 hash bits into one value.
///
/// A new codebook can add a variant here. The decode loop reads the variant
/// from one `Codebook` value. Thus the loop stays branch-free.
#[derive(Debug, Clone, Copy)]
enum Reducer {
    /// Add the two `f16` halves of the hash word.
    SumF16Halves,
}

/// The data of one trellis codebook.
///
/// The decode equation is:
/// `v = reduce(((code * mul + add) & mask) ^ xor) * scale + bias`.
#[derive(Debug, Clone, Copy)]
struct Codebook {
    mul: u32,
    add: u32,
    mask: u32,
    xor: u32,
    reducer: Reducer,
    scale: f32,
    bias: f32,
}

/// Codebook 1 (`decode_3inst<1>`). The tests compare it with the released
/// checkpoint. Do not change these values.
const CODEBOOK_MCG: Codebook = Codebook {
    mul: MCG_MULT,
    add: 0,
    mask: LOP3_AND,
    xor: LOP3_XOR,
    reducer: Reducer::SumF16Halves,
    scale: 1.0,
    bias: 0.0,
};

/// The codebook slot for each value of the `escha_config[3]` flag.
///
/// The slots 0 and 2 exist in exllamav3. Their decode steps have no
/// verification here. Thus the slots hold no data, and the loader rejects
/// them. To fill a slot, get reference codes and values from exllamav3.
/// Then add a test like `unpack_tile_matches_reference`.
const CODEBOOK_SLOTS: [Option<Codebook>; 3] = [None, Some(CODEBOOK_MCG), None];

/// Give the codebook for one config flag, or an error.
fn codebook_for_flag(flag: i32) -> Result<Codebook, ModelError> {
    let index = usize::try_from(flag).ok();
    match index.and_then(|i| CODEBOOK_SLOTS.get(i)) {
        Some(&Some(codebook)) => Ok(codebook),
        Some(&None) => Err(ModelError::UnsupportedModel(format!(
            "eschamoe codebook flag {flag} is unverified: this module has no reference decode \
             data for it; supply exllamav3 codes and values to enable it"
        ))),
        None => Err(ModelError::UnsupportedModel(format!(
            "eschamoe codebook flag {flag} unknown: known flags are 0..=2, verified flag is 1"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// The layout of one quantized projection. The data comes from the
/// `escha_config` tensor.
///
/// The tensor holds nine values in this sequence: the tile edge, the value K
/// (the number of bits for each weight), the number of bits, the MCG codebook
/// flag, the number of experts, the input size, the output size, the padded
/// input size, and the padded output size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EschaSpec {
    pub k: usize,
    pub mcg: bool,
    pub num_experts: i32,
    pub in_features: i32,
    pub out_features: i32,
}

impl EschaSpec {
    /// Read the layout from the `escha_config` tensor of nine values.
    ///
    /// # Pad note
    ///
    /// This function rejects a checkpoint with `in_p > in` or `out_p > out`.
    /// The pad layout is unverified, not impossible. The working hypothesis:
    /// the quantizer adds zero rows and columns at the end, before the
    /// trellis step. The safe strip would then be: decode the full matrix,
    /// apply the Hadamard, apply the scales, transpose, and slice
    /// `[..out, ..in]`. The blockwise Hadamard mixes pad rows into real
    /// rows. Thus a slice before the Hadamard is wrong. To settle it,
    /// quantize a small padded matrix with exllamav3. Then compare the two
    /// strip orders against the source matrix.
    pub fn from_config(config: &[i32]) -> Result<Self, ModelError> {
        let [
            tile,
            k,
            _bits,
            mcg,
            num_experts,
            in_features,
            out_features,
            in_p,
            out_p,
        ] = *<&[i32; 9]>::try_from(config).map_err(|_| {
            ModelError::ShapeMismatch(format!(
                "escha_config must have 9 elements, got {}",
                config.len()
            ))
        })?;

        if tile != TILE_I32 {
            return Err(ModelError::UnsupportedModel(format!(
                "eschamoe tile size {tile} unsupported (only {TILE})"
            )));
        }
        codebook_for_flag(mcg)?;
        if !(1..=8).contains(&k) {
            return Err(ModelError::UnsupportedModel(format!(
                "eschamoe K={k} out of range 1..=8"
            )));
        }
        for (name, dim, padded) in [("in", in_features, in_p), ("out", out_features, out_p)] {
            if padded < dim {
                return Err(ModelError::ShapeMismatch(format!(
                    "eschamoe {name} padded size {padded} is less than {name}_features {dim}"
                )));
            }
            if padded % TILE_I32 != 0 {
                return Err(ModelError::ShapeMismatch(format!(
                    "eschamoe {name} padded size {padded} is not a multiple of the tile edge \
                     {TILE}"
                )));
            }
            had_size(padded, name)?;
        }
        // Refer to the pad note above. The pad layout is unverified. A decode
        // with a guessed layout could give wrong weights. Thus the function
        // rejects it.
        if in_p != in_features || out_p != out_features {
            return Err(ModelError::UnsupportedModel(format!(
                "eschamoe padded dims {in_features}->{in_p}, {out_features}->{out_p}: the pad \
                 layout is unverified (see the pad note on EschaSpec::from_config)"
            )));
        }

        Ok(Self {
            k: usize::try_from(k).unwrap_or(0),
            // Only flag 1 passes the codebook gate above. Thus `mcg` is true.
            mcg: true,
            num_experts,
            in_features,
            out_features,
        })
    }

    /// Give the number of tiles on the input axis and the output axis.
    pub const fn tiles(&self) -> (usize, usize) {
        (
            self.in_features as usize / TILE,
            self.out_features as usize / TILE,
        )
    }

    /// Give the number of `u16` words in one packed tile.
    pub const fn words_per_tile(&self) -> usize {
        TILE * self.k
    }
}

/// The expert axis of one trellis projection.
///
/// An expert checkpoint stores each projection with a leading expert axis.
/// A dense checkpoint stores one matrix and has no expert axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertAxis {
    /// No expert axis. The code tensor has rank 3, and each scale vector
    /// has rank 1.
    Dense,
    /// A leading expert axis with this number of experts. The code tensor
    /// has rank 4, and each scale vector has rank 2.
    Experts(i32),
}

impl ExpertAxis {
    /// Derive the axis from `escha_config[4]` and the code tensor rank.
    ///
    /// The two sources must agree. A dense code tensor needs a config value
    /// of zero or one. An expert code tensor needs a config value of one or
    /// more. On disagreement, the error names both values.
    pub fn derive(config_experts: i32, code_rank: usize) -> Result<Self, ModelError> {
        match (code_rank, config_experts) {
            (3, 0 | 1) => Ok(Self::Dense),
            (4, experts) if experts >= 1 => Ok(Self::Experts(experts)),
            (3, experts) => Err(ModelError::ShapeMismatch(format!(
                "escha_config[4] gives {experts} experts, but escha_code has rank 3 (the dense \
                 layout permits 0 or 1)"
            ))),
            (4, experts) => Err(ModelError::ShapeMismatch(format!(
                "escha_config[4] gives {experts} experts, but escha_code has rank 4 (the expert \
                 layout needs 1 or more)"
            ))),
            (rank, experts) => Err(ModelError::ShapeMismatch(format!(
                "escha_code has rank {rank} with {experts} experts in escha_config[4]; the rank \
                 must be 3 (dense) or 4 (experts)"
            ))),
        }
    }

    /// Give the number of matrices to decode.
    pub const fn count(self) -> i32 {
        match self {
            Self::Dense => 1,
            Self::Experts(experts) => experts,
        }
    }
}

/// The length convention of one scale vector.
///
/// A checkpoint can store each scale vector with the logical length or with
/// the padded length. The loader records the convention here. The record
/// matters for a future pad strip. Refer to the pad note on
/// [`EschaSpec::from_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLen {
    /// The vector length equals the feature count.
    Logical,
    /// The vector length equals the padded feature count.
    Padded,
}

/// The validated layout of one trellis projection.
#[derive(Debug, Clone, Copy)]
pub struct TrellisLayout {
    /// The parsed config values.
    pub spec: EschaSpec,
    /// The expert axis of the six tensors.
    pub expert_axis: ExpertAxis,
    /// The length convention of `rin` and `s_in`.
    pub scale_len_in: ScaleLen,
    /// The length convention of `rout` and `s_out`.
    pub scale_len_out: ScaleLen,
}

// ---------------------------------------------------------------------------
// Trellis decode
// ---------------------------------------------------------------------------

/// Give the element order in a tile of 16 by 16. EXL3 writes the elements in
/// this order (`exl3_lib::tensor_core_perm`).
///
/// Item `i` of the result is the row-major position of stored element `i`.
fn tile_perm() -> [u8; TILE_ELEMS] {
    let mut perm = [0u8; TILE_ELEMS];
    for t in 0..32 {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let rows = [r0, r0 + 1, r0 + 8, r0 + 9];
        for (j, c) in [c0, c0 + 8].into_iter().enumerate() {
            for (i, r) in rows.into_iter().enumerate() {
                // r, c < 16 so r * 16 + c < 256 and the cast cannot truncate.
                perm[t * 8 + j * 4 + i] = (r * TILE + c) as u8;
            }
        }
    }
    perm
}

/// Change a trellis code of 16 bits into its codebook value.
///
/// The hash step multiplies the code, adds a constant, and applies a mask
/// and an XOR. The reduce step changes the hash word into one value. Last,
/// the scale and the bias apply. For codebook 1, the values have an
/// approximate Gaussian distribution.
#[inline]
fn decode_code(code: u16, cb: &Codebook) -> f32 {
    let x = (u32::from(code).wrapping_mul(cb.mul).wrapping_add(cb.add) & cb.mask) ^ cb.xor;
    let reduced = match cb.reducer {
        Reducer::SumF16Halves => {
            f32::from(f16::from_bits(x as u16)) + f32::from(f16::from_bits((x >> 16) as u16))
        }
    };
    reduced.mul_add(cb.scale, cb.bias)
}

/// Give the four hash constants of the codebook for the GPU decode.
///
/// The result holds the multiply, the add, the mask, and the XOR values.
/// The Metal kernel supports only the verified codebook 1. That codebook
/// adds the two `f16` halves and has a neutral scale and bias. For any
/// other configuration, the function gives `None`. The kernel must then
/// refuse the decode.
// Phase 3 connects the kernel to the forward path. Until then, only the
// tests use this chain. The allow keeps the lib target quiet.
#[allow(dead_code)]
pub(crate) fn gpu_codebook(spec: &EschaSpec) -> Option<[u32; 4]> {
    if !spec.mcg {
        return None;
    }
    let cb = CODEBOOK_MCG;
    let neutral = cb.scale.to_bits() == 1.0f32.to_bits() && cb.bias.to_bits() == 0.0f32.to_bits();
    (matches!(cb.reducer, Reducer::SumF16Halves) && neutral)
        .then_some([cb.mul, cb.add, cb.mask, cb.xor])
}

/// Change a trellis code with codebook 1. The name comes from the CUDA
/// function `decode_3inst<1>`. The tests use this helper.
#[cfg(test)]
fn decode_3inst(code: u16) -> f32 {
    decode_code(code, &CODEBOOK_MCG)
}

/// Get the 256 trellis codes from one packed tile.
///
/// Each code is a window of 16 bits into a circular bit stream. The stream has
/// `256 * K` bits. The step between two codes is `K` bits. Thus two adjacent
/// codes have `16 - K` bits in common.
///
/// This function is a copy of `unpack_trellis_kernel`. In that kernel, thread
/// `t` gives code `2t` and code `2t + 1`.
fn unpack_tile(packed: &[u16], k: usize, codes: &mut [u16; TILE_ELEMS]) {
    let n_words = k * TILE_ELEMS / 32;
    // Read two little-endian `u16` values as one `u32`. The CUDA kernel does
    // the same.
    let word = |index: usize| -> u32 {
        let lo = (index % n_words) * 2;
        u32::from(packed[lo]) | (u32::from(packed[lo + 1]) << 16)
    };

    for t in 0..TILE_ELEMS / 2 {
        // The term `+ TILE_ELEMS * k` is the wrap of the circular stream. It
        // comes before `- 16`. Thus the unsigned result stays 0 or more when
        // `t` is 0.
        let b0 = t * 2 * k + k + TILE_ELEMS * k - 16;
        let b2 = b0 + k + 16;
        let i0 = b0 / 32;
        let i1 = (b2 - 1) / 32;
        let s1 = (i1 + 1) * 32 - b2;

        let pair = (u64::from(word(i0)) << 32) | u64::from(word(i1));
        let w1 = (pair >> s1) as u32;
        codes[2 * t] = ((w1 >> k) & 0xFFFF) as u16;
        codes[2 * t + 1] = (w1 & 0xFFFF) as u16;
    }
}

/// Decode the packed codes of one expert into an `[in, out]` f16 matrix.
///
/// The result is `Ŵ`. The function [`dequant_expert`] applies the Hadamard and
/// the channel scales after this step. Those operations are small, and the GPU
/// does them.
fn decode_expert_codes(packed: &[u16], spec: &EschaSpec) -> Vec<f16> {
    let (tiles_k, tiles_n) = spec.tiles();
    let words = spec.words_per_tile();
    let perm = tile_perm();
    let row_len = spec.out_features as usize;

    // Only codebook 1 passes `EschaSpec::from_config`. Refer to
    // `CODEBOOK_SLOTS`. The loop reads the codebook data, not a flag.
    let cb = CODEBOOK_MCG;
    let mut out = vec![f16::ZERO; tiles_k * TILE * row_len];
    let mut codes = [0u16; TILE_ELEMS];

    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let base = (tk * tiles_n + tn) * words;
            unpack_tile(&packed[base..base + words], spec.k, &mut codes);
            for (i, &code) in codes.iter().enumerate() {
                // Move each element from the stored order to its row-major
                // position in the tile.
                let slot = usize::from(perm[i]);
                let (r, c) = (slot / TILE, slot % TILE);
                out[(tk * TILE + r) * row_len + tn * TILE + c] =
                    f16::from_f32(decode_code(code, &cb));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Give the Hadamard block size for one axis length.
///
/// The module supports one size today. The value is per axis. Thus a future
/// checkpoint can use a different size on each axis. The function gives an
/// error when no supported size divides the length. Do not derive the size
/// from the `rin` values: `|rin|` is not a scaled sign vector.
fn had_size(dim: i32, name: &str) -> Result<i32, ModelError> {
    HAD_SIZES
        .into_iter()
        .find(|&block| dim > 0 && dim % block == 0)
        .ok_or_else(|| {
            ModelError::UnsupportedModel(format!(
                "eschamoe {name} size {dim}: no supported Hadamard block {HAD_SIZES:?} divides it"
            ))
        })
}

/// Apply an orthonormal Hadamard on `axis`. The block size comes from the
/// axis length. Refer to [`had_size`].
fn had_blockwise(x: &Array, axis: i32) -> Result<Array, ModelError> {
    let moved = ops::swap_axes(x, axis, -1)?;
    let shape = moved.shape().to_vec();
    let last = *shape.last().unwrap_or(&0);
    let block = had_size(last, "Hadamard axis")?;
    let blocked = moved.reshape(&[-1, last / block, block])?;
    let transformed = blocked.hadamard_transform(None)?;
    Ok(ops::swap_axes(&transformed.reshape(&shape)?, axis, -1)?)
}

/// Build the weight of one expert as an `[out, in]` matrix.
///
/// The parameter `code` is the `[in/16, out/16, 16K]` slice of that expert.
/// The four scale vectors are its `[in]` and `[out]` slices.
///
/// In the released checkpoint, all values of `s_in` and `s_out` are 1.0. This
/// function applies them because a different checkpoint can have other
/// values.
pub fn dequant_expert(
    code: &Array,
    rin: &Array,
    rout: &Array,
    s_in: &Array,
    s_out: &Array,
    spec: &EschaSpec,
) -> Result<Array, ModelError> {
    let (tiles_k, tiles_n) = spec.tiles();
    let expected = [
        i32::try_from(tiles_k).unwrap_or(i32::MAX),
        i32::try_from(tiles_n).unwrap_or(i32::MAX),
        i32::try_from(spec.words_per_tile()).unwrap_or(i32::MAX),
    ];
    if code.shape() != expected {
        return Err(ModelError::ShapeMismatch(format!(
            "escha_code expected {expected:?}, got {:?}",
            code.shape()
        )));
    }

    let packed: Vec<u16> = code.as_dtype(Dtype::Uint16)?.as_slice::<u16>().to_vec();
    let decoded = decode_expert_codes(&packed, spec);
    let w = Array::from_slice(&decoded, &[spec.in_features, spec.out_features]);

    // The scales apply outside the Hadamard on the two axes. Refer to the
    // module documentation.
    let scale_in = rin.multiply(s_in)?.reshape(&[spec.in_features, 1])?;
    let scale_out = rout.multiply(s_out)?.reshape(&[1, spec.out_features])?;
    let rotated = had_blockwise(&had_blockwise(&w, 0)?, 1)?;
    let scaled = rotated.multiply(&scale_in)?.multiply(&scale_out)?;
    Ok(ops::swap_axes(&scaled, 0, 1)?)
}

// ---------------------------------------------------------------------------
// Native expert forward
// ---------------------------------------------------------------------------

/// One expert projection in native trellis form.
///
/// The struct keeps the packed code words resident. It folds the four scale
/// vectors into two at load time. Refer to `factored_matvec_matches_dequant_expert`
/// for the proof of the factored form.
#[derive(Debug, Clone)]
pub struct EschaProj {
    /// The trellis words. Shape `[E, in/16, out/16, 16K]`, dtype int16.
    pub code: Array,
    /// The folded input scale `rin * s_in`. Shape `[E, in]`, dtype f32.
    pub su: Array,
    /// The folded output scale `rout * s_out`. Shape `[E, out]`, dtype f32.
    pub sv: Array,
    /// The tile layout of `code`.
    pub spec: EschaSpec,
}

/// The stacked expert projections of one `SwitchMLP` block.
///
/// The checkpoint stores gate and up as one fused tensor. Thus the block
/// holds two projections, not three.
#[derive(Debug, Clone)]
pub struct EschaSwitchMlp {
    /// The fused gate+up projection. Its output size is `2 * intermediate`.
    pub gate_up: EschaProj,
    /// The down projection.
    pub down: EschaProj,
}

impl EschaProj {
    /// The row limit for the matvec kernel path.
    ///
    /// The matvec kernel decodes the expert weight again for each row. The
    /// scratch path decodes each selected expert once. At 32 rows the two
    /// decode costs are equal in the worst case. Below the limit, the matvec
    /// path also avoids one CPU sync and the scratch buffer.
    const GATHER_QMV_MAX_ROWS: i32 = 32;

    /// Fold the scale vectors and keep the code words.
    pub fn new(
        code: Array,
        rin: &Array,
        rout: &Array,
        s_in: &Array,
        s_out: &Array,
        spec: EschaSpec,
    ) -> Result<Self, ModelError> {
        let su = rin
            .as_dtype(Dtype::Float32)?
            .multiply(s_in.as_dtype(Dtype::Float32)?)?;
        let sv = rout
            .as_dtype(Dtype::Float32)?
            .multiply(s_out.as_dtype(Dtype::Float32)?)?;
        Ok(Self { code, su, sv, spec })
    }

    /// Apply the projection to each row with its own expert.
    ///
    /// The input `x` has shape `[rows, in]` in a float dtype. The input
    /// `eids` has shape `[rows]`, dtype uint32. Sorted ids give the best
    /// speed on the scratch path. Unsorted ids stay correct on both paths.
    /// The result has shape `[rows, out]`, dtype f32.
    pub fn gather_forward(&self, x: &Array, eids: &Array) -> Result<Array, ModelError> {
        let rows = *x.shape().first().ok_or_else(|| {
            ModelError::ShapeMismatch("gather_forward input must be [rows, in]".to_owned())
        })?;
        let su_rows = self.su.take_axis(eids, 0)?;
        let sv_rows = self.sv.take_axis(eids, 0)?;
        let xh = had_blockwise(&x.as_dtype(Dtype::Float32)?.multiply(&su_rows)?, -1)?;
        let y_pre = if rows <= Self::GATHER_QMV_MAX_ROWS {
            crate::metal_kernel::eschamoe_gather_qmv(&xh, &self.code, eids, &self.spec)?
        } else {
            self.scratch_matmul(&xh, eids)?
        };
        Ok(had_blockwise(&y_pre, -1)?.multiply(&sv_rows)?)
    }

    /// Apply the projection through a decoded scratch weight.
    ///
    /// The function splits the rows into runs with one expert each. It
    /// decodes the expert of each run once. Then one matmul serves the whole
    /// run.
    // ponytail: the ceiling is the decode bandwidth. Each prefill call writes
    // and reads up to E_used * in * out half words of scratch. A native
    // trellis GEMM kernel would remove the scratch traffic. Build one when
    // prefill throughput becomes the bottleneck.
    fn scratch_matmul(&self, xh: &Array, eids: &Array) -> Result<Array, ModelError> {
        let ids: Vec<u32> = eids.as_slice::<u32>().to_vec();
        let mut run_experts: Vec<u32> = Vec::new();
        let mut boundaries: Vec<i32> = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            if run_experts.last() == Some(&id) {
                continue;
            }
            run_experts.push(id);
            if i > 0 {
                boundaries.push(
                    i32::try_from(i).map_err(|_| {
                        ModelError::ShapeMismatch("row count exceeds i32".to_owned())
                    })?,
                );
            }
        }
        let parts = xh.split_axis(&boundaries, Some(0))?;
        let mut segments: Vec<Array> = Vec::with_capacity(parts.len());
        for (part, &expert) in parts.iter().zip(&run_experts) {
            let sel = Array::from_slice(&[expert], &[1]);
            let code_e = self.code.take_axis(&sel, 0)?.squeeze_axes(&[0])?;
            let w_e = crate::metal_kernel::eschamoe_dequant_tiles(&code_e, &self.spec)?
                .as_dtype(Dtype::Float32)?;
            segments.push(ops::matmul(part, &w_e)?);
        }
        let refs: Vec<&Array> = segments.iter().collect();
        Ok(ops::concatenate_axis(&refs, 0)?)
    }
}

/// Dequantize a per-output-channel int8 tensor: `w[o, i] * scale[o]`.
///
/// Used for every non-expert projection in the checkpoint (`weight_int8` +
/// `weight_scale`).
pub fn dequant_int8(weight: &Array, scale: &Array) -> Result<Array, ModelError> {
    let rows = *weight
        .shape()
        .first()
        .ok_or_else(|| ModelError::ShapeMismatch("int8 weight must be at least 1-D".to_owned()))?;
    if scale.shape() != [rows] {
        return Err(ModelError::ShapeMismatch(format!(
            "weight_scale expected [{rows}], got {:?}",
            scale.shape()
        )));
    }
    let per_row = scale.as_dtype(Dtype::Float32)?.reshape(&[rows, 1])?;
    Ok(weight.as_dtype(Dtype::Float32)?.multiply(&per_row)?)
}

// ---------------------------------------------------------------------------
// Checkpoint keys
// ---------------------------------------------------------------------------

/// The suffixes of the six tensors of one trellis projection.
pub const CODE_SUFFIX: &str = ".escha_code";
pub const CONFIG_SUFFIX: &str = ".escha_config";
pub const ESCHA_SUFFIXES: [&str; 6] = [
    ".escha_code",
    ".escha_config",
    ".escha_rin",
    ".escha_rout",
    ".escha_s_in",
    ".escha_s_out",
];
/// The suffixes of an int8 projection. It has one scale for each output
/// channel.
pub const INT8_WEIGHT_SUFFIX: &str = ".weight_int8";
pub const INT8_SCALE_SUFFIX: &str = ".weight_scale";

/// Change an eschamoe key to the format that the Qwen3.5 loader accepts.
///
/// Escha uses the names of transformers v5, for example
/// `model.language_model.layers.N...`. The loader in [`crate::qwen3_next`]
/// accepts the mlx-community sequence, for example
/// `language_model.model.layers.N...`. The loader then removes the first part
/// to get the parameter path `model.layers.N...`.
///
/// This function moves the first two parts. Thus the loader can read an escha
/// checkpoint with no other change.
///
/// The keys that start with `mtp.` are already correct. They do not change.
pub fn normalize_key(key: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    key.strip_prefix("model.language_model.").map_or_else(
        || {
            if key.starts_with("lm_head.") {
                Cow::Owned(format!("language_model.{key}"))
            } else {
                Cow::Borrowed(key)
            }
        },
        |rest| Cow::Owned(format!("language_model.model.{rest}")),
    )
}

/// The storage type of a checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    /// One of the six trellis tensors. It has the projection prefix.
    Trellis,
    /// A `weight_int8` tensor or a `weight_scale` tensor.
    Int8,
    /// A bf16 tensor. Examples are the norms, the router gate, and the GDN
    /// values.
    Dense,
}

/// Give the storage type of a key and its projection prefix.
///
/// For a dense tensor, the prefix is the full key.
pub fn classify(key: &str) -> (Storage, &str) {
    for suffix in ESCHA_SUFFIXES {
        if let Some(prefix) = key.strip_suffix(suffix) {
            return (Storage::Trellis, prefix);
        }
    }
    for suffix in [INT8_WEIGHT_SUFFIX, INT8_SCALE_SUFFIX] {
        if let Some(prefix) = key.strip_suffix(suffix) {
            return (Storage::Int8, prefix);
        }
    }
    (Storage::Dense, key)
}

/// Whether a key holds an `RMSNorm` weight the checkpoint stores as `w - 1`.
///
/// Escha centres its norm weights on zero. Every `RMSNorm` follows that
/// convention except the gated norm of the GDN block, which keeps the plain
/// weight. Measured against `mlx-community/Qwen3.6-35B-A3B-4bit`, all 101
/// offset tensors match at `|escha + 1 - base| <= 0.008`, the bf16 step, and
/// the 30 `linear_attn.norm` tensors match exactly with no offset.
///
/// A norm is the only 1-D weight whose last name part holds `norm`, so the
/// name gives the answer with no list of layers to maintain.
fn is_offset_rmsnorm(key: &str) -> bool {
    let Some(module) = key.strip_suffix(".weight") else {
        return false;
    };
    !module.ends_with(".linear_attn.norm")
        && module
            .rsplit('.')
            .next()
            .is_some_and(|part| part.contains("norm"))
}

// ---------------------------------------------------------------------------
// Checkpoint conversion
// ---------------------------------------------------------------------------

/// The affine quantization values for one projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineTarget {
    pub group_size: i32,
    pub bits: i32,
}

/// The affine layout that the conversion code makes.
///
/// An eschamoe checkpoint has no MLX `quantization` block. Its
/// `quantization_config` block gives the trellis bit rate, for example 2.0.
/// That value is not an affine bit rate, and the model must not use it. Thus
/// the loader sets this layout for the model and the conversion code. The
/// values agree with `mlx-community/Qwen3.6-35B-A3B-4bit`.
pub const CONVERSION_TARGET: AffineTarget = AffineTarget {
    group_size: 64,
    bits: 4,
};

impl Default for AffineTarget {
    fn default() -> Self {
        CONVERSION_TARGET
    }
}

impl AffineTarget {
    /// Read the values from the `quantization` block of `config.json`.
    ///
    /// The block has a `group_size` value and a `bits` value. It can also have
    /// a different pair of values for each parameter path. The mlx-community
    /// checkpoints use this layout. For example, they set 8 bits for
    /// `mlp.gate`.
    pub fn resolve(quantization: Option<&serde_json::Value>, path: &str) -> Self {
        let Some(cfg) = quantization else {
            return Self::default();
        };
        let field = |value: &serde_json::Value, key: &str| -> Option<i32> {
            value.get(key)?.as_i64()?.try_into().ok()
        };
        let base = Self {
            group_size: field(cfg, "group_size").unwrap_or(64),
            bits: field(cfg, "bits").unwrap_or(4),
        };
        cfg.get(path).map_or(base, |over| Self {
            group_size: field(over, "group_size").unwrap_or(base.group_size),
            bits: field(over, "bits").unwrap_or(base.bits),
        })
    }
}

/// The two parts of the escha `gate_up_proj` tensor on the output axis.
///
/// The checkpoint `mlx-community/Qwen3.6-35B-A3B-4bit` holds the two parts as
/// separate tensors. A test compared them with the escha tensor. The first
/// part agrees with `gate_proj`, and the second part agrees with `up_proj`.
/// The cosine value for each pair is 0.966. The cosine value for the two other
/// pairs is near 0.00.
pub const FUSED_GATE_UP: [&str; 2] = ["gate_proj", "up_proj"];

/// An affine tensor. It has three parts: the packed weight, the scales, and
/// the biases.
type Triple = [Array; 3];

/// Quantize an `[out, in]` matrix into the three affine tensors.
///
/// MLX gives a short message when this operation fails. Thus the function adds
/// the tensor name, the shape, and the layout to the error. A person can then
/// find the cause.
fn quantize_affine_named(
    name: &str,
    dense: &Array,
    target: AffineTarget,
) -> Result<Triple, ModelError> {
    let last = dense.shape().last().copied().unwrap_or(0);
    if last % target.group_size != 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "{name}: last dimension {last} is not a multiple of group_size {}",
            target.group_size
        )));
    }
    let (weight, scales, biases) =
        ops::quantize(dense, target.group_size, target.bits).map_err(|e| {
            ModelError::ShapeMismatch(format!(
                "{name}: quantize shape {:?} group_size={} bits={} failed: {e}",
                dense.shape(),
                target.group_size,
                target.bits
            ))
        })?;
    Ok((weight, scales, biases).into())
}

fn quantize_affine(dense: &Array, target: AffineTarget) -> Result<Triple, ModelError> {
    quantize_affine_named("tensor", dense, target)
}

/// Put the affine tensors of all experts into `[E, ...]` tensors.
fn stack_triples(per_expert: &[Triple]) -> Result<Triple, ModelError> {
    let mut stacked = Vec::with_capacity(3);
    for axis in 0..3 {
        let expanded: Vec<Array> = per_expert
            .iter()
            .map(|t| t[axis].expand_dims(0))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&Array> = expanded.iter().collect();
        stacked.push(ops::concatenate_axis(&refs, 0)?);
    }
    stacked
        .try_into()
        .map_err(|_| ModelError::ShapeMismatch("expected three stacked tensors".to_owned()))
}

/// Give the three parameter entries for one affine tensor.
fn triple_entries(target: &str, [w, s, b]: Triple) -> [(String, Array); 3] {
    [
        (format!("{target}.weight"), w),
        (format!("{target}.scales"), s),
        (format!("{target}.biases"), b),
    ]
}

/// The six tensors of one trellis projection.
///
/// The `prefix` field holds the checkpoint name of the projection, for example
/// `model.layers.7.mlp.experts.gate_up_proj`. The checks name it in their
/// messages, so a warning points at one tensor of the checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct TrellisGroup<'a> {
    pub prefix: &'a str,
    pub code: &'a Array,
    pub config: &'a Array,
    pub rin: &'a Array,
    pub rout: &'a Array,
    pub s_in: &'a Array,
    pub s_out: &'a Array,
}

impl TrellisGroup<'_> {
    pub fn spec(&self) -> Result<EschaSpec, ModelError> {
        let config: Vec<i32> = self.config.as_dtype(Dtype::Int32)?.as_slice().to_vec();
        EschaSpec::from_config(&config)
    }

    /// Check the six tensors against the config and give the full layout.
    ///
    /// The checks are cheap and run before any decode:
    ///
    /// - the expert axis: the config value and the code rank must agree.
    /// - the bit budget: the code bits must equal `experts * in_p * out_p * K`.
    /// - the scale vectors: each shape must match the expert axis and one
    ///   feature length.
    /// - the `|rout|` mean: a value far from 1.0 gives a warning, not an
    ///   error. The released checkpoint keeps the mean near 1.0.
    pub fn validate(&self) -> Result<TrellisLayout, ModelError> {
        let spec = self.spec()?;
        let expert_axis = ExpertAxis::derive(spec.num_experts, self.code.ndim())?;
        if let ExpertAxis::Experts(experts) = expert_axis {
            let leading = self.code.shape().first().copied().unwrap_or(0);
            if leading != experts {
                return Err(ModelError::ShapeMismatch(format!(
                    "escha_code leading dim {leading} does not equal the {experts} experts of \
                     escha_config[4]"
                )));
            }
        }
        check_bit_budget(self.code, &spec, expert_axis)?;
        // `from_config` rejects pads. Thus each padded length equals the
        // logical length here, and the pair below carries the same value.
        let scale_len_in = check_scale_pair(
            ("escha_rin", self.rin),
            ("escha_s_in", self.s_in),
            expert_axis,
            spec.in_features,
            spec.in_features,
        )?;
        let scale_len_out = check_scale_pair(
            ("escha_rout", self.rout),
            ("escha_s_out", self.s_out),
            expert_axis,
            spec.out_features,
            spec.out_features,
        )?;
        if let Some(mean) = rout_mean_if_off(self.rout)? {
            tracing::warn!(
                mean,
                projection = self.prefix,
                "eschamoe escha_rout |mean| is far from the expected 1.0; the checkpoint may \
                 not follow the released convention"
            );
        }
        Ok(TrellisLayout {
            spec,
            expert_axis,
            scale_len_in,
            scale_len_out,
        })
    }
}

/// Check the total bit budget of the code tensor.
///
/// Each stored word holds 16 bits. The trellis needs `K` bits for each
/// weight. Thus the code bits must equal `experts * in_p * out_p * K`.
fn check_bit_budget(code: &Array, spec: &EschaSpec, axis: ExpertAxis) -> Result<(), ModelError> {
    let have = code.shape().iter().map(|&d| i64::from(d)).product::<i64>() * 16;
    let need = i64::from(axis.count())
        * i64::from(spec.in_features)
        * i64::from(spec.out_features)
        * i64::try_from(spec.k).unwrap_or(i64::MAX);
    if have != need {
        return Err(ModelError::ShapeMismatch(format!(
            "escha_code holds {have} bits, expected {need} bits ({experts} experts x {in_f} x \
             {out_f} x K={k})",
            experts = axis.count(),
            in_f = spec.in_features,
            out_f = spec.out_features,
            k = spec.k
        )));
    }
    Ok(())
}

/// Check one pair of scale vectors on one axis. The two vectors must use the
/// same length convention. Give that convention.
fn check_scale_pair(
    first: (&str, &Array),
    second: (&str, &Array),
    axis: ExpertAxis,
    logical: i32,
    padded: i32,
) -> Result<ScaleLen, ModelError> {
    let len_first = check_scale(first.0, first.1, axis, logical, padded)?;
    let len_second = check_scale(second.0, second.1, axis, logical, padded)?;
    if len_first == len_second {
        Ok(len_first)
    } else {
        Err(ModelError::ShapeMismatch(format!(
            "{} uses the {len_first:?} length but {} uses the {len_second:?} length",
            first.0, second.0
        )))
    }
}

/// Check one scale vector against the expert axis and the feature lengths.
fn check_scale(
    name: &str,
    vector: &Array,
    axis: ExpertAxis,
    logical: i32,
    padded: i32,
) -> Result<ScaleLen, ModelError> {
    let shape = vector.shape();
    let last = match (axis, shape) {
        (ExpertAxis::Dense, [len]) => Some(*len),
        (ExpertAxis::Experts(experts), [lead, len]) if *lead == experts => Some(*len),
        (ExpertAxis::Dense | ExpertAxis::Experts(_), _) => None,
    };
    match last {
        Some(len) if len == logical => Ok(ScaleLen::Logical),
        Some(len) if len == padded => Ok(ScaleLen::Padded),
        Some(_) | None => {
            let expected = match axis {
                ExpertAxis::Dense => format!("[{logical}]"),
                ExpertAxis::Experts(experts) => format!("[{experts}, {logical}]"),
            };
            Err(ModelError::ShapeMismatch(format!(
                "{name} expected {expected} (or the padded length {padded}), got {shape:?}"
            )))
        }
    }
}

/// Give the mean of `|rout|` when it is far from 1.0.
///
/// The released checkpoint keeps the mean near 1.0. Six tensors gave values
/// between 0.99971 and 1.00011. A future checkpoint can drop the convention.
/// Thus the caller warns and does not fail.
fn rout_mean_if_off(rout: &Array) -> Result<Option<f32>, ModelError> {
    let mean = rout
        .as_dtype(Dtype::Float32)?
        .abs()?
        .mean(None)?
        .item::<f32>();
    Ok(((mean - 1.0).abs() > ROUT_MEAN_TOL).then_some(mean))
}

/// Decode a trellis projection and quantize it to the affine format.
///
/// For an expert projection, the function decodes each expert and stacks the
/// results on a leading `[E, ...]` axis. For a dense projection, the function
/// decodes one matrix. Each dense result is then 2-D, with no expert axis.
///
/// The parameter `parts` divides the output axis into equal parts. Use 2 for
/// the escha `gate_up_proj` tensor. Refer to [`FUSED_GATE_UP`]. Use 1 for the
/// `down_proj` tensor. The function gives three affine tensors for each part.
///
/// The function decodes one expert at a time and keeps only the quantized
/// result. The full set of experts in the f32 format is too large. For one
/// projection, a `[256, 1024, 2048]` f32 tensor uses 2 GiB. One expert uses
/// some MiB.
pub fn convert_trellis(
    group: TrellisGroup<'_>,
    parts: i32,
    target: AffineTarget,
) -> Result<Vec<Triple>, ModelError> {
    let layout = group.validate()?;
    let spec = layout.spec;
    if parts < 1 || spec.out_features % parts != 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "cannot split {} output features into {parts} parts",
            spec.out_features
        )));
    }
    let rows = spec.out_features / parts;

    match layout.expert_axis {
        ExpertAxis::Dense => {
            let dense = dequant_expert(
                group.code,
                group.rin,
                group.rout,
                group.s_in,
                group.s_out,
                &spec,
            )?;
            (0..parts)
                .map(|part| {
                    let lo = rows * part;
                    quantize_affine(&dense.index((lo..lo + rows, ..)), target)
                })
                .collect()
        }
        ExpertAxis::Experts(experts) => {
            let mut stacks: Vec<Vec<Triple>> =
                vec![Vec::new(); usize::try_from(parts).unwrap_or(1)];
            for expert in 0..experts {
                let dense = dequant_expert(
                    &group.code.index(expert),
                    &group.rin.index(expert),
                    &group.rout.index(expert),
                    &group.s_in.index(expert),
                    &group.s_out.index(expert),
                    &spec,
                )?;
                for (part, stack) in stacks.iter_mut().enumerate() {
                    let lo = rows * i32::try_from(part).unwrap_or(0);
                    stack.push(quantize_affine(&dense.index((lo..lo + rows, ..)), target)?);
                }
            }
            stacks.iter().map(|stack| stack_triples(stack)).collect()
        }
    }
}

/// Give the parameter names of one trellis projection.
///
/// An expert projection must sit under `mlp.experts`. It maps to the
/// `switch_mlp` module of the model. The tensor `gate_up_proj` holds two
/// projections. Thus it gives two names. Refer to [`FUSED_GATE_UP`].
///
/// A dense projection keeps its own prefix. A fused dense `gate_up_proj`
/// tensor gives the two names next to it.
fn projection_targets(prefix: &str, axis: ExpertAxis) -> Result<Vec<String>, ModelError> {
    match (prefix.rsplit_once(".mlp.experts."), axis) {
        (Some((stem, proj)), ExpertAxis::Experts(_)) => Ok(match proj {
            "gate_up_proj" => FUSED_GATE_UP
                .iter()
                .map(|half| format!("{stem}.mlp.switch_mlp.{half}"))
                .collect(),
            other => vec![format!("{stem}.mlp.switch_mlp.{other}")],
        }),
        (Some(_), ExpertAxis::Dense) => Err(ModelError::UnsupportedModel(format!(
            "dense trellis tensor under mlp.experts: {prefix}"
        ))),
        (None, ExpertAxis::Dense) => Ok(prefix.strip_suffix(".gate_up_proj").map_or_else(
            || vec![prefix.to_owned()],
            |stem| {
                FUSED_GATE_UP
                    .iter()
                    .map(|half| format!("{stem}.{half}"))
                    .collect()
            },
        )),
        (None, ExpertAxis::Experts(_)) => Err(ModelError::UnsupportedModel(format!(
            "expert trellis tensor outside mlp.experts: {prefix}"
        ))),
    }
}

/// Change a full eschamoe checkpoint into the affine tensors that the Qwen3.5
/// loader accepts.
///
/// The result keys use the mlx-community format, for example
/// `language_model.model.layers.N...`. Thus the caller can send them to the
/// standard parameter code and the GDN fusion code.
///
/// The function operates on three storage types:
///
/// - the trellis tensors, dense or with experts. [`convert_trellis`] divides
///   them and quantizes them.
/// - the int8 tensors. The function decodes them and quantizes them.
/// - the bf16 tensors. The function does not change them.
///
/// The sequence of the int8 step is important. The GDN fusion code looks for
/// the names `.weight`, `.scales`, and `.biases`. Thus this function must make
/// those three tensors before the fusion code starts.
pub fn convert_checkpoint(
    model_dir: &std::path::Path,
    quantization: Option<&serde_json::Value>,
) -> Result<Vec<(String, Array)>, ModelError> {
    Ok(convert_checkpoint_impl(model_dir, quantization, false)?.0)
}

/// Whether the native expert path is on. Set `HIGGS_ESCHA_NATIVE=0` for the
/// affine path.
///
/// The native path keeps each expert projection in its trellis form and reads
/// it with the Metal kernel. The affine path decodes every expert and
/// requantizes the result to 4 bits. For the 35B release the native path holds
/// 11.2 GB and loads in about 6 s; the affine path holds 21.7 GB and takes
/// about 140 s, which crowds a 32 GB machine. Thus the native path is the
/// default, and the affine path stays available for comparison.
pub fn native_mode() -> bool {
    !std::env::var("HIGGS_ESCHA_NATIVE").is_ok_and(|v| v == "0")
}

/// The tensor list and the native expert weights of one conversion.
pub type ConvertedCheckpoint = (Vec<(String, Array)>, Vec<(usize, EschaSwitchMlp)>);

/// Convert a checkpoint and honor [`native_mode`].
///
/// In native mode, the expert projections do not decode. They stay in the
/// trellis form as [`EschaSwitchMlp`] values. The second result gives one
/// value for each layer. The first result then holds no expert affine
/// tensors. All other tensors convert as in [`convert_checkpoint`].
pub fn convert_checkpoint_auto(
    model_dir: &std::path::Path,
    quantization: Option<&serde_json::Value>,
) -> Result<ConvertedCheckpoint, ModelError> {
    convert_checkpoint_impl(model_dir, quantization, native_mode())
}

/// Give the layer index and the projection name of one expert prefix.
///
/// The prefix must sit in the main decoder stack. For example,
/// `model.layers.7.mlp.experts.gate_up_proj` gives `(7, "gate_up_proj")`.
/// Other prefixes give `None`, and the caller decodes them to affine.
fn expert_layer_target(prefix: &str) -> Option<(usize, &str)> {
    let (stem, rest) = prefix.split_once(".layers.")?;
    if stem != "model" && stem != "language_model.model" {
        return None;
    }
    let (index, tail) = rest.split_once('.')?;
    let layer = index.parse::<usize>().ok()?;
    let proj = tail.strip_prefix("mlp.experts.")?;
    matches!(proj, "gate_up_proj" | "down_proj").then_some((layer, proj))
}

// The group loop must keep the take and eval sequence in one place. This
// bounds the peak memory. Thus the function exceeds the line limit.
#[allow(clippy::too_many_lines)]
fn convert_checkpoint_impl(
    model_dir: &std::path::Path,
    quantization: Option<&serde_json::Value>,
    native: bool,
) -> Result<ConvertedCheckpoint, ModelError> {
    use std::collections::HashMap;

    let mut raw: HashMap<String, Array> = HashMap::new();
    for file in crate::collect_safetensors_files(model_dir)? {
        let loaded = Array::load_safetensors(file.to_str().unwrap_or_default())
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
        for (key, value) in loaded {
            raw.insert(normalize_key(&key).into_owned(), value);
        }
    }

    // Group by projection so the six trellis tensors (or the two int8 ones)
    // are converted together.
    let mut groups: HashMap<(Storage, String), Vec<String>> = HashMap::new();
    for key in raw.keys() {
        let (storage, prefix) = classify(key);
        groups
            .entry((storage, prefix.to_owned()))
            .or_default()
            .push(key.clone());
    }

    let mut out: Vec<(String, Array)> = Vec::with_capacity(raw.len());
    // The native expert projections of each layer. The tuple holds the
    // fused gate+up projection and then the down projection.
    #[allow(clippy::type_complexity)]
    let mut native_map: HashMap<usize, (Option<EschaProj>, Option<EschaProj>)> = HashMap::new();

    // Each group makes its result and then frees its source tensors. A full
    // checkpoint holds more than 11 GiB of packed data, and the affine result
    // is larger again. If this loop kept both, the process would use too much
    // memory. `take` removes each source tensor when the code reads it, and
    // `eval` completes the MLX operations. MLX keeps the input of an
    // incomplete operation, so the `eval` call is necessary to free it.
    for ((storage, prefix), keys) in &groups {
        let mut made: Vec<(String, Array)> = Vec::with_capacity(6);
        let mut take = |key: &str| -> Result<Array, ModelError> {
            raw.remove(key)
                .ok_or_else(|| ModelError::MissingWeight(key.to_owned()))
        };

        match storage {
            Storage::Dense => {
                for key in keys {
                    let value = take(key)?;
                    // Escha keeps some projections in the bf16 format, for
                    // example `mlp.gate` and the GDN `in_proj_a`. The model
                    // makes a quantized layer for each of them, and that layer
                    // needs a scales tensor and a biases tensor. Thus this code
                    // quantizes each 2-D weight. The other dense tensors are
                    // norms, the 3-D conv1d weight, or 1-D vectors. Only the
                    // offset norms change; the rest go to the model as they
                    // are.
                    match key.strip_suffix(".weight").filter(|_| value.ndim() == 2) {
                        Some(module) => {
                            let target = AffineTarget::resolve(quantization, module);
                            made.extend(triple_entries(
                                module,
                                quantize_affine_named(module, &value, target)?,
                            ));
                        }
                        // Escha keeps the GDN conv1d weight in the PyTorch
                        // sequence `[out, in/groups, kernel]`. MLX needs the
                        // sequence `[out, kernel, in/groups]`. Thus this code
                        // exchanges the last two axes.
                        None if key.ends_with(".conv1d.weight") && value.ndim() == 3 => {
                            made.push((key.clone(), ops::swap_axes(&value, 1, 2)?));
                        }
                        // An offset norm holds `w - 1`. Restore `w`, and keep
                        // the checkpoint dtype so the model reads it as it
                        // reads every other norm.
                        None if is_offset_rmsnorm(key) => {
                            let one = Array::from_f32(1.0).as_dtype(value.dtype())?;
                            made.push((key.clone(), ops::add(&value, &one)?));
                        }
                        None => made.push((key.clone(), value)),
                    }
                }
            }
            Storage::Int8 => {
                let weight = take(&format!("{prefix}{INT8_WEIGHT_SUFFIX}"))?;
                let scale = take(&format!("{prefix}{INT8_SCALE_SUFFIX}"))?;
                let dense = dequant_int8(&weight, &scale)?;
                let target = AffineTarget::resolve(quantization, prefix);
                made.extend(triple_entries(
                    prefix,
                    quantize_affine_named(prefix, &dense, target)?,
                ));
            }
            Storage::Trellis => {
                let code = take(&format!("{prefix}.escha_code"))?;
                let config = take(&format!("{prefix}.escha_config"))?;
                let rin = take(&format!("{prefix}.escha_rin"))?;
                let rout = take(&format!("{prefix}.escha_rout"))?;
                let s_in = take(&format!("{prefix}.escha_s_in"))?;
                let s_out = take(&format!("{prefix}.escha_s_out"))?;
                let group = TrellisGroup {
                    prefix,
                    code: &code,
                    config: &config,
                    rin: &rin,
                    rout: &rout,
                    s_in: &s_in,
                    s_out: &s_out,
                };
                let axis = ExpertAxis::derive(group.spec()?.num_experts, code.ndim())?;
                let native_target = (native && matches!(axis, ExpertAxis::Experts(_)))
                    .then(|| expert_layer_target(prefix))
                    .flatten();
                if let Some((layer, proj)) = native_target {
                    // Keep the expert projection in the trellis form. Do
                    // not decode it. The eval call makes the folded scales
                    // concrete and frees the source vectors.
                    let spec = group.validate()?.spec;
                    let built = EschaProj::new(code, &rin, &rout, &s_in, &s_out, spec)?;
                    mlx_rs::transforms::eval([&built.code, &built.su, &built.sv])?;
                    let slots = native_map.entry(layer).or_default();
                    let slot = if proj == "gate_up_proj" {
                        &mut slots.0
                    } else {
                        &mut slots.1
                    };
                    *slot = Some(built);
                    continue;
                }
                let targets = projection_targets(prefix, axis)?;
                let splits = i32::try_from(targets.len()).unwrap_or(1);
                let affine = AffineTarget::resolve(quantization, prefix);
                let converted = convert_trellis(group, splits, affine)?;
                for (target, stacked) in targets.iter().zip(converted) {
                    made.extend(triple_entries(target, stacked));
                }
            }
        }

        mlx_rs::transforms::eval(made.iter().map(|(_, a)| a))?;
        out.extend(made);
    }

    let mut natives: Vec<(usize, EschaSwitchMlp)> = Vec::with_capacity(native_map.len());
    for (layer, slots) in native_map {
        match slots {
            (Some(gate_up), Some(down)) => {
                natives.push((layer, EschaSwitchMlp { gate_up, down }));
            }
            _ => {
                return Err(ModelError::MissingWeight(format!(
                    "layer {layer} misses one of its two native expert projections"
                )));
            }
        }
    }

    tracing::info!(
        tensors = out.len(),
        groups = groups.len(),
        native_layers = natives.len(),
        "Converted eschamoe checkpoint to affine"
    );
    Ok((out, natives))
}

/// Whether `model_dir` holds an eschamoe checkpoint.
///
/// `quantize_config.json` is the authority; `config.json`'s
/// `quantization_config.quant_method` is accepted as a fallback because the two
/// duplicate the field.
pub fn is_eschamoe_checkpoint(model_dir: &std::path::Path) -> Result<bool, ModelError> {
    let method = |value: &serde_json::Value| -> Option<String> {
        value
            .get("quant_method")
            .or_else(|| value.get("quantization_config")?.get("quant_method"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    for name in ["quantize_config.json", "config.json"] {
        let path = model_dir.join(name);
        if !path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        if let Some(found) = method(&value) {
            return Ok(found == "eschamoe");
        }
    }
    Ok(false)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr
)]
mod tests {
    use mlx_rs::ops::indexing::IndexOp;

    use super::*;

    fn spec(k: usize, in_f: i32, out_f: i32) -> EschaSpec {
        EschaSpec {
            k,
            mcg: true,
            num_experts: 1,
            in_features: in_f,
            out_features: out_f,
        }
    }

    /// Pack `codes` into the `[tiles_k, tiles_n, 16K]` i16 tensor the loader sees.
    fn code_array(codes: &[u16], spec: &EschaSpec) -> Array {
        let (tk, tn) = spec.tiles();
        let signed: Vec<i16> = codes.iter().map(|&c| c.cast_signed()).collect();
        let dims = [tk, tn, spec.words_per_tile()].map(|d| i32::try_from(d).unwrap());
        Array::from_slice(&signed, &dims)
    }

    /// Give the directory of the test checkpoint.
    fn test_model_dir() -> std::path::PathBuf {
        std::env::var("HIGGS_ESCHA_TEST_MODEL")
            .unwrap_or_else(|_| {
                format!(
                    "{}/AI-Models/escha-subset",
                    std::env::var("HOME").unwrap_or_default()
                )
            })
            .into()
    }

    fn pseudo_random(len: usize, seed: u64) -> Vec<u16> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (state >> 33) as u16
            })
            .collect()
    }

    /// The owned tensors of one synthetic projection.
    struct OwnedGroup {
        code: Array,
        config: Array,
        rin: Array,
        rout: Array,
        s_in: Array,
        s_out: Array,
    }

    impl OwnedGroup {
        const fn as_group(&self) -> TrellisGroup<'_> {
            TrellisGroup {
                prefix: "synthetic",
                code: &self.code,
                config: &self.config,
                rin: &self.rin,
                rout: &self.rout,
                s_in: &self.s_in,
                s_out: &self.s_out,
            }
        }
    }

    /// Make one synthetic projection. `experts` selects the layout: `None`
    /// gives the dense rank-3 layout, `Some(e)` gives the `[e, ...]` layout.
    /// The scale values vary by channel with a zero-mean offset pattern.
    fn synth_group(experts: Option<i32>, k: usize, in_f: i32, out_f: i32) -> OwnedGroup {
        const OFFSETS: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let s = spec(k, in_f, out_f);
        let (tk, tn) = s.tiles();
        let words = s.words_per_tile();
        let e = experts.unwrap_or(1);
        let n = e as usize * tk * tn * words;
        let signed: Vec<i16> = pseudo_random(n, 0xDA7A)
            .iter()
            .map(|&c| c.cast_signed())
            .collect();
        let mut dims = vec![
            i32::try_from(tk).unwrap(),
            i32::try_from(tn).unwrap(),
            i32::try_from(words).unwrap(),
        ];
        if experts.is_some() {
            dims.insert(0, e);
        }
        let channel = |len: i32, step: f32| -> Array {
            let vals: Vec<f32> = (0..e * len)
                .map(|i| OFFSETS[(i % 5) as usize].mul_add(step, 1.0))
                .collect();
            let shape = if experts.is_some() {
                vec![e, len]
            } else {
                vec![len]
            };
            Array::from_slice(&vals, &shape)
        };
        let k_i32 = i32::try_from(k).unwrap();
        OwnedGroup {
            code: Array::from_slice(&signed, &dims),
            config: Array::from_slice(
                &[
                    16,
                    k_i32,
                    k_i32,
                    1,
                    experts.unwrap_or(0),
                    in_f,
                    out_f,
                    in_f,
                    out_f,
                ],
                &[9],
            ),
            rin: channel(in_f, 0.125),
            rout: channel(out_f, 0.0625),
            s_in: channel(in_f, 0.25),
            s_out: channel(out_f, 0.5),
        }
    }

    /// Test the basic property of a circular window of bits.
    ///
    /// Two adjacent codes come from the same bit stream. The distance between
    /// them is `K` bits. Thus the two codes must have the same `16 - K` bits
    /// in common. This is true for all 256 codes, and the last code connects
    /// to the first code.
    ///
    /// This property is true for all packed data. Thus the test does not need
    /// a reference. An error of one bit in the offset breaks the property. The
    /// test `unpack_tile_matches_reference` uses data from a checkpoint.
    #[test]
    fn unpack_tile_windows_overlap_circularly() {
        for k in [2usize, 3, 4] {
            let packed = pseudo_random(TILE * k, 0x5EED ^ k as u64);
            let mut codes = [0u16; TILE_ELEMS];
            unpack_tile(&packed, k, &mut codes);

            let shared = (1u16 << (16 - k)) - 1;
            for i in 0..TILE_ELEMS {
                let next = codes[(i + 1) % TILE_ELEMS];
                assert_eq!(
                    codes[i] & shared,
                    next >> k,
                    "K={k}: codes {i} and {} do not share their {} overlapping bits",
                    (i + 1) % TILE_ELEMS,
                    16 - k
                );
            }
            assert!(
                codes.iter().any(|&c| c != codes[0]),
                "K={k}: decode produced a constant stream"
            );
        }
    }

    /// Test the decode with known data from the released checkpoint.
    ///
    /// The data is tile [0, 0] of expert 0 of the tensor
    /// `layers.0.mlp.experts.gate_up_proj`, with K equal to 2. The tool
    /// `tools/escha_ref.py` gives the codes and the codebook values. A test
    /// compared that tool with the unquantized model. The cosine value was
    /// 0.97.
    #[test]
    fn unpack_tile_matches_reference() {
        const PACKED: [u16; 32] = [
            17306, 23673, 17721, 8298, 30284, 37780, 11190, 65313, 62638, 7569, 24420, 46439,
            33879, 51555, 44741, 64370, 7988, 27160, 22813, 57471, 57599, 33229, 54809, 28588,
            45070, 5627, 62389, 10510, 41066, 20219, 12929, 50458,
        ];
        const EXPECT_CODES: [u16; 8] = [51717, 10261, 41047, 33116, 1393, 5575, 22302, 23673];
        const EXPECT_VALS: [f32; 8] = [
            -0.998_169, -1.437_866, -0.343_384, -1.734_619, 0.371_948, 1.424_805, 0.638_672,
            0.843_262,
        ];

        let mut codes = [0u16; TILE_ELEMS];
        unpack_tile(&PACKED, 2, &mut codes);
        assert_eq!(&codes[..8], &EXPECT_CODES, "trellis codes diverge");

        for (i, (&code, &want)) in codes.iter().zip(EXPECT_VALS.iter()).enumerate() {
            let got = decode_3inst(code);
            assert!(
                (got - want).abs() < 1e-5,
                "codebook value {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn tile_perm_is_a_permutation() {
        let perm = tile_perm();
        let mut seen = [false; TILE_ELEMS];
        for &p in &perm {
            assert!(!seen[usize::from(p)], "duplicate index {p}");
            seen[usize::from(p)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn decode_3inst_is_zero_mean_and_bounded() {
        // The codebook values must have an approximate Gaussian distribution.
        // The mean must be near zero, and the values must stay in a small
        // range. If not, the trellis search cannot give good results.
        let mut sum = 0.0f64;
        let mut max = 0.0f32;
        for code in 0..=u16::MAX {
            let v = decode_3inst(code);
            assert!(v.is_finite(), "code {code} decoded to {v}");
            sum += f64::from(v);
            max = max.max(v.abs());
        }
        let mean = sum / f64::from(u32::from(u16::MAX) + 1);
        assert!(mean.abs() < 0.02, "codebook mean {mean} not centred");
        assert!(
            (1.0..8.0).contains(&max),
            "codebook range {max} implausible"
        );
    }

    #[test]
    fn spec_rejects_padded_and_unaligned_shapes() {
        assert!(EschaSpec::from_config(&[16, 2, 2, 1, 256, 2048, 1024, 2048, 1024]).is_ok());
        // Padded dims are not handled.
        assert!(EschaSpec::from_config(&[16, 2, 2, 1, 256, 2000, 1024, 2048, 1024]).is_err());
        // in_features not a multiple of the Hadamard block.
        assert!(EschaSpec::from_config(&[16, 2, 2, 1, 256, 2032, 1024, 2032, 1024]).is_err());
        // Non-MCG codebook.
        assert!(EschaSpec::from_config(&[16, 2, 2, 0, 256, 2048, 1024, 2048, 1024]).is_err());
        // Wrong arity.
        assert!(EschaSpec::from_config(&[16, 2, 2, 1]).is_err());
    }

    #[test]
    fn spec_derives_tile_counts() {
        let s = EschaSpec::from_config(&[16, 3, 3, 1, 256, 512, 2048, 512, 2048]).unwrap();
        assert_eq!(s.k, 3);
        assert_eq!(s.tiles(), (32, 128));
        assert_eq!(s.words_per_tile(), 48);
    }

    /// Test the Hadamard operation on the two axes.
    ///
    /// An orthonormal Hadamard is its own inverse. Thus two calls to
    /// `had_blockwise` on one axis must give the input again. The test finds
    /// an error in the shape change, in the axis order, or in the scale.
    #[test]
    fn had_blockwise_is_an_involution_on_both_axes() {
        let n = (HAD_BLOCK * 2) as usize;
        let data: Vec<f32> = pseudo_random(n * n, 19)
            .into_iter()
            .map(|v| f32::from(v) / 32768.0 - 1.0)
            .collect();
        let x = Array::from_slice(&data, &[HAD_BLOCK * 2, HAD_BLOCK * 2]);

        for axis in [0, 1] {
            let back = had_blockwise(&had_blockwise(&x, axis).unwrap(), axis).unwrap();
            let err = back
                .subtract(&x)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(err < 1e-4, "axis {axis} round-trip error {err}");
        }
    }

    /// Test the axis of each scale and the transpose step.
    ///
    /// The function `dequant_expert` gives an `[out, in]` matrix. The vectors
    /// `rin` and `s_in` scale the input axis. The vectors `rout` and `s_out`
    /// scale the output axis. The test sets one scale at a time. An incorrect
    /// axis or an absent transpose gives an incorrect shape or an incorrect
    /// column.
    #[test]
    fn dequant_expert_orients_scales_and_transpose() {
        let (in_f, out_f) = (HAD_BLOCK * 2, HAD_BLOCK);
        let s = spec(2, in_f, out_f);
        let (tk, tn) = s.tiles();
        let codes = pseudo_random(tk * tn * s.words_per_tile(), 23);
        let code = code_array(&codes, &s);

        let ones_in = Array::from_slice(&vec![1.0f32; in_f as usize], &[in_f]);
        let ones_out = Array::from_slice(&vec![1.0f32; out_f as usize], &[out_f]);
        let base = dequant_expert(&code, &ones_in, &ones_out, &ones_in, &ones_out, &s).unwrap();
        assert_eq!(base.shape(), [out_f, in_f], "result must be [out, in]");

        // If the scale of one input channel is 2.0, only that column must
        // change by a factor of 2.0.
        let mut rin = vec![1.0f32; in_f as usize];
        rin[3] = 2.0;
        let scaled = dequant_expert(
            &code,
            &Array::from_slice(&rin, &[in_f]),
            &ones_out,
            &ones_in,
            &ones_out,
            &s,
        )
        .unwrap();
        let ratio = scaled.divide(&base).unwrap();
        let col = ratio.index((.., 3)).mean(None).unwrap().item::<f32>();
        let other = ratio.index((.., 4)).mean(None).unwrap().item::<f32>();
        assert!((col - 2.0).abs() < 1e-3, "input scale hit column 3: {col}");
        assert!((other - 1.0).abs() < 1e-3, "column 4 disturbed: {other}");

        // The f32 scales must compose with the f16 ones, not be ignored.
        let mut s_out = vec![1.0f32; out_f as usize];
        s_out[5] = 3.0;
        let scaled_out = dequant_expert(
            &code,
            &ones_in,
            &ones_out,
            &ones_in,
            &Array::from_slice(&s_out, &[out_f]),
            &s,
        )
        .unwrap();
        let row = scaled_out
            .divide(&base)
            .unwrap()
            .index((5, ..))
            .mean(None)
            .unwrap()
            .item::<f32>();
        assert!((row - 3.0).abs() < 1e-3, "s_out scale hit row 5: {row}");
    }

    /// Test all decode steps together against the `NumPy` reference. The test
    /// uses data from a checkpoint.
    ///
    /// The steps are the bit decode, the codebook, the tile order, the two
    /// Hadamard operations, the two scale pairs, and the transpose.
    ///
    /// To make the data file, use the command
    /// `python3 tools/escha_ref.py fixture`. If the file is not available, the
    /// test stops with no error. Thus a new copy of the repository passes.
    #[test]
    fn dequant_expert_matches_numpy_reference() {
        const FIXTURE: &str = "/tmp/escha_fixture.safetensors";
        if !std::path::Path::new(FIXTURE).exists() {
            eprintln!("skipping: {FIXTURE} absent (python3 tools/escha_ref.py fixture)");
            return;
        }
        let t = Array::load_safetensors(FIXTURE).unwrap();
        let get = |k: &str| t.get(k).unwrap_or_else(|| panic!("fixture missing {k}"));

        let expected = get("expected");
        let (out_f, in_f) = (expected.shape()[0], expected.shape()[1]);
        let s = EschaSpec {
            k: get("code").shape()[2] as usize / TILE,
            mcg: true,
            num_experts: 1,
            in_features: in_f,
            out_features: out_f,
        };

        let got = dequant_expert(
            get("code"),
            get("rin"),
            get("rout"),
            get("s_in"),
            get("s_out"),
            &s,
        )
        .unwrap();
        assert_eq!(got.shape(), expected.shape());

        // The f16 format of the codebook values causes most of the error.
        // Thus the test compares the error with the range of the reference.
        let diff = got.subtract(expected).unwrap().abs().unwrap();
        let max_err = diff.max(None).unwrap().item::<f32>();
        let scale = expected.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(
            max_err < 1e-3 * scale,
            "max |diff| {max_err} exceeds 1e-3 of reference range {scale}"
        );
    }

    #[test]
    fn dequant_expert_rejects_mismatched_code_shape() {
        let s = spec(2, HAD_BLOCK, HAD_BLOCK);
        let bad = Array::from_slice(&vec![0i16; 8 * 8 * 16], &[8, 8, 16]);
        let ones = Array::from_slice(&vec![1.0f32; HAD_BLOCK as usize], &[HAD_BLOCK]);
        assert!(dequant_expert(&bad, &ones, &ones, &ones, &ones, &s).is_err());
    }

    /// Test that the offset applies to every norm the checkpoint holds and to
    /// nothing else. The `w - 1` convention was measured against
    /// `mlx-community/Qwen3.6-35B-A3B-4bit`: 101 tensors match after the
    /// offset, the 30 `linear_attn.norm` tensors match without it.
    #[test]
    fn is_offset_rmsnorm_splits_the_norms_from_everything_else() {
        // Every norm key shape the checkpoint holds, in the normalized form.
        // The GDN gated norm is the one norm that keeps the plain weight.
        let offset = [
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.39.post_attention_layernorm.weight",
            "language_model.model.layers.3.self_attn.q_norm.weight",
            "language_model.model.layers.3.self_attn.k_norm.weight",
            "language_model.model.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ];
        let plain = [
            "language_model.model.layers.0.linear_attn.norm.weight",
            "language_model.model.layers.0.linear_attn.A_log",
            "language_model.model.layers.0.linear_attn.dt_bias",
            "language_model.model.layers.0.linear_attn.conv1d.weight",
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.mlp.shared_expert_gate.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight_int8",
            "language_model.model.embed_tokens.weight_int8",
            "lm_head.weight_int8",
        ];
        for key in offset {
            assert!(is_offset_rmsnorm(key), "{key} holds w - 1");
        }
        for key in plain {
            assert!(!is_offset_rmsnorm(key), "{key} holds the plain value");
        }
    }

    /// Test that each key format of the checkpoint gives a correct parameter
    /// path. The key names come from `model.safetensors.index.json`.
    #[test]
    fn normalize_key_maps_every_checkpoint_shape() {
        let cases = [
            (
                "model.language_model.layers.0.mlp.experts.gate_up_proj.escha_code",
                "model.layers.0.mlp.experts.gate_up_proj.escha_code",
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight_int8",
                "model.layers.3.self_attn.q_proj.weight_int8",
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
                "model.layers.0.linear_attn.in_proj_qkv.weight_scale",
            ),
            (
                "model.language_model.embed_tokens.weight_int8",
                "model.embed_tokens.weight_int8",
            ),
            ("model.language_model.norm.weight", "model.norm.weight"),
            ("lm_head.weight_int8", "lm_head.weight_int8"),
            // MTP sidecar keys are already in the expected form.
            ("mtp.fc.weight", "mtp.fc.weight"),
            (
                "mtp.pre_fc_norm_hidden.weight",
                "mtp.pre_fc_norm_hidden.weight",
            ),
        ];
        for (raw, want) in cases {
            let normalized = normalize_key(raw);
            let stripped = normalized
                .strip_prefix("language_model.")
                .unwrap_or(&normalized);
            assert_eq!(stripped, want, "{raw}");
        }
    }

    #[test]
    fn classify_splits_storage_and_recovers_prefix() {
        let proj = "model.layers.0.mlp.experts.down_proj";
        for suffix in ESCHA_SUFFIXES {
            assert_eq!(
                classify(&format!("{proj}{suffix}")),
                (Storage::Trellis, proj)
            );
        }
        let attn = "model.layers.3.self_attn.q_proj";
        for suffix in [INT8_WEIGHT_SUFFIX, INT8_SCALE_SUFFIX] {
            assert_eq!(classify(&format!("{attn}{suffix}")), (Storage::Int8, attn));
        }
        // A plain tensor keeps its whole key, and `.weight` must not be
        // mistaken for the int8 pair.
        let norm = "model.layers.0.input_layernorm.weight";
        assert_eq!(classify(norm), (Storage::Dense, norm));
        let gate = "model.layers.0.mlp.gate.weight";
        assert_eq!(classify(gate), (Storage::Dense, gate));
    }

    #[test]
    fn is_eschamoe_checkpoint_reads_either_config() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_eschamoe_checkpoint(dir.path()).unwrap());

        let write = |name: &str, body: &str| {
            std::fs::write(dir.path().join(name), body).unwrap();
        };
        write(
            "config.json",
            r#"{"quantization_config":{"quant_method":"eschamoe"}}"#,
        );
        assert!(is_eschamoe_checkpoint(dir.path()).unwrap());

        // quantize_config.json wins when both are present.
        write("quantize_config.json", r#"{"quant_method":"awq"}"#);
        assert!(!is_eschamoe_checkpoint(dir.path()).unwrap());
        write(
            "quantize_config.json",
            r#"{"quant_method":"eschamoe","bits":2.0}"#,
        );
        assert!(is_eschamoe_checkpoint(dir.path()).unwrap());

        // Malformed config surfaces as an error rather than a silent false.
        write("quantize_config.json", "not json");
        assert!(is_eschamoe_checkpoint(dir.path()).is_err());
    }

    #[test]
    fn affine_target_resolves_overrides() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{"group_size": 64, "bits": 4, "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}}"#,
        )
        .unwrap();
        let base =
            AffineTarget::resolve(Some(&cfg), "language_model.model.layers.0.self_attn.q_proj");
        assert_eq!(
            base,
            AffineTarget {
                group_size: 64,
                bits: 4
            }
        );

        let gate = AffineTarget::resolve(Some(&cfg), "language_model.model.layers.0.mlp.gate");
        assert_eq!(
            gate,
            AffineTarget {
                group_size: 64,
                bits: 8
            }
        );

        // No quantization block at all falls back to the MLX default.
        assert_eq!(
            AffineTarget::resolve(None, "anything"),
            AffineTarget::default()
        );
    }

    /// Test the conversion of a trellis projection from a checkpoint.
    ///
    /// The test decodes the affine result and compares it with the output of
    /// `dequant_expert`. It does this for the two parts of the `gate_up`
    /// tensor.
    ///
    /// To make the checkpoint, use the command
    /// `python3 tools/escha_subset.py <dir> 4 32`. If the checkpoint is not
    /// available, the test stops with no error.
    #[test]
    fn convert_trellis_round_trips_through_affine() {
        let dir = std::env::var("HIGGS_ESCHA_TEST_MODEL").unwrap_or_else(|_| {
            format!(
                "{}/AI-Models/escha-subset",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let path = std::path::Path::new(&dir).join("model.safetensors");
        if !path.exists() {
            eprintln!(
                "skipping: {} absent (tools/escha_subset.py)",
                path.display()
            );
            return;
        }
        let t = Array::load_safetensors(path.to_str().unwrap_or_default()).unwrap();
        let pfx = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let get = |s: &str| t.get(&format!("{pfx}{s}")).unwrap();
        let group = TrellisGroup {
            prefix: pfx,
            code: get(".escha_code"),
            config: get(".escha_config"),
            rin: get(".escha_rin"),
            rout: get(".escha_rout"),
            s_in: get(".escha_s_in"),
            s_out: get(".escha_s_out"),
        };

        let spec = group.spec().unwrap();
        let target = AffineTarget::default();
        let parts = convert_trellis(group, 2, target).unwrap();
        assert_eq!(parts.len(), 2, "fused gate_up yields gate and up");

        let rows = spec.out_features / 2;
        let reference = dequant_expert(
            &group.code.index(0),
            &group.rin.index(0),
            &group.rout.index(0),
            &group.s_in.index(0),
            &group.s_out.index(0),
            &spec,
        )
        .unwrap();

        for (part, [w, s, b]) in parts.iter().enumerate() {
            assert_eq!(
                w.shape()[0],
                spec.num_experts,
                "part {part} lost the expert axis"
            );

            let got = ops::dequantize(
                w.index(0),
                s.index(0),
                &b.index(0),
                target.group_size,
                target.bits,
            )
            .unwrap();

            let lo = rows * i32::try_from(part).unwrap();
            let want = reference.index((lo..lo + rows, ..));
            assert_eq!(got.shape(), want.shape());

            // The source data has 2 bits, and the result has 4 bits. Thus the
            // quantization error must be much less than the reference range.
            let err = got
                .subtract(&want)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let scale = want.abs().unwrap().max(None).unwrap().item::<f32>();
            eprintln!(
                "part {part}: affine requant err {err:.6} range {scale:.6} rel {:.3e}",
                err / scale
            );
            assert!(
                err < 0.1 * scale,
                "part {part}: requantization error {err} vs range {scale}"
            );
        }
    }

    /// Load a real eschamoe checkpoint and do one forward operation.
    ///
    /// This test covers the full chain: the format detection, the key change,
    /// the conversion of the trellis tensors and the int8 tensors, the GDN
    /// fusion, and the forward code. It stops with no error if the checkpoint
    /// is not available.
    #[test]
    fn eschamoe_checkpoint_loads_and_runs_forward() {
        let dir = test_model_dir();
        // A checkpoint can hold one file or a set of shards with an index.
        let present = dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists();
        if !present {
            eprintln!("skipping: no checkpoint at {}", dir.display());
            return;
        }
        eprintln!("loading {}", dir.display());
        let mut model = crate::qwen3_next::load_qwen3_5_moe_model(&dir)
            .unwrap_or_else(|e| panic!("load failed: {e}"));

        let tokens = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]);
        let mut cache = model.make_cache();
        let logits = model
            .forward(&tokens, None, &mut cache)
            .unwrap_or_else(|e| panic!("forward failed: {e}"));

        // `forward` returns the logits of the last position only.
        let vocab = *logits.shape().last().unwrap();
        assert_eq!(logits.shape(), [1, 1, vocab]);
        let peak = logits.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(peak.is_finite() && peak > 0.0, "logits degenerate: {peak}");
    }

    #[test]
    fn convert_trellis_rejects_uneven_split() {
        let dir = std::env::var("HIGGS_ESCHA_TEST_MODEL").unwrap_or_else(|_| {
            format!(
                "{}/AI-Models/escha-subset",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        let path = std::path::Path::new(&dir).join("model.safetensors");
        if !path.exists() {
            return;
        }
        let t = Array::load_safetensors(path.to_str().unwrap_or_default()).unwrap();
        let pfx = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let get = |s: &str| t.get(&format!("{pfx}{s}")).unwrap();
        let group = TrellisGroup {
            prefix: pfx,
            code: get(".escha_code"),
            config: get(".escha_config"),
            rin: get(".escha_rin"),
            rout: get(".escha_rout"),
            s_in: get(".escha_s_in"),
            s_out: get(".escha_s_out"),
        };
        assert!(convert_trellis(group, 0, AffineTarget::default()).is_err());
        assert!(convert_trellis(group, 3, AffineTarget::default()).is_err());
    }

    #[test]
    fn dequant_int8_scales_per_output_row() {
        let w = Array::from_slice::<i8>(&[1, 2, -3, 4], &[2, 2]);
        let s = Array::from_slice::<f32>(&[0.5, 2.0], &[2]);
        let got = dequant_int8(&w, &s).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[0.5, 1.0, -6.0, 8.0]);
        // Mismatched scale length must be rejected, not broadcast.
        assert!(dequant_int8(&w, &Array::from_slice::<f32>(&[0.5], &[1])).is_err());
    }

    #[test]
    fn decode_expert_codes_fills_every_slot() {
        // The decode must write to all positions of all tiles. An error in
        // the element order leaves some positions at zero.
        let s = spec(2, 32, 48);
        let (tk, tn) = s.tiles();
        let packed = pseudo_random(tk * tn * s.words_per_tile(), 11);
        let w = decode_expert_codes(&packed, &s);

        assert_eq!(w.len(), (s.in_features * s.out_features) as usize);
        assert!(
            w.iter().all(|v| v.to_f32() != 0.0),
            "unfilled slots present"
        );
    }

    #[test]
    fn decode_expert_codes_places_tiles_by_position() {
        // Tile (tk, tn) must go to row `16*tk` and column `16*tn`. Thus one
        // tile with different codes must change only one block of 16 by 16.
        // The test finds an error in the change from [tk, tn, 16, 16] to
        // [in, out].
        let s = spec(2, 32, 32);
        let words = s.words_per_tile();
        let mut packed = vec![0u16; 4 * words];
        packed[words..2 * words].copy_from_slice(&pseudo_random(words, 3)); // tile (0,1)
        let w = decode_expert_codes(&packed, &s);

        let row_len = s.out_features as usize;
        let block = |r0: usize, c0: usize| -> Vec<f32> {
            (0..TILE)
                .flat_map(|r| (0..TILE).map(move |c| (r, c)))
                .map(|(r, c)| w[(r0 + r) * row_len + c0 + c].to_f32())
                .collect()
        };
        let uniform = |v: &[f32]| v.iter().all(|x| (x - v[0]).abs() < 1e-6);

        assert!(!uniform(&block(0, 16)), "tile (0,1) should carry the data");
        for (r0, c0) in [(0, 0), (16, 0), (16, 16)] {
            assert!(uniform(&block(r0, c0)), "tile at ({r0},{c0}) was disturbed");
        }
    }

    /// Test the dense decode path against the expert path.
    ///
    /// A dense projection is one matrix. The same data with a leading axis of
    /// one expert must give the same values. The dense result must have no
    /// expert axis.
    #[test]
    fn convert_trellis_dense_matches_batch_of_one() {
        let (in_f, out_f) = (HAD_BLOCK, HAD_BLOCK);
        let dense = synth_group(None, 2, in_f, out_f);
        let moe = synth_group(Some(1), 2, in_f, out_f);
        let target = AffineTarget::default();

        let layout = dense.as_group().validate().unwrap();
        assert_eq!(layout.expert_axis, ExpertAxis::Dense);
        assert_eq!(layout.scale_len_in, ScaleLen::Logical);
        assert_eq!(layout.scale_len_out, ScaleLen::Logical);

        let dense_parts = convert_trellis(dense.as_group(), 1, target).unwrap();
        let moe_parts = convert_trellis(moe.as_group(), 1, target).unwrap();
        assert_eq!(dense_parts.len(), 1);

        for (i, (d, m)) in dense_parts[0].iter().zip(moe_parts[0].iter()).enumerate() {
            assert_eq!(d.ndim() + 1, m.ndim(), "tensor {i} kept an expert axis");
            let err = d
                .as_dtype(Dtype::Float32)
                .unwrap()
                .subtract(m.index(0).as_dtype(Dtype::Float32).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(err < 1e-6, "tensor {i} diverges from batch-of-1: {err}");
        }
    }

    /// Test the load checks: the axis agreement, the leading dim, the bit
    /// budget, and the scale lengths. Each case must fail with an error that
    /// names the bad value.
    #[test]
    fn validate_rejects_axis_budget_and_scale_mismatches() {
        let base_config = |experts: i32| {
            Array::from_slice(
                &[
                    16, 2, 2, 1, experts, HAD_BLOCK, HAD_BLOCK, HAD_BLOCK, HAD_BLOCK,
                ],
                &[9],
            )
        };

        // The config gives four experts, but the code tensor has rank 3.
        let mut claims_experts = synth_group(None, 2, HAD_BLOCK, HAD_BLOCK);
        claims_experts.config = base_config(4);
        let axis_msg = claims_experts
            .as_group()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            axis_msg.contains("4 experts") && axis_msg.contains("rank 3"),
            "{axis_msg}"
        );

        // The config gives zero experts, but the code tensor has rank 4.
        let mut claims_dense = synth_group(Some(2), 2, HAD_BLOCK, HAD_BLOCK);
        claims_dense.config = base_config(0);
        let rank_msg = claims_dense.as_group().validate().unwrap_err().to_string();
        assert!(
            rank_msg.contains("0 experts") && rank_msg.contains("rank 4"),
            "{rank_msg}"
        );

        // The code leading dim must equal the expert count.
        let mut wrong_lead = synth_group(Some(2), 2, HAD_BLOCK, HAD_BLOCK);
        wrong_lead.config = base_config(3);
        let lead_msg = wrong_lead.as_group().validate().unwrap_err().to_string();
        assert!(
            lead_msg.contains("leading dim 2") && lead_msg.contains('3'),
            "{lead_msg}"
        );

        // A short code tensor breaks the bit budget.
        let mut short_code = synth_group(None, 2, HAD_BLOCK, HAD_BLOCK);
        short_code.code = short_code.code.index((.., .., 0..16));
        let bits_msg = short_code.as_group().validate().unwrap_err().to_string();
        assert!(
            bits_msg.contains("escha_code") && bits_msg.contains("bits"),
            "{bits_msg}"
        );

        // A wrong scale length names the vector.
        let mut long_rin = synth_group(None, 2, HAD_BLOCK, HAD_BLOCK);
        let len = HAD_BLOCK + TILE_I32;
        long_rin.rin = Array::from_slice(&vec![1.0f32; len as usize], &[len]);
        let rin_msg = long_rin.as_group().validate().unwrap_err().to_string();
        assert!(rin_msg.contains("escha_rin"), "{rin_msg}");
    }

    /// Test the codebook gate. The flags 0 and 2 exist but have no verified
    /// decode data. An unknown flag gives a different error.
    #[test]
    fn spec_rejects_unverified_and_unknown_codebooks() {
        for flag in [0, 2] {
            let msg = EschaSpec::from_config(&[16, 2, 2, flag, 256, 2048, 1024, 2048, 1024])
                .unwrap_err()
                .to_string();
            assert!(msg.contains("unverified"), "flag {flag}: {msg}");
        }
        let msg = EschaSpec::from_config(&[16, 2, 2, 7, 256, 2048, 1024, 2048, 1024])
            .unwrap_err()
            .to_string();
        assert!(msg.contains("unknown"), "{msg}");
    }

    /// Test the `|rout|` mean check. A mean of 1.0 passes. A large deviation
    /// gives the mean back, and the loader warns without failure.
    #[test]
    fn rout_mean_check_warns_and_does_not_fail() {
        let good = Array::from_slice(&[1.04f32, -0.96, 1.0, -1.0], &[4]);
        assert!(rout_mean_if_off(&good).unwrap().is_none());

        let bad = Array::from_slice(&[2.0f32; 8], &[8]);
        let mean = rout_mean_if_off(&bad).unwrap().unwrap();
        assert!((mean - 2.0).abs() < 1e-6, "reported mean {mean}");

        // The check warns only. A group with a large mean still validates.
        let mut off = synth_group(None, 2, HAD_BLOCK, HAD_BLOCK);
        off.rout = Array::from_slice(&vec![2.0f32; HAD_BLOCK as usize], &[HAD_BLOCK]);
        assert!(off.as_group().validate().is_ok());
    }

    /// Test the activation-side factorization of the dequant equation.
    ///
    /// The reconstruction is `W = (H.Ŵ.H * rin[:,None] * rout[None,:]).T`.
    /// `H` is symmetric and orthonormal. Thus `y = x @ W.T` factors to:
    /// scale `x` by `rin*s_in`, apply the blockwise Hadamard, multiply by
    /// `Ŵ`, apply the blockwise Hadamard again, and scale by `rout*s_out`.
    /// The Phase 3 kernel uses this form. The Hadamard never touches the
    /// weights at inference.
    #[test]
    fn factored_matvec_matches_dequant_expert() {
        let (in_f, out_f) = (HAD_BLOCK * 2, HAD_BLOCK);
        let g = synth_group(None, 2, in_f, out_f);
        let s = g.as_group().spec().unwrap();

        // The reference: y_ref = x @ W.T with the oracle weight.
        let w = dequant_expert(&g.code, &g.rin, &g.rout, &g.s_in, &g.s_out, &s).unwrap();
        let x_vals: Vec<f32> = pseudo_random(in_f as usize, 0xFAC7)
            .into_iter()
            .map(|v| f32::from(v) / 32768.0 - 1.0)
            .collect();
        let x = Array::from_slice(&x_vals, &[1, in_f]);
        let y_ref = ops::matmul(&x, ops::swap_axes(&w, 0, 1).unwrap()).unwrap();

        // The factored form. `Ŵ` comes from the same trellis decode.
        let packed: Vec<u16> = g
            .code
            .as_dtype(Dtype::Uint16)
            .unwrap()
            .as_slice::<u16>()
            .to_vec();
        let w_hat = Array::from_slice(&decode_expert_codes(&packed, &s), &[in_f, out_f]);
        let su = g
            .rin
            .multiply(&g.s_in)
            .unwrap()
            .reshape(&[1, in_f])
            .unwrap();
        let sv = g
            .rout
            .multiply(&g.s_out)
            .unwrap()
            .reshape(&[1, out_f])
            .unwrap();
        let xh = had_blockwise(&x.multiply(&su).unwrap(), -1).unwrap();
        let y_pre = ops::matmul(&xh, &w_hat).unwrap();
        let y = had_blockwise(&y_pre, -1).unwrap().multiply(&sv).unwrap();

        let err = y
            .subtract(&y_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        let scale = y_ref.abs().unwrap().max(None).unwrap().item::<f32>();
        eprintln!("factorization: max |y - y_ref| = {err} (scale {scale})");
        assert!(
            err <= 2e-3 * scale,
            "factored form diverges: {err} vs scale {scale}"
        );
    }

    /// The native gather forward must match the oracle on both row paths.
    ///
    /// The matvec path takes six rows with unsorted ids. The scratch path
    /// takes 40 sorted rows. Both compare against `dequant_expert` row by
    /// row.
    #[test]
    fn escha_proj_gather_forward_matches_oracle() {
        let (in_f, out_f) = (HAD_BLOCK * 2, HAD_BLOCK);
        let g = synth_group(Some(4), 2, in_f, out_f);
        let s = g.as_group().spec().unwrap();
        let proj = EschaProj::new(g.code.clone(), &g.rin, &g.rout, &g.s_in, &g.s_out, s).unwrap();

        let slice = |t: &Array, e: u32| {
            let sel = Array::from_slice(&[e], &[1]);
            t.take_axis(&sel, 0).unwrap().squeeze_axes(&[0]).unwrap()
        };
        let oracle = |x: &Array, ids: &[u32]| {
            let rows: Vec<Array> = ids
                .iter()
                .enumerate()
                .map(|(i, &e)| {
                    let w = dequant_expert(
                        &slice(&g.code, e),
                        &slice(&g.rin, e),
                        &slice(&g.rout, e),
                        &slice(&g.s_in, e),
                        &slice(&g.s_out, e),
                        &s,
                    )
                    .unwrap();
                    let sel = Array::from_slice(&[u32::try_from(i).unwrap()], &[1]);
                    let xi = x.take_axis(&sel, 0).unwrap();
                    ops::matmul(&xi, ops::swap_axes(&w, 0, 1).unwrap()).unwrap()
                })
                .collect();
            let refs: Vec<&Array> = rows.iter().collect();
            ops::concatenate_axis(&refs, 0).unwrap()
        };
        let check = |ids: &[u32], label: &str| {
            let n = i32::try_from(ids.len()).unwrap();
            let x_vals: Vec<f32> = pseudo_random(ids.len() * in_f as usize, 0x9A7E)
                .into_iter()
                .map(|v| f32::from(v) / 32768.0 - 1.0)
                .collect();
            let x = Array::from_slice(&x_vals, &[n, in_f]);
            let eids = Array::from_slice(ids, &[n]);
            let got = proj.gather_forward(&x, &eids).unwrap();
            assert_eq!(got.shape(), [n, out_f], "{label}");
            let want = oracle(&x, ids);
            let err = got
                .subtract(&want)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let scale = want.abs().unwrap().max(None).unwrap().item::<f32>();
            eprintln!("{label}: max |native - oracle| = {err} (scale {scale})");
            assert!(err <= 2e-3 * scale, "{label}: {err} vs scale {scale}");
        };

        // Six rows, unsorted ids: the matvec kernel path.
        check(&[2, 0, 3, 1, 2, 0], "matvec path");
        // Forty sorted rows: the scratch decode path.
        let sorted: Vec<u32> = (0..40u32).map(|i| i / 10).collect();
        check(&sorted, "scratch path");
    }

    /// Compare the GPU tile decode with the CPU reference for one spec.
    ///
    /// The gate is one f16 ULP at the value scale of the reference. The two
    /// paths round the same f32 value to f16. Thus the expected difference
    /// is zero.
    fn assert_kernel_matches_cpu(packed: &[u16], s: &EschaSpec, label: &str) {
        let code = code_array(packed, s);
        let want = decode_expert_codes(packed, s);
        let got = crate::metal_kernel::eschamoe_dequant_tiles(&code, s).unwrap();
        assert_eq!(got.shape(), [s.in_features, s.out_features], "{label}");
        assert_eq!(got.dtype(), Dtype::Float16, "{label}");

        let got_v = got.as_slice::<f16>();
        assert_eq!(got_v.len(), want.len(), "{label}");
        let diff = got_v
            .iter()
            .zip(&want)
            .map(|(g, w)| (g.to_f32() - w.to_f32()).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
        let ulp = scale * f16::EPSILON.to_f32();
        eprintln!("{label}: max |gpu - cpu| = {diff} (scale {scale}, one ulp {ulp})");
        assert!(
            diff <= ulp,
            "{label}: diff {diff} exceeds one f16 ULP {ulp}"
        );
    }

    /// Compare the GPU decode with the CPU decode on random tiles.
    #[test]
    fn eschamoe_dequant_tiles_matches_cpu_for_k2_k3_k4() {
        for k in [2usize, 3, 4] {
            let s = spec(k, 32, 48);
            let (tk, tn) = s.tiles();
            let packed = pseudo_random(tk * tn * s.words_per_tile(), 0xE5C4 ^ k as u64);
            assert_kernel_matches_cpu(&packed, &s, &format!("K={k} random"));
        }
    }

    /// Compare the GPU decode with the CPU decode on checkpoint data.
    ///
    /// The data is expert 0 of `layers.0.mlp.experts.gate_up_proj`. If the
    /// checkpoint is not available, the test stops with no error.
    #[test]
    fn eschamoe_dequant_tiles_matches_cpu_on_checkpoint() {
        let path = test_model_dir().join("model.safetensors");
        if !path.exists() {
            eprintln!("skipping: {} absent", path.display());
            return;
        }
        let t = Array::load_safetensors(path.to_str().unwrap()).unwrap();
        let pfx = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let get = |sfx: &str| t.get(&format!("{pfx}{sfx}")).unwrap();

        let config: Vec<i32> = get(".escha_config")
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice()
            .to_vec();
        let s = EschaSpec::from_config(&config).unwrap();
        let packed: Vec<u16> = get(".escha_code")
            .index(0)
            .as_dtype(Dtype::Uint16)
            .unwrap()
            .as_slice()
            .to_vec();
        assert_kernel_matches_cpu(&packed, &s, "checkpoint expert 0");
    }

    /// Test the input gate of the GPU decode.
    ///
    /// The kernel must refuse an unverified codebook, a wrong shape, and a
    /// wrong dtype. A refusal is an error, not incorrect output.
    #[test]
    fn eschamoe_dequant_tiles_rejects_bad_inputs() {
        let s = spec(2, 32, 32);
        let (tk, tn) = s.tiles();
        let packed = pseudo_random(tk * tn * s.words_per_tile(), 0xBAD);
        let code = code_array(&packed, &s);

        let mut off_book = s;
        off_book.mcg = false;
        assert!(crate::metal_kernel::eschamoe_dequant_tiles(&code, &off_book).is_err());

        let short = code.index((.., .., 0..16));
        assert!(crate::metal_kernel::eschamoe_dequant_tiles(&short, &s).is_err());

        let wide = code.as_dtype(Dtype::Float32).unwrap();
        assert!(crate::metal_kernel::eschamoe_dequant_tiles(&wide, &s).is_err());
    }

    /// Build the transformed row and the CPU product of one expert.
    ///
    /// The row is `had(x * rin * s_in)`. The product is `row @ Ŵ`. The
    /// gather kernel must give the same product.
    fn factored_row_and_ref(
        x: &Array,
        code: &Array,
        rin: &Array,
        s_in: &Array,
        e: i32,
        s: &EschaSpec,
    ) -> (Vec<f32>, Vec<f32>) {
        let in_f = s.in_features;
        let su = rin
            .index(e)
            .multiply(s_in.index(e))
            .unwrap()
            .reshape(&[1, in_f])
            .unwrap();
        let xh = had_blockwise(&x.multiply(&su).unwrap(), -1).unwrap();
        let packed: Vec<u16> = code
            .index(e)
            .as_dtype(Dtype::Uint16)
            .unwrap()
            .as_slice()
            .to_vec();
        let w_hat = Array::from_slice(&decode_expert_codes(&packed, s), &[in_f, s.out_features]);
        let y = ops::matmul(&xh, &w_hat).unwrap();
        (xh.as_slice::<f32>().to_vec(), y.as_slice::<f32>().to_vec())
    }

    /// Compare the gather kernel with the factored CPU form.
    ///
    /// Four synthetic experts share one token. Each row carries the input
    /// scales and the input Hadamard of its expert. The expert order is
    /// not sorted. Thus the test also checks the expert-id indirection.
    #[test]
    fn eschamoe_gather_qmv_matches_factored_cpu() {
        let (in_f, out_f) = (HAD_BLOCK * 2, HAD_BLOCK);
        let g = synth_group(Some(4), 2, in_f, out_f);
        let s = g.as_group().spec().unwrap();
        let ids: [u32; 3] = [3, 0, 2];

        let x_vals: Vec<f32> = pseudo_random(in_f as usize, 0x9A7E)
            .into_iter()
            .map(|v| f32::from(v) / 32768.0 - 1.0)
            .collect();
        let x = Array::from_slice(&x_vals, &[1, in_f]);

        let mut xh_all = Vec::new();
        let mut y_ref = Vec::new();
        for &e in &ids {
            let (row, y) = factored_row_and_ref(&x, &g.code, &g.rin, &g.s_in, e.cast_signed(), &s);
            xh_all.extend(row);
            y_ref.extend(y);
        }
        let rows = i32::try_from(ids.len()).unwrap();
        let xh = Array::from_slice(&xh_all, &[rows, in_f]);
        let ids_arr = Array::from_slice(&ids, &[rows]);

        let got = crate::metal_kernel::eschamoe_gather_qmv(&xh, &g.code, &ids_arr, &s).unwrap();
        assert_eq!(got.shape(), [rows, out_f]);
        let got_v = got.as_slice::<f32>();
        let diff = got_v
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("gather_qmv synth: max |gpu - cpu| = {diff} (scale {scale})");
        assert!(
            diff <= 2e-3 * scale,
            "gather kernel diverges: {diff} vs scale {scale}"
        );
    }

    /// Compare the gather kernel with the CPU form on checkpoint data.
    ///
    /// The data is `layers.0.mlp.experts.gate_up_proj`. The rows hold raw
    /// random activations. Thus the test checks the kernel contract
    /// `y_pre = xh @ Ŵ` on real trellis bits. If the checkpoint is not
    /// available, the test stops with no error.
    #[test]
    fn eschamoe_gather_qmv_matches_cpu_on_checkpoint() {
        let path = test_model_dir().join("model.safetensors");
        if !path.exists() {
            eprintln!("skipping: {} absent", path.display());
            return;
        }
        let t = Array::load_safetensors(path.to_str().unwrap()).unwrap();
        let pfx = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let get = |sfx: &str| t.get(&format!("{pfx}{sfx}")).unwrap();

        let config: Vec<i32> = get(".escha_config")
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice()
            .to_vec();
        let s = EschaSpec::from_config(&config).unwrap();
        let code = get(".escha_code");
        let (in_f, out_f) = (s.in_features, s.out_features);
        let ids: [u32; 3] = [0, 5, 17];

        let x_vals: Vec<f32> = pseudo_random(in_f as usize, 0xC0DE)
            .into_iter()
            .map(|v| f32::from(v) / 32768.0 - 1.0)
            .collect();
        let x = Array::from_slice(&x_vals, &[1, in_f]);

        let mut xh_all = Vec::new();
        let mut y_ref: Vec<f32> = Vec::new();
        for &e in &ids {
            let packed: Vec<u16> = code
                .index(e.cast_signed())
                .as_dtype(Dtype::Uint16)
                .unwrap()
                .as_slice()
                .to_vec();
            let w_hat = Array::from_slice(&decode_expert_codes(&packed, &s), &[in_f, out_f]);
            let y = ops::matmul(&x, &w_hat).unwrap();
            xh_all.extend_from_slice(&x_vals);
            y_ref.extend(y.as_slice::<f32>());
        }
        let rows = i32::try_from(ids.len()).unwrap();
        let xh = Array::from_slice(&xh_all, &[rows, in_f]);
        let ids_arr = Array::from_slice(&ids, &[rows]);

        let got = crate::metal_kernel::eschamoe_gather_qmv(&xh, code, &ids_arr, &s).unwrap();
        assert_eq!(got.shape(), [rows, out_f]);
        let got_v = got.as_slice::<f32>();
        let diff = got_v
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("gather_qmv checkpoint: max |gpu - cpu| = {diff} (scale {scale})");
        assert!(
            diff <= 2e-3 * scale,
            "gather kernel diverges: {diff} vs scale {scale}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::print_stderr, clippy::unwrap_used)]
mod convert_dump {
    use super::*;

    /// Print the converted tensors of layer 0. Set `HIGGS_ESCHA_DUMP` to run.
    #[test]
    fn dump_converted_shapes() {
        if std::env::var("HIGGS_ESCHA_DUMP").is_err() {
            return;
        }
        let dir: std::path::PathBuf = std::env::var("HIGGS_ESCHA_TEST_MODEL").unwrap().into();
        let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&text).unwrap();
        let out = convert_checkpoint(&dir, cfg.get("quantization")).unwrap();
        let mut names: Vec<_> = out
            .iter()
            .filter(|(k, v)| v.shape() == [256, 64] || k.contains("mtp"))
            .map(|(k, v)| (k.clone(), v.shape().to_vec()))
            .collect();
        names.sort();
        for (k, s) in names {
            eprintln!("CONV {k} {s:?}");
        }
    }
}
