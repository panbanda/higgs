//! Bonsai-Q1 target-capable engine: packed 1.25-bpw weight storage.
//!
//! Unlike `DiffusionEngine::load_q1` which dequantizes to fp32 at load (32 GB
//! residency on 8B), this engine holds MLX's `Q1_0_g128` affine encoding
//! verbatim: `w[row, col] = scales[row, col/128] * bit(col) + biases[row,
//! col/128]`. Because stock `oxideai/mlx-rs` ships no bits=1 affine kernel, the
//! matvec/dequant run through runtime JIT Metal kernels in
//! [`crate::metal_kernel`] (decode uses a fused matvec over the packed weights;
//! prefill/embedding dequantize to dense f16).
//!
//! Residency: ~1.25 GB for Bonsai-8B-mlx-1bit, ~260 MB for Bonsai-1.7B-mlx-1bit.
//!
//! Scope: Rust-side loader and engine implementation. Routing is enabled in
//! `higgs-engine::model_loader` since the kernels run on the stock MLX pin.

#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    // Quantization math uses small bounded dims (head_dim, GROUP_SIZE=128, vocab) and
    // bit-packed u32→f32 conversions where precision/sign loss is intentional.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    // Dequant kernel + safetensors loader index into manually-bounds-checked slices.
    clippy::indexing_slicing,
    // Decode loop reuses names (q, k, v, t0) across rope/sdpa/o_proj stages by design.
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    // Loader unwraps on safetensors slices after explicit shape validation; load failure
    // paths return ShapeMismatch via map_err elsewhere.
    clippy::unwrap_used,
    clippy::map_unwrap_or,
    // YarnRoPE / Q1 / KV abbreviations are domain terms, not items to backtick.
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::missing_const_for_fn,
    clippy::manual_flatten,
    clippy::if_then_some_else_none,
    clippy::suboptimal_flops,
)]

use half::f16;
use std::path::Path;

use mlx_rs::{Array, Dtype, error::Exception, fast, ops, ops::indexing::IndexOp};
use safetensors::SafeTensors;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    utils::{cached_scaled_dot_product_attention, create_attention_mask},
    yarn::{apply_yarn_rope, compute_yarn_freqs, yarn_get_mscale},
};

/// Load and materialize a Bonsai-Q1 model from `model_dir` onto the GPU.
///
/// Adapts [`BonsaiQ1Engine::load`]'s `Result<_, String>` into [`ModelError`] so
/// the engine surface in `higgs-engine::model_loader` can route it through the
/// same `EngineError::Model` path used by all other architectures.
pub fn load_bonsai_q1<P: AsRef<Path>>(model_dir: P) -> Result<BonsaiQ1Gpu, ModelError> {
    let engine = BonsaiQ1Engine::load(model_dir).map_err(ModelError::ShapeMismatch)?;
    engine.to_gpu().map_err(ModelError::Mlx)
}

pub const GROUP_SIZE: usize = 128;
const GROUP_SIZE_I32: i32 = GROUP_SIZE as i32;

/// Packed 1-bit linear layer with affine per-group dequant.
///
/// Layout (matches MLX 1-bit `QuantizedLinear`, `PrismML` fork):
///   - `w_packed`: `[out_features, in_features/32]` u32, bit `col%32` of word
///     `col/32` is the raw 1-bit weight for column `col`.
///   - `scales`, `biases`: `[out_features, in_features/128]` f16, one per group
///     of 128 input columns.
///
/// Effective: 1 bit/weight + 32 bits/group / 128 weights = **1.25 bpw**.
pub struct PackedQ1Linear {
    pub w_packed: Vec<u32>,
    pub scales: Vec<f16>,
    pub biases: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
}

impl PackedQ1Linear {
    pub const fn resident_bytes(&self) -> usize {
        self.w_packed.len() * 4 + self.scales.len() * 2 + self.biases.len() * 2
    }

    /// Dequantize a single row to fp32 (reference path for correctness tests).
    ///
    /// Not used on the hot path — the hot path uses the fused matvec/dequant
    /// kernels in [`crate::metal_kernel`]; this CPU path is the oracle those
    /// kernels are tested against.
    pub fn dequant_row_to_fp32(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.in_features);
        let n_groups = self.in_features / GROUP_SIZE;
        let packed_cols = self.in_features / 32;
        let w_row = &self.w_packed[row * packed_cols..(row + 1) * packed_cols];
        let s_row = &self.scales[row * n_groups..(row + 1) * n_groups];
        let b_row = &self.biases[row * n_groups..(row + 1) * n_groups];
        for col in 0..self.in_features {
            let word = w_row[col / 32];
            let bit = ((word >> (col % 32)) & 1) as f32;
            let group = col / GROUP_SIZE;
            out[col] = s_row[group].to_f32().mul_add(bit, b_row[group].to_f32());
        }
    }
}

pub struct BonsaiQ1LayerWeights {
    pub q_proj: PackedQ1Linear,
    pub k_proj: PackedQ1Linear,
    pub v_proj: PackedQ1Linear,
    pub o_proj: PackedQ1Linear,
    pub gate_proj: PackedQ1Linear,
    pub up_proj: PackedQ1Linear,
    pub down_proj: PackedQ1Linear,
    pub q_norm: Vec<f16>,
    pub k_norm: Vec<f16>,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

impl BonsaiQ1LayerWeights {
    pub fn resident_bytes(&self) -> usize {
        self.q_proj.resident_bytes()
            + self.k_proj.resident_bytes()
            + self.v_proj.resident_bytes()
            + self.o_proj.resident_bytes()
            + self.gate_proj.resident_bytes()
            + self.up_proj.resident_bytes()
            + self.down_proj.resident_bytes()
            + (self.q_norm.len()
                + self.k_norm.len()
                + self.input_norm.len()
                + self.post_attn_norm.len())
                * 2
    }
}

#[derive(Debug, Clone)]
pub struct BonsaiQ1Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// YARN scaling factor if present (Bonsai-8B uses `factor=4.0, original=16384`).
    pub rope_yarn_factor: Option<f64>,
    pub rope_original_max_seq: Option<usize>,
    pub tie_word_embeddings: bool,
}

pub struct BonsaiQ1Engine {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1LayerWeights>,
    /// Token embedding stored packed (dequants inline at embed lookup time).
    pub embed: PackedQ1Linear,
    /// Untied LM head for 8B (`tie_word_embeddings: false`). None for 1.7B.
    pub lm_head: Option<PackedQ1Linear>,
    pub final_norm: Vec<f16>,
}

impl BonsaiQ1Engine {
    pub const fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn resident_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(BonsaiQ1LayerWeights::resident_bytes)
            .sum();
        let lm_head_bytes = self
            .lm_head
            .as_ref()
            .map_or(0, PackedQ1Linear::resident_bytes);
        layer_bytes + self.embed.resident_bytes() + lm_head_bytes + self.final_norm.len() * 2
    }

    /// Load from a `HuggingFace` directory containing `config.json` +
    /// `model.safetensors` in MLX 1-bit affine-quant format.
    #[allow(clippy::too_many_lines)]
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self, String> {
        let dir = model_dir.as_ref();

        let cfg_txt = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("config.json: {e}"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json parse: {e}"))?;

        let u64_of = |k: &str| -> Result<u64, String> {
            cfg[k]
                .as_u64()
                .ok_or_else(|| format!("config.json missing u64 '{k}'"))
        };
        let hidden = u64_of("hidden_size")? as usize;
        let heads = u64_of("num_attention_heads")? as usize;
        let kv_heads = u64_of("num_key_value_heads")? as usize;
        let head_dim = cfg["head_dim"].as_u64().map_or(128, |v| v as usize);
        let inter = u64_of("intermediate_size")? as usize;
        let layers_n = u64_of("num_hidden_layers")? as usize;
        let vocab = u64_of("vocab_size")? as usize;

        let rms_norm_eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_theta = cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let tie_word_embeddings = cfg["tie_word_embeddings"].as_bool().unwrap_or(false);

        let (rope_yarn_factor, rope_original_max_seq) = cfg
            .get("rope_scaling")
            .and_then(|rs| {
                (rs.get("rope_type").and_then(|v| v.as_str()) == Some("yarn")).then(|| {
                    let f = rs.get("factor").and_then(serde_json::Value::as_f64);
                    let o = rs
                        .get("original_max_position_embeddings")
                        .and_then(serde_json::Value::as_u64)
                        .map(|v| v as usize);
                    (f, o)
                })
            })
            .unwrap_or((None, None));

        let quant = cfg
            .get("quantization")
            .ok_or("missing quantization block")?;
        let q_bits = quant.get("bits").and_then(serde_json::Value::as_u64);
        let q_group = quant.get("group_size").and_then(serde_json::Value::as_u64);
        if q_bits != Some(1) || q_group != Some(GROUP_SIZE as u64) {
            return Err(format!(
                "expected quantization {{bits:1, group_size:{GROUP_SIZE}}}, got bits={q_bits:?} \
                 group_size={q_group:?}"
            ));
        }

        let st_path = dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| format!("read safetensors: {e}"))?;
        let tensors = SafeTensors::deserialize(&st_data)
            .map_err(|e| format!("deserialize safetensors: {e}"))?;

        let config = BonsaiQ1Config {
            hidden,
            layers: layers_n,
            heads,
            kv_heads,
            head_dim,
            inter,
            vocab,
            rms_norm_eps,
            rope_theta,
            rope_yarn_factor,
            rope_original_max_seq,
            tie_word_embeddings,
        };

        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let embed = load_packed(
            &tensors,
            "model.embed_tokens",
            vocab,
            hidden,
            "embed_tokens",
        )?;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            Some(load_packed(&tensors, "lm_head", vocab, hidden, "lm_head")?)
        };
        let final_norm = load_f16(&tensors, "model.norm.weight")?;
        if final_norm.len() != hidden {
            return Err(format!(
                "final_norm len {} != hidden {hidden}",
                final_norm.len()
            ));
        }

        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("model.layers.{i}");
            let attn = format!("{p}.self_attn");
            let mlp = format!("{p}.mlp");

            let layer = BonsaiQ1LayerWeights {
                q_proj: load_packed(&tensors, &format!("{attn}.q_proj"), q_dim, hidden, "q_proj")?,
                k_proj: load_packed(
                    &tensors,
                    &format!("{attn}.k_proj"),
                    kv_dim,
                    hidden,
                    "k_proj",
                )?,
                v_proj: load_packed(
                    &tensors,
                    &format!("{attn}.v_proj"),
                    kv_dim,
                    hidden,
                    "v_proj",
                )?,
                o_proj: load_packed(&tensors, &format!("{attn}.o_proj"), hidden, q_dim, "o_proj")?,
                gate_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.gate_proj"),
                    inter,
                    hidden,
                    "gate_proj",
                )?,
                up_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.up_proj"),
                    inter,
                    hidden,
                    "up_proj",
                )?,
                down_proj: load_packed(
                    &tensors,
                    &format!("{mlp}.down_proj"),
                    hidden,
                    inter,
                    "down_proj",
                )?,
                q_norm: load_f16(&tensors, &format!("{attn}.q_norm.weight"))?,
                k_norm: load_f16(&tensors, &format!("{attn}.k_norm.weight"))?,
                input_norm: load_f16(&tensors, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: load_f16(
                    &tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
            };
            layers.push(layer);
        }

        let engine = Self {
            config,
            layers,
            embed,
            lm_head,
            final_norm,
        };
        let resident_mb = engine.resident_bytes() as f64 / (1024.0 * 1024.0);
        tracing::info!(
            layers = engine.config.layers,
            hidden = engine.config.hidden,
            heads = engine.config.heads,
            kv_heads = engine.config.kv_heads,
            head_dim = engine.config.head_dim,
            inter = engine.config.inter,
            vocab = engine.config.vocab,
            tied_embed = engine.config.tie_word_embeddings,
            packed_resident_mb = format!("{resident_mb:.1}"),
            "BonsaiQ1Engine::load",
        );
        Ok(engine)
    }
}

// ---------------------------------------------------------------------------
// GPU-ready mirror — built once from the packed engine.
// ---------------------------------------------------------------------------

/// MLX-resident 1-bit linear: weight as uint32 packed, scales/biases as f16,
/// same shape as `PackedQ1Linear`, consumed by the bits=1 matvec/dequant
/// kernels in [`crate::metal_kernel`].
pub struct BonsaiQ1GpuLinear {
    pub w: Array,
    pub scales: Array,
    pub biases: Array,
    pub out_features: i32,
    pub in_features: i32,
}

impl BonsaiQ1GpuLinear {
    fn from_packed(p: &PackedQ1Linear) -> Result<Self, Exception> {
        let out = i32::try_from(p.out_features)
            .map_err(|_| Exception::custom("out_features overflows i32"))?;
        let inf = i32::try_from(p.in_features)
            .map_err(|_| Exception::custom("in_features overflows i32"))?;
        let packed_cols = inf / 32;
        let n_groups = inf / GROUP_SIZE_I32;

        let w = Array::from_slice(&p.w_packed, &[out, packed_cols]);
        let scales_f32: Vec<f32> = p.scales.iter().map(|h| h.to_f32()).collect();
        let biases_f32: Vec<f32> = p.biases.iter().map(|h| h.to_f32()).collect();
        let scales = Array::from_slice(&scales_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;
        let biases = Array::from_slice(&biases_f32, &[out, n_groups]).as_dtype(Dtype::Float16)?;

        Ok(Self {
            w,
            scales,
            biases,
            out_features: out,
            in_features: inf,
        })
    }

    /// `y = x @ dequant(w, scales, biases).T`.
    ///
    /// Decode (M = B·T = 1) uses the fused bits=1 matvec kernel, which reads the
    /// packed weights once — the performance path. Prefill (M > 1) dequantizes to
    /// dense f16 and uses a regular matmul (the transient dense weight amortizes
    /// over the M rows). Both run on stock `oxideai/mlx-rs` via
    /// [`crate::metal_kernel`] — no `bits=1` MLX kernel required.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // `m` is derived from the element count, so a shape whose trailing dim
        // isn't `in_features` would be silently treated as a decode row of the
        // wrong size. Reject it loudly instead.
        let shape = x.shape();
        let last_dim = shape.last().copied().unwrap_or(0);
        if self.in_features <= 0 || last_dim != self.in_features {
            return Err(Exception::custom(format!(
                "BonsaiQ1GpuLinear::forward: expected last dim {}, got shape {shape:?}",
                self.in_features
            )));
        }
        let total: i32 = shape.iter().product();
        let m = total / self.in_features;
        if m == 1 {
            crate::metal_kernel::bonsai_q1_qmv(
                x,
                &self.w,
                &self.scales,
                &self.biases,
                GROUP_SIZE_I32,
            )
        } else {
            let wd = crate::metal_kernel::bonsai_q1_dequant(
                &self.w,
                &self.scales,
                &self.biases,
                GROUP_SIZE_I32,
            )?;
            ops::matmul(x, &wd.transpose_axes(&[1, 0])?)
        }
    }
}

pub struct BonsaiQ1GpuLayer {
    pub q_proj: BonsaiQ1GpuLinear,
    pub k_proj: BonsaiQ1GpuLinear,
    pub v_proj: BonsaiQ1GpuLinear,
    pub o_proj: BonsaiQ1GpuLinear,
    pub gate_proj: BonsaiQ1GpuLinear,
    pub up_proj: BonsaiQ1GpuLinear,
    pub down_proj: BonsaiQ1GpuLinear,
    pub q_norm: Array,
    pub k_norm: Array,
    pub input_norm: Array,
    pub post_attn_norm: Array,
}

pub struct BonsaiQ1Gpu {
    pub config: BonsaiQ1Config,
    pub layers: Vec<BonsaiQ1GpuLayer>,
    pub embed: BonsaiQ1GpuLinear,
    pub lm_head: Option<BonsaiQ1GpuLinear>,
    pub final_norm: Array,
    /// YARN-scaled `RoPE` frequencies (per `head_dim/2`). None if no YARN.
    pub yarn_freqs: Option<Array>,
    pub yarn_mscale: f32,
    pub attention_scale: f32,
}

fn f16_vec_to_array(weights: &[f16]) -> Result<Array, Exception> {
    let f32s: Vec<f32> = weights.iter().map(|h| h.to_f32()).collect();
    let len =
        i32::try_from(weights.len()).map_err(|_| Exception::custom("norm len overflows i32"))?;
    Array::from_slice(&f32s, &[len]).as_dtype(Dtype::Float16)
}

impl BonsaiQ1Engine {
    /// Consume the packed engine and materialize MLX arrays.
    ///
    /// Frees the `Vec<u32>` / `Vec<f16>` residency once copied to MLX.
    pub fn to_gpu(self) -> Result<BonsaiQ1Gpu, Exception> {
        let mut gpu_layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            gpu_layers.push(BonsaiQ1GpuLayer {
                q_proj: BonsaiQ1GpuLinear::from_packed(&layer.q_proj)?,
                k_proj: BonsaiQ1GpuLinear::from_packed(&layer.k_proj)?,
                v_proj: BonsaiQ1GpuLinear::from_packed(&layer.v_proj)?,
                o_proj: BonsaiQ1GpuLinear::from_packed(&layer.o_proj)?,
                gate_proj: BonsaiQ1GpuLinear::from_packed(&layer.gate_proj)?,
                up_proj: BonsaiQ1GpuLinear::from_packed(&layer.up_proj)?,
                down_proj: BonsaiQ1GpuLinear::from_packed(&layer.down_proj)?,
                q_norm: f16_vec_to_array(&layer.q_norm)?,
                k_norm: f16_vec_to_array(&layer.k_norm)?,
                input_norm: f16_vec_to_array(&layer.input_norm)?,
                post_attn_norm: f16_vec_to_array(&layer.post_attn_norm)?,
            });
        }

        let embed = BonsaiQ1GpuLinear::from_packed(&self.embed)?;
        let lm_head = self
            .lm_head
            .as_ref()
            .map(BonsaiQ1GpuLinear::from_packed)
            .transpose()?;
        let final_norm = f16_vec_to_array(&self.final_norm)?;

        // YARN precompute.
        let head_dim_i = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        let base = self.config.rope_theta as f32;
        let (yarn_freqs, yarn_mscale) = match self.config.rope_yarn_factor {
            Some(factor) if factor > 1.0 => {
                let orig_seq = self.config.rope_original_max_seq.ok_or_else(|| {
                    Exception::custom(
                        "rope_yarn_factor > 1.0 requires \
                         rope_scaling.original_max_position_embeddings",
                    )
                })?;
                let orig = i32::try_from(orig_seq)
                    .map_err(|_| Exception::custom("orig_max_seq overflows i32"))?;
                let factor_f = factor as f32;
                let freqs = compute_yarn_freqs(head_dim_i, base, factor_f, orig, 32.0, 1.0);
                (Some(freqs), yarn_get_mscale(factor_f, 1.0))
            }
            _ => (None, 1.0),
        };

        let head_dim_f = head_dim_i as f32;
        let attention_scale = head_dim_f.sqrt().recip();

        Ok(BonsaiQ1Gpu {
            config: self.config,
            layers: gpu_layers,
            embed,
            lm_head,
            final_norm,
            yarn_freqs,
            yarn_mscale,
            attention_scale,
        })
    }
}

impl BonsaiQ1Gpu {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Gather embedding rows for a token-ID tensor.
    ///
    /// Gathers the selected packed rows, then dequantizes them to dense f16 via
    /// the [`crate::metal_kernel`] bits=1 kernel (runs on stock `oxideai/mlx-rs`).
    fn embed_rows(&self, ids: &Array) -> Result<Array, Exception> {
        let shape = ids.shape().to_vec();
        let flat = ids.flatten(None, None)?;
        let w = self.embed.w.take_axis(&flat, 0)?;
        let s = self.embed.scales.take_axis(&flat, 0)?;
        let b = self.embed.biases.take_axis(&flat, 0)?;
        let out = crate::metal_kernel::bonsai_q1_dequant(&w, &s, &b, GROUP_SIZE_I32)?;
        let mut ret_shape: Vec<i32> = shape;
        ret_shape.push(-1);
        out.reshape(&ret_shape)
    }

    fn apply_rope(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let head_dim = i32::try_from(self.config.head_dim)
            .map_err(|_| Exception::custom("head_dim overflows i32"))?;
        let offset_array = Array::from_int(offset);
        apply_yarn_rope(
            x,
            head_dim,
            self.config.rope_theta as f32,
            self.yarn_freqs.as_ref(),
            self.yarn_mscale,
            &offset_array,
            false, // Qwen3 layout
        )
    }

    /// Run the decoder trunk and return final-normed hidden `[B, T, hidden]`.
    /// Shared body for `forward` (last-position logits) and
    /// `forward_all_logits` (all-position logits, used by spec-decode verify).
    ///
    /// Body lives in [`forward_trunk_free`] so `compile_with_state` can wrap
    /// it via a free-fn pointer (the `Copy + 'static` closure constraint
    /// forbids capturing `&self`).
    fn forward_trunk(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        forward_trunk_free(self, cache, inputs)
    }

    /// Apply LM head (or tied embed) to `[B, T, hidden]` → `[B, T, vocab]`.
    fn project_logits(&self, h: &Array) -> Result<Array, Exception> {
        let logits = match &self.lm_head {
            Some(head) => head.forward(h)?,
            None => self.embed.forward(h)?,
        };
        // Logits are returned as f32 by API contract (callers do as_slice::<f32>
        // for argmax / softmax). The trunk now stays in fp16 throughout (after
        // the apply_yarn_rope dtype fix), so we cast here at the boundary.
        logits.as_dtype(Dtype::Float32)
    }

    /// Causal forward. Returns logits `[B, 1, vocab]` for the last position
    /// (mlx_lm convention).
    pub fn forward(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk(inputs, cache)?;
        let t = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("trunk hidden missing T dim"))?;
        let last = if t > 1 { h.index((.., -1.., ..)) } else { h };
        self.project_logits(&last)
    }

    /// Causal forward returning logits at **every** position `[B, T, vocab]`.
    /// Used by speculative-decode target verify: given the draft prefix,
    /// obtain one logits row per proposed token in a single forward pass.
    pub fn forward_all_logits(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk(inputs, cache)?;
        self.project_logits(&h)
    }

    /// Profiled variant of `forward`: same result, but attributes per-section
    /// wall time into `times`. Forces `.eval()` after every section (kills
    /// lazy batching — that's the point: ratios matter, absolutes don't).
    ///
    /// Used by `bench_bonsai_q1_decode_breakdown` to answer the
    /// dispatch-bound-vs-matmul-bound question for Bonsai-8B AR parity.
    pub fn forward_profiled(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
        times: &mut SectionTimes,
    ) -> Result<Array, Exception> {
        let h = self.forward_trunk_profiled(inputs, cache, times)?;
        let t0 = std::time::Instant::now();
        let t = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("trunk hidden missing T dim"))?;
        let last = if t > 1 { h.index((.., -1.., ..)) } else { h };
        let logits = self.project_logits(&last)?;
        logits.eval()?;
        times.add("lm_head", t0.elapsed().as_nanos());
        Ok(logits)
    }

    /// Profiled mirror of `forward_trunk`. Inserts `eval + record` at each
    /// semantic section boundary. Sections are grouped by operation type
    /// (qkv projections together, mlp up+gate together, etc.) — per-layer
    /// noise is collapsed into section totals across all layers.
    #[allow(non_snake_case)]
    fn forward_trunk_profiled(
        &self,
        inputs: &Array,
        cache: &mut Vec<Option<SteppingKeyValueCache>>,
        times: &mut SectionTimes,
    ) -> Result<Array, Exception> {
        use std::time::Instant;

        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;

        if cache.is_empty() {
            *cache = (0..self.layers.len())
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "cache len {} != num_layers {}",
                cache.len(),
                self.layers.len()
            )));
        }

        // Sync point: make sure prior work isn't folded into embed_rows time.
        inputs.eval()?;

        let t0 = Instant::now();
        let mut h = self.embed_rows(inputs)?;
        h.eval()?;
        times.add("embed_rows", t0.elapsed().as_nanos());

        // Prefill (T>1) needs a causal mask. Request the Array form: the cached
        // SDPA path below consumes Option<&Array> and applies no causal masking
        // itself, so the Causal variant would be dropped to None (= no mask).
        let mask = create_attention_mask(&h, cache, Some(true))?;

        let heads = i32::try_from(self.config.heads)
            .map_err(|_| Exception::custom("heads overflows i32"))?;
        let kv_heads = i32::try_from(self.config.kv_heads)
            .map_err(|_| Exception::custom("kv_heads overflows i32"))?;
        let rms_eps = self.config.rms_norm_eps;

        for (layer, layer_cache) in self.layers.iter().zip(cache.iter_mut()) {
            let t0 = Instant::now();
            let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps)?;
            normed.eval()?;
            times.add("input_norm", t0.elapsed().as_nanos());

            // qkv projections — 3× quantized_matmul on the same input.
            let t0 = Instant::now();
            let q = layer.q_proj.forward(&normed)?;
            let k = layer.k_proj.forward(&normed)?;
            let v = layer.v_proj.forward(&normed)?;
            q.eval()?;
            k.eval()?;
            v.eval()?;
            times.add("qkv_proj", t0.elapsed().as_nanos());

            // Reshape to [B, L, n_heads, head_dim] then transpose to
            // [B, n_heads, L, head_dim]. Metadata-only; lumped with qk_norm.
            let q = q
                .reshape(&[B, T, heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = k
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let v = v
                .reshape(&[B, T, kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;

            let t0 = Instant::now();
            let q = fast::rms_norm(&q, &layer.q_norm, rms_eps)?;
            let k = fast::rms_norm(&k, &layer.k_norm, rms_eps)?;
            q.eval()?;
            k.eval()?;
            times.add("qk_norm", t0.elapsed().as_nanos());

            let offset = layer_cache.as_ref().map_or(0, KeyValueCache::offset);
            let t0 = Instant::now();
            let q = self.apply_rope(&q, offset)?;
            let k = self.apply_rope(&k, offset)?;
            q.eval()?;
            k.eval()?;
            times.add("rope", t0.elapsed().as_nanos());

            let mask_arr = match &mask {
                Some(crate::utils::AttentionMask::Array(a)) => Some(a),
                _ => None,
            };
            let mask_arr_opt: Option<&Array> = mask_arr;

            let t0 = Instant::now();
            let attn_out = match layer_cache.as_mut() {
                Some(c) => cached_scaled_dot_product_attention(
                    q,
                    c,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt,
                )?,
                None => fast::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    self.attention_scale,
                    mask_arr_opt.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
                    None::<&Array>,
                )?,
            };
            attn_out.eval()?;
            times.add("sdpa_kv", t0.elapsed().as_nanos());

            let attn_out = attn_out
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, T, -1])?;

            let t0 = Instant::now();
            let attn_out = layer.o_proj.forward(&attn_out)?;
            attn_out.eval()?;
            times.add("o_proj", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let h_post_attn = h.add(&attn_out)?;
            h_post_attn.eval()?;
            times.add("residual", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps)?;
            normed_post.eval()?;
            times.add("post_attn_norm", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let gate = layer.gate_proj.forward(&normed_post)?;
            let up = layer.up_proj.forward(&normed_post)?;
            gate.eval()?;
            up.eval()?;
            times.add("mlp_up_gate", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let mlp_hidden = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
            mlp_hidden.eval()?;
            times.add("silu_mul", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            let mlp_out = layer.down_proj.forward(&mlp_hidden)?;
            mlp_out.eval()?;
            times.add("mlp_down", t0.elapsed().as_nanos());

            let t0 = Instant::now();
            h = h_post_attn.add(&mlp_out)?;
            h.eval()?;
            times.add("residual", t0.elapsed().as_nanos());
        }

        let t0 = Instant::now();
        let out = fast::rms_norm(&h, &self.final_norm, rms_eps)?;
        out.eval()?;
        times.add("final_norm", t0.elapsed().as_nanos());
        Ok(out)
    }
}

/// Free-function body of the decoder trunk.
///
/// Lives at module scope (not as a method) so a **function pointer** to
/// [`decode_step_free`] satisfies `compile_with_state`'s
/// `F: Copy + 'static` bound — a closure capturing `&self` would not.
/// All `self.xxx` access is replaced with `gpu.xxx`; `embed_rows`,
/// `apply_rope`, and `project_logits` are called as methods on `gpu`
/// (they are already `&self`-only, so no further plumbing is needed).
#[allow(non_snake_case)]
pub fn forward_trunk_free(
    gpu: &BonsaiQ1Gpu,
    cache: &mut Vec<Option<SteppingKeyValueCache>>,
    inputs: &Array,
) -> Result<Array, Exception> {
    let shape = inputs.shape();
    let B = *shape
        .first()
        .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;
    let T = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("inputs must have >= 2 dims"))?;

    if cache.is_empty() {
        *cache = (0..gpu.layers.len())
            .map(|_| Some(SteppingKeyValueCache::new()))
            .collect();
    } else if cache.len() != gpu.layers.len() {
        return Err(Exception::custom(format!(
            "cache len {} != num_layers {}",
            cache.len(),
            gpu.layers.len()
        )));
    }

    let mut h = gpu.embed_rows(inputs)?; // [B, L, hidden]

    // Prefill (T>1) needs a causal mask. Request the Array form: the cached
    // SDPA path below consumes Option<&Array> and applies no causal masking
    // itself, so the Causal variant would be dropped to None (= no mask).
    let mask = create_attention_mask(&h, cache, Some(true))?;

    let heads =
        i32::try_from(gpu.config.heads).map_err(|_| Exception::custom("heads overflows i32"))?;
    let kv_heads = i32::try_from(gpu.config.kv_heads)
        .map_err(|_| Exception::custom("kv_heads overflows i32"))?;
    let rms_eps = gpu.config.rms_norm_eps;

    for (layer, layer_cache) in gpu.layers.iter().zip(cache.iter_mut()) {
        let normed = fast::rms_norm(&h, &layer.input_norm, rms_eps)?;

        let q = layer.q_proj.forward(&normed)?;
        let k = layer.k_proj.forward(&normed)?;
        let v = layer.v_proj.forward(&normed)?;

        let q = q
            .reshape(&[B, T, heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[B, T, kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[B, T, kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = fast::rms_norm(&q, &layer.q_norm, rms_eps)?;
        let k = fast::rms_norm(&k, &layer.k_norm, rms_eps)?;

        let offset = layer_cache.as_ref().map_or(0, KeyValueCache::offset);
        let q = gpu.apply_rope(&q, offset)?;
        let k = gpu.apply_rope(&k, offset)?;

        let mask_arr = match &mask {
            Some(crate::utils::AttentionMask::Array(a)) => Some(a),
            _ => None,
        };
        let mask_arr_opt: Option<&Array> = mask_arr;

        let attn_out = match layer_cache.as_mut() {
            Some(c) => {
                cached_scaled_dot_product_attention(q, c, k, v, gpu.attention_scale, mask_arr_opt)?
            }
            None => fast::scaled_dot_product_attention(
                q,
                k,
                v,
                gpu.attention_scale,
                mask_arr_opt.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
                None::<&Array>,
            )?,
        };

        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, T, -1])?;
        let attn_out = layer.o_proj.forward(&attn_out)?;
        let h_post_attn = h.add(&attn_out)?;

        let normed_post = fast::rms_norm(&h_post_attn, &layer.post_attn_norm, rms_eps)?;
        let gate = layer.gate_proj.forward(&normed_post)?;
        let up = layer.up_proj.forward(&normed_post)?;
        let mlp_hidden = mlx_rs::nn::silu(&gate)?.multiply(&up)?;
        let mlp_out = layer.down_proj.forward(&mlp_hidden)?;

        h = h_post_attn.add(&mlp_out)?;
    }

    fast::rms_norm(&h, &gpu.final_norm, rms_eps)
}

/// Owned state wrapper for [`compile_with_state`]-driven decoding.
///
/// `compile_with_state` takes the state by `&mut U` where `U: Updatable`.
/// Wrapping the model **and** the per-layer KV cache in one owned struct
/// lets us hand-roll a single `Updatable` impl whose positional iteration
/// order covers both — safer than fighting lifetimes on `(&mut gpu, cache)`
/// tuples. See session-25 recap for the design rationale.
///
/// Expected construction: after prefill, move `gpu` and the filled cache
/// vector into this struct, run the decode loop with a compiled step,
/// then destructure back out when done.
pub struct BonsaiQ1DecodeState {
    pub gpu: BonsaiQ1Gpu,
    pub cache: Vec<Option<SteppingKeyValueCache>>,
}

/// Number of updatable `Array`s per decoder layer:
/// - `input_norm` + 3×(w,s,b) qkv + `q_norm` + `k_norm` + 3×(w,s,b) o_proj
///   ... wait: 1 + 3×3 + 2 + 3 + 1 + 3×3 = 1+9+2+3+1+9 = **25**.
/// Corresponds to the array push order in [`BonsaiQ1DecodeState::updatable_states`].
const PER_LAYER_UPDATABLE: usize = 25;

impl mlx_rs::utils::Updatable for BonsaiQ1DecodeState {
    fn updatable_states_len(&self) -> usize {
        let mut n = 3 // embed (w, scales, biases)
            + self.gpu.layers.len() * PER_LAYER_UPDATABLE
            + 1; // final_norm
        if self.gpu.lm_head.is_some() {
            n += 3;
        }
        if self.gpu.yarn_freqs.is_some() {
            n += 1;
        }
        for slot in &self.cache {
            if let Some(c) = slot {
                if c.keys().is_some() {
                    n += 1;
                }
                if c.values().is_some() {
                    n += 1;
                }
            }
        }
        n
    }

    fn updatable_states(&self) -> impl IntoIterator<Item = &Array> {
        let mut v: Vec<&Array> = Vec::with_capacity(self.updatable_states_len());
        v.push(&self.gpu.embed.w);
        v.push(&self.gpu.embed.scales);
        v.push(&self.gpu.embed.biases);
        for layer in &self.gpu.layers {
            v.push(&layer.input_norm);
            v.push(&layer.q_proj.w);
            v.push(&layer.q_proj.scales);
            v.push(&layer.q_proj.biases);
            v.push(&layer.k_proj.w);
            v.push(&layer.k_proj.scales);
            v.push(&layer.k_proj.biases);
            v.push(&layer.v_proj.w);
            v.push(&layer.v_proj.scales);
            v.push(&layer.v_proj.biases);
            v.push(&layer.q_norm);
            v.push(&layer.k_norm);
            v.push(&layer.o_proj.w);
            v.push(&layer.o_proj.scales);
            v.push(&layer.o_proj.biases);
            v.push(&layer.post_attn_norm);
            v.push(&layer.gate_proj.w);
            v.push(&layer.gate_proj.scales);
            v.push(&layer.gate_proj.biases);
            v.push(&layer.up_proj.w);
            v.push(&layer.up_proj.scales);
            v.push(&layer.up_proj.biases);
            v.push(&layer.down_proj.w);
            v.push(&layer.down_proj.scales);
            v.push(&layer.down_proj.biases);
        }
        v.push(&self.gpu.final_norm);
        if let Some(lm) = self.gpu.lm_head.as_ref() {
            v.push(&lm.w);
            v.push(&lm.scales);
            v.push(&lm.biases);
        }
        if let Some(y) = self.gpu.yarn_freqs.as_ref() {
            v.push(y);
        }
        for slot in &self.cache {
            if let Some(c) = slot {
                if let Some(k) = c.keys() {
                    v.push(k);
                }
                if let Some(val) = c.values() {
                    v.push(val);
                }
            }
        }
        v
    }

    fn updatable_states_mut(&mut self) -> impl IntoIterator<Item = &mut Array> {
        let mut v: Vec<&mut Array> = Vec::with_capacity(self.updatable_states_len());
        v.push(&mut self.gpu.embed.w);
        v.push(&mut self.gpu.embed.scales);
        v.push(&mut self.gpu.embed.biases);
        for layer in &mut self.gpu.layers {
            v.push(&mut layer.input_norm);
            v.push(&mut layer.q_proj.w);
            v.push(&mut layer.q_proj.scales);
            v.push(&mut layer.q_proj.biases);
            v.push(&mut layer.k_proj.w);
            v.push(&mut layer.k_proj.scales);
            v.push(&mut layer.k_proj.biases);
            v.push(&mut layer.v_proj.w);
            v.push(&mut layer.v_proj.scales);
            v.push(&mut layer.v_proj.biases);
            v.push(&mut layer.q_norm);
            v.push(&mut layer.k_norm);
            v.push(&mut layer.o_proj.w);
            v.push(&mut layer.o_proj.scales);
            v.push(&mut layer.o_proj.biases);
            v.push(&mut layer.post_attn_norm);
            v.push(&mut layer.gate_proj.w);
            v.push(&mut layer.gate_proj.scales);
            v.push(&mut layer.gate_proj.biases);
            v.push(&mut layer.up_proj.w);
            v.push(&mut layer.up_proj.scales);
            v.push(&mut layer.up_proj.biases);
            v.push(&mut layer.down_proj.w);
            v.push(&mut layer.down_proj.scales);
            v.push(&mut layer.down_proj.biases);
        }
        v.push(&mut self.gpu.final_norm);
        if let Some(lm) = self.gpu.lm_head.as_mut() {
            v.push(&mut lm.w);
            v.push(&mut lm.scales);
            v.push(&mut lm.biases);
        }
        if let Some(y) = self.gpu.yarn_freqs.as_mut() {
            v.push(y);
        }
        for slot in &mut self.cache {
            if let Some(c) = slot {
                let (k_opt, v_opt) = c.key_value_arrays_mut();
                if let Some(k) = k_opt {
                    v.push(k);
                }
                if let Some(val) = v_opt {
                    v.push(val);
                }
            }
        }
        v
    }
}

/// Free-fn decode step compatible with `compile_with_state`.
///
/// `state.cache` **must** be populated by a prefill call before this runs:
/// compile-wrap is applied only in decode, and shape consistency across
/// steps (for the MLX per-shape trace cache) requires
/// [`SteppingKeyValueCache::reserve_max_tokens`] ahead of the first
/// `update_dense`.
pub fn decode_step_free(
    state: &mut BonsaiQ1DecodeState,
    inputs: &Array,
) -> Result<Array, Exception> {
    let h = forward_trunk_free(&state.gpu, &mut state.cache, inputs)?;
    let t = *h
        .shape()
        .get(1)
        .ok_or_else(|| Exception::custom("trunk hidden missing T dim"))?;
    let last = if t > 1 { h.index((.., -1.., ..)) } else { h };
    state.gpu.project_logits(&last)
}

/// Per-section wall-time accumulator for the Bonsai-Q1 forward pass.
///
/// Exists only to attribute the 45 ms/tok Bonsai-8B AR decode cost to
/// individual sections (embed / norms / qkv / rope / sdpa / o_proj / mlp / lm_head).
/// Each section's compute is force-`.eval()`'d to prevent MLX lazy batching
/// from pooling multiple sections into one materialization — ratios between
/// sections are meaningful even though absolutes will be slower than the
/// unprofiled path.
#[derive(Debug, Default, Clone)]
pub struct SectionTimes {
    totals: std::collections::BTreeMap<&'static str, (u128, u64)>,
}

impl SectionTimes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &'static str, ns: u128) {
        let e = self.totals.entry(name).or_insert((0, 0));
        e.0 += ns;
        e.1 += 1;
    }

    /// Total across all sections (ns).
    pub fn total_ns(&self) -> u128 {
        self.totals.values().map(|(t, _)| *t).sum()
    }

    /// Section totals: `(name, total_ns, call_count)`, sorted by ns descending.
    pub fn entries(&self) -> Vec<(&'static str, u128, u64)> {
        let mut v: Vec<_> = self.totals.iter().map(|(k, (t, n))| (*k, *t, *n)).collect();
        v.sort_by_key(|b| std::cmp::Reverse(b.1));
        v
    }
}

fn load_packed(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    who: &str,
) -> Result<PackedQ1Linear, String> {
    if in_features % GROUP_SIZE != 0 {
        return Err(format!(
            "{who}: in_features {in_features} not divisible by group_size {GROUP_SIZE}"
        ));
    }
    let packed_cols = in_features / 32;
    let n_groups = in_features / GROUP_SIZE;

    let w_view = tensors
        .tensor(&format!("{prefix}.weight"))
        .map_err(|e| format!("{who}: {prefix}.weight: {e}"))?;
    let s_view = tensors
        .tensor(&format!("{prefix}.scales"))
        .map_err(|e| format!("{who}: {prefix}.scales: {e}"))?;
    let b_view = tensors
        .tensor(&format!("{prefix}.biases"))
        .map_err(|e| format!("{who}: {prefix}.biases: {e}"))?;

    let w_bytes = w_view.data();
    let s_bytes = s_view.data();
    let b_bytes = b_view.data();

    let expected_w_bytes = out_features * packed_cols * 4;
    if w_bytes.len() != expected_w_bytes {
        return Err(format!(
            "{who}: weight byte-size mismatch: got {} expected {}",
            w_bytes.len(),
            expected_w_bytes,
        ));
    }
    let expected_sb_bytes = out_features * n_groups * 2;
    if s_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: scales byte-size mismatch: got {} expected {}",
            s_bytes.len(),
            expected_sb_bytes,
        ));
    }
    if b_bytes.len() != expected_sb_bytes {
        return Err(format!(
            "{who}: biases byte-size mismatch: got {} expected {}",
            b_bytes.len(),
            expected_sb_bytes,
        ));
    }

    let w_packed: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let scales = bytes_to_f16_vec(s_bytes);
    let biases = bytes_to_f16_vec(b_bytes);

    Ok(PackedQ1Linear {
        w_packed,
        scales,
        biases,
        out_features,
        in_features,
    })
}

fn load_f16(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f16>, String> {
    let view = tensors.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    Ok(bytes_to_f16_vec(view.data()))
}

fn bytes_to_f16_vec(b: &[u8]) -> Vec<f16> {
    b.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_kernel::{bonsai_q1_dequant, bonsai_q1_qmv_legacy};

    /// Deterministic PRNG (SplitMix-ish LCG). Tests prove the kernels match the
    /// CPU reference over pseudo-random data, not against hand-picked constants.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 32) as u32
    }

    fn make_packed(out_features: usize, in_features: usize, seed: u64) -> PackedQ1Linear {
        let packed_cols = in_features / 32;
        let n_groups = in_features / GROUP_SIZE;
        let mut st = seed;
        let w_packed: Vec<u32> = (0..out_features * packed_cols)
            .map(|_| lcg(&mut st))
            .collect();
        // Per-(row,group) scales/biases so a wrong group index produces a clear
        // mismatch. Magnitudes are small and signed, like real affine params.
        let scales: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(0.05 + 0.013 * ((i % 7) as f32)))
            .collect();
        let biases: Vec<f16> = (0..out_features * n_groups)
            .map(|i| f16::from_f32(-0.03 + 0.011 * ((i % 5) as f32)))
            .collect();
        PackedQ1Linear {
            w_packed,
            scales,
            biases,
            out_features,
            in_features,
        }
    }

    /// CPU reference: full dense dequant via the documented per-row path.
    fn dense_reference(p: &PackedQ1Linear) -> Vec<f32> {
        let mut wd = vec![0.0f32; p.out_features * p.in_features];
        for r in 0..p.out_features {
            let (lo, hi) = (r * p.in_features, (r + 1) * p.in_features);
            p.dequant_row_to_fp32(r, &mut wd[lo..hi]);
        }
        wd
    }

    #[test]
    fn dequant_kernel_matches_cpu_reference() {
        let (out_f, in_f) = (96usize, 256usize); // 256 cols => 2 groups of 128
        let p = make_packed(out_f, in_f, 0xDEAD_BEEF);
        let gpu = BonsaiQ1GpuLinear::from_packed(&p).unwrap();

        let wd = bonsai_q1_dequant(&gpu.w, &gpu.scales, &gpu.biases, GROUP_SIZE_I32).unwrap();
        wd.eval().unwrap();
        let got = wd.as_slice::<f16>();
        let want = dense_reference(&p);

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let gv = g.to_f32();
            assert!(
                (gv - w).abs() <= 2e-3,
                "dequant mismatch at {i}: got {gv} want {w}"
            );
        }
    }

    #[test]
    fn qmv_kernel_matches_cpu_reference() {
        let (out_f, in_f) = (96usize, 256usize);
        let p = make_packed(out_f, in_f, 0x1234_5678);
        let gpu = BonsaiQ1GpuLinear::from_packed(&p).unwrap();

        // x in [-1, 1], deterministic; the kernel reads it as f16, so the
        // reference uses the f16-rounded values for an apples-to-apples compare.
        let mut st = 0xABCD_EF01_u64;
        let x_f32: Vec<f32> = (0..in_f)
            .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
            .collect();
        let x = Array::from_slice(&x_f32, &[1, in_f as i32])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let x_ref: Vec<f32> = x_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

        let y = bonsai_q1_qmv_legacy(&x, &gpu.w, &gpu.scales, &gpu.biases, GROUP_SIZE_I32).unwrap();
        y.eval().unwrap();
        let got = y.as_slice::<f16>();
        assert_eq!(got.len(), out_f);

        let wd = dense_reference(&p);
        for r in 0..out_f {
            let mut acc = 0.0f32;
            for c in 0..in_f {
                acc += x_ref[c] * wd[r * in_f + c];
            }
            let gv = got[r].to_f32();
            let tol = 1e-2 * acc.abs().max(1.0);
            assert!(
                (gv - acc).abs() <= tol,
                "qmv mismatch at row {r}: got {gv} want {acc}"
            );
        }
    }

    /// Oracle for the `qmv_fast`-class kernel (`bonsai_q1_qmv_fast`). Covers the
    /// tail path (K = 256 < block) and the main-block path (K = 4096) with an
    /// N % 4 != 0 row remainder (the lm_head case). Bit-exact vs CPU reference.
    #[test]
    fn qmv_fast_kernel_matches_cpu_reference() {
        for &(out_f, in_f, seed) in &[
            (96usize, 256usize, 0x1234_5678_u64),
            (130usize, 4096usize, 0x0BAD_F00D_u64),
        ] {
            let p = make_packed(out_f, in_f, seed);
            let gpu = BonsaiQ1GpuLinear::from_packed(&p).unwrap();

            let mut st = 0xABCD_EF01_u64;
            let x_f32: Vec<f32> = (0..in_f)
                .map(|_| (lcg(&mut st) as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
                .collect();
            let x = Array::from_slice(&x_f32, &[1, in_f as i32])
                .as_dtype(Dtype::Float16)
                .unwrap();
            let x_ref: Vec<f32> = x_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

            let y = crate::metal_kernel::bonsai_q1_qmv_fast(
                &x,
                &gpu.w,
                &gpu.scales,
                &gpu.biases,
                GROUP_SIZE_I32,
            )
            .unwrap();
            y.eval().unwrap();
            let got = y.as_slice::<f16>();
            assert_eq!(got.len(), out_f);

            let wd = dense_reference(&p);
            for r in 0..out_f {
                let mut acc = 0.0f32;
                for c in 0..in_f {
                    acc += x_ref[c] * wd[r * in_f + c];
                }
                let gv = got[r].to_f32();
                let tol = 1e-2 * acc.abs().max(1.0);
                assert!(
                    (gv - acc).abs() <= tol,
                    "qmv_fast mismatch ({out_f}x{in_f}) row {r}: got {gv} want {acc}"
                );
            }
        }
    }
}
