//! `DFlash` block-diffusion drafter for speculative decoding.
//!
//! A 0.5B drafter that produces 16 draft tokens per round via a single
//! non-causal forward pass. The Modal drafter checkpoints verified by the
//! loader tests use 8 target-layer taps: `[1, 6, 11, 16, 22, 27, 32, 37]`.
//!
//! Architecture: 6 decoder layers with dual-stream attention —
//! Q from noise embedding, K/V from `concat(target_hidden, noise)`.
//! No `embed_tokens` or `lm_head` — uses the target model's `lm_head`.
//!
//! Reference checkpoints: `modal-labs/Qwen3.6-35B-A3B-DFlash` and
//! `modal-labs/Qwen3.5-9B-DFlash`.
use std::path::Path;

use mlx_rs::{
    Array, builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn, ops,
};
use serde::Deserialize;

use crate::{error::ModelError, utils::apply_rope};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct DFlashSubConfig {
    target_layer_ids: Vec<usize>,
    #[serde(default)]
    mask_token_id: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    pub vocab_size: i32,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    dflash_config: DFlashSubConfig,
}

impl DFlashConfig {
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }

    pub const fn num_taps(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    pub fn mask_token_id(&self) -> i32 {
        self.dflash_config.mask_token_id.unwrap_or(248_070)
    }
}

const fn default_head_dim() -> i32 {
    128
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1e7
}

const fn default_block_size() -> i32 {
    16
}

/// Runtime decode `block_size` used at inference.
///
/// Overridable via `HIGGS_DFLASH_BLOCK_SIZE`. Diverges from the drafter's
/// trained `block_size` (16) because acceptance rate plateaus at ~3 tokens
/// and smaller blocks amortize verify cost better.
pub const DEFAULT_DECODE_BLOCK_SIZE: i32 = 4;

// ---------------------------------------------------------------------------
// SwiGLU MLP (non-quantized)
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashMLP {
    #[param]
    gate_proj: nn::Linear,
    #[param]
    up_proj: nn::Linear,
    #[param]
    down_proj: nn::Linear,
}

impl DFlashMLP {
    fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::sigmoid(&gate)?.multiply(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// DFlash dual-stream attention
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashAttention {
    #[param]
    q_proj: nn::Linear,
    #[param]
    k_proj: nn::Linear,
    #[param]
    v_proj: nn::Linear,
    #[param]
    o_proj: nn::Linear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    scale: f32,
    is_sliding: bool,
    sliding_window: i32,
}

impl DFlashAttention {
    fn new(config: &DFlashConfig, layer_idx: usize) -> Result<Self, Exception> {
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let hidden = config.hidden_size;
        let is_sliding = config
            .layer_types
            .as_ref()
            .and_then(|lt| lt.get(layer_idx))
            .is_some_and(|t| t == "sliding_attention");
        let sliding_window = config.sliding_window.unwrap_or(0);

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, n_heads * head_dim)
                .bias(false)
                .build()?,
            k_proj: nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(n_heads * head_dim, hidden)
                .bias(false)
                .build()?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(config.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv_heads,
            head_dim,
            scale: f32::from(
                i16::try_from(head_dim)
                    .map_err(|_| Exception::custom("head_dim out of i16 range"))?,
            )
            .sqrt()
            .recip(),
            is_sliding,
            sliding_window,
        })
    }

    /// Dual-stream attention: Q from noise, K/V from `concat(target, noise)`.
    ///
    /// `noise`: `[B, block_size, hidden]` — the 16 draft positions.
    /// `target_hidden`: `[B, ctx_len, hidden]` — projected+normed tap states.
    /// `cache`: optional (K, V) from prior rounds, shape `[B, n_kv, cached_len, head_dim]`.
    ///   Post-RoPE K and raw V. Updated in-place with the new K/V appended.
    /// `cache_offset`: absolute position offset for `RoPE` (= cached seq length).
    #[allow(non_snake_case, clippy::shadow_reuse)]
    fn forward(
        &mut self,
        noise: &Array,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        use mlx_rs::ops::indexing::IndexOp;

        let B = *noise
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("need 3D"))?;
        let q_len = *noise
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("need 3D"))?;
        let ctx_len = *target_hidden
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("need 3D"))?;
        // Q from noise only
        let q = self.q_proj.forward(noise)?;
        let q = q.reshape(&[B, q_len, self.num_attention_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;

        // K/V from context (target_hidden) — SEPARATE from noise
        let ctx_k = self.k_proj.forward(target_hidden)?;
        let ctx_v = self.v_proj.forward(target_hidden)?;
        let ctx_k = ctx_k.reshape(&[B, ctx_len, self.num_key_value_heads, self.head_dim])?;
        let ctx_k = self.k_norm.forward(&ctx_k)?.transpose_axes(&[0, 2, 1, 3])?;
        let ctx_v = ctx_v
            .reshape(&[B, ctx_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // K/V from noise — freshly computed every round, never cached
        let noise_k = self.k_proj.forward(noise)?;
        let noise_v = self.v_proj.forward(noise)?;
        let noise_k = noise_k.reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?;
        let noise_k = self
            .k_norm
            .forward(&noise_k)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let noise_v = noise_v
            .reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with absolute positions:
        // Context K: [cache_offset .. cache_offset + ctx_len]
        // Noise K + Q: [cache_offset + ctx_len .. cache_offset + ctx_len + q_len]
        let q = apply_rope(&q, &self.rope, cache_offset + ctx_len)?;
        let ctx_k = apply_rope(&ctx_k, &self.rope, cache_offset)?;
        let noise_k = apply_rope(&noise_k, &self.rope, cache_offset + ctx_len)?;

        // Cache stores ONLY context K/V (append to prior rounds)
        let (ctx_k, ctx_v) = if let Some((k_cached, v_cached)) = cache.as_ref() {
            (
                ops::concatenate_axis(&[k_cached, &ctx_k], 2)?,
                ops::concatenate_axis(&[v_cached, &ctx_v], 2)?,
            )
        } else {
            (ctx_k, ctx_v)
        };
        let (ctx_k, ctx_v) = if self.is_sliding && self.sliding_window > 1 {
            let keep = self.sliding_window - 1;
            let len = ctx_k.shape().get(2).copied().unwrap_or(0);
            if len > keep {
                let skip = len - keep;
                (
                    ctx_k.index((.., .., skip.., ..)),
                    ctx_v.index((.., .., skip.., ..)),
                )
            } else {
                (ctx_k, ctx_v)
            }
        } else {
            (ctx_k, ctx_v)
        };
        *cache = Some((ctx_k.clone(), ctx_v.clone()));

        // Attention over cached_context + fresh_noise
        let k = ops::concatenate_axis(&[&ctx_k, &noise_k], 2)?;
        let v = ops::concatenate_axis(&[&ctx_v, &noise_v], 2)?;

        // Non-causal SDPA (no mask)
        let output = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            self.scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        // [B, n_heads, q_len, head_dim] -> [B, q_len, n_heads * head_dim]
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[B, q_len, -1])?;
        self.o_proj.forward(&output)
    }
}

// ---------------------------------------------------------------------------
// DFlash decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashDecoderLayer {
    #[param]
    self_attn: DFlashAttention,
    #[param]
    mlp: DFlashMLP,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DFlashDecoderLayer {
    fn new(config: &DFlashConfig, layer_idx: usize) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: DFlashAttention::new(config, layer_idx)?,
            mlp: DFlashMLP::new(config.hidden_size, config.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn forward(
        &mut self,
        noise: &Array,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(noise)?;
        let attn_out = self
            .self_attn
            .forward(&normed, target_hidden, cache, cache_offset)?;
        let h = noise.add(attn_out)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// DFlash drafter (top-level)
// ---------------------------------------------------------------------------

/// `DFlash` block-diffusion drafter.
///
/// Produces `block_size` (16) draft tokens per round. Does NOT have its own
/// `embed_tokens` or `lm_head` — uses the target model's `lm_head` on the output.
#[derive(Debug, ModuleParameters)]
pub struct DFlashDrafter {
    #[param]
    fc: nn::Linear,
    #[param]
    hidden_norm: nn::RmsNorm,
    #[param]
    layers: Vec<DFlashDecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
    pub config: DFlashConfig,
}

impl DFlashDrafter {
    pub fn new(config: DFlashConfig) -> Result<Self, Exception> {
        let fc_in = i32::try_from(config.num_taps())
            .map_err(|e| Exception::custom(format!("num_taps too large for i32: {e}")))?
            * config.hidden_size;
        let layers = (0..config.num_hidden_layers)
            .map(|i| DFlashDecoderLayer::new(&config, usize::try_from(i).unwrap_or(0)))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            fc: nn::LinearBuilder::new(fc_in, config.hidden_size)
                .bias(false)
                .build()?,
            hidden_norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            config,
        })
    }

    /// Create an empty per-layer KV cache for the drafter.
    pub fn make_cache(&self) -> Vec<Option<(Array, Array)>> {
        vec![None; self.layers.len()]
    }

    /// Run the drafter forward pass.
    ///
    /// - `noise`: `[B, block_size, hidden_size]` — embedded block tokens.
    /// - `taps`: slice of hidden states from the target model at tap layers,
    ///   each `[B, T, target_hidden_size]`. Concatenated along the last dim,
    ///   projected via `fc`, then normalized.
    /// - `cache`: per-layer KV cache. Grows each round; crop after verify.
    ///
    /// Returns `[B, block_size, hidden_size]` — pass to target's `lm_head` for logits.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        noise: &Array,
        taps: &[Array],
        cache: &mut [Option<(Array, Array)>],
    ) -> Result<Array, Exception> {
        if taps.len() != self.config.num_taps() {
            return Err(Exception::custom(format!(
                "expected {} taps, got {}",
                self.config.num_taps(),
                taps.len()
            )));
        }

        // Cache offset = max cached seq length (0 on first round)
        let cache_offset = cache
            .iter()
            .filter_map(|c| c.as_ref())
            .filter_map(|(k, _)| k.shape().get(2).copied())
            .max()
            .unwrap_or(0);

        // Concatenate tap hidden states: [B, T, num_taps * hidden_size]
        let tap_refs: Vec<&Array> = taps.iter().collect();
        let target_cat = ops::concatenate_axis(&tap_refs, -1)?;

        // Project + norm: [B, T, hidden_size]
        let target_projected = self.fc.forward(&target_cat)?;
        let target_hidden = self.hidden_norm.forward(&target_projected)?;

        let mut h = noise.clone();
        for (layer, lc) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, &target_hidden, lc, cache_offset)?;
        }

        self.norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// GDN state save/restore for hybrid models (Qwen3.5)
// ---------------------------------------------------------------------------

/// Saved state for all GDN/linear-attention layers in the target model.
///
/// Much smaller than cloning the full KV cache — only stores `conv_state`,
/// `ssm_state`, and offset for each `ArraysCache` layer.
pub struct GdnStateBackup {
    states: Vec<(Option<Array>, Option<Array>, i32)>,
}

impl GdnStateBackup {
    /// Save GDN (`ArraysCache`) state from all layers. Call BEFORE verify forward.
    /// KV layers are not saved — they use cheap offset-based rollback instead.
    pub fn save(kv_cache: &[Option<crate::qwen3_next::LayerCache>]) -> Result<Self, Exception> {
        let mut states = Vec::with_capacity(kv_cache.len());
        for lc in kv_cache {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.eval_arrays()?;
                    states.push((ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset));
                }
                _ => states.push((None, None, 0)),
            }
        }
        Ok(Self { states })
    }

    /// Restore GDN state and rollback KV offsets. On rejection, call this
    /// BEFORE re-running the accepted tokens.
    pub fn restore_and_rollback(
        &self,
        kv_cache: &mut [Option<crate::qwen3_next::LayerCache>],
        rollback: i32,
    ) {
        for (lc, (conv, ssm, offset)) in kv_cache.iter_mut().zip(self.states.iter()) {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.conv_state.clone_from(conv);
                    ac.ssm_state.clone_from(ssm);
                    ac.offset = *offset;
                }
                Some(crate::qwen3_next::LayerCache::KV(kv)) if rollback > 0 => {
                    kv.trim_by(rollback.unsigned_abs().try_into().unwrap_or(usize::MAX));
                }
                Some(crate::qwen3_next::LayerCache::KV(_)) | None => {}
            }
        }
    }
}

/// Crop the drafter KV cache to `keep_len` along the sequence dim.
///
/// Called after verify to discard rejected positions.
/// Cache tensors have shape `[B, n_kv_heads, seq_len, head_dim]`.
pub fn crop_drafter_cache(cache: &mut [Option<(Array, Array)>], keep_len: i32) {
    use mlx_rs::ops::indexing::IndexOp;
    for (k, v) in cache.iter_mut().filter_map(Option::as_mut) {
        *k = k.index((.., .., ..keep_len, ..));
        *v = v.index((.., .., ..keep_len, ..));
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load a `DFlash` drafter from a directory containing `config.json` + `model.safetensors`.
pub fn load_dflash_drafter(model_path: &Path) -> Result<DFlashDrafter, ModelError> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("reading config.json: {e}"))))?;
    let config: DFlashConfig = serde_json::from_str(&config_str)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("parsing config.json: {e}"))))?;

    let mut drafter = DFlashDrafter::new(config)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    crate::load_safetensors_weights(&mut drafter, model_path)?;

    Ok(drafter)
}

// ---------------------------------------------------------------------------
// Speculative-decode acceptance helper
// ---------------------------------------------------------------------------

/// Greedy speculative-decode acceptance.
///
/// Compares each drafted token to the target's verified argmax. Accepts the
/// longest matching prefix and appends one bonus token (the target's argmax at
/// the position immediately after the rejected draft, or after the last accept
/// if all tokens accepted).
///
/// Returns at least 1 token, at most `draft.len() + 1`.
///
/// # Panics in debug
/// Panics if `verify_argmax.len() != draft.len() + 1`.
pub fn accept_prefix(draft: &[u32], verify_argmax: &[u32]) -> Vec<u32> {
    debug_assert_eq!(verify_argmax.len(), draft.len() + 1);

    let accepted = draft
        .iter()
        .zip(verify_argmax.iter())
        .take_while(|(d, v)| **d == **v)
        .count();
    let mut out: Vec<u32> = draft.get(..accepted).unwrap_or_default().to_vec();
    if let Some(&bonus) = verify_argmax.get(accepted) {
        out.push(bonus);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The full DFlash test suite (~3.8K lines, 30+ end-to-end tests covering the
// draft-verify loop against a real target model) lives on `feat/magic-canvas`
// and will be ported in a follow-up PR alongside the engine glue
// (`SimpleEngine::generate_dflash_inner`).
//
// What's tested here: only the pure helpers (`accept_prefix`) — the
// rest depends on MLX-loaded model weights which aren't part of unit-test
// surface.

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::accept_prefix;

    #[test]
    fn accept_prefix_full_match_returns_draft_plus_bonus() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2, 3, 4];
        assert_eq!(accept_prefix(&draft, &verify), vec![1, 2, 3, 4]);
    }

    #[test]
    fn accept_prefix_first_token_rejects_returns_correction() {
        let draft = vec![1, 2, 3];
        let verify = vec![9, 5, 5, 5];
        assert_eq!(accept_prefix(&draft, &verify), vec![9]);
    }

    #[test]
    fn accept_prefix_partial_match_returns_prefix_plus_correction() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2, 9, 0];
        assert_eq!(accept_prefix(&draft, &verify), vec![1, 2, 9]);
    }

    #[test]
    fn accept_prefix_empty_draft_returns_single_verify_token() {
        let draft: Vec<u32> = vec![];
        let verify = vec![42];
        assert_eq!(accept_prefix(&draft, &verify), vec![42]);
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn accept_prefix_mismatched_lengths_panic_in_debug() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2]; // wrong length
        let _ = accept_prefix(&draft, &verify);
    }

    /// Sliding-window eviction: a tiny random drafter with layer 0 = sliding
    /// (window 4) and layer 1 = full. Driving several rounds must cap the
    /// sliding layer's context KV at `sliding_window - 1` while the full layer
    /// accumulates all context — and the rope offset (max cache length across
    /// layers) must keep advancing absolutely. No weights needed.
    #[test]
    fn sliding_layer_cache_caps_while_full_layer_grows() {
        use super::{DFlashConfig, DFlashDrafter, DFlashSubConfig};
        use mlx_rs::Array;

        let hidden = 8i32;
        let config = DFlashConfig {
            hidden_size: hidden,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            block_size: 4,
            vocab_size: 32,
            layer_types: Some(vec![
                "sliding_attention".to_owned(),
                "full_attention".to_owned(),
            ]),
            sliding_window: Some(4),
            dflash_config: DFlashSubConfig {
                target_layer_ids: vec![0],
                mask_token_id: Some(1),
            },
        };
        let mut drafter = DFlashDrafter::new(config).unwrap();
        let mut cache = drafter.make_cache();

        let ctx_len = 2i32; // context positions added per round
        let rounds = 6;
        let zeros = |t: i32| {
            let n = usize::try_from(t * hidden).unwrap();
            Array::from_slice(&vec![0f32; n], &[1, t, hidden])
        };
        for _ in 0..rounds {
            let taps = vec![zeros(ctx_len)]; // num_taps = 1
            let noise = zeros(4); // block_size noise positions
            drafter.forward(&noise, &taps, &mut cache).unwrap();
        }

        let sliding_len = cache
            .first()
            .unwrap()
            .as_ref()
            .unwrap()
            .0
            .shape()
            .get(2)
            .copied()
            .unwrap();
        let full_len = cache
            .get(1)
            .unwrap()
            .as_ref()
            .unwrap()
            .0
            .shape()
            .get(2)
            .copied()
            .unwrap();
        assert!(
            sliding_len <= 3,
            "sliding cache must cap at window-1=3, got {sliding_len}"
        );
        assert_eq!(
            full_len,
            ctx_len * rounds,
            "full-attn cache must accumulate all {} context positions, got {full_len}",
            ctx_len * rounds
        );
        assert!(
            full_len > sliding_len,
            "eviction must make the sliding cache shorter than the full one"
        );
    }

    /// The "drop-in drafter, no MLX port" claim: the config-driven loader must
    /// parse Modal's config.json and populate every module from Modal's bf16
    /// safetensors by key name. Gated on the real weights dir.
    #[test]
    #[ignore = "load: set HIGGS_DFLASH_DRAFTER_DIR to the Modal drafter snapshot dir"]
    #[allow(clippy::print_stderr)]
    fn loads_modal_drafter_against_real_weights() {
        let Ok(dir) = std::env::var("HIGGS_DFLASH_DRAFTER_DIR") else {
            eprintln!("skip: set HIGGS_DFLASH_DRAFTER_DIR");
            return;
        };
        let d = super::load_dflash_drafter(std::path::Path::new(&dir)).unwrap();
        let c = &d.config;
        assert_eq!(c.num_hidden_layers, 6, "layers");
        assert_eq!(c.hidden_size, 2048, "hidden");
        assert_eq!(c.num_taps(), 8, "taps");
        assert_eq!(
            c.target_layer_ids(),
            &[1, 6, 11, 16, 22, 27, 32, 37],
            "tap ids"
        );
        assert_eq!(c.mask_token_id(), 248_077, "mask token");
        eprintln!(
            "Modal drafter loaded OK: {} layers, hidden {}, {} taps, fc_in {}",
            c.num_hidden_layers,
            c.hidden_size,
            c.num_taps(),
            i32::try_from(c.num_taps()).unwrap() * c.hidden_size
        );
    }
}
