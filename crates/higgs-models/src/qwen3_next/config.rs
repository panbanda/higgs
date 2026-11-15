//! Model arguments for Qwen3-Next and Qwen3.5 checkpoints, and parsing them from `config.json`.

use crate::error::ModelError;
use serde::Deserialize;
use std::path::Path;

const fn default_full_attention_interval() -> i32 {
    4
}

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Match Python mlx-lm default: `norm_topk_prob: bool = True`.
/// Without normalization, `MoE` expert scores sum to ~0.39 instead of 1.0,
/// producing 0.39x output magnitude and degenerate generation.
const fn default_norm_topk_prob() -> bool {
    true
}

pub use crate::quant_config::QuantizationSettings as QuantizationConfig;

/// Configuration for the Qwen3-Next / Qwen3.5 hybrid architecture.
///
/// Supports hybrid SSM/attention transformers with optional Sparse `MoE`.
/// Every `full_attention_interval`-th layer uses full attention, all other
/// layers use `GatedDeltaNet` (SSM-like linear attention). `MoE` layers are
/// enabled when `decoder_sparse_step > 0` and `num_experts > 0`.
///
/// Key fields:
/// - `norm_topk_prob` — normalize top-k expert scores (default `true`).
/// - `gate_quantization` — optional quantization override for `MoE` gate weights.
/// - `use_separate_gdn_projections` — when `true`, GDN layers use 4 separate
///   projection matrices; when `false` (default), projections are fused to 2
///   combined matrices for fewer GPU dispatches.
#[allow(clippy::struct_excessive_bools)]
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
    /// Normalize top-k expert scores to sum to 1.0 before weighting outputs.
    /// Defaults to `true` to match Python mlx-lm. Setting to `false` scales
    /// `MoE` output by the raw softmax scores (~0.39x), causing degenerate output.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Per-layer quantization override for router gate / `shared_expert_gate`.
    /// When absent, uses the global quantization config.
    #[serde(default)]
    pub gate_quantization: Option<QuantizationConfig>,

    /// Use separate GDN projections (qwen3.5-style) instead of combined (qwen3_next-style).
    #[serde(default)]
    pub use_separate_gdn_projections: bool,

    /// Number of MTP (Multi-Token Prediction) hidden layers.
    /// 0 = no MTP head, 1 = one transformer layer for next-next-token prediction.
    #[serde(default)]
    pub mtp_num_hidden_layers: i32,

    /// Use dense projection tensors for the MTP head.
    ///
    /// Some Qwen3.5 MTP sidecars ship full-precision `mtp.*.weight` tensors
    /// rather than quantized `weight/scales/biases` triples. This is set by the
    /// loader after inspecting checkpoint keys; it is not expected in configs.
    #[serde(default)]
    pub use_dense_mtp: bool,

    /// Use an MoE-structured MTP head (Qwen3.6-A3B style).
    ///
    /// These sidecars ship the MTP layer as a full `MoE` decoder layer
    /// (`mlp.gate`, `mlp.switch_mlp.*`, `mlp.shared_expert*`) with a quantized
    /// `fc`. Set by the loader after inspecting checkpoint keys; not expected
    /// in configs.
    #[serde(default)]
    pub use_moe_mtp: bool,
}

/// Load model args from config.json.
pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;
    load_qwen3_next_args_from_value(config)
}

fn gate_quantization_override(config: &serde_json::Value) -> Option<serde_json::Value> {
    let quant = config.get("quantization")?;
    for key in [
        "model.layers.0.mlp.gate",
        "language_model.model.layers.0.mlp.gate",
    ] {
        if let Some(gate_q) = quant.get(key) {
            // `true` means "quantize with the scalar defaults" (see
            // QuantizationSettings::deserialize) — identical to the key
            // being absent. Treat it as such rather than injecting it into
            // `gate_quantization`, whose type requires an object with
            // group_size/bits and would fail to deserialize a bare bool.
            if gate_q == &serde_json::Value::Bool(true) {
                continue;
            }
            return Some(gate_q.clone());
        }
    }
    None
}

pub fn load_qwen3_next_args_from_value(
    mut config: serde_json::Value,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let gate_override = gate_quantization_override(&config);
    let map = config
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("config.json root is not an object".into()))?;
    if !map.contains_key("gate_quantization") {
        if let Some(gate_q) = gate_override {
            map.insert("gate_quantization".to_owned(), gate_q);
        }
    }
    Ok(serde_json::from_value(config)?)
}

/// Load model args from a Qwen3.5-MoE VLM config.json.
///
/// Qwen3.5-MoE uses the same architecture as `Qwen3Next` (hybrid
/// `GatedDeltaNet` + full attention + sparse `MoE` with shared expert) but ships
/// as a VLM with config nested under `text_config` and rope parameters nested
/// under `rope_parameters`.
pub(super) fn load_qwen3_5_moe_text_config_args<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;
    load_qwen3_5_text_config_args_from_value(&config)
}

pub fn load_qwen3_5_text_config_args_from_value(
    config: &serde_json::Value,
) -> Result<Qwen3NextModelArgs, ModelError> {
    let text_config = config
        .get("text_config")
        .ok_or_else(|| ModelError::UnsupportedModel("missing text_config in config.json".into()))?;

    let mut obj = text_config.clone();
    let map = obj
        .as_object_mut()
        .ok_or_else(|| ModelError::UnsupportedModel("text_config is not an object".into()))?;

    // Flatten rope_parameters into top-level fields
    if let Some(rope_params) = text_config.get("rope_parameters") {
        if let Some(theta) = rope_params.get("rope_theta") {
            map.entry("rope_theta").or_insert_with(|| theta.clone());
        }
        if let Some(prf) = rope_params.get("partial_rotary_factor") {
            map.entry("partial_rotary_factor")
                .or_insert_with(|| prf.clone());
        }
    }

    // Merge top-level quantization config
    if let Some(quant) = config.get("quantization") {
        map.entry("quantization").or_insert_with(|| quant.clone());
    }

    // Merge top-level tie_word_embeddings
    if let Some(tie) = config.get("tie_word_embeddings") {
        map.entry("tie_word_embeddings")
            .or_insert_with(|| tie.clone());
    }

    // Set decoder_sparse_step=1 only for MoE models (num_experts > 0).
    // Dense models (qwen3_5) use standard FFN and must keep decoder_sparse_step=0.
    let has_experts = text_config
        .get("num_experts")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0)
        > 0;
    if has_experts {
        map.entry("decoder_sparse_step")
            .or_insert(serde_json::Value::from(1));
    }

    // intermediate_size is unused when all layers are MoE;
    // for dense models, keep whatever value is in text_config.
    if has_experts {
        map.entry("intermediate_size")
            .or_insert(serde_json::Value::from(0));
    }

    // When HIGGS_SEPARATE_GDN_PROJ is set, or when per-layer GDN BA quantization
    // disagrees on bit-width / group_size between in_proj_a and in_proj_b (common
    // in Unsloth dynamic quants), construct the model with separate GDN
    // projection fields so the direct weight loader can match them. Otherwise,
    // construct with fused fields (weights are rearranged at load time).
    let mixed_ba_layers = qwen3_5_mixed_ba_quantization_layers(config, text_config);
    let config_requests_separate = map
        .get("use_separate_gdn_projections")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let use_separate = config_requests_separate
        || std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok()
        || !mixed_ba_layers.is_empty();
    map.insert(
        "use_separate_gdn_projections".to_owned(),
        serde_json::Value::from(use_separate),
    );
    if !mixed_ba_layers.is_empty() {
        tracing::info!(
            layers = ?mixed_ba_layers,
            "Detected mixed-bit GDN BA projections; using separate GDN projections"
        );
    }

    // Detect per-layer gate quantization override from top-level quantization config
    if let Some(gate_q) = gate_quantization_override(config) {
        map.insert("gate_quantization".to_owned(), gate_q);
    }

    Ok(serde_json::from_value(obj)?)
}

/// Parse a `{group_size, bits}` quantization spec from a JSON node.
fn qwen3_5_quantization_config(value: &serde_json::Value) -> Option<QuantizationConfig> {
    Some(QuantizationConfig::new(
        i32::try_from(value.get("group_size")?.as_i64()?).ok()?,
        i32::try_from(value.get("bits")?.as_i64()?).ok()?,
    ))
}

/// Scan the per-layer `quantization` map and return layer indices where the GDN
/// `in_proj_a` and `in_proj_b` projections disagree on bit-width or group size.
/// Such layers cannot be fused into a single `in_proj_ba` matrix without
/// dequantizing, so the loader must fall back to separate GDN projections.
fn qwen3_5_mixed_ba_quantization_layers(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
) -> Vec<i32> {
    let Some(quant) = config.get("quantization") else {
        return Vec::new();
    };
    let Some(default_quant) = qwen3_5_quantization_config(quant) else {
        return Vec::new();
    };
    let Some(num_hidden_layers) = text_config
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
    else {
        return Vec::new();
    };

    (0..num_hidden_layers)
        .filter(|layer_idx| {
            let prefixes = [
                format!("language_model.model.layers.{layer_idx}.linear_attn"),
                format!("model.layers.{layer_idx}.linear_attn"),
            ];
            let projection_quantization = |projection: &str| {
                prefixes
                    .iter()
                    .find_map(|prefix| {
                        quant
                            .get(format!("{prefix}.{projection}"))
                            .and_then(qwen3_5_quantization_config)
                    })
                    .unwrap_or_else(|| default_quant.clone())
            };
            let a_quant = projection_quantization("in_proj_a");
            let b_quant = projection_quantization("in_proj_b");
            // The qkvz fusion pair has the same constraint: `in_proj_qkv` and
            // `in_proj_z` are concatenated into `in_proj_qkvz`, which is only
            // possible when their packed (quantized) shapes agree. Mixed-
            // precision quants (e.g. OptiQ) assign different bit-widths per
            // projection on sensitive layers, so check both fusion pairs.
            let qkv_quant = projection_quantization("in_proj_qkv");
            let z_quant = projection_quantization("in_proj_z");
            a_quant.bits != b_quant.bits
                || a_quant.group_size != b_quant.group_size
                || qkv_quant.bits != z_quant.bits
                || qkv_quant.group_size != z_quant.group_size
        })
        .collect()
}
