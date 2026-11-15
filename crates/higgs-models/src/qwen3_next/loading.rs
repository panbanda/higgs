//! Entry points that build and load Qwen3-Next and Qwen3.5 models from a checkpoint directory.

use super::{
    checkpoint::maybe_disable_mtp_without_checkpoint_weights,
    config::{Qwen3NextModelArgs, load_model_args, load_qwen3_5_moe_text_config_args},
    model::Qwen3NextCausalLM,
    weights::{
        GdnDims, load_qwen3_5_moe_weights_direct, load_qwen3_5_moe_weights_fused,
        load_qwen3_next_weights,
    },
};
use crate::error::ModelError;
use std::path::Path;

/// Load a `Qwen3Next` model from a directory containing safetensors + config.json.
pub fn load_qwen3_next_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;
    load_qwen3_next_model_with_args(model_path, args)
}

pub fn load_qwen3_next_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        "Loading qwen3_next model"
    );

    let quantization = args.quantization.clone();
    let mut model = Qwen3NextCausalLM::new(args)?;

    // Backbone keys match model params directly, but the MTP sidecar may need
    // remapping: `maybe_disable_mtp_without_checkpoint_weights` can select the
    // dense or MoE head (params `dense_mtp.*` / `moe_mtp.*`) while the checkpoint
    // still ships the head under the `mtp.*` namespace. The plain loader can't
    // bridge that, so it would silently leave the draft head uninitialized.
    load_qwen3_next_weights(&mut model, model_path, quantization.as_ref())?;

    tracing::info!("Qwen3Next model loaded successfully");
    Ok(model)
}

/// Load a Qwen3.5 dense model (VLM wrapper around `Qwen3Next` architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys. Unlike [`load_qwen3_5_moe_model`], does NOT force
/// `decoder_sparse_step=1` or attempt `MoE` gate fusion.
pub fn load_qwen3_5_model<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_qwen3_5_moe_text_config_args(model_path)?;
    load_qwen3_5_model_with_args(model_path, args)
}

pub fn load_qwen3_5_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5 dense model (VLM text backbone via qwen3_next)"
    );

    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    gdn_dims.validate()?;
    let model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

    tracing::info!("Qwen3.5 dense model loaded successfully");
    Ok(model)
}

/// Load a Qwen3.5-MoE model (VLM wrapper around `Qwen3Next` architecture).
///
/// Reads `text_config` for model args, strips `language_model.` prefix from
/// safetensors weight keys.
pub fn load_qwen3_5_moe_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_qwen3_5_moe_text_config_args(model_path)?;
    load_qwen3_5_moe_model_with_args(model_path, args)
}

pub fn load_qwen3_5_moe_model_with_args(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
) -> Result<Qwen3NextCausalLM, ModelError> {
    maybe_disable_mtp_without_checkpoint_weights(&mut args, model_path)?;

    tracing::info!(
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        full_attention_interval = args.full_attention_interval,
        "Loading qwen3_5_moe model (VLM text backbone via qwen3_next)"
    );

    // Save GDN dimensions before args is moved
    let gdn_dims = GdnDims {
        num_k_heads: args.linear_num_key_heads,
        num_v_heads: args.linear_num_value_heads,
        head_k_dim: args.linear_key_head_dim,
        head_v_dim: args.linear_value_head_dim,
    };
    gdn_dims.validate()?;
    // Load weights with GDN projection rearrangement: flat (qkv,z,b,a)
    // → per-head-grouped (qkvz,ba) for fused 2-dispatch forward path.
    // Respects use_separate_gdn_projections (set by HIGGS_SEPARATE_GDN_PROJ env
    // var or mixed-bit BA detection in load_qwen3_5_moe_text_config_args), and
    // falls back to separate projections at runtime if fusion finds a
    // shape-incompatible BA pair.
    let model = load_qwen3_5_model_with_gdn_fallback(model_path, args, &gdn_dims)?;

    tracing::info!("Qwen3.5-MoE model loaded successfully");
    Ok(model)
}

/// Build a `Qwen3NextCausalLM` and load weights, choosing fused or separate GDN
/// projections. When the config (or env var) requests separate projections, use
/// the direct loader. Otherwise try the fused loader; if it reports a mixed-bit
/// `in_proj_ba` shape mismatch, rebuild the model with separate projections and
/// retry via the direct loader.
fn load_qwen3_5_model_with_gdn_fallback(
    model_path: &Path,
    mut args: Qwen3NextModelArgs,
    gdn_dims: &GdnDims,
) -> Result<Qwen3NextCausalLM, ModelError> {
    let quantization = args.quantization.clone();
    let force_separate =
        args.use_separate_gdn_projections || std::env::var("HIGGS_SEPARATE_GDN_PROJ").is_ok();
    if force_separate {
        args.use_separate_gdn_projections = true;
        let mut model = Qwen3NextCausalLM::new(args)?;
        load_qwen3_5_moe_weights_direct(&mut model, model_path, quantization.as_ref())?;
        tracing::info!("Using SEPARATE GDN projections (4 dispatches per layer)");
        return Ok(model);
    }

    let mut fused_model = Qwen3NextCausalLM::new(args.clone())?;
    match load_qwen3_5_moe_weights_fused(
        &mut fused_model,
        model_path,
        gdn_dims,
        quantization.as_ref(),
    ) {
        Ok(()) => {
            tracing::info!("Using FUSED GDN projections (2 dispatches per layer)");
            Ok(fused_model)
        }
        Err(err) if is_mixed_bit_gdn_ba_fusion_error(&err) => {
            tracing::warn!(
                error = %err,
                "Detected mixed-bit GDN BA projection shapes; retrying with separate GDN projections"
            );
            args.use_separate_gdn_projections = true;
            let mut separate_model = Qwen3NextCausalLM::new(args)?;
            load_qwen3_5_moe_weights_direct(
                &mut separate_model,
                model_path,
                quantization.as_ref(),
            )?;
            tracing::info!(
                "Using SEPARATE GDN projections (4 dispatches per layer, mixed-bit fallback)"
            );
            Ok(separate_model)
        }
        Err(err) => Err(err),
    }
}

/// Returns true when the supplied error is the mixed-bit BA fusion error raised
/// by [`load_qwen3_5_moe_weights_fused`] when `in_proj_a` and `in_proj_b` have
/// incompatible packed inner shapes.
fn is_mixed_bit_gdn_ba_fusion_error(err: &ModelError) -> bool {
    matches!(
        err,
        ModelError::ShapeMismatch(message)
            if message.contains("in_proj_ba")
                && message.contains("requires separate GDN projections")
    )
}
