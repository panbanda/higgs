//! Tests for checkpoint inspection and model loading.

use super::valid_causal_lm_args;
use crate::{
    error::ModelError,
    qwen3_next::{
        checkpoint::{
            MtpWeightLayout, checkpoint_has_mtp_weights, checkpoint_mtp_weight_layout,
            dense_mtp_param_key, dense_mtp_rmsnorm_weight_key,
            maybe_disable_mtp_without_checkpoint_weights, moe_mtp_param_key,
            mtp_weight_layout_from_keys, normalize_sidecar_mtp_key, qwen35_checkpoint_param_key,
        },
        config::{QuantizationConfig, load_qwen3_5_moe_text_config_args},
        model::Qwen3NextCausalLM,
        weights::{
            can_concatenate0_shapes, ensure_all_model_params_loaded, load_qwen3_next_weights,
            placeholder_param_names,
        },
    },
};
use mlx_rs::Array;
use std::path::Path;

#[test]
fn test_placeholder_param_names_finds_shape_one_tensors() {
    let loaded = Array::from_slice(&[1.0f32, 2.0], &[2]);
    let placeholder = Array::from_slice(&[0.0f32], &[1]);
    let names =
        placeholder_param_names([("loaded.weight", &loaded), ("missing.weight", &placeholder)]);
    assert_eq!(names, vec!["missing.weight".to_owned()]);
}

#[test]
fn test_ensure_all_model_params_loaded_errors_on_placeholders() {
    let loaded = Array::from_slice(&[1.0f32, 2.0], &[2]);
    let placeholder = Array::from_slice(&[0.0f32], &[1]);
    let err = ensure_all_model_params_loaded([
        ("loaded.weight", &loaded),
        ("missing.weight", &placeholder),
    ])
    .unwrap_err();
    if let ModelError::MissingWeight(message) = err {
        assert!(message.contains("1 model params"));
        assert!(message.contains("missing.weight"));
    } else {
        panic!("expected MissingWeight");
    }
}

#[test]
fn test_ensure_all_model_params_loaded_accepts_fully_loaded_params() {
    let weight = Array::from_slice(&[1.0f32, 2.0], &[2]);
    ensure_all_model_params_loaded([("loaded.weight", &weight)]).unwrap();
}

/// A malformed `scales` width must fail at load time through the custom
/// `load_qwen3_next_weights` shard loop, not surface as an opaque Metal
/// kernel error on first forward. This exercises the same
/// `validate_quantized_tensor_widths` check that the generic loaders in
/// `lib.rs` already run, wired into the qwen3_next-specific loader.
#[test]
fn test_load_qwen3_next_weights_rejects_malformed_scales_width() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"quantization": {"group_size": 64, "bits": 4}}"#,
    )
    .unwrap();

    // weight: [4, 8] packed i.e. bits=4, group_size=64 implies scales
    // width should be 8*32/4/64 = 1, but we provide width 2.
    let weight_data = vec![0_u8; 4 * 8 * 4];
    let scales_data = vec![0_u8; 4 * 2 * 4];
    let weight_tensor = safetensors::tensor::TensorView::new(
        safetensors::tensor::Dtype::F32,
        vec![4, 8],
        &weight_data,
    )
    .unwrap();
    let scales_tensor = safetensors::tensor::TensorView::new(
        safetensors::tensor::Dtype::F32,
        vec![4, 2],
        &scales_data,
    )
    .unwrap();
    safetensors::serialize_to_file(
        [
            ("model.layers.0.self_attn.q_proj.weight", weight_tensor),
            ("model.layers.0.self_attn.q_proj.scales", scales_tensor),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();

    let mut model = Qwen3NextCausalLM::new(valid_causal_lm_args()).unwrap();
    let quantization: QuantizationConfig =
        serde_json::from_str(r#"{"group_size": 64, "bits": 4}"#).unwrap();
    let err = load_qwen3_next_weights(&mut model, dir.path(), Some(&quantization)).unwrap_err();
    assert!(
        err.to_string().contains("model.layers.0.self_attn.q_proj"),
        "unexpected error: {err}"
    );
}

/// Write a qwen3.5-style VLM config.json (with text_config) and parse it.
fn write_qwen35_config(dir: &std::path::Path, text_config_json: &str) {
    let config = format!(r#"{{"text_config": {text_config_json}, "tie_word_embeddings": false}}"#);
    std::fs::write(dir.join("config.json"), config).unwrap();
}

/// Helper: minimal qwen3.5 text_config JSON for a dense (non-MoE) model.
fn qwen35_dense_text_config() -> &'static str {
    r#"{
            "model_type": "qwen3_5",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0
        }"#
}

/// Helper: minimal qwen3.5 text_config JSON for an MoE model.
fn qwen35_moe_text_config() -> &'static str {
    r#"{
            "model_type": "qwen3_5_moe",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 256,
            "moe_intermediate_size": 128,
            "norm_topk_prob": true
        }"#
}

fn write_weight_index(dir: &std::path::Path, keys: &[&str]) {
    let weight_map = keys
        .iter()
        .map(|key| format!(r#""{key}": "model-00001-of-00001.safetensors""#))
        .collect::<Vec<_>>()
        .join(", ");
    let index = format!(r#"{{"metadata": {{}}, "weight_map": {{{weight_map}}}}}"#);
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
}

fn write_safetensors_file(dir: &std::path::Path, file_name: &str, key: &str) {
    let data = [0_u8; 4];
    let tensor =
        safetensors::tensor::TensorView::new(safetensors::tensor::Dtype::F32, vec![1], &data)
            .unwrap();
    safetensors::serialize_to_file([(key, tensor)], None, &dir.join(file_name)).unwrap();
}

#[test]
fn test_load_qwen35_moe_text_config_moe_sets_decoder_sparse_step() {
    let dir = tempfile::tempdir().unwrap();
    write_qwen35_config(dir.path(), qwen35_moe_text_config());
    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
    assert_eq!(
        args.decoder_sparse_step, 1,
        "MoE model should get decoder_sparse_step=1"
    );
    assert!(args.num_experts > 0);
}

#[test]
fn test_load_qwen35_dense_text_config_no_forced_moe() {
    let dir = tempfile::tempdir().unwrap();
    write_qwen35_config(dir.path(), qwen35_dense_text_config());
    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
    // Dense models (num_experts=0) must NOT get decoder_sparse_step=1,
    // otherwise every layer tries to create SparseMoeBlock and fails.
    assert_eq!(
        args.decoder_sparse_step, 0,
        "Dense model should NOT get decoder_sparse_step=1"
    );
    assert_eq!(args.num_experts, 0);
}

#[test]
fn test_checkpoint_has_mtp_weights_detects_prefixed_keys() {
    let dir = tempfile::tempdir().unwrap();
    write_weight_index(
        dir.path(),
        &["language_model.mtp.layers.0.self_attn.q_proj.weight"],
    );
    assert!(checkpoint_has_mtp_weights(dir.path()).unwrap());
}

#[test]
fn test_checkpoint_has_mtp_weights_detects_auxiliary_mtp_file() {
    let dir = tempfile::tempdir().unwrap();
    write_weight_index(
        dir.path(),
        &["language_model.model.layers.0.input_layernorm.weight"],
    );
    write_safetensors_file(
        dir.path(),
        "model-mtp.safetensors",
        "language_model.mtp.layers.0.self_attn.q_proj.weight",
    );

    assert!(checkpoint_has_mtp_weights(dir.path()).unwrap());
}

#[test]
fn test_checkpoint_mtp_weight_layout_detects_quantized_indexed_keys() {
    let dir = tempfile::tempdir().unwrap();
    write_weight_index(
        dir.path(),
        &[
            "language_model.mtp.layers.0.self_attn.q_proj.weight",
            "language_model.mtp.layers.0.self_attn.q_proj.scales",
            "language_model.mtp.layers.0.self_attn.q_proj.biases",
        ],
    );

    assert_eq!(
        checkpoint_mtp_weight_layout(dir.path()).unwrap(),
        MtpWeightLayout::Quantized
    );
}

#[test]
fn test_mtp_layout_detects_moe_structured_head() {
    // Qwen3.6-A3B style: the MTP layer is a full MoE layer.
    let layout = mtp_weight_layout_from_keys([
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.scales",
        "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
        "mtp.fc.weight",
    ]);
    assert_eq!(layout, MtpWeightLayout::MoeQuantized);
}

#[test]
fn test_moe_mtp_param_key_remaps_mtp_prefix() {
    assert_eq!(
        moe_mtp_param_key("mtp.layers.0.mlp.gate.weight").as_deref(),
        Some("moe_mtp.layers.0.mlp.gate.weight")
    );
    assert_eq!(
        moe_mtp_param_key("mtp.fc.scales").as_deref(),
        Some("moe_mtp.fc.scales")
    );
    assert!(moe_mtp_param_key("model.layers.0.mlp.gate.weight").is_none());
}

#[test]
fn test_normalize_sidecar_mtp_key_prefixes_aux_files_only() {
    let aux = Path::new("/models/x/mtp.safetensors");
    let main = Path::new("/models/x/model-00001-of-00004.safetensors");
    // Unprefixed keys from the sidecar get the mtp. prefix.
    assert_eq!(
        normalize_sidecar_mtp_key(aux, "fc.weight".to_owned()),
        "mtp.fc.weight"
    );
    assert_eq!(
        normalize_sidecar_mtp_key(aux, "layers.0.mlp.gate.weight".to_owned()),
        "mtp.layers.0.mlp.gate.weight"
    );
    // Already-prefixed sidecar keys are unchanged.
    assert_eq!(
        normalize_sidecar_mtp_key(aux, "mtp.fc.weight".to_owned()),
        "mtp.fc.weight"
    );
    // Already-namespaced sidecar keys (e.g. `language_model.mtp.*`) must NOT
    // be over-prefixed into unmatchable `mtp.language_model.mtp.*`.
    assert_eq!(
        normalize_sidecar_mtp_key(aux, "language_model.mtp.layers.0.fc.weight".to_owned()),
        "language_model.mtp.layers.0.fc.weight"
    );
    // Keys from main shards are never touched.
    assert_eq!(
        normalize_sidecar_mtp_key(main, "fc.weight".to_owned()),
        "fc.weight"
    );
}

#[test]
fn test_checkpoint_mtp_weight_layout_detects_dense_auxiliary_mtp_file() {
    let dir = tempfile::tempdir().unwrap();
    write_weight_index(
        dir.path(),
        &["language_model.model.layers.0.input_layernorm.weight"],
    );
    write_safetensors_file(
        dir.path(),
        "model-mtp.safetensors",
        "mtp.layers.0.self_attn.q_proj.weight",
    );

    assert_eq!(
        checkpoint_mtp_weight_layout(dir.path()).unwrap(),
        MtpWeightLayout::Dense
    );
}

#[test]
fn test_qwen35_checkpoint_key_accepts_unprefixed_mtp_sidecar() {
    assert_eq!(
        qwen35_checkpoint_param_key("mtp.layers.0.self_attn.q_proj.weight"),
        Some("mtp.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(
        qwen35_checkpoint_param_key("language_model.model.layers.0.input_layernorm.weight"),
        Some("model.layers.0.input_layernorm.weight")
    );
    assert_eq!(qwen35_checkpoint_param_key("vision_tower.foo"), None);
}

#[test]
fn test_dense_mtp_param_key_remaps_mtp_namespace() {
    assert_eq!(
        dense_mtp_param_key("mtp.layers.0.self_attn.q_proj.weight").as_deref(),
        Some("dense_mtp.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(dense_mtp_param_key("model.layers.0.foo"), None);
}

#[test]
fn test_dense_mtp_rmsnorm_weight_keys_require_plus_one() {
    assert!(dense_mtp_rmsnorm_weight_key(
        "mtp.layers.0.input_layernorm.weight"
    ));
    assert!(dense_mtp_rmsnorm_weight_key(
        "mtp.layers.0.self_attn.q_norm.weight"
    ));
    assert!(dense_mtp_rmsnorm_weight_key(
        "mtp.pre_fc_norm_hidden.weight"
    ));
    assert!(!dense_mtp_rmsnorm_weight_key(
        "mtp.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn test_maybe_disable_mtp_without_checkpoint_weights_turns_off_missing_mtp() {
    let dir = tempfile::tempdir().unwrap();
    write_qwen35_config(dir.path(), qwen35_moe_text_config());
    write_weight_index(
        dir.path(),
        &["language_model.model.layers.0.input_layernorm.weight"],
    );
    let mut args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
    args.mtp_num_hidden_layers = 1;
    maybe_disable_mtp_without_checkpoint_weights(&mut args, dir.path()).unwrap();
    assert_eq!(args.mtp_num_hidden_layers, 0);
}

#[test]
fn test_maybe_disable_mtp_without_checkpoint_weights_preserves_present_mtp() {
    let dir = tempfile::tempdir().unwrap();
    write_qwen35_config(dir.path(), qwen35_moe_text_config());
    write_weight_index(
        dir.path(),
        &["language_model.mtp.layers.0.self_attn.q_proj.weight"],
    );
    let mut args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();
    args.mtp_num_hidden_layers = 1;
    maybe_disable_mtp_without_checkpoint_weights(&mut args, dir.path()).unwrap();
    assert_eq!(args.mtp_num_hidden_layers, 1);
}

#[test]
fn test_load_qwen35_mixed_ba_quantization_forces_separate_gdn() {
    let dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
        qwen35_dense_text_config()
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();

    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

    assert!(
        args.use_separate_gdn_projections,
        "mixed-bit in_proj_a/in_proj_b must force separate GDN projections"
    );
}

#[test]
fn test_load_qwen35_mixed_ba_quantization_supports_unprefixed_layer_keys() {
    let dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
        qwen35_dense_text_config()
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();

    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

    assert!(
        args.use_separate_gdn_projections,
        "unprefixed mixed-bit in_proj_a/in_proj_b must force separate GDN projections"
    );
}

#[test]
fn test_load_qwen35_mixed_qkvz_quantization_forces_separate_gdn() {
    // Mixed-precision quants (e.g. OptiQ) can also put `in_proj_qkv` and
    // `in_proj_z` at different bit-widths — that breaks the qkvz fusion
    // concat exactly like a mixed BA pair does.
    let dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.2.linear_attn.in_proj_z": {{
                        "group_size": 64,
                        "bits": 8,
                        "mode": "affine"
                    }}
                }}
            }}"#,
        qwen35_dense_text_config()
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();

    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

    assert!(
        args.use_separate_gdn_projections,
        "mixed-bit in_proj_qkv/in_proj_z must force separate GDN projections"
    );
}

#[test]
fn test_load_qwen35_matching_ba_quantization_keeps_fused_gdn() {
    let dir = tempfile::tempdir().unwrap();
    let config = format!(
        r#"{{
                "text_config": {},
                "tie_word_embeddings": false,
                "quantization": {{
                    "group_size": 64,
                    "bits": 2,
                    "mode": "affine",
                    "language_model.model.layers.1.linear_attn.in_proj_a": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }},
                    "language_model.model.layers.1.linear_attn.in_proj_b": {{
                        "group_size": 64,
                        "bits": 5,
                        "mode": "affine"
                    }}
                }}
            }}"#,
        qwen35_dense_text_config()
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();

    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

    assert!(
        !args.use_separate_gdn_projections,
        "matching BA overrides should keep the fused GDN loader path"
    );
}

#[test]
fn test_load_qwen35_explicit_separate_gdn_config_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let mut text_config = qwen35_dense_text_config().trim_end_matches('}').to_owned();
    text_config.push_str(
        r#",
            "use_separate_gdn_projections": true
        }"#,
    );
    write_qwen35_config(dir.path(), &text_config);

    let args = load_qwen3_5_moe_text_config_args(dir.path()).unwrap();

    assert!(
        args.use_separate_gdn_projections,
        "explicit use_separate_gdn_projections=true must not be overwritten"
    );
}

#[test]
fn test_can_concatenate0_detects_quantized_inner_shape_mismatch() {
    assert!(
        !can_concatenate0_shapes(&[48, 320], &[48, 800]),
        "different packed inner dims must block BA fusion"
    );
    assert!(
        can_concatenate0_shapes(&[48, 320], &[96, 320]),
        "axis-0 size may differ because fusion concatenates rows"
    );
}
