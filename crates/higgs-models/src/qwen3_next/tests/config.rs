//! Tests for model argument parsing.

use crate::qwen3_next::{
    config::{Qwen3NextModelArgs, load_model_args, load_qwen3_next_args_from_value},
    model::Qwen3NextCausalLM,
};

#[test]
fn test_config_deserialization() {
    let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 262144,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "full_attention_interval": 4,
            "tie_word_embeddings": false,
            "quantization": { "group_size": 64, "bits": 4 }
        }"#;

    let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.model_type, "qwen3_next");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_hidden_layers, 48);
    assert_eq!(args.head_dim, 256);
    assert_eq!(args.num_experts, 512);
    assert_eq!(args.num_experts_per_tok, 10);
    assert_eq!(args.full_attention_interval, 4);
    assert_eq!(args.linear_conv_kernel_dim, 4);
    assert!(!args.tie_word_embeddings);
    assert!(args.norm_topk_prob);
    let qc = args.quantization.unwrap();
    assert_eq!(qc.group_size, 64);
    assert_eq!(qc.bits, 4);
}

#[test]
fn test_config_deserialization_missing_optional_fields() {
    // Only required fields; all serde(default) fields should get defaults
    let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
    let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
    assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
    assert!((args.partial_rotary_factor - 1.0).abs() < f32::EPSILON);
    assert_eq!(args.full_attention_interval, 4);
    assert!(!args.tie_word_embeddings);
    assert!(!args.attention_bias);
    assert!(args.rope_scaling.is_none());
    assert!(args.quantization.is_none());
    assert_eq!(args.linear_num_value_heads, 0);
    assert_eq!(args.linear_num_key_heads, 0);
    assert_eq!(args.linear_key_head_dim, 0);
    assert_eq!(args.linear_value_head_dim, 0);
    assert_eq!(args.linear_conv_kernel_dim, 0);
    assert_eq!(args.num_experts, 0);
    assert_eq!(args.num_experts_per_tok, 0);
    assert_eq!(args.decoder_sparse_step, 0);
    assert!(args.norm_topk_prob);
    assert_eq!(args.mlp_only_layers, Vec::<i32>::new());
}

#[test]
fn test_config_deserialization_quantization_null() {
    let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "quantization": null
        }"#;
    let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
    assert!(args.quantization.is_none());
}

#[test]
fn test_load_qwen3_next_args_injects_gate_quantization_override() {
    let config = serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": 32,
        "moe_intermediate_size": 32,
        "full_attention_interval": 4,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.mlp.gate": {
                "group_size": 64,
                "bits": 8
            }
        }
    });

    let args = load_qwen3_next_args_from_value(config).unwrap();
    let gate_q = args.gate_quantization.unwrap();
    assert_eq!(gate_q.group_size, 64);
    assert_eq!(gate_q.bits, 8);
}

#[test]
fn test_load_qwen3_next_args_treats_true_gate_override_as_default() {
    // Real predicate-map checkpoints list every tensor path explicitly,
    // including the layer-0 router gate as `true` ("use the scalar
    // defaults"). That must NOT be injected into `gate_quantization`
    // (whose type requires a {group_size, bits} object) — it should be
    // treated exactly like the key being absent.
    let config = serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": 32,
        "moe_intermediate_size": 32,
        "full_attention_interval": 4,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.mlp.gate": true,
            "model.layers.0.mlp.shared_expert_gate": true,
            "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
            "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
            "model.embed_tokens": true,
            "lm_head": false
        }
    });

    let args = load_qwen3_next_args_from_value(config).unwrap();
    assert!(
        args.gate_quantization.is_none(),
        "a `true` override at the scanned layer must not populate gate_quantization"
    );

    // Loads end-to-end with the gate at default (scalar) quantization.
    let model = Qwen3NextCausalLM::new(args).unwrap();
    assert_eq!(model.args.num_hidden_layers, 2);
}

#[test]
fn test_real_checkpoint_style_config_loads_end_to_end() {
    // Full pipeline (config.json -> args -> model) for a config shaped
    // like a real mlx-community Qwen3-Next 5-bit checkpoint: per-layer
    // router gate overrides at every MoE layer, not just layer 0.
    let config = serde_json::json!({
        "model_type": "qwen3_next",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "shared_expert_intermediate_size": 32,
        "moe_intermediate_size": 32,
        "full_attention_interval": 4,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
            "model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
            "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
            "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
            "model.embed_tokens": true,
            "lm_head": false
        }
    });

    let args = load_qwen3_next_args_from_value(config).unwrap();
    Qwen3NextCausalLM::new(args).unwrap();
}

#[test]
fn test_load_model_args_happy_path() {
    let dir = tempfile::tempdir().unwrap();
    let config = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 262144
        }"#;
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    let args = load_model_args(dir.path()).unwrap();
    assert_eq!(args.model_type, "qwen3_next");
    assert_eq!(args.hidden_size, 2048);
}

#[test]
fn test_load_model_args_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let result = load_model_args(dir.path());
    assert!(result.is_err());
}

#[test]
fn test_load_model_args_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), "{{bad json").unwrap();
    let result = load_model_args(dir.path());
    assert!(result.is_err());
}
