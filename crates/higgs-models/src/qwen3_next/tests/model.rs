//! Tests for model construction and cache layout.

use super::valid_causal_lm_args;
use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    qwen3_next::{
        cache::{ArraysCache, LayerCache},
        config::QuantizationConfig,
        model::Qwen3NextCausalLM,
    },
};

#[test]
fn test_causal_lm_rejects_zero_full_attention_interval() {
    let mut args = valid_causal_lm_args();
    args.full_attention_interval = 0;
    let result = Qwen3NextCausalLM::new(args);
    assert!(
        result.is_err(),
        "Should reject full_attention_interval == 0"
    );
}

#[test]
fn test_causal_lm_rejects_zero_linear_key_heads() {
    let mut args = valid_causal_lm_args();
    args.linear_num_key_heads = 0;
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_err(), "Should reject linear_num_key_heads == 0");
}

#[test]
fn test_causal_lm_rejects_zero_linear_value_heads() {
    let mut args = valid_causal_lm_args();
    args.linear_num_value_heads = 0;
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_err(), "Should reject linear_num_value_heads == 0");
}

#[test]
fn test_causal_lm_rejects_unsupported_per_tensor_override() {
    // self_attn projections are always built from the scalar
    // group_size/bits fallback; an override for one must be rejected
    // rather than silently ignored.
    let mut args = valid_causal_lm_args();
    args.quantization = Some(
        serde_json::from_str(
            r#"{"group_size": 64, "bits": 4, "model.layers.0.self_attn.q_proj": false}"#,
        )
        .unwrap(),
    );
    let err = Qwen3NextCausalLM::new(args).unwrap_err();
    assert!(
        err.to_string().contains("model.layers.0.self_attn.q_proj"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_causal_lm_accepts_honored_expert_projection_override() {
    // valid_causal_lm_args() has num_experts=4; resolve_uniform requires
    // every expert in the fused group to share the override.
    let mut args = valid_causal_lm_args();
    args.quantization = Some(
        serde_json::from_str(
            r#"{
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.0.mlp.experts.0.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.1.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.2.gate_proj": {"group_size": 32, "bits": 8},
                    "model.layers.0.mlp.experts.3.gate_proj": {"group_size": 32, "bits": 8}
                }"#,
        )
        .unwrap(),
    );
    Qwen3NextCausalLM::new(args).unwrap();
}

#[test]
fn test_causal_lm_accepts_real_checkpoint_per_layer_router_gate_overrides() {
    // Mirrors real mlx-community Qwen3-Next 5-bit checkpoints (e.g.
    // Qwen3-Next-80B-A3B-Thinking-5bit), which carry a per-layer 8-bit
    // router override for both `mlp.gate` and `mlp.shared_expert_gate`
    // at every MoE layer, not just layer 0.
    let mut args = valid_causal_lm_args();
    args.quantization = Some(
        serde_json::from_str(
            r#"{
                    "group_size": 64,
                    "bits": 4,
                    "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.0.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.1.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.2.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.2.mlp.shared_expert_gate": {"group_size": 64, "bits": 8},
                    "model.layers.3.mlp.gate": {"group_size": 64, "bits": 8},
                    "model.layers.3.mlp.shared_expert_gate": {"group_size": 64, "bits": 8}
                }"#,
        )
        .unwrap(),
    );
    Qwen3NextCausalLM::new(args).unwrap();
}

#[test]
fn test_causal_lm_rejects_zero_conv_kernel_dim() {
    let mut args = valid_causal_lm_args();
    args.linear_conv_kernel_dim = 0;
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_err(), "Should reject linear_conv_kernel_dim == 0");
}

#[test]
fn test_layer_cache_variants() {
    let kv = LayerCache::KV(SteppingKeyValueCache::new());
    let arrays = LayerCache::Arrays(ArraysCache::new());
    match &kv {
        LayerCache::KV(c) => assert_eq!(c.offset(), 0),
        LayerCache::Arrays(_) => panic!("Expected KV variant"),
    }
    match &arrays {
        LayerCache::Arrays(c) => assert_eq!(c.offset, 0),
        LayerCache::KV(_) => panic!("Expected Arrays variant"),
    }
}

#[test]
fn test_causal_lm_valid_construction() {
    let args = valid_causal_lm_args();
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_ok());
    let model = result.unwrap();
    assert_eq!(model.args.model_type, "qwen3_next");
}

#[test]
fn test_causal_lm_make_cache_layer_types() {
    let args = valid_causal_lm_args();
    let model = Qwen3NextCausalLM::new(args).unwrap();
    let cache = model.make_cache();
    // 4 layers, full_attention_interval=4, so layers 0,1,2 are linear, layer 3 is full attention
    assert_eq!(cache.len(), 4);
    for (i, layer_cache) in cache.iter().enumerate() {
        let lc = layer_cache.as_ref().unwrap();
        let is_linear = (i + 1) % 4 != 0;
        if is_linear {
            assert!(
                matches!(lc, LayerCache::Arrays(_)),
                "Layer {i} should be Arrays (linear)"
            );
        } else {
            assert!(
                matches!(lc, LayerCache::KV(_)),
                "Layer {i} should be KV (full attention)"
            );
        }
    }
}

#[test]
fn test_causal_lm_negative_full_attention_interval() {
    let mut args = valid_causal_lm_args();
    args.full_attention_interval = -1;
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_err());
}

#[test]
fn test_causal_lm_with_quantization() {
    let mut args = valid_causal_lm_args();
    args.quantization = Some(QuantizationConfig::new(32, 8));
    let result = Qwen3NextCausalLM::new(args);
    assert!(result.is_ok());
}

#[test]
fn test_causal_lm_with_tied_embeddings() {
    let mut args = valid_causal_lm_args();
    args.tie_word_embeddings = true;
    let model = Qwen3NextCausalLM::new(args).unwrap();
    assert!(model.lm_head.is_none());
}

#[test]
fn test_causal_lm_without_tied_embeddings() {
    let mut args = valid_causal_lm_args();
    args.tie_word_embeddings = false;
    let model = Qwen3NextCausalLM::new(args).unwrap();
    assert!(model.lm_head.is_some());
}

/// GQA ratio: `num_v_heads` must be divisible by `num_k_heads`.
/// This validates the assumption used in test/bench GDN recurrence loops.
#[test]
fn test_gqa_ratio_divisibility() {
    let args = valid_causal_lm_args();
    let hv = args.linear_num_value_heads;
    let hk = args.linear_num_key_heads;
    assert!(
        hk > 0 && hv % hk == 0,
        "linear_num_value_heads ({hv}) must be divisible by linear_num_key_heads ({hk})"
    );
}
