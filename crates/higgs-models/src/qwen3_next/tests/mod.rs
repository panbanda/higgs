//! Shared fixtures for the `qwen3_next` tests.

mod bench_dispatch;
mod bench_gdn_moe;
mod bench_model;
mod bench_module_vs_inline;
mod bench_moe;
mod config;
mod gdn;
mod linear;
mod loading;
mod mlp;
mod model;
mod moe;
mod mtp;
mod prefill;

use crate::qwen3_next::config::Qwen3NextModelArgs;
use mlx_rs::Array;

/// Minimal args for tests that only care about `MoE` fields.
pub(super) fn minimal_qwen3_next_args() -> Qwen3NextModelArgs {
    serde_json::from_str(
        r#"{
                "model_type": "qwen3_next",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 1e-06,
                "vocab_size": 1024,
                "max_position_embeddings": 512,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
    )
    .unwrap()
}

/// Full args suitable for `Qwen3NextCausalLM::new()` validation tests.
pub(super) fn valid_causal_lm_args() -> Qwen3NextModelArgs {
    serde_json::from_str(
        r#"{
                "model_type": "qwen3_next",
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
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "shared_expert_intermediate_size": 256,
                "moe_intermediate_size": 128,
                "norm_topk_prob": true
            }"#,
    )
    .unwrap()
}

/// Quantize a float matrix and return (weight, scales, biases) suitable for
/// `gather_qmm` / `quantized_matmul`.
pub(super) fn quantize_weights(w: &Array, group_size: i32, bits: i32) -> (Array, Array, Array) {
    let (qw, scales, biases) = crate::test_support::quantize_affine(w, group_size, bits).unwrap();
    (qw, scales, biases)
}
