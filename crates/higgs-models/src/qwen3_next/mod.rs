//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (`MoE`).
//! Every `full_attention_interval`-th layer uses full attention (`Qwen3NextAttention`),
//! all other layers use `GatedDeltaNet` (SSM-like linear attention).
//! All layers use Sparse `MoE` for the feed-forward block.

mod activations;
mod attention;
mod cache;
mod cached_kernel;
mod checkpoint;
mod config;
mod decoder;
mod ffi;
mod flags;
mod gated_delta_kernel;
mod gated_delta_net;
mod linear;
mod loading;
mod mlp;
mod model;
mod moe;
mod mtp;
mod qgemv;
mod sparse_prefill;
mod weights;

pub(crate) use activations::swiglu;
pub use attention::Qwen3NextAttention;
pub(crate) use cache::chronological_conv_history;
pub use cache::{ArraysCache, LayerCache};
pub use checkpoint::checkpoint_has_mtp_head;
pub use config::{QuantizationConfig, Qwen3NextModelArgs, load_model_args};
pub(crate) use config::{
    load_qwen3_5_text_config_args_from_value, load_qwen3_next_args_from_value,
};
pub(crate) use linear::{QEmbedding, QLinear};
pub use loading::{load_qwen3_5_model, load_qwen3_5_moe_model, load_qwen3_next_model};
pub(crate) use loading::{
    load_qwen3_5_model_with_args, load_qwen3_5_moe_model_with_args, load_qwen3_next_model_with_args,
};
pub use mlp::Qwen3NextMLP;
pub(crate) use mlp::new_mlp_projections;
pub use model::Qwen3NextCausalLM;
pub(crate) use moe::SwitchMlpWeights;
pub use mtp::{MoeMtpHead, MtpHead};

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::shadow_unrelated,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::needless_for_each,
    clippy::needless_collect,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_range_loop,
    clippy::manual_flatten,
    clippy::unnecessary_map_or,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    clippy::explicit_iter_loop,
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr,
    clippy::str_to_string,
    clippy::if_then_some_else_none,
    clippy::ignore_without_reason,
    clippy::unreadable_literal,
    clippy::cast_possible_wrap,
    clippy::useless_conversion,
    clippy::manual_assert,
    clippy::option_if_let_else,
    clippy::used_underscore_binding,
    clippy::redundant_clone
)]
#[cfg(test)]
mod tests;
