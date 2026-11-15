//! Engine crate for Higgs local inference, including MLX-backed MTP decoding.

pub mod chat_template;
pub mod constrained;
pub mod decode_plan;
pub mod detokenizer;
pub mod disk_prefix_store;
pub mod engine;
pub mod error;
pub(crate) mod generation_support;
pub mod inference;
mod mlx_memory;
pub mod mlx_tuning;
pub mod model_loader;
pub mod mtp;
pub mod paged_prefix_cache;
pub mod prefix_checkpoints;
pub mod reasoning_parser;
pub mod speculative;
pub mod tool_format;
pub mod tool_parser;

pub use tokenizers;
