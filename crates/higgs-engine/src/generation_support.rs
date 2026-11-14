//! Model setup and helpers for the engine: thinking support, chat prompts,
//! prefix-cache keys and embeddings.

use std::path::Path;
use std::sync::Arc;

use higgs_models::{AnyCache, AnyModel, turboquant::KvCacheConfig};
use mlx_rs::{
    Array, Dtype, Stream,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    with_stream,
};
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    error::EngineError,
    paged_prefix_cache::PagedPrefixMatch,
    prefix_checkpoints::RecurrentCheckpoint,
};

/// Whether the model can reason inside `<think>` tags, and the token that closes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingSupport {
    pub enabled: bool,
    /// Resolved from the tokenizer. `None` when it has no single `</think>` token.
    pub close_token: Option<u32>,
}

impl ThinkingSupport {
    /// Detect thinking support for a checkpoint.
    ///
    /// `HIGGS_ENABLE_THINKING` overrides detection. Thinking is disabled when
    /// the tokenizer cannot encode `</think>` as one token, since forcing an
    /// unknown id would index outside the embedding table.
    pub fn resolve(model_dir: &Path, tokenizer: &Tokenizer) -> Self {
        let mut enabled = std::env::var("HIGGS_ENABLE_THINKING")
            .ok()
            .as_deref()
            .map_or_else(
                || detect_thinking_support(model_dir),
                |value| {
                    crate::mlx_tuning::parse_enabled_flag(Some(value))
                        .unwrap_or_else(|| detect_thinking_support(model_dir))
                },
            );
        let close_token = tokenizer
            .encode("</think>", false)
            .ok()
            .and_then(|encoding| {
                if let [single] = encoding.get_ids() {
                    Some(*single)
                } else {
                    None
                }
            });
        if enabled && close_token.is_none() {
            tracing::warn!("Tokenizer has no single </think> token; disabling thinking mode");
            enabled = false;
        }
        if enabled {
            tracing::info!(think_close_token = close_token, "Thinking mode enabled");
        }
        Self {
            enabled,
            close_token,
        }
    }
}

/// Render `messages` through the chat template and tokenize the result.
pub fn tokenize_chat_prompt(
    template: Option<&ChatTemplateRenderer>,
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> Result<Vec<u32>, EngineError> {
    let renderer = template.ok_or_else(|| {
        EngineError::Template(
            "This model has no chat template; use /v1/completions instead".to_owned(),
        )
    })?;
    let prompt = renderer.apply_with_thinking(messages, tools, true, enable_thinking)?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| EngineError::Tokenization(e.to_string()))?;
    Ok(encoding.get_ids().to_vec())
}

/// Number of trailing tokens `add_generation_prompt=true` appends to a chat
/// prompt (for example `<|im_start|>assistant\n<think>\n`).
///
/// Prefix-cache keys leave these tokens out so a conversation's next turn,
/// which renders that stretch differently, still shares the history.
pub fn generation_prompt_suffix_len(
    template: Option<&ChatTemplateRenderer>,
    tokenizer: &Tokenizer,
    enable_thinking: bool,
) -> usize {
    template
        .and_then(|renderer| {
            let probe = vec![ChatMessage {
                role: "user".to_owned(),
                content: "x".to_owned(),
                tool_calls: None,
            }];
            let with_generation = renderer
                .apply_with_thinking(&probe, None, true, enable_thinking)
                .ok()?;
            let without_generation = renderer
                .apply_with_thinking(&probe, None, false, enable_thinking)
                .ok()?;
            let tokens_with = tokenizer.encode(with_generation.as_str(), false).ok()?;
            let tokens_without = tokenizer.encode(without_generation.as_str(), false).ok()?;
            let suffix = tokens_with
                .get_ids()
                .len()
                .saturating_sub(tokens_without.get_ids().len());
            tracing::info!(
                gen_prompt_suffix_len = suffix,
                "Computed generation prompt suffix length for prefix cache"
            );
            Some(suffix)
        })
        .unwrap_or(0)
}

/// The part of a prompt a prefix cache is keyed on.
pub fn prefix_cache_key(prompt_tokens: &[u32], generation_suffix_len: usize) -> &[u32] {
    prompt_tokens
        .get(..prompt_tokens.len().saturating_sub(generation_suffix_len))
        .unwrap_or(prompt_tokens)
}

/// Where prefill resumes, the cache to resume from, and the recurrent
/// checkpoints that remain valid for the longer entry this prefill will store.
pub struct PrefixResume {
    pub resume_at: usize,
    pub cache: AnyCache,
    pub checkpoints: Vec<Arc<RecurrentCheckpoint>>,
}

/// Where prefill resumes, and the cache to resume from, given the longest
/// cached prefix of `prompt_tokens`.
///
/// A match covering the whole prompt is not reused: logits are not cached, so
/// at least one token must run through the model to produce the first
/// completion token.
pub fn resume_from_prefix<MakeCache: FnOnce() -> Result<AnyCache, EngineError>>(
    prefix_match: Option<PagedPrefixMatch>,
    prompt_tokens: &[u32],
    fresh_cache: MakeCache,
) -> Result<PrefixResume, EngineError> {
    match prefix_match {
        Some(matched) if matched.prefix_len < prompt_tokens.len() => {
            tracing::debug!(
                prefix_len = matched.prefix_len,
                total_len = prompt_tokens.len(),
                "Prefix cache hit — reusing cached prefix"
            );
            Ok(PrefixResume {
                resume_at: matched.prefix_len,
                cache: matched.cache,
                checkpoints: matched.checkpoints,
            })
        }
        Some(_) => {
            tracing::debug!(
                prompt_len = prompt_tokens.len(),
                "Full prefix hit requires cached logits; falling back to fresh prefill"
            );
            Ok(PrefixResume {
                resume_at: 0,
                cache: fresh_cache()?,
                checkpoints: Vec::new(),
            })
        }
        None => Ok(PrefixResume {
            resume_at: 0,
            cache: fresh_cache()?,
            checkpoints: Vec::new(),
        }),
    }
}

/// Embed `token_ids`: one forward pass, mean-pooled over the sequence and
/// L2-normalized. Runs on a fresh cache, so it cannot disturb other requests.
pub fn embed_tokens(
    model: &mut AnyModel,
    kv_cache_config: KvCacheConfig,
    token_ids: &[u32],
) -> Result<Vec<f32>, EngineError> {
    if token_ids.is_empty() {
        return Err(EngineError::Generation("Input is empty".to_owned()));
    }
    with_stream(&Stream::new(), || {
        let input = Array::from(token_ids).index(NewAxis);
        let mut cache = model
            .make_cache_with_config(kv_cache_config)
            .map_err(EngineError::Mlx)?;
        let hidden = model
            .forward_hidden(&input, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let pooled = hidden.mean_axes(&[1], false).map_err(EngineError::Mlx)?;
        // The model may compute in bfloat16.
        let pooled_f32 = pooled.as_dtype(Dtype::Float32).map_err(EngineError::Mlx)?;
        eval([&pooled_f32]).map_err(EngineError::Mlx)?;
        Ok(l2_normalized(pooled_f32.as_slice::<f32>()))
    })
}

fn l2_normalized(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter().map(|value| value / norm).collect()
    } else {
        values.to_vec()
    }
}

/// Whether the model's chat template references the `enable_thinking` toggle,
/// read from `chat_template.jinja` or `tokenizer_config.json`'s `chat_template`
/// (a string, or a `{name, template}` array).
fn chat_template_mentions_enable_thinking(model_dir: &Path) -> bool {
    const MARKER: &str = "enable_thinking";
    if let Ok(jinja) = std::fs::read_to_string(model_dir.join("chat_template.jinja")) {
        return jinja.contains(MARKER);
    }
    let Ok(config_text) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_text) else {
        return false;
    };
    let template = config.get("chat_template");
    if let Some(text) = template.and_then(|value| value.as_str()) {
        return text.contains(MARKER);
    }
    if let Some(entries) = template.and_then(|value| value.as_array()) {
        return entries.iter().any(|entry| {
            entry
                .get("template")
                .and_then(|text| text.as_str())
                .is_some_and(|text| text.contains(MARKER))
        });
    }
    false
}

/// Whether the model *supports* a thinking toggle (capability, not default).
///
/// The per-request default — e.g. Qwen3.6 reasons off unless asked — is decided
/// separately by `model_defaults_to_non_thinking` in the router; this only
/// answers "can it think at all".
///
/// Signals, in order:
/// 1. the chat template exposes an `enable_thinking` switch — the model
///    author's own marker, which covers Qwen3.5/3.6, `MiniCPM5`, and future
///    reasoning models without hardcoding model types; or
/// 2. a known reasoning `model_type`.
///
/// The caller additionally requires a single-token `</think>`, so a stray
/// mention can't enable thinking for a model that wasn't trained for it.
pub fn detect_thinking_support(model_dir: &Path) -> bool {
    if chat_template_mentions_enable_thinking(model_dir) {
        return true;
    }
    let Ok(config_text) = std::fs::read_to_string(model_dir.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_text) else {
        return false;
    };
    // Check both top-level and nested text_config (VLM wrappers).
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|text_config| text_config.get("model_type"))
                .and_then(|value| value.as_str())
        });
    matches!(model_type, Some("qwen3_5" | "qwen3_5_moe"))
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_drops_the_generation_suffix() {
        let prompt = [1, 2, 3, 4, 5, 6];
        assert_eq!(prefix_cache_key(&prompt, 2), &[1, 2, 3, 4]);
        assert_eq!(prefix_cache_key(&prompt, 0), &prompt);
    }

    #[test]
    fn cache_key_never_underflows() {
        let prompt = [1, 2, 3];
        assert!(prefix_cache_key(&prompt, 3).is_empty());
        assert!(prefix_cache_key(&prompt, 10).is_empty());
    }

    fn prefix_match(prefix_len: usize) -> PagedPrefixMatch {
        PagedPrefixMatch {
            prefix_len,
            cache: AnyCache::KV(Vec::new()),
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn partial_prefix_hit_resumes_after_the_prefix() {
        let prompt = [1, 2, 3, 4, 5];
        let resumed = resume_from_prefix(Some(prefix_match(3)), &prompt, || {
            Err(EngineError::Generation("no fresh cache needed".to_owned()))
        })
        .unwrap();
        assert_eq!(resumed.resume_at, 3);
    }

    #[test]
    fn full_prefix_hit_replays_the_prompt_from_scratch() {
        let prompt = [1, 2, 3];
        let resumed = resume_from_prefix(Some(prefix_match(3)), &prompt, || {
            Ok(AnyCache::KV(Vec::new()))
        })
        .unwrap();
        assert_eq!(resumed.resume_at, 0);
    }

    #[test]
    fn miss_starts_from_a_fresh_cache() {
        let resumed = resume_from_prefix(None, &[1, 2], || Ok(AnyCache::KV(Vec::new()))).unwrap();
        assert_eq!(resumed.resume_at, 0);
    }

    #[test]
    fn l2_normalization_gives_unit_length() {
        let normalized = l2_normalized(&[3.0, 4.0]);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalization_leaves_zero_vector_alone() {
        assert_eq!(l2_normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    /// Write a config.json file into the given directory with the provided JSON content.
    fn write_config(dir: &std::path::Path, json: &str) {
        std::fs::write(dir.join("config.json"), json).unwrap();
    }

    // --- detect_thinking_support tests ---

    /// MiniCPM5-style: a non-reasoning `model_type` (llama) but a chat template
    /// that exposes the `enable_thinking` switch ⇒ thinking-capable.
    #[test]
    fn detect_thinking_from_template_marker() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- if enable_thinking %}<think>\n{%- endif %}",
        )
        .unwrap();
        assert!(detect_thinking_support(dir.path()));
    }

    /// Qwen3.5 reasoning `model_type` is detected even without a template file.
    #[test]
    fn detect_thinking_from_model_type() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "qwen3_5_moe"}"#);
        assert!(detect_thinking_support(dir.path()));
    }

    /// A plain Llama (no reasoning `model_type`, no `enable_thinking` in the
    /// template) must NOT be treated as a thinking model.
    #[test]
    fn no_thinking_for_plain_llama() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), r#"{"model_type": "llama"}"#);
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{%- for m in messages %}{{ m.content }}{%- endfor %}",
        )
        .unwrap();
        assert!(!detect_thinking_support(dir.path()));
    }
}
