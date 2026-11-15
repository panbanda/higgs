//! How a model's requests are decoded: together in one batched forward pass,
//! or one request at a time.
//!
//! Every model is served by the same engine. What differs is whether it can
//! decode several requests per forward pass, which the engine decides when it
//! loads the model. [`DecodeFacts`] is the same decision as a pure function of
//! the checkpoint and configuration, so `higgs doctor`, the load log and the
//! model status API can report it without loading weights.

use std::fmt;
use std::path::Path;

use higgs_engine::model_loader;

use crate::config::{BatchMode, ExperimentalConfig, ModelConfig};

/// What decides whether a model batches its decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct DecodeFacts {
    /// The architecture implements batched decode and the checkpoint has no MTP
    /// head, whose speculative decode drafts one sequence at a time.
    pub architecture_batches: bool,
    /// `TurboQuant` KV compression is configured; its per-request quantized
    /// caches cannot share a batched cache.
    pub turboquant_kv: bool,
    /// `batch = false` caps the model at one request at a time.
    pub serialized_by_config: bool,
    /// The checkpoint carries an MTP head and the MLX profile enables
    /// speculative decode.
    pub mtp_speculative_decode: bool,
}

impl DecodeFacts {
    /// Facts for a model whose checkpoint is on disk at `resolved`.
    pub fn inspect(
        model: &ModelConfig,
        experimental: &ExperimentalConfig,
        resolved: &Path,
    ) -> Result<Self, String> {
        let inspected = model_loader::ModelConfig::from_dir(resolved)
            .map_err(|e| format!("failed to inspect {}: {e}", resolved.display()))?;
        let has_mtp_head =
            inspected.capabilities.mtp && model_loader::checkpoint_has_mtp_head(resolved);
        let profile_enables_mtp = higgs_engine::mlx_tuning::resolve_runtime_tuning_with(
            resolved,
            model.profile,
            &experimental.tuning_overrides(),
        )
        .enable_mtp();
        Ok(Self {
            architecture_batches: inspected.capabilities.batch_engine && !has_mtp_head,
            turboquant_kv: model.kv_cache.is_turboquant(),
            serialized_by_config: model.batch == BatchMode::Off,
            mtp_speculative_decode: has_mtp_head && profile_enables_mtp,
        })
    }
}

/// Why a model decodes one request at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequentialReason {
    /// `batch = false` or `--no-batch`.
    Requested,
    UnsupportedArchitecture,
    TurboQuantKv,
}

impl fmt::Display for SequentialReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Requested => "batch = false limits the model to one request at a time",
            Self::UnsupportedArchitecture => {
                "the architecture (or the checkpoint's MTP head) has no batched decode"
            }
            Self::TurboQuantKv => "TurboQuant KV caches cannot be batched",
        })
    }
}

/// How a model's requests are decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStrategy {
    Batched,
    /// Requests queue and decode one at a time.
    Sequential(SequentialReason),
}

impl DecodeStrategy {
    /// `batched` or `sequential`, as reported by the status API.
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Batched => "batched",
            Self::Sequential(_) => "sequential",
        }
    }
}

/// The strategy the facts imply. The first applicable reason wins.
pub const fn decode_strategy(facts: &DecodeFacts) -> DecodeStrategy {
    if facts.serialized_by_config {
        DecodeStrategy::Sequential(SequentialReason::Requested)
    } else if !facts.architecture_batches {
        DecodeStrategy::Sequential(SequentialReason::UnsupportedArchitecture)
    } else if facts.turboquant_kv {
        DecodeStrategy::Sequential(SequentialReason::TurboQuantKv)
    } else {
        DecodeStrategy::Batched
    }
}

/// The strategy a model will get, or `None` when its checkpoint cannot be
/// inspected yet (for example a Hugging Face model not downloaded).
pub fn planned_strategy(
    model: &ModelConfig,
    experimental: &ExperimentalConfig,
) -> Option<DecodeStrategy> {
    if model.batch == BatchMode::Off {
        return Some(DecodeStrategy::Sequential(SequentialReason::Requested));
    }
    let resolved = crate::model_resolver::resolve(&model.path).ok()?;
    let facts = DecodeFacts::inspect(model, experimental, &resolved).ok()?;
    Some(decode_strategy(&facts))
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    const BATCHABLE: DecodeFacts = DecodeFacts {
        architecture_batches: true,
        turboquant_kv: false,
        serialized_by_config: false,
        mtp_speculative_decode: false,
    };

    #[test]
    fn a_plain_batchable_model_batches() {
        assert_eq!(decode_strategy(&BATCHABLE), DecodeStrategy::Batched);
    }

    #[test]
    fn each_limit_serializes_with_its_own_reason() {
        let cases = [
            (
                DecodeFacts {
                    architecture_batches: false,
                    ..BATCHABLE
                },
                SequentialReason::UnsupportedArchitecture,
            ),
            (
                DecodeFacts {
                    turboquant_kv: true,
                    ..BATCHABLE
                },
                SequentialReason::TurboQuantKv,
            ),
            (
                DecodeFacts {
                    serialized_by_config: true,
                    ..BATCHABLE
                },
                SequentialReason::Requested,
            ),
        ];
        for (facts, reason) in cases {
            assert_eq!(decode_strategy(&facts), DecodeStrategy::Sequential(reason));
        }
    }

    #[test]
    fn the_configured_limit_is_reported_first() {
        let facts = DecodeFacts {
            architecture_batches: false,
            turboquant_kv: true,
            serialized_by_config: true,
            ..BATCHABLE
        };
        assert_eq!(
            decode_strategy(&facts),
            DecodeStrategy::Sequential(SequentialReason::Requested)
        );
    }

    #[test]
    fn speculative_decode_does_not_change_the_strategy() {
        let facts = DecodeFacts {
            mtp_speculative_decode: true,
            ..BATCHABLE
        };
        assert_eq!(decode_strategy(&facts), DecodeStrategy::Batched);
    }

    #[test]
    fn kind_names_the_strategy() {
        assert_eq!(DecodeStrategy::Batched.kind(), "batched");
        assert_eq!(
            DecodeStrategy::Sequential(SequentialReason::TurboQuantKv).kind(),
            "sequential"
        );
    }

    fn model_config(toml: &str) -> ModelConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        crate::config::load_config_file(&path, None)
            .unwrap()
            .models
            .remove(0)
    }

    fn write_checkpoint(config_json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();
        dir
    }

    const LLAMA: &str = r#"{"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5, "vocab_size": 100}"#;

    #[test]
    fn inspect_reads_the_facts_from_the_checkpoint_and_config() {
        let checkpoint = write_checkpoint(LLAMA);
        let model = model_config(&format!(
            "[[models]]\npath = \"{}\"\nbatch = false\n[models.kv_cache]\nmode = \"turboquant\"\n",
            checkpoint.path().display()
        ));
        let facts = DecodeFacts::inspect(&model, &ExperimentalConfig::default(), checkpoint.path())
            .unwrap();

        assert!(facts.architecture_batches);
        assert!(facts.turboquant_kv);
        assert!(facts.serialized_by_config);
        assert!(!facts.mtp_speculative_decode);
    }

    #[test]
    fn plan_for_batch_false_needs_no_checkpoint() {
        let off = model_config("[[models]]\npath = \"org/missing\"\nbatch = false\n");
        let auto_mode = model_config("[[models]]\npath = \"org/never-downloaded\"\n");

        assert_eq!(
            planned_strategy(&off, &ExperimentalConfig::default()),
            Some(DecodeStrategy::Sequential(SequentialReason::Requested))
        );
        assert_eq!(
            planned_strategy(&auto_mode, &ExperimentalConfig::default()),
            None
        );
    }
}
