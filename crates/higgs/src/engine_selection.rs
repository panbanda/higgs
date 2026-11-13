//! Chooses between the batch and simple engines for a model.
//!
//! `batch = "auto"` (the default) serves a model with the batch engine unless
//! the model, or a setting the configuration actually uses, needs something
//! only the simple engine provides. The decision is a pure function of
//! [`BatchFacts`] so it can be tested and reported (by `higgs doctor` and the
//! load log) without loading weights.

use std::fmt;
use std::path::Path;

use higgs_engine::{mlx_tuning::resolve_runtime_tuning_with, model_loader};

use crate::config::{BatchMode, ExperimentalConfig, ModelConfig};

/// What the engine choice depends on, for one model under one configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BatchFacts {
    /// The architecture implements batched decode.
    pub architecture_batches: bool,
    /// `TurboQuant` KV compression is configured.
    pub turboquant_kv: bool,
    /// A disk prefix store (`prefix_cache.disk_dir`) is configured.
    pub disk_prefix_store: bool,
    /// The architecture takes image inputs.
    pub vision: bool,
    /// The checkpoint carries an MTP head and the MLX profile enables
    /// speculative decode, so the simple engine would use it.
    pub mtp_speculative_decode: bool,
    /// Experimental prompt-lookup decoding is switched on (`experimental.prompt_lookup`).
    pub prompt_lookup: bool,
}

impl BatchFacts {
    /// Facts for a model whose checkpoint is on disk at `resolved`.
    pub fn inspect(
        model: &ModelConfig,
        experimental: &ExperimentalConfig,
        resolved: &Path,
    ) -> Result<Self, String> {
        let inspected = model_loader::ModelConfig::from_dir(resolved)
            .map_err(|e| format!("failed to inspect {}: {e}", resolved.display()))?;
        let mtp_speculative_decode = inspected.capabilities.mtp
            && resolve_runtime_tuning_with(
                resolved,
                model.profile,
                &experimental.tuning_overrides(),
            )
            .enable_mtp()
            && model_loader::checkpoint_has_mtp_head(resolved);
        Ok(Self {
            architecture_batches: inspected.capabilities.batch_engine,
            turboquant_kv: model.kv_cache.is_turboquant(),
            disk_prefix_store: model.prefix_cache.disk_dir.is_some(),
            vision: inspected.capabilities.vision,
            mtp_speculative_decode,
            prompt_lookup: experimental.prompt_lookup,
        })
    }
}

/// Why the simple engine serves a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleReason {
    /// `batch = false` or `--no-batch`.
    Requested,
    UnsupportedArchitecture,
    TurboQuantKv,
    DiskPrefixStore,
    Vision,
    MtpSpeculativeDecode,
    PromptLookup,
}

impl fmt::Display for SimpleReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Requested => "batch is turned off for this model",
            Self::UnsupportedArchitecture => "the architecture has no batched decode",
            Self::TurboQuantKv => "TurboQuant KV cache is configured (simple engine only)",
            Self::DiskPrefixStore => "a disk prefix store (prefix_cache.disk_dir) is configured (simple engine only)",
            Self::Vision => "the model takes image inputs (simple engine only)",
            Self::MtpSpeculativeDecode => {
                "the checkpoint has an MTP head and speculative decode is enabled (simple engine only)"
            }
            Self::PromptLookup => "prompt-lookup decoding is enabled (experimental.prompt_lookup or HIGGS_PROMPT_LOOKUP; simple engine only)",
        })
    }
}

/// The engine a model gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineChoice {
    Batch,
    Simple(SimpleReason),
}

impl EngineChoice {
    /// `batch` or `simple`, as reported by the API and the status views.
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Simple(_) => "simple",
        }
    }
}

/// Pick the engine for a model.
///
/// `BatchMode::On` always answers `Batch`: an unsupported model must fail
/// loudly when it loads, not be silently served by the other engine.
/// `Auto` answers `Batch` only when no simple-engine-only feature applies.
pub const fn choose_engine(mode: BatchMode, facts: &BatchFacts) -> EngineChoice {
    match mode {
        BatchMode::On => EngineChoice::Batch,
        BatchMode::Off => EngineChoice::Simple(SimpleReason::Requested),
        BatchMode::Auto => {
            if !facts.architecture_batches {
                EngineChoice::Simple(SimpleReason::UnsupportedArchitecture)
            } else if facts.turboquant_kv {
                EngineChoice::Simple(SimpleReason::TurboQuantKv)
            } else if facts.disk_prefix_store {
                EngineChoice::Simple(SimpleReason::DiskPrefixStore)
            } else if facts.vision {
                EngineChoice::Simple(SimpleReason::Vision)
            } else if facts.mtp_speculative_decode {
                EngineChoice::Simple(SimpleReason::MtpSpeculativeDecode)
            } else if facts.prompt_lookup {
                EngineChoice::Simple(SimpleReason::PromptLookup)
            } else {
                EngineChoice::Batch
            }
        }
    }
}

/// The engine a model will get, or `None` when its checkpoint cannot be
/// inspected yet (for example a Hugging Face model not downloaded).
pub fn planned_engine(
    model: &ModelConfig,
    experimental: &ExperimentalConfig,
) -> Option<EngineChoice> {
    match model.batch {
        BatchMode::On => Some(EngineChoice::Batch),
        BatchMode::Off => Some(EngineChoice::Simple(SimpleReason::Requested)),
        BatchMode::Auto => {
            let resolved = crate::model_resolver::resolve(&model.path).ok()?;
            let facts = BatchFacts::inspect(model, experimental, &resolved).ok()?;
            Some(choose_engine(BatchMode::Auto, &facts))
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    const BATCHABLE: BatchFacts = BatchFacts {
        architecture_batches: true,
        turboquant_kv: false,
        disk_prefix_store: false,
        vision: false,
        mtp_speculative_decode: false,
        prompt_lookup: false,
    };

    fn auto(facts: BatchFacts) -> EngineChoice {
        choose_engine(BatchMode::Auto, &facts)
    }

    #[test]
    fn auto_batches_a_plain_batchable_model() {
        assert_eq!(auto(BATCHABLE), EngineChoice::Batch);
    }

    #[test]
    fn auto_falls_back_for_an_architecture_without_batched_decode() {
        let facts = BatchFacts {
            architecture_batches: false,
            ..BATCHABLE
        };
        assert_eq!(
            auto(facts),
            EngineChoice::Simple(SimpleReason::UnsupportedArchitecture)
        );
    }

    #[test]
    fn auto_falls_back_for_each_simple_only_feature() {
        let cases = [
            (
                BatchFacts {
                    turboquant_kv: true,
                    ..BATCHABLE
                },
                SimpleReason::TurboQuantKv,
            ),
            (
                BatchFacts {
                    disk_prefix_store: true,
                    ..BATCHABLE
                },
                SimpleReason::DiskPrefixStore,
            ),
            (
                BatchFacts {
                    vision: true,
                    ..BATCHABLE
                },
                SimpleReason::Vision,
            ),
            (
                BatchFacts {
                    mtp_speculative_decode: true,
                    ..BATCHABLE
                },
                SimpleReason::MtpSpeculativeDecode,
            ),
            (
                BatchFacts {
                    prompt_lookup: true,
                    ..BATCHABLE
                },
                SimpleReason::PromptLookup,
            ),
        ];
        for (facts, reason) in cases {
            assert_eq!(auto(facts), EngineChoice::Simple(reason));
        }
    }

    #[test]
    fn auto_reports_the_first_applicable_rule() {
        let many_problems = BatchFacts {
            architecture_batches: false,
            turboquant_kv: true,
            vision: true,
            ..BATCHABLE
        };
        assert_eq!(
            auto(many_problems),
            EngineChoice::Simple(SimpleReason::UnsupportedArchitecture)
        );
        let two_features = BatchFacts {
            turboquant_kv: true,
            disk_prefix_store: true,
            ..BATCHABLE
        };
        assert_eq!(
            auto(two_features),
            EngineChoice::Simple(SimpleReason::TurboQuantKv)
        );
    }

    #[test]
    fn explicit_modes_ignore_the_facts() {
        let unsupported = BatchFacts {
            architecture_batches: false,
            turboquant_kv: true,
            ..BATCHABLE
        };
        assert_eq!(
            choose_engine(BatchMode::On, &unsupported),
            EngineChoice::Batch
        );
        assert_eq!(
            choose_engine(BatchMode::Off, &BATCHABLE),
            EngineChoice::Simple(SimpleReason::Requested)
        );
    }

    #[test]
    fn kind_names_the_engine() {
        assert_eq!(EngineChoice::Batch.kind(), "batch");
        assert_eq!(EngineChoice::Simple(SimpleReason::Vision).kind(), "simple");
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
            "[[models]]\npath = \"{}\"\n[models.prefix_cache]\ndisk_dir = \"/tmp/higgs-test-kv\"\n",
            checkpoint.path().display()
        ));
        let facts =
            BatchFacts::inspect(&model, &ExperimentalConfig::default(), checkpoint.path()).unwrap();

        assert!(facts.architecture_batches);
        assert!(facts.disk_prefix_store);
        assert!(!facts.turboquant_kv);
        assert!(!facts.vision);
        assert!(!facts.mtp_speculative_decode);
    }

    #[test]
    fn inspect_sees_turboquant() {
        let checkpoint = write_checkpoint(LLAMA);
        let model = model_config(&format!(
            "[[models]]\npath = \"{}\"\n[models.kv_cache]\nmode = \"turboquant\"\n",
            checkpoint.path().display()
        ));
        let facts =
            BatchFacts::inspect(&model, &ExperimentalConfig::default(), checkpoint.path()).unwrap();

        assert!(facts.turboquant_kv);
        assert_eq!(
            choose_engine(model.batch, &facts),
            EngineChoice::Simple(SimpleReason::TurboQuantKv)
        );
    }

    #[test]
    fn plan_for_explicit_modes_needs_no_checkpoint() {
        let on = model_config("[[models]]\npath = \"org/missing\"\nbatch = true\n");
        let off = model_config("[[models]]\npath = \"org/missing\"\nbatch = false\n");
        let auto_mode = model_config("[[models]]\npath = \"org/never-downloaded\"\n");
        let local = ExperimentalConfig::default();

        assert_eq!(planned_engine(&on, &local), Some(EngineChoice::Batch));
        assert_eq!(
            planned_engine(&off, &local),
            Some(EngineChoice::Simple(SimpleReason::Requested))
        );
        assert_eq!(planned_engine(&auto_mode, &local), None);
    }
}
