use std::path::{Path, PathBuf};

use higgs_models::{
    AnyModel,
    adapter::{self, Capabilities, ModelFamily, ModelVersion},
    load_tokenizer as shared_load_tokenizer,
};

use crate::error::EngineError;

/// Configuration for loading a model from a directory.
#[derive(Debug)]
pub struct ModelConfig {
    /// Local checkpoint directory used for model and tokenizer loading.
    pub model_dir: PathBuf,
    /// Effective model type, using nested `text_config.model_type` for wrappers.
    pub model_type: String,
    /// Stable identifier of the adapter selected for this checkpoint.
    pub adapter_id: &'static str,
    /// Broad family detected from the effective model type.
    pub family: ModelFamily,
    /// Numeric version parsed from `model_type`, or `None` when no numeric version parses.
    pub version: Option<ModelVersion>,
    /// Capabilities of the resolved adapter implementation, not claims copied from the checkpoint.
    pub capabilities: Capabilities,
}

impl ModelConfig {
    /// Detect model type and create a config from a model directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref().to_path_buf();
        let detected = adapter::detect(&model_dir)?;
        let resolved = adapter::resolve(&detected)?;
        let info = resolved.describe();

        Ok(Self {
            model_dir,
            model_type: detected.model_type,
            adapter_id: info.id,
            family: detected.family,
            version: detected.version,
            capabilities: info.capabilities,
        })
    }
}

/// Load a model from a directory, auto-detecting the architecture.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<AnyModel, EngineError> {
    let detected = adapter::detect(model_dir.as_ref()).map_err(EngineError::Model)?;
    let resolved = adapter::resolve(&detected).map_err(EngineError::Model)?;
    resolved.load(&detected).map_err(EngineError::Model)
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<tokenizers::Tokenizer, EngineError> {
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use higgs_models::error::ModelError;

    /// Create a temp dir with a config.json containing the given `model_type` and
    /// return the `ModelConfig` result.
    fn config_for_model(model_type: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    /// Write arbitrary content to config.json in a temp dir and return
    /// the `ModelConfig` result.
    fn config_from_raw(content: &str) -> (tempfile::TempDir, Result<ModelConfig, EngineError>) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), content).unwrap();
        let result = ModelConfig::from_dir(dir.path());
        (dir, result)
    }

    #[test]
    fn model_config_from_dir_qwen2() {
        let (dir, result) = config_for_model("qwen2");
        let config = result.unwrap();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.model_dir, dir.path());
    }

    #[test]
    fn model_config_from_dir_qwen3() {
        let (_dir, result) = config_for_model("qwen3");
        assert_eq!(result.unwrap().model_type, "qwen3");
    }

    #[test]
    fn model_config_from_dir_llama() {
        let (_dir, result) = config_for_model("llama");
        assert_eq!(result.unwrap().model_type, "llama");
    }

    #[test]
    fn model_config_from_dir_mistral() {
        let (_dir, result) = config_for_model("mistral");
        assert_eq!(result.unwrap().model_type, "mistral");
    }

    #[test]
    fn model_config_from_dir_qwen3_next() {
        let (_dir, result) = config_for_model("qwen3_next");
        assert_eq!(result.unwrap().model_type, "qwen3_next");
    }

    #[test]
    fn model_config_from_dir_qwen3_moe() {
        let (_dir, result) = config_for_model("qwen3_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_moe");
    }

    #[test]
    fn model_config_from_dir_gemma2() {
        let (_dir, result) = config_for_model("gemma2");
        assert_eq!(result.unwrap().model_type, "gemma2");
    }

    #[test]
    fn model_config_from_dir_gemma3() {
        let (_dir, result) = config_for_model("gemma3");
        assert_eq!(result.unwrap().model_type, "gemma3");
    }

    #[test]
    fn model_config_from_dir_gemma3_text() {
        let (_dir, result) = config_for_model("gemma3_text");
        assert_eq!(result.unwrap().model_type, "gemma3_text");
    }

    #[test]
    fn model_config_from_dir_gemma4() {
        let (_dir, result) = config_for_model("gemma4");
        assert_eq!(result.unwrap().model_type, "gemma4");
    }

    #[test]
    fn model_config_from_dir_gemma4_text() {
        let (_dir, result) = config_for_model("gemma4_text");
        assert_eq!(result.unwrap().model_type, "gemma4_text");
    }

    #[test]
    fn model_config_from_dir_phi3() {
        let (_dir, result) = config_for_model("phi3");
        assert_eq!(result.unwrap().model_type, "phi3");
    }

    #[test]
    fn model_config_from_dir_starcoder2() {
        let (_dir, result) = config_for_model("starcoder2");
        assert_eq!(result.unwrap().model_type, "starcoder2");
    }

    #[test]
    fn model_config_from_dir_deepseek_v2() {
        let (_dir, result) = config_for_model("deepseek_v2");
        assert_eq!(result.unwrap().model_type, "deepseek_v2");
    }

    #[test]
    fn model_config_from_dir_qwen3_5() {
        let (_dir, result) = config_for_model("qwen3_5");
        assert_eq!(result.unwrap().model_type, "qwen3_5");
    }

    #[test]
    fn model_config_from_dir_qwen3_5_moe() {
        let (_dir, result) = config_for_model("qwen3_5_moe");
        assert_eq!(result.unwrap().model_type, "qwen3_5_moe");
    }

    #[test]
    fn model_config_from_dir_unsupported_model_type() {
        let (_dir, result) = config_for_model("gpt2");
        match result {
            Err(e) => assert!(e.to_string().contains("gpt2")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn model_config_from_dir_missing_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let err = ModelConfig::from_dir(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Io(_))));
    }

    #[test]
    fn model_config_from_dir_invalid_json() {
        let (_dir, result) = config_from_raw("not valid json {{{");
        let err = result.unwrap_err();
        assert!(matches!(err, EngineError::Model(ModelError::Json(_))));
    }

    #[test]
    fn model_config_from_dir_missing_model_type_field() {
        let (_dir, result) = config_from_raw(r#"{"vocab_size": 32000, "hidden_size": 4096}"#);
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            EngineError::Model(ModelError::UnsupportedModel(_))
        ));
    }

    #[test]
    fn load_model_routes_bonsai_q1_to_packed_engine() {
        // A bits=1 / group=128 qwen3 config now routes to the packed Bonsai-Q1
        // engine (its bits=1 kernels live in higgs-models::metal_kernel) instead
        // of being rejected up front. With no weights in the dir the load still
        // fails inside the engine — but it must no longer be gated out, and the
        // old "requires MLX bits=1" guard error must be gone.
        let (dir, _result) = config_from_raw(
            r#"{
                "model_type": "qwen3",
                "quantization": {"bits": 1, "group_size": 128}
            }"#,
        );
        match load_model(dir.path()) {
            Ok(_) => panic!("expected load failure: config-only dir has no weights"),
            Err(EngineError::Model(ModelError::UnsupportedModel(_))) => {
                panic!("Bonsai-Q1 must route to the packed engine, not be rejected as unsupported")
            }
            Err(err) => assert!(
                !err.to_string().contains("requires MLX bits=1"),
                "stale bits=1 guard error should be gone, got: {err}"
            ),
        }
    }

    #[test]
    fn load_tokenizer_missing_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        match load_tokenizer(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Tokenization error")),
            Ok(_) => panic!("Expected error for missing tokenizer.json"),
        }
    }
}
