use std::path::{Path, PathBuf};

use higgs_models::{
    AnyModel, dflash, error::ModelError, load_tokenizer as shared_load_tokenizer, registry,
    transformer,
};

use crate::error::EngineError;

/// Configuration for loading a model from a directory.
#[derive(Debug)]
pub struct ModelConfig {
    pub model_dir: PathBuf,
    pub model_type: String,
}

impl ModelConfig {
    /// Detect model type and create a config from a model directory.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref().to_path_buf();
        let model_type = registry::detect_model_type(&model_dir)?;

        if !registry::is_supported(&model_type) {
            return Err(EngineError::Model(
                higgs_models::error::ModelError::UnsupportedModel(model_type),
            ));
        }

        Ok(Self {
            model_dir,
            model_type,
        })
    }
}

/// Load a model from a directory, auto-detecting the architecture.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<AnyModel, EngineError> {
    let config = ModelConfig::from_dir(&model_dir)?;

    match config.model_type.as_str() {
        "qwen2" | "qwen3" | "llama" | "mistral" => {
            // Packed 1.25-bpw Bonsai-Q1 checkpoints declare model_type="qwen3"
            // but the weights are quantized to bits=1. Route them to the
            // dedicated packed engine, whose bits=1 matvec/dequant run through
            // runtime JIT Metal kernels (higgs-models::metal_kernel) — so it
            // runs on stock oxideai/mlx-rs with no forked bits=1 MLX kernel.
            if is_bonsai_q1(&config.model_dir)? {
                let gpu = higgs_models::bonsai_q1::load_bonsai_q1(&config.model_dir)
                    .map_err(EngineError::Model)?;
                return Ok(AnyModel::BonsaiQ1(gpu));
            }
            let model = transformer::load_model(&config.model_dir).map_err(EngineError::Model)?;
            Ok(AnyModel::Transformer(model))
        }
        "qwen3_next" => {
            let model = higgs_models::qwen3_next::load_qwen3_next_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_5" => {
            let model = higgs_models::qwen3_next::load_qwen3_5_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_5_moe" => {
            let model = higgs_models::qwen3_next::load_qwen3_5_moe_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        "qwen3_moe" => {
            let model = higgs_models::qwen3_moe::load_qwen3_moe_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Moe(model))
        }
        "gemma2" => {
            let model = higgs_models::gemma2::load_gemma2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Gemma2(model))
        }
        "gemma3" | "gemma3_text" => {
            let model = higgs_models::gemma3::load_gemma3_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Gemma3(model))
        }
        "gemma4" | "gemma4_text" | "gemma4_unified" => {
            let model = higgs_models::gemma4::load_gemma4_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Gemma4(model))
        }
        "phi3" => {
            let model = higgs_models::phi3::load_phi3_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Phi3(model))
        }
        "starcoder2" => {
            let model = higgs_models::starcoder2::load_starcoder2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Starcoder2(model))
        }
        "llava-qwen2" => {
            let model = higgs_models::llava_qwen2::load_llava_qwen2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::LlavaQwen2(model))
        }
        "deepseek_v2" => {
            let model = higgs_models::deepseek_v2::load_deepseek_v2_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::DeepSeekV2(model))
        }
        other => Err(EngineError::Model(
            higgs_models::error::ModelError::UnsupportedModel(other.to_owned()),
        )),
    }
}

/// Peek into `config.json` to detect packed 1-bit Bonsai-Q1 checkpoints.
///
/// Returns `true` for Qwen3-shaped `quantization.bits == 1` checkpoints using
/// the expected group size. Returns `false` for any other model type or
/// quantization config. A missing / malformed `config.json` propagates as an
/// IO / JSON error — we never mask it.
fn is_bonsai_q1(dir: &Path) -> Result<bool, EngineError> {
    let cfg_path = dir.join("config.json");
    let txt = std::fs::read_to_string(&cfg_path).map_err(|e| {
        EngineError::Model(higgs_models::error::ModelError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", cfg_path.display()),
        )))
    })?;
    let cfg: serde_json::Value = serde_json::from_str(&txt)
        .map_err(|e| EngineError::Model(higgs_models::error::ModelError::Json(e)))?;
    let bonsai_group_size = u64::try_from(higgs_models::bonsai_q1::GROUP_SIZE)
        .map_err(|e| EngineError::Model(ModelError::ShapeMismatch(e.to_string())))?;
    Ok(
        cfg.get("model_type").and_then(serde_json::Value::as_str) == Some("qwen3")
            && cfg
                .get("quantization")
                .and_then(|q| q.get("bits"))
                .and_then(serde_json::Value::as_u64)
                == Some(1)
            && cfg
                .get("quantization")
                .and_then(|q| q.get("group_size"))
                .and_then(serde_json::Value::as_u64)
                == Some(bonsai_group_size),
    )
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<tokenizers::Tokenizer, EngineError> {
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

/// Load a `DFlash` drafter from a model directory.
///
/// The drafter is a small block-diffusion model that the `SimpleEngine` uses to
/// propose draft tokens which the target verifies in a single forward.
pub fn load_dflash_drafter<P: AsRef<Path>, T: AsRef<Path>>(
    model_dir: P,
    target_dir: T,
) -> Result<dflash::DFlashDrafter, EngineError> {
    dflash::load_dflash_drafter_for_target(model_dir.as_ref(), target_dir.as_ref())
        .map_err(EngineError::Model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
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
    fn is_bonsai_q1_requires_qwen3_model_type_and_group_size() {
        let (qwen3_dir, _qwen3_result) = config_from_raw(
            r#"{
                "model_type": "qwen3",
                "quantization": {"bits": 1, "group_size": 128}
            }"#,
        );
        assert!(is_bonsai_q1(qwen3_dir.path()).unwrap());

        let (llama_dir, _llama_result) = config_from_raw(
            r#"{
                "model_type": "llama",
                "quantization": {"bits": 1, "group_size": 128}
            }"#,
        );
        assert!(!is_bonsai_q1(llama_dir.path()).unwrap());

        let (wrong_group_dir, _wrong_group_result) = config_from_raw(
            r#"{
                "model_type": "qwen3",
                "quantization": {"bits": 1, "group_size": 64}
            }"#,
        );
        assert!(!is_bonsai_q1(wrong_group_dir.path()).unwrap());

        let (q4_dir, _q4_result) = config_from_raw(
            r#"{
                "model_type": "qwen3",
                "quantization": {"bits": 4, "group_size": 128}
            }"#,
        );
        assert!(
            !is_bonsai_q1(q4_dir.path()).unwrap(),
            "regular Q4 Qwen3 must not be misclassified as Bonsai-Q1"
        );
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
