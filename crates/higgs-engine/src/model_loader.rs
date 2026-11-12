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

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::indexing_slicing
)]
#[cfg(test)]
mod tests {
    use super::*;
    use higgs_models::error::ModelError;

    /// Largest absolute difference between two same-shaped arrays, relative
    /// to the largest magnitude in `expected`.
    fn relative_difference(expected: &mlx_rs::Array, actual: &mlx_rs::Array) -> f32 {
        let to_vec = |array: &mlx_rs::Array| {
            array
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap()
                .contiguous()
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        let expected_values = to_vec(expected);
        let actual_values = to_vec(actual);
        assert_eq!(expected_values.len(), actual_values.len());
        let largest = expected_values
            .iter()
            .fold(0.0_f32, |max, x| max.max(x.abs()));
        let difference = expected_values
            .iter()
            .zip(&actual_values)
            .fold(0.0_f32, |max, (a, b)| max.max((a - b).abs()));
        difference / largest
    }

    /// Batched prefill and decode must reproduce one-at-a-time prefill and
    /// decode for every architecture `supports_batched_decode` admits. Needs a
    /// real checkpoint:
    /// `HIGGS_MODEL_PATH=<model dir> cargo test -p higgs-engine -- --ignored batched_decode_matches`
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    fn batched_decode_matches_unbatched_decode() {
        use mlx_rs::{Array, ops::indexing::IndexOp};

        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = load_model(&model_path).unwrap();
        assert!(
            model.supports_batched_decode(),
            "this architecture does not claim batched decode support"
        );

        // Different lengths exercise left padding, masking, and per-row positions.
        let prompts: [&[u32]; 3] = [
            &[101, 2002, 3003, 404, 505, 6006, 707, 88, 99],
            &[909, 808, 7007],
            &[1100, 1200, 130, 1400, 150],
        ];
        let steps: [[u32; 3]; 3] = [[210, 2200, 230], [3100, 320, 3300], [41, 4200, 430]];
        let longest = prompts.iter().map(|p| p.len()).max().unwrap();

        let mut padded = Vec::new();
        for prompt in prompts {
            padded.extend(std::iter::repeat_n(0_u32, longest - prompt.len()));
            padded.extend_from_slice(prompt);
        }
        let lengths: Vec<i32> = prompts
            .iter()
            .map(|p| i32::try_from(p.len()).unwrap())
            .collect();
        let (batched_prefill, mut batch) = model
            .prefill_batched(
                &Array::from_slice(&padded, &[3, i32::try_from(longest).unwrap()]),
                &lengths,
            )
            .unwrap();
        let batched_steps: Vec<Array> = steps
            .iter()
            .map(|tokens| {
                let logits = model
                    .forward_batched(&Array::from_slice(tokens, &[3, 1]), &mut batch)
                    .unwrap();
                logits.eval().unwrap();
                logits
            })
            .collect();

        let mut worst = 0.0_f32;
        for (row, prompt) in prompts.iter().enumerate() {
            let row_index = i32::try_from(row).unwrap();
            let row_of = |logits: &Array| logits.index((row_index..row_index + 1, -1, ..));
            let mut cache = model.make_cache().unwrap();
            let length = i32::try_from(prompt.len()).unwrap();
            let prefill = model
                .forward(&Array::from_slice(prompt, &[1, length]), None, &mut cache)
                .unwrap();
            worst = worst.max(relative_difference(
                &prefill.index((.., -1, ..)),
                &row_of(&batched_prefill),
            ));
            for (step, batched) in steps.iter().zip(&batched_steps) {
                let logits = model
                    .forward(&Array::from_slice(&[step[row]], &[1, 1]), None, &mut cache)
                    .unwrap();
                worst = worst.max(relative_difference(
                    &logits.index((.., -1, ..)),
                    &row_of(batched),
                ));
            }
        }
        println!("largest logit difference relative to the largest logit: {worst}");
        // Half-precision kernels round differently by batch size (about 1.5%
        // on Qwen3-1.7B-4bit, 0.4% on Llama-3.2-1B-4bit). A positional or
        // masking bug is an order of magnitude larger.
        assert!(
            worst < 0.03,
            "batched logits differ from unbatched by {worst} of the largest logit"
        );
        assert!(!worst.is_nan(), "batched decode produced NaN logits");
    }

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
