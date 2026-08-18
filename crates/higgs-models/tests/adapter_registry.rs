#![allow(clippy::panic, clippy::tests_outside_test_module, clippy::unwrap_used)]

use std::io::Write;

use higgs_models::adapter::{self, ModelFamily, ModelVersion};

fn write_config(value: &serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(value).unwrap(),
    )
    .unwrap();
    dir
}

fn complete_config(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "model_type": model_type,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 152_064,
        "intermediate_size": 11_008,
        "head_dim": 128,
        "max_position_embeddings": 131_072,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 0.000_001
    })
}

fn wrapped_config(top_model_type: &str, nested_model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "model_type": top_model_type,
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": complete_config(nested_model_type)
    })
}

#[test]
fn detects_plain_config() {
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5Model"]
    }));
    let detected = adapter::detect(dir.path()).unwrap();

    assert_eq!(detected.model_type, "qwen3_5");
    assert_eq!(detected.wrapper_model_type, None);
    assert_eq!(detected.family, ModelFamily::Qwen);
    assert_eq!(detected.version, Some(ModelVersion { major: 3, minor: 5 }));
}

#[test]
fn parses_family_versions_from_known_type_shapes() {
    for (model_type, family, version) in [
        (
            "gemma4_text",
            ModelFamily::Gemma,
            ModelVersion { major: 4, minor: 0 },
        ),
        (
            "deepseek_v2",
            ModelFamily::DeepSeek,
            ModelVersion { major: 2, minor: 0 },
        ),
    ] {
        let dir = write_config(&serde_json::json!({"model_type": model_type}));
        let detected = adapter::detect(dir.path()).unwrap();
        assert_eq!(detected.family, family);
        assert_eq!(detected.version, Some(version));
    }
}

#[test]
fn detects_nested_text_config_for_conditional_generation_wrapper() {
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": { "model_type": "qwen3_9" }
    }));
    let detected = adapter::detect(dir.path()).unwrap();

    assert_eq!(detected.wrapper_model_type.as_deref(), Some("qwen3_5"));
    assert_eq!(detected.model_type, "qwen3_9");
    assert_eq!(detected.version, Some(ModelVersion { major: 3, minor: 9 }));
}

#[test]
fn nested_text_config_is_always_a_resolution_candidate() {
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3_5",
        "architectures": ["NonstandardWrapperModel"],
        "text_config": complete_config("unknown_text_backbone")
    }));
    let detected = adapter::detect(dir.path()).unwrap();

    assert_eq!(detected.model_type, "unknown_text_backbone");
    assert_eq!(detected.wrapper_model_type.as_deref(), Some("qwen3_5"));
    assert_eq!(adapter::resolve(&detected).unwrap().id(), "qwen3.5-dense");
}

#[test]
fn exact_wrapper_fallback_validates_unknown_nested_config() {
    let mut text_config = complete_config("unknown_text_backbone");
    text_config.as_object_mut().unwrap().remove("hidden_size");
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3_5",
        "architectures": ["NonstandardWrapperModel"],
        "text_config": text_config
    }));
    let detected = adapter::detect(dir.path()).unwrap();
    let error = match adapter::resolve(&detected) {
        Err(error) => error,
        Ok(adapter) => panic!("unexpected adapter: {}", adapter.id()),
    };

    assert!(error.to_string().contains("hidden_size"));
}

#[test]
fn qwen38_wrapper_text_alias_resolves_to_dense_adapter() {
    let dir = write_config(&wrapped_config("qwen3_5", "qwen3_5_text"));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();

    assert_eq!(detected.model_type, "qwen3_5_text");
    assert_eq!(detected.wrapper_model_type.as_deref(), Some("qwen3_5"));
    assert_eq!(resolved.id(), "qwen3.5-dense");
    assert!(!adapter::is_untested_version(resolved, &detected));
}

#[test]
fn exact_wrapper_candidate_beats_tolerant_nested_candidate() {
    let dir = write_config(&wrapped_config("qwen3_5", "qwen3_9_text"));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();

    assert_eq!(resolved.id(), "qwen3.5-dense");
    assert!(!adapter::is_untested_version(resolved, &detected));
}

#[test]
fn future_qwen_wrapper_text_alias_resolves_tolerantly() {
    let dir = write_config(&wrapped_config("qwen3_9", "qwen3_9_text"));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();

    assert_eq!(resolved.id(), "qwen3.5-dense");
    assert!(adapter::is_untested_version(resolved, &detected));
}

#[test]
fn future_qwen_text_moe_alias_resolves_tolerantly() {
    let mut config = wrapped_config("qwen3_9_text_moe", "qwen3_9_text_moe");
    let text_config = config
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
        .unwrap();
    text_config.insert("num_experts".into(), 128.into());
    text_config.insert("num_experts_per_tok".into(), 8.into());
    text_config.insert("shared_expert_intermediate_size".into(), 1_024.into());
    text_config.insert("moe_intermediate_size".into(), 768.into());
    let dir = write_config(&config);
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();

    assert_eq!(resolved.id(), "qwen3.5-moe");
    assert!(adapter::is_untested_version(resolved, &detected));
}

#[test]
fn missing_model_type_is_an_error() {
    let dir = write_config(&serde_json::json!({"vocab_size": 32000}));
    let error = adapter::detect(dir.path()).unwrap_err();
    assert!(error.to_string().contains("missing model_type"));
}

#[test]
fn oversized_config_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut file = std::fs::File::create(dir.path().join("config.json")).unwrap();
    file.set_len(adapter::MAX_CONFIG_SIZE + 1).unwrap();
    file.flush().unwrap();

    let error = adapter::detect(dir.path()).unwrap_err();
    assert!(error.to_string().contains("config.json too large"));
}

#[test]
fn bonsai_is_more_specific_than_transformer_dense() {
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3",
        "quantization": {"bits": 1, "group_size": 128}
    }));
    let detected = adapter::detect(dir.path()).unwrap();
    assert_eq!(
        adapter::resolve(&detected).unwrap().id(),
        "bonsai-q1-packed"
    );
}

#[test]
fn qwen_moe_does_not_resolve_to_dense() {
    let dir = write_config(&complete_config("qwen3_5_moe"));
    let detected = adapter::detect(dir.path()).unwrap();
    assert_eq!(adapter::resolve(&detected).unwrap().id(), "qwen3.5-moe");
}

#[test]
fn exact_match_beats_version_tolerant_match() {
    let dir = write_config(&complete_config("gemma4_text"));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();
    assert_eq!(resolved.id(), "gemma4-text");
    assert!(!adapter::is_untested_version(resolved, &detected));
}

#[test]
fn future_qwen_dense_resolves_after_structural_check() {
    let dir = write_config(&complete_config("qwen3_9"));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();

    assert_eq!(resolved.id(), "qwen3.5-dense");
    assert!(adapter::is_untested_version(resolved, &detected));
}

#[test]
fn future_qwen_missing_required_field_is_rejected() {
    let mut config = complete_config("qwen3_9");
    config.as_object_mut().unwrap().remove("hidden_size");
    let dir = write_config(&config);
    let detected = adapter::detect(dir.path()).unwrap();
    let error = match adapter::resolve(&detected) {
        Err(error) => error,
        Ok(adapter) => panic!("unexpected adapter: {}", adapter.id()),
    };

    assert!(error.to_string().contains("hidden_size"));
}

#[test]
fn unknown_family_error_lists_supported_ranges() {
    let dir = write_config(&complete_config("mamba"));
    let detected = adapter::detect(dir.path()).unwrap();
    let error = match adapter::resolve(&detected) {
        Err(error) => error,
        Ok(adapter) => panic!("unexpected adapter: {}", adapter.id()),
    };

    assert!(error.to_string().contains("mamba"));
    assert!(error.to_string().contains("Qwen"));
    assert!(error.to_string().contains("Gemma"));
}

#[test]
fn supported_adapter_ids_are_unique() {
    let supported = adapter::supported();
    assert!(!supported.is_empty());
    let mut ids = supported.iter().map(|info| info.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), supported.len());
}

#[test]
fn adapter_load_uses_the_already_parsed_config() {
    let dir = write_config(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 0.000_001,
        "vocab_size": 256,
        "max_position_embeddings": 1024
    }));
    let detected = adapter::detect(dir.path()).unwrap();
    let resolved = adapter::resolve(&detected).unwrap();
    std::fs::remove_file(dir.path().join("config.json")).unwrap();

    let Err(error) = resolved.load(&detected) else {
        panic!("config-only checkpoint unexpectedly loaded");
    };
    assert!(
        !matches!(error, higgs_models::error::ModelError::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound),
        "adapter reopened config.json instead of using DetectedModel::raw"
    );
}
