//! Checks on a resolved configuration that make it unusable, collected rather than stopping at
//! the first. Range checks that only warn live in `higgs doctor`.

use std::collections::HashSet;
use std::path::Path;

use higgs_engine::model_loader;

use super::settings::{HiggsConfig, ModelConfig};
use super::types::BatchMode;

pub fn batch_unsupported_message(model_path: &str) -> String {
    format!(
        "batch=true is only supported for models with batched decode (llama, mistral, qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_5_moe, phi3, starcoder2, gemma2, gemma3, gemma4); {model_path} is not supported"
    )
}

pub fn resolved_model_supports_batch(model_dir: &Path) -> Result<bool, String> {
    let inspected = model_loader::ModelConfig::from_dir(model_dir)
        .map_err(|e| format!("failed to inspect {}: {e}", model_dir.display()))?;
    Ok(inspected.capabilities.batch_engine)
}

fn batch_support_for_model_path(model_path: &str) -> Result<Option<bool>, String> {
    match crate::model_resolver::resolve(model_path) {
        Ok(resolved) => resolved_model_supports_batch(&resolved).map(Some),
        Err(err) if batch_support_check_can_be_deferred(model_path, &err) => Ok(None),
        Err(err) => Err(err),
    }
}

fn batch_support_check_can_be_deferred(model_path: &str, err: &str) -> bool {
    crate::model_resolver::is_hf_model_id(model_path)
        && (err
            == format!(
                "model '{model_path}' is not an existing directory and was not found in the HuggingFace cache"
            )
            || (err.starts_with(&format!("could not read HF cache ref for '{model_path}':"))
                && err.contains("No such file or directory")))
}

fn model_problems(model: &ModelConfig, problems: &mut Vec<String>) {
    if model.path.trim().is_empty() {
        problems.push("model path must not be empty or whitespace-only".to_owned());
        return;
    }
    if model
        .name
        .as_ref()
        .is_some_and(|name| name.trim().is_empty())
    {
        problems.push("model name must not be empty or whitespace-only".to_owned());
    }
    for alias in &model.aliases {
        if alias.trim().is_empty() {
            problems.push(format!(
                "model {}: aliases must not be empty or whitespace-only",
                model.path
            ));
        }
    }
    if let Err(error) = model.kv_cache.validate() {
        problems.push(error.to_string());
    }
    if model.batch == BatchMode::On && model.kv_cache.is_turboquant() {
        problems.push(format!(
            "TurboQuant is not supported with batch=true for model {}",
            model.path
        ));
    }
    if model.batch == BatchMode::On {
        match batch_support_for_model_path(&model.path) {
            Ok(Some(false)) => problems.push(batch_unsupported_message(&model.path)),
            Ok(Some(true) | None) => {}
            Err(error) => problems.push(error),
        }
    }
}

/// The names API clients can use for each model (served name first, then its aliases) and any
/// collisions among them.
fn name_problems(config: &HiggsConfig, problems: &mut Vec<String>) {
    let mut seen_paths = HashSet::new();
    let mut seen_names = HashSet::new();
    for model in &config.models {
        if !seen_paths.insert(model.path.as_str()) {
            problems.push(format!("duplicate model path: {}", model.path));
        }
        if let Some(name) = &model.name {
            if !seen_names.insert(name.as_str()) {
                problems.push(format!("duplicate model name: {name}"));
            }
        }
    }
    let mut claimed: HashSet<String> = config
        .models
        .iter()
        .map(crate::engine_loader::served_name)
        .collect();
    for model in &config.models {
        let own_name = crate::engine_loader::served_name(model);
        for alias in &model.aliases {
            if *alias == own_name {
                problems.push(format!(
                    "alias '{alias}' of model {} is the model's own name",
                    model.path
                ));
            } else if !claimed.insert(alias.clone()) {
                problems.push(format!(
                    "alias '{alias}' of model {} collides with another model name or alias",
                    model.path
                ));
            }
        }
    }
}

/// Every reason `config` cannot be served. Empty when it can.
pub fn problems(config: &HiggsConfig, simple_mode: bool) -> Vec<String> {
    let mut problems = Vec::new();
    if simple_mode {
        if config.models.is_empty() {
            problems.push("at least one --model is required".to_owned());
        }
    } else if config.models.is_empty() && config.providers.is_empty() {
        problems
            .push("config must define at least one [[models]] entry or [providers.*]".to_owned());
    }

    if config.memory.limit.is_some_and(|limit| limit.bytes() == 0) {
        problems.push("memory.limit must be greater than zero".to_owned());
    }
    if config.memory.max_loaded_models == Some(0) {
        problems.push("memory.max_loaded_models must be at least 1".to_owned());
    }
    if config.server.request_timeout.is_zero() {
        problems.push("server.request_timeout must be longer than zero".to_owned());
    }

    for model in &config.models {
        model_problems(model, &mut problems);
    }
    name_problems(config, &mut problems);

    for route in &config.routes {
        if route.provider != "higgs" && !config.providers.contains_key(&route.provider) {
            problems.push(format!(
                "route references unknown provider '{}'",
                route.provider
            ));
        }
    }
    if config.routing.default_provider != "higgs"
        && !config
            .providers
            .contains_key(&config.routing.default_provider)
    {
        problems.push(format!(
            "routing.default '{}' not found in providers",
            config.routing.default_provider
        ));
    }
    problems
}
