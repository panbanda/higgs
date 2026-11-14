//! Checks on a resolved configuration that make it unusable, collected rather than stopping at
//! the first. Range checks that only warn live in `higgs doctor`.

use super::settings::{HiggsConfig, ModelConfig};
use std::collections::HashSet;

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
    for problem in model.prefix_cache.range_problems() {
        problems.push(format!("model {}: {problem}", model.path));
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
