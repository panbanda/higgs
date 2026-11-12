//! Builds, sizes and frees the real engines behind the model pool.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use higgs_engine::mlx_tuning::resolve_runtime_tuning;

use crate::config::{self, BatchMode, HiggsConfig, LoadMode, LocalConfig, ModelConfig};
use crate::engine_selection::{
    BatchFacts, EngineChoice, SimpleReason, choose_engine, planned_engine,
};
use crate::model_pool::{ModelLoader, ModelPool, ModelSpec, PoolLimits};
use crate::model_resolver;
use crate::state::Engine;

/// Loads engines for the models named in the config.
pub struct EngineLoader {
    models: HashMap<String, ModelConfig>,
    local: LocalConfig,
}

impl EngineLoader {
    fn model(&self, name: &str) -> Result<&ModelConfig, String> {
        self.models
            .get(name)
            .ok_or_else(|| format!("model '{name}' is not configured"))
    }
}

impl ModelLoader<Engine> for EngineLoader {
    fn load(&self, name: &str) -> Result<Engine, String> {
        let model = self.model(name)?;
        let resolved = model_resolver::resolve(&model.path)
            .map_err(|error| format!("cannot find '{}': {error}", model.path))?;
        load_engine(model, &self.local, &resolved)
    }

    fn unload(&self, name: &str, model: Arc<Engine>) {
        // MLX arrays are bound to the thread that built them, so an engine frees its model on
        // its own worker thread, and merely dropping the handle returns before that happens.
        // Shutting down joins the worker, so the memory is back before the pool clears the MLX
        // buffer cache and reports the model unloaded.
        match Arc::try_unwrap(model) {
            Ok(engine) => engine.shutdown(),
            Err(still_shared) => {
                tracing::warn!(
                    model = name,
                    "engine is still referenced; its memory is freed when the last reference drops"
                );
                drop(still_shared);
            }
        }
    }

    fn estimate_bytes(&self, name: &str) -> Option<u64> {
        model_resolver::estimated_footprint_bytes(&self.models.get(name)?.path)
    }

    fn memory_in_use(&self) -> Option<u64> {
        mlx_rs::memory::active_memory()
            .ok()
            .and_then(|bytes| u64::try_from(bytes).ok())
    }

    fn release_cached_memory(&self) {
        if let Err(error) = mlx_rs::memory::clear_cache() {
            tracing::warn!(%error, "failed to clear the MLX buffer cache");
        }
    }
}

/// Build one engine from a resolved model directory.
pub fn load_engine(
    model: &ModelConfig,
    local: &LocalConfig,
    resolved: &Path,
) -> Result<Engine, String> {
    tracing::info!(model = %model.path, resolved = %resolved.display(), "Loading model");
    let choice = match model.batch {
        BatchMode::On => {
            if !config::resolved_model_supports_batch(resolved)? {
                return Err(config::batch_unsupported_message(&model.path));
            }
            EngineChoice::Batch
        }
        BatchMode::Off => EngineChoice::Simple(SimpleReason::Requested),
        BatchMode::Auto => {
            let facts = BatchFacts::inspect(model, local, resolved)?;
            let choice = choose_engine(BatchMode::Auto, &facts);
            match choice {
                EngineChoice::Batch => {
                    tracing::info!(model = %model.path, "batch = auto: using the batch engine");
                }
                EngineChoice::Simple(reason) => {
                    tracing::info!(model = %model.path, %reason, "batch = auto: using the simple engine");
                }
            }
            choice
        }
    };
    let kv_cache_config = model.kv_cache_config();
    let tuning = resolve_runtime_tuning(resolved, model.requested_mlx_profile(local));
    let engine = match choice {
        EngineChoice::Batch => Engine::load_batch(
            resolved,
            kv_cache_config,
            &tuning,
            local.raise_wired_limit,
            model.prefill_yield_tokens,
            model.effective_thinking_budget(local),
        ),
        EngineChoice::Simple(_) => Engine::load_simple(
            resolved,
            kv_cache_config,
            tuning,
            local.raise_wired_limit,
            model.kv_disk_dir.as_deref().map(Path::new),
            model.kv_disk_space_mb.saturating_mul(1024 * 1024),
            model.effective_thinking_budget(local),
        ),
    }
    .map_err(|error| error.to_string())?;
    tracing::info!(model_name = %engine.model_name(), "Model loaded");
    Ok(engine)
}

/// The name API clients use for a model: its configured name, else the name its engine reports.
pub fn served_name(model: &ModelConfig) -> String {
    if let Some(name) = &model.name {
        return name.clone();
    }
    match model_resolver::resolve(&model.path) {
        Ok(directory) => higgs_engine::simple::derive_model_name(&directory),
        Err(_) if model_resolver::is_hf_model_id(&model.path) => model.path.clone(),
        Err(_) => higgs_engine::simple::derive_model_name(Path::new(&model.path)),
    }
}

/// Lifecycle settings for one model.
///
/// The auto-router classifier is consulted on every `model = "auto"` request, so it always
/// loads at startup and is never evicted or expired.
pub fn model_spec(config: &HiggsConfig, model: &ModelConfig, name: String) -> ModelSpec {
    let is_classifier = config.auto_router.enabled && config.auto_router.model == name;
    ModelSpec {
        engine_kind: planned_engine(model, &config.local).map_or("auto", EngineChoice::kind),
        load: if is_classifier {
            LoadMode::Startup
        } else {
            model.effective_load(&config.local)
        },
        pinned: model.pinned || is_classifier,
        idle_ttl: if is_classifier {
            None
        } else {
            model.effective_idle_ttl(&config.local)
        },
        name,
    }
}

/// Configured models that load when the server starts.
pub fn startup_models(config: &HiggsConfig) -> Vec<&ModelConfig> {
    config
        .models
        .iter()
        .filter(|model| model_spec(config, model, served_name(model)).load == LoadMode::Startup)
        .collect()
}

/// Build the pool for every `[[models]]` entry. Nothing is loaded yet.
pub fn build_pool(config: &HiggsConfig) -> Result<Arc<ModelPool<Engine>>, String> {
    let mut specs = Vec::new();
    let mut models = HashMap::new();
    for model in &config.models {
        let name = served_name(model);
        if models.insert(name.clone(), model.clone()).is_some() {
            return Err(format!(
                "model name collision: two model paths resolve to the same name '{name}'"
            ));
        }
        specs.push(model_spec(config, model, name));
    }
    let limits = PoolLimits {
        max_loaded_models: config.local.max_loaded_models,
        memory_ceiling_bytes: config
            .local
            .memory_ceiling_bytes(crate::routes::system::physical_memory_bytes()),
    };
    let loader = EngineLoader {
        models,
        local: config.local.clone(),
    };
    Ok(ModelPool::new(specs, Arc::new(loader), limits))
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{MemoryGuard, load_config_file};

    fn config_from_toml(toml: &str) -> HiggsConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        load_config_file(&path, None).unwrap()
    }

    #[test]
    fn configured_name_is_the_served_name() {
        let config = config_from_toml("[[models]]\npath = \"org/model\"\nname = \"coder\"\n");

        assert_eq!(served_name(config.models.first().unwrap()), "coder");
    }

    #[test]
    fn undownloaded_hf_model_is_served_under_its_repo_id() {
        let config = config_from_toml("[[models]]\npath = \"no-such-org/never-downloaded\"\n");

        assert_eq!(
            served_name(config.models.first().unwrap()),
            "no-such-org/never-downloaded"
        );
    }

    #[test]
    fn local_directory_is_served_under_its_leaf_name() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("my-local-model");
        std::fs::create_dir(&model_dir).unwrap();
        let config = config_from_toml(&format!("[[models]]\npath = \"{}\"\n", model_dir.display()));

        assert_eq!(
            served_name(config.models.first().unwrap()),
            "my-local-model"
        );
    }

    #[test]
    fn spec_applies_local_defaults_and_model_overrides() {
        let config = config_from_toml(
            r#"
            [local]
            load = "on_demand"
            idle_ttl_seconds = 300

            [[models]]
            path = "a/default"

            [[models]]
            path = "b/override"
            load = "startup"
            idle_ttl_seconds = 0
            pinned = true
            batch = false
            "#,
        );
        let specs: Vec<ModelSpec> = config
            .models
            .iter()
            .map(|model| model_spec(&config, model, served_name(model)))
            .collect();
        let default = specs.first().unwrap();
        let overridden = specs.get(1).unwrap();

        assert_eq!(default.load, LoadMode::OnDemand);
        assert_eq!(default.idle_ttl, Some(std::time::Duration::from_secs(300)));
        assert!(!default.pinned);
        assert_eq!(overridden.load, LoadMode::Startup);
        assert_eq!(overridden.idle_ttl, None);
        assert!(overridden.pinned);
    }

    #[test]
    fn auto_router_classifier_always_loads_at_startup_and_is_never_evicted() {
        let config = config_from_toml(
            r#"
            [local]
            load = "on_demand"
            idle_ttl_seconds = 60

            [[models]]
            path = "org/router"
            name = "router"
            load = "on_demand"

            [[models]]
            path = "org/chat"
            name = "chat"

            [auto_router]
            enabled = true
            model = "router"
            "#,
        );

        let router = model_spec(&config, config.models.first().unwrap(), "router".to_owned());
        let chat = model_spec(&config, config.models.get(1).unwrap(), "chat".to_owned());

        assert_eq!(router.load, LoadMode::Startup);
        assert!(router.pinned);
        assert_eq!(router.idle_ttl, None);
        assert_eq!(chat.load, LoadMode::OnDemand);
        assert!(!chat.pinned);
    }

    #[test]
    fn startup_models_are_those_not_deferred() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "org/eager"
            name = "eager"

            [[models]]
            path = "org/lazy"
            name = "lazy"
            load = "on_demand"
            "#,
        );

        let names: Vec<&str> = startup_models(&config)
            .into_iter()
            .map(|model| model.path.as_str())
            .collect();

        assert_eq!(names, ["org/eager"]);
    }

    #[test]
    fn pool_lists_every_model_unloaded_and_reports_the_limits() {
        let config = config_from_toml(
            r#"
            [local]
            load = "on_demand"
            max_loaded_models = 1

            [[models]]
            path = "org/one"
            name = "one"
            batch = true

            [[models]]
            path = "org/two"
            name = "two"
            batch = false

            [[models]]
            path = "org/three-never-downloaded"
            name = "three"
            "#,
        );

        let pool = build_pool(&config).unwrap();
        let statuses = pool.statuses();

        assert_eq!(statuses.len(), 3);
        assert!(statuses.iter().all(|status| {
            status.state == crate::model_pool::ModelState::Unloaded
                && status.load == LoadMode::OnDemand
        }));
        assert_eq!(pool.status("one").unwrap().engine, "batch");
        assert_eq!(pool.status("two").unwrap().engine, "simple");
        assert_eq!(pool.status("three").unwrap().engine, "auto");
    }

    #[test]
    fn two_models_with_the_same_served_name_collide() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "org-a/same"
            name = "same"

            [[models]]
            path = "org-b/other"
            name = "same2"
            "#,
        );
        let mut colliding = config.clone();
        colliding.models.get_mut(1).unwrap().name = Some("same".to_owned());

        let error = build_pool(&colliding).err().unwrap();

        assert!(error.contains("model name collision"), "{error}");
        assert!(build_pool(&config).is_ok());
    }

    #[test]
    fn memory_guard_is_off_by_default_in_the_pool_limits() {
        assert_eq!(LocalConfig::default().memory_guard, MemoryGuard::Off);
    }

    #[test]
    fn estimate_is_the_weight_size_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0_u8; 4096]).unwrap();
        let config = config_from_toml(&format!(
            "[[models]]\npath = \"{}\"\nname = \"sized\"\n",
            dir.path().display()
        ));
        let loader = EngineLoader {
            models: config
                .models
                .iter()
                .map(|model| (served_name(model), model.clone()))
                .collect(),
            local: config.local.clone(),
        };

        assert_eq!(loader.estimate_bytes("sized"), Some(4096));
        assert_eq!(loader.estimate_bytes("unknown"), None);
    }

    #[test]
    fn loading_a_missing_model_directory_reports_the_path() {
        let config =
            config_from_toml("[[models]]\npath = \"/nonexistent/higgs/model\"\nname = \"gone\"\n");
        let loader = EngineLoader {
            models: [("gone".to_owned(), config.models.first().unwrap().clone())].into(),
            local: config.local.clone(),
        };

        let error = loader.load("gone").err().unwrap();

        assert!(error.contains("/nonexistent/higgs/model"), "{error}");
    }
}
