//! Builds, sizes and frees the real engines behind the model pool.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use higgs_engine::inference::{DiskPrefixConfig, EngineOptions, PrefixCacheOptions};
use higgs_engine::mlx_tuning::resolve_runtime_tuning_with;

use crate::config::{
    BatchMode, ExperimentalConfig, HiggsConfig, LoadMode, MemoryConfig, ModelConfig,
    PrefixCacheConfig,
};
use crate::decode_strategy::{DecodeFacts, DecodeStrategy, decode_strategy, planned_strategy};
use crate::model_pool::{ModelLoader, ModelPool, ModelSpec, PoolLimits};
use crate::model_resolver;
use crate::state::Engine;

/// Loads engines for the models named in the config.
pub struct EngineLoader {
    models: HashMap<String, ModelConfig>,
    memory: MemoryConfig,
    experimental: ExperimentalConfig,
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
        load_engine(model, &self.memory, &self.experimental, &resolved)
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
    memory: &MemoryConfig,
    experimental: &ExperimentalConfig,
    resolved: &Path,
) -> Result<Engine, String> {
    tracing::info!(model = %model.path, resolved = %resolved.display(), "Loading model");
    if let Ok(facts) = DecodeFacts::inspect(model, experimental, resolved) {
        match decode_strategy(&facts) {
            DecodeStrategy::Batched => {
                tracing::info!(model = %model.path, "requests decode together in batched passes");
            }
            DecodeStrategy::Sequential(reason) => {
                tracing::info!(model = %model.path, %reason, "requests decode one at a time");
            }
        }
    }
    let tuning =
        resolve_runtime_tuning_with(resolved, model.profile, &experimental.tuning_overrides());
    let engine = Engine::load(
        resolved,
        EngineOptions {
            kv_cache: model.kv_cache,
            tuning,
            raise_wired_limit: memory.raise_wired_limit,
            prefill_yield_tokens: model.prefill_yield_tokens,
            default_thinking_budget: model.thinking_budget,
            disk_prefix: model
                .prefix_cache
                .disk_dir
                .as_deref()
                .map(|dir| DiskPrefixConfig {
                    dir: Path::new(dir).to_path_buf(),
                    budget: model.prefix_cache.disk_budget.bytes(),
                }),
            prefix_cache: prefix_cache_options(&model.prefix_cache),
            max_concurrent_requests: (model.batch == BatchMode::Off).then_some(1),
        },
    )
    .map_err(|error| error.to_string())?;
    tracing::info!(model_name = %engine.model_name(), "Model loaded");
    Ok(engine)
}

fn prefix_cache_options(config: &PrefixCacheConfig) -> PrefixCacheOptions {
    let count = |value: u32| usize::try_from(value).unwrap_or(usize::MAX);
    PrefixCacheOptions {
        max_entries: count(config.max_entries),
        checkpoint_interval_tokens: count(config.checkpoint_interval_tokens),
        max_checkpoints_per_entry: count(config.max_checkpoints_per_entry),
        checkpoint_budget_bytes: usize::try_from(config.checkpoint_budget.bytes())
            .unwrap_or(usize::MAX),
    }
}

/// The name API clients use for a model: its configured name, else the name its engine reports.
pub fn served_name(model: &ModelConfig) -> String {
    if let Some(name) = &model.name {
        return name.clone();
    }
    match model_resolver::resolve(&model.path) {
        Ok(directory) => higgs_engine::model_loader::derive_model_name(&directory),
        Err(_) if model_resolver::is_hf_model_id(&model.path) => model.path.clone(),
        Err(_) => higgs_engine::model_loader::derive_model_name(Path::new(&model.path)),
    }
}

/// Lifecycle settings for one model.
///
/// The auto-router classifier is consulted on every `model = "auto"` request, so it always
/// loads at startup and is never evicted or expired.
pub fn model_spec(config: &HiggsConfig, model: &ModelConfig, name: String) -> ModelSpec {
    let is_classifier = config.routing.auto.enabled && config.routing.auto.model == name;
    ModelSpec {
        decode: planned_strategy(model, &config.experimental)
            .map_or("unknown", DecodeStrategy::kind),
        load: if is_classifier {
            LoadMode::Startup
        } else {
            model.load
        },
        pinned: model.pinned || is_classifier,
        idle_ttl: if is_classifier { None } else { model.idle_ttl },
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
        max_loaded_models: config.memory.max_loaded_models,
        memory_ceiling_bytes: config
            .memory
            .ceiling_bytes(crate::routes::system::physical_memory_bytes()),
    };
    let loader = EngineLoader {
        models,
        memory: config.memory.clone(),
        experimental: config.experimental.clone(),
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
    fn default_prefix_cache_config_maps_to_the_engine_defaults() {
        assert_eq!(
            prefix_cache_options(&PrefixCacheConfig::default()),
            PrefixCacheOptions::default()
        );
    }

    #[test]
    fn configured_prefix_cache_sizes_reach_the_engine_options() {
        let config = config_from_toml(
            "[[models]]\npath = \"org/m\"\n[models.prefix_cache]\nmax_entries = 3\ncheckpoint_interval_tokens = 2048\nmax_checkpoints_per_entry = 5\ncheckpoint_budget = \"1GB\"\n",
        );
        let options = prefix_cache_options(&config.models.first().unwrap().prefix_cache);
        assert_eq!(options.max_entries, 3);
        assert_eq!(options.checkpoint_interval_tokens, 2048);
        assert_eq!(options.max_checkpoints_per_entry, 5);
        assert_eq!(options.checkpoint_budget_bytes, 1 << 30);
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
            [defaults]
            load = "on_demand"
            idle_ttl = "5m"

            [[models]]
            path = "a/default"

            [[models]]
            path = "b/override"
            load = "startup"
            idle_ttl = "0"
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
            [defaults]
            load = "on_demand"
            idle_ttl = "1m"

            [[models]]
            path = "org/router"
            name = "router"
            load = "on_demand"

            [[models]]
            path = "org/chat"
            name = "chat"

            [routing.auto]
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
            [defaults]
            load = "on_demand"
            [memory]
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
        assert_eq!(pool.status("one").unwrap().engine, "unknown");
        assert_eq!(pool.status("two").unwrap().engine, "sequential");
        assert_eq!(pool.status("three").unwrap().engine, "unknown");
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
        assert_eq!(MemoryConfig::default().guard, MemoryGuard::Off);
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
            memory: config.memory.clone(),
            experimental: config.experimental.clone(),
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
            memory: config.memory.clone(),
            experimental: config.experimental.clone(),
        };

        let error = loader.load("gone").err().unwrap();

        assert!(error.contains("/nonexistent/higgs/model"), "{error}");
    }
}
