use std::sync::Arc;

use crate::config::HiggsConfig;
use crate::metrics::MetricsStore;
use crate::router::Router;

/// Shared application state available to all route handlers.
pub struct AppState {
    /// Routes model names to local engines or remote providers.
    pub router: Router,
    /// Full server configuration.
    pub config: HiggsConfig,
    /// HTTP client for proxying requests to remote providers.
    pub http_client: reqwest::Client,
    /// Request metrics (present in config mode, absent in simple mode).
    pub metrics: Option<Arc<MetricsStore>>,
    /// Tools the server can run for a model (MCP servers).
    pub tools: Arc<crate::tools::ToolHub>,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;

#[cfg(test)]
#[allow(clippy::panic)]
pub(crate) mod test_support {
    use std::sync::{Arc, Mutex};

    use crate::backend::LoadedEngine;
    use crate::config::LoadMode;
    use crate::model_pool::{ModelLoader, ModelPool, ModelSpec, PoolLimits};
    use crate::test_support::StubEngine;

    /// Loads [`StubEngine`]s and records what the pool asked of it.
    #[derive(Default)]
    pub struct StubLoader {
        pub fail_with: Mutex<Option<String>>,
        pub events: Mutex<Vec<String>>,
    }

    impl ModelLoader<LoadedEngine> for StubLoader {
        fn load(&self, name: &str) -> Result<LoadedEngine, String> {
            let scripted_failure = self
                .fail_with
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if let Some(reason) = scripted_failure {
                return Err(reason);
            }
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!("load:{name}"));
            Ok(Box::new(StubEngine::named(name)))
        }

        fn unload(&self, name: &str, model: Arc<LoadedEngine>) {
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!("unload:{name}"));
            drop(model);
        }

        fn estimate_bytes(&self, _name: &str) -> Option<u64> {
            None
        }

        fn memory_in_use(&self) -> Option<u64> {
            None
        }

        fn release_cached_memory(&self) {}
    }

    pub fn stub_spec(name: &str) -> ModelSpec {
        ModelSpec {
            name: name.to_owned(),
            decode: "sequential",
            load: LoadMode::OnDemand,
            pinned: false,
            idle_ttl: None,
        }
    }

    pub fn stub_pool_with(
        specs: Vec<ModelSpec>,
        limits: PoolLimits,
    ) -> (Arc<ModelPool<LoadedEngine>>, Arc<StubLoader>) {
        let loader = Arc::new(StubLoader::default());
        let shared_loader: Arc<dyn ModelLoader<LoadedEngine>> = Arc::<StubLoader>::clone(&loader);
        (ModelPool::new(specs, shared_loader, limits), loader)
    }

    /// App state over a pool of stub engines, with the default config.
    pub fn stub_state(
        specs: Vec<ModelSpec>,
        limits: PoolLimits,
        metrics: Option<Arc<crate::metrics::MetricsStore>>,
    ) -> (super::SharedState, Arc<StubLoader>) {
        let (pool, loader) = stub_pool_with(specs, limits);
        let config = crate::config::HiggsConfig::default();
        let router = crate::router::Router::new(&config, pool)
            .unwrap_or_else(|error| panic!("default config must build a router: {error}"));
        let state = Arc::new(super::AppState {
            router,
            config,
            http_client: reqwest::Client::new(),
            metrics,
            tools: Arc::new(crate::tools::ToolHub::disabled()),
        });
        (state, loader)
    }
}
