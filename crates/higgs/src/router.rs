use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use regex::Regex;
use tracing::{info, warn};

use crate::backend::{EngineLease, LoadedEngine};
use crate::config::{ApiFormat, HiggsConfig, LoadMode};
use crate::model_pool::{ModelError, ModelPool, ModelSpec};

/// Why a model name could not be resolved to a servable route.
#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    /// Nothing is configured to serve the requested model.
    #[error("{0}")]
    NotFound(String),
    /// The model is configured but could not be made ready.
    #[error(transparent)]
    Model(#[from] ModelError),
}

/// How a model name was resolved to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMethod {
    /// Direct lookup by model name in local engines (no route matched).
    Direct,
    /// Matched a regex pattern route.
    Pattern,
    /// Selected by the auto-router AI classifier.
    Auto,
    /// Fell through to the default provider.
    Default,
}

/// Outcome of resolving a model name through the routing table.
pub enum ResolvedRoute {
    /// Serve locally via a loaded MLX engine.
    Higgs {
        engine: EngineLease,
        model_name: String,
        routing_method: RoutingMethod,
    },
    /// Forward to a remote provider.
    Remote {
        provider_name: String,
        provider_url: String,
        provider_format: ApiFormat,
        model_rewrite: Option<String>,
        strip_auth: bool,
        api_key: Option<crate::config::Secret>,
        stub_count_tokens: bool,
        routing_method: RoutingMethod,
    },
}

/// A named route candidate for auto-routing classification.
#[derive(Clone)]
pub struct RouteCandidate {
    pub name: String,
    pub description: String,
}

// -- Internal types --------------------------------------------------------

#[derive(Clone)]
enum RouteTarget {
    Higgs {
        model_rewrite: Option<String>,
    },
    Remote {
        provider_name: String,
        provider_url: String,
        provider_format: ApiFormat,
        model_rewrite: Option<String>,
        strip_auth: bool,
        api_key: Option<crate::config::Secret>,
        stub_count_tokens: bool,
    },
}

struct CompiledRoute {
    pattern: Regex,
    target: RouteTarget,
}

struct AutoRouteEntry {
    name: String,
    target: RouteTarget,
}

/// Routes model names to local engines or remote providers.
///
/// Resolution order:
/// 1. If `model == "auto"`, try auto-routing classification
/// 2. Direct engine lookup by model name
/// 3. Pattern matching (first match wins)
/// 4. With one local model and nothing else configured, that model
/// 5. Default provider fallback
pub struct Router {
    pool: Arc<ModelPool<LoadedEngine>>,
    compiled_routes: Vec<CompiledRoute>,
    auto_routes: Vec<AutoRouteEntry>,
    auto_candidates: Vec<RouteCandidate>,
    auto_router_model: Option<String>,
    auto_router_force: bool,
    auto_router_timeout: std::time::Duration,
    default_target: RouteTarget,
    /// Extra request names (`aliases`) mapped to the served name of their model.
    aliases: HashMap<String, String>,
    /// True when the config has no routes and no remote providers, so a request for an
    /// unknown name can only mean the one local model, if there is exactly one.
    local_only: bool,
    /// Requested names already logged as mapped to the single local model.
    announced_fallback_names: Mutex<HashSet<String>>,
}

/// Cap on remembered names, so clients that invent names cannot grow the set without bound.
const MAX_ANNOUNCED_FALLBACK_NAMES: usize = 64;

impl Router {
    /// Build a router from the unified config and the pool that serves its local models.
    pub fn new(config: &HiggsConfig, pool: Arc<ModelPool<LoadedEngine>>) -> Result<Self, String> {
        let mut compiled_routes = Vec::new();
        let mut auto_routes = Vec::new();
        let mut auto_candidates = Vec::new();
        let mut seen_names = HashSet::new();

        for route in &config.routes {
            if route.pattern.is_none() && route.description.is_none() {
                return Err(format!(
                    "route for provider '{}' has neither pattern nor description",
                    route.provider
                ));
            }

            if route.description.is_some() && route.name.is_none() {
                return Err(format!(
                    "route for provider '{}' has description but no name",
                    route.provider
                ));
            }

            let target = build_route_target(&route.provider, route.model.clone(), config)?;

            if let Some(ref pattern_str) = route.pattern {
                let pattern = Regex::new(pattern_str)
                    .map_err(|e| format!("invalid regex '{pattern_str}': {e}"))?;
                compiled_routes.push(CompiledRoute {
                    pattern,
                    target: target.clone(),
                });
            }

            if let (Some(name), Some(description)) = (&route.name, &route.description) {
                if !seen_names.insert(name.clone()) {
                    return Err(format!("duplicate route name '{name}'"));
                }
                auto_routes.push(AutoRouteEntry {
                    name: name.clone(),
                    target,
                });
                auto_candidates.push(RouteCandidate {
                    name: name.clone(),
                    description: description.clone(),
                });
            }
        }

        let auto_router_model = if config.routing.auto.enabled {
            if config.routing.auto.model.is_empty() {
                return Err("routing.auto.enabled is true but model is empty".to_owned());
            }
            if auto_candidates.is_empty() {
                warn!("routing.auto is enabled but no routes have descriptions");
            }
            let auto_model = &config.routing.auto.model;
            if !pool.contains(auto_model) {
                return Err(format!(
                    "routing.auto.model '{auto_model}' not found among loaded models"
                ));
            }
            Some(auto_model.clone())
        } else {
            None
        };

        let default_target = build_route_target(&config.routing.default_provider, None, config)?;

        let mut aliases = HashMap::new();
        for model in &config.models {
            let served = crate::engine_loader::served_name(model);
            for alias in &model.aliases {
                aliases.insert(alias.clone(), served.clone());
            }
        }

        Ok(Self {
            pool,
            compiled_routes,
            auto_routes,
            auto_candidates,
            auto_router_model,
            auto_router_force: config.routing.auto.force,
            auto_router_timeout: config.routing.auto.timeout,
            default_target,
            aliases,
            local_only: config.routes.is_empty()
                && config.providers.is_empty()
                && config.routing.default_provider == crate::config::DEFAULT_ROUTING_PROVIDER,
            announced_fallback_names: Mutex::new(HashSet::new()),
        })
    }

    /// Resolve a model name to a route.
    ///
    /// Pass `messages` for auto-routing support. Used when `model == "auto"` or
    /// when `force` mode is enabled.
    pub async fn resolve(
        &self,
        model: &str,
        messages: Option<&[serde_json::Value]>,
    ) -> Result<ResolvedRoute, RouteError> {
        if self.auto_router_force || model == "auto" {
            if let Some(resolved) = self.try_auto_route(messages).await {
                return Ok(resolved);
            }
            if model == "auto" {
                return self
                    .resolve_target(&self.default_target, model, RoutingMethod::Default)
                    .await;
            }
            // force mode: auto-routing returned nothing, fall through to normal resolution
        }

        // Direct engine lookup, under the model's name or one of its aliases
        let canonical = self.canonical_name(model);
        if self.pool.contains(canonical) {
            return Ok(ResolvedRoute::Higgs {
                engine: self.pool.acquire(canonical).await?,
                model_name: canonical.to_owned(),
                routing_method: RoutingMethod::Direct,
            });
        }

        // Pattern matching (first match wins)
        for route in &self.compiled_routes {
            if route.pattern.is_match(model) {
                return self
                    .resolve_target(&route.target, model, RoutingMethod::Pattern)
                    .await;
            }
        }

        if let Some(only_model) = self.sole_local_model() {
            self.announce_fallback(model, only_model);
            return Ok(ResolvedRoute::Higgs {
                engine: self.pool.acquire(only_model).await?,
                model_name: only_model.to_owned(),
                routing_method: RoutingMethod::Default,
            });
        }

        // Default fallback
        self.resolve_target(&self.default_target, model, RoutingMethod::Default)
            .await
    }

    /// The only local model, when the server is configured for local serving alone and
    /// has exactly one. Clients such as Claude Code send fixed vendor model names that
    /// no local model carries; there is nothing else such a request could mean.
    fn sole_local_model(&self) -> Option<&str> {
        if !self.local_only {
            return None;
        }
        let mut names = self.pool.names();
        let only = names.next()?;
        names.next().is_none().then_some(only)
    }

    fn announce_fallback(&self, requested: &str, served: &str) {
        let mut announced = self
            .announced_fallback_names
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if announced.len() < MAX_ANNOUNCED_FALLBACK_NAMES && announced.insert(requested.to_owned())
        {
            info!(
                requested,
                served, "unknown model name served by the only local model"
            );
        }
    }

    /// The served name a request name refers to: itself, or the model it is an alias of.
    fn canonical_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.aliases.get(name).map_or(name, String::as_str)
    }

    /// Build a router over engines the caller already loaded.
    ///
    /// Such engines are pinned: the pool has no way to rebuild one once it is unloaded.
    pub fn from_config(
        config: &HiggsConfig,
        engines: std::collections::HashMap<String, Arc<LoadedEngine>>,
    ) -> Result<Self, String> {
        let mut resident: Vec<(ModelSpec, Arc<LoadedEngine>)> = engines
            .into_iter()
            .map(|(name, engine)| {
                let spec = ModelSpec {
                    name,
                    decode: engine.decode_kind(),
                    load: LoadMode::Startup,
                    pinned: true,
                    idle_ttl: None,
                };
                (spec, engine)
            })
            .collect();
        resident.sort_by(|(left, _), (right, _)| left.name.cmp(&right.name));
        Self::new(config, ModelPool::from_resident(resident))
    }

    /// The pool that loads, tracks and unloads the configured local models.
    pub const fn pool(&self) -> &Arc<ModelPool<LoadedEngine>> {
        &self.pool
    }

    // -- Private helpers ---------------------------------------------------

    async fn try_auto_route(
        &self,
        messages: Option<&[serde_json::Value]>,
    ) -> Option<ResolvedRoute> {
        let classifier_model = self.auto_router_model.as_ref()?;
        let msg_slice = messages?;
        if self.auto_candidates.is_empty() || msg_slice.is_empty() {
            return None;
        }

        let classifier = match self.pool.acquire(classifier_model).await {
            Ok(lease) => lease,
            Err(error) => {
                warn!(%error, "auto-router classifier unavailable; using normal routing");
                return None;
            }
        };
        let candidates = self.auto_candidates.clone();
        let messages_owned = msg_slice.to_vec();

        let timeout = self.auto_router_timeout;
        let timeout_result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                crate::auto_router::classify_local(&**classifier, &candidates, &messages_owned)
            }),
        )
        .await;
        let name = if let Ok(join_result) = timeout_result {
            join_result.ok()??
        } else {
            warn!("auto-router classification timed out after {timeout:?}");
            return None;
        };

        let entry = self.auto_routes.iter().find(|r| r.name == name)?;
        self.resolve_target(&entry.target, "auto", RoutingMethod::Auto)
            .await
            .ok()
    }

    async fn resolve_target(
        &self,
        target: &RouteTarget,
        model: &str,
        method: RoutingMethod,
    ) -> Result<ResolvedRoute, RouteError> {
        match target {
            RouteTarget::Higgs { model_rewrite } => {
                let lookup_name = self.canonical_name(model_rewrite.as_deref().unwrap_or(model));
                let resolved_name = if self.pool.contains(lookup_name) {
                    lookup_name
                } else if model == "auto" {
                    // "auto" is a virtual model name; prefer a model that is already loaded
                    self.pool.preferred_name().ok_or_else(|| {
                        RouteError::NotFound(
                            "no local models configured for default route".to_owned(),
                        )
                    })?
                } else {
                    return Err(RouteError::NotFound(format!(
                        "model '{lookup_name}' not found among loaded local models"
                    )));
                };
                Ok(ResolvedRoute::Higgs {
                    engine: self.pool.acquire(resolved_name).await?,
                    model_name: resolved_name.to_owned(),
                    routing_method: method,
                })
            }
            RouteTarget::Remote {
                provider_name,
                provider_url,
                provider_format,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
            } => {
                if model == "auto" && model_rewrite.is_none() {
                    return Err(RouteError::NotFound(
                        "cannot forward virtual model name \"auto\" to remote provider; \
                         configure a model rewrite on the default route or add auto-router routes"
                            .to_owned(),
                    ));
                }
                Ok(ResolvedRoute::Remote {
                    provider_name: provider_name.clone(),
                    provider_url: provider_url.clone(),
                    provider_format: *provider_format,
                    model_rewrite: model_rewrite.clone(),
                    strip_auth: *strip_auth,
                    api_key: api_key.clone(),
                    stub_count_tokens: *stub_count_tokens,
                    routing_method: method,
                })
            }
        }
    }
}

fn build_route_target(
    provider_name: &str,
    model_rewrite: Option<String>,
    config: &HiggsConfig,
) -> Result<RouteTarget, String> {
    if provider_name == "higgs" {
        return Ok(RouteTarget::Higgs { model_rewrite });
    }
    let provider = config
        .providers
        .get(provider_name)
        .ok_or_else(|| format!("route provider '{provider_name}' not found in providers"))?;
    Ok(RouteTarget::Remote {
        provider_name: provider_name.to_owned(),
        provider_url: provider.url.clone(),
        provider_format: provider.format,
        model_rewrite,
        strip_auth: provider.strip_auth,
        api_key: provider.api_key.clone(),
        stub_count_tokens: provider.stub_count_tokens,
    })
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::shadow_unrelated
)]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::config::load_config_file;

    fn config_from_toml(toml: &str) -> HiggsConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        load_config_file(&path, None).unwrap()
    }

    fn router_from_toml(toml: &str) -> Router {
        let config = config_from_toml(toml);
        Router::from_config(&config, HashMap::new()).unwrap()
    }

    fn production_toml() -> &'static str {
        r#"
        [provider.anthropic]
        url = "https://api.anthropic.com"
        format = "anthropic"

        [provider.ollama]
        url = "http://localhost:11434"
        strip_auth = true
        api_key = "ollama"
        stub_count_tokens = true

        [[routes]]
        pattern = "opus"
        provider = "anthropic"

        [[routes]]
        pattern = "sonnet|haiku"
        provider = "ollama"
        model = "qwen3-coder:30b"

        [default]
        provider = "anthropic"
        "#
    }

    #[tokio::test]
    async fn remote_pattern_resolves_to_anthropic() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("claude-opus-4-6", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_name,
                provider_url,
                provider_format,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
                routing_method,
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(provider_format, ApiFormat::Anthropic);
                assert_eq!(model_rewrite, None);
                assert!(!strip_auth);
                assert_eq!(api_key, None);
                assert!(!stub_count_tokens);
                assert_eq!(routing_method, RoutingMethod::Pattern);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn remote_pattern_with_model_rewrite() {
        let router = router_from_toml(production_toml());
        let route = router
            .resolve("claude-sonnet-4-5-20250929", None)
            .await
            .unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_url,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
                ..
            } => {
                assert_eq!(provider_url, "http://localhost:11434");
                assert_eq!(model_rewrite.as_deref(), Some("qwen3-coder:30b"));
                assert!(strip_auth);
                assert_eq!(
                    api_key.as_ref().map(crate::config::Secret::expose),
                    Some("ollama")
                );
                assert!(stub_count_tokens);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn unmatched_model_falls_to_default() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("some-unknown-model", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_name,
                provider_url,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn empty_model_falls_to_default() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_url,
                routing_method,
                ..
            } => {
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn first_matching_route_wins() {
        let router = router_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [provider.b]
            url = "http://b"
            [[routes]]
            pattern = "opus"
            provider = "a"
            [[routes]]
            pattern = "opus"
            provider = "b"
            [default]
            provider = "a"
            "#,
        );
        let route = router.resolve("opus", None).await.unwrap();
        match route {
            ResolvedRoute::Remote { provider_url, .. } => {
                assert_eq!(provider_url, "http://a");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[test]
    fn invalid_regex_returns_error() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "[invalid"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("invalid regex"), "got: {err}");
    }

    #[test]
    fn missing_route_provider_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "test"
            provider = "nonexistent"
            [default]
            provider = "a"
            "#,
        )
        .unwrap();
        // Config validation catches this before Router::from_config
        let result = load_config_file(&path, None);
        assert!(result.is_err());
    }

    #[test]
    fn missing_default_provider_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "x"
            provider = "a"
            [default]
            provider = "nonexistent"
            "#,
        )
        .unwrap();
        let result = load_config_file(&path, None);
        assert!(result.is_err());
    }

    #[test]
    fn description_without_name_errors() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            description = "some task"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("description but no name"), "got: {err}");
    }

    #[test]
    fn route_without_pattern_or_description_errors() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(
            err.contains("neither pattern nor description"),
            "got: {err}"
        );
    }

    #[test]
    fn duplicate_route_names_error() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            name = "coding"
            description = "code tasks"
            pattern = "opus"
            provider = "a"
            [[routes]]
            name = "coding"
            description = "other code tasks"
            pattern = "sonnet"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("duplicate route name"), "got: {err}");
    }

    #[test]
    fn auto_candidates_built_from_descriptions() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [provider.b]
            url = "http://b"
            [[routes]]
            name = "coding"
            description = "code tasks"
            pattern = "opus"
            provider = "a"
            [[routes]]
            pattern = "sonnet"
            provider = "b"
            [default]
            provider = "a"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert_eq!(router.auto_candidates.len(), 1);
        assert_eq!(router.auto_candidates[0].name, "coding");
        assert_eq!(router.auto_routes.len(), 1);
        assert_eq!(router.compiled_routes.len(), 2);
    }

    #[test]
    fn description_only_route_not_in_pattern_routes() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            name = "coding"
            description = "code tasks"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert_eq!(router.compiled_routes.len(), 0);
        assert_eq!(router.auto_candidates.len(), 1);
    }

    #[tokio::test]
    async fn higgs_route_errors_when_model_not_found() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            [[routes]]
            pattern = "Llama.*"
            provider = "higgs"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("Llama-3.2-1B", None).await;
        match result {
            Err(e) => assert!(
                e.to_string()
                    .contains("not found among loaded local models"),
                "got: {e}"
            ),
            Ok(_) => panic!("expected error for missing local model"),
        }
    }

    #[tokio::test]
    async fn exact_local_model_beats_regex_route() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            name = "llama"

            [provider.remote]
            url = "http://127.0.0.1:9"

            [[routes]]
            pattern = "llama"
            provider = "remote"

            [default]
            provider = "remote"
            "#,
        );
        let mut engines = HashMap::new();
        engines.insert(
            "llama".to_owned(),
            Arc::new(crate::test_support::StubEngine::named("llama").into_loaded()),
        );

        let router = Router::from_config(&config, engines).unwrap();
        let result = router.resolve("llama", None).await.unwrap();

        match result {
            ResolvedRoute::Higgs {
                model_name,
                routing_method,
                ..
            } => {
                assert_eq!(model_name, "llama");
                assert_eq!(routing_method, RoutingMethod::Direct);
            }
            ResolvedRoute::Remote { .. } => panic!("expected exact local model to win"),
        }
    }

    fn stub_engines(names: &[&str]) -> HashMap<String, Arc<LoadedEngine>> {
        names
            .iter()
            .map(|name| {
                (
                    (*name).to_owned(),
                    Arc::new(crate::test_support::StubEngine::named(name).into_loaded()),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn only_local_model_serves_unknown_names() {
        let config = config_from_toml("[[models]]\npath = \"some/model\"\nname = \"solo\"\n");
        let router = Router::from_config(&config, stub_engines(&["solo"])).unwrap();
        for requested in ["claude-sonnet-4-5", "gpt-4o", "claude-sonnet-4-5"] {
            match router.resolve(requested, None).await.unwrap() {
                ResolvedRoute::Higgs { model_name, .. } => assert_eq!(model_name, "solo"),
                ResolvedRoute::Remote { .. } => panic!("expected the local model"),
            }
        }
        assert_eq!(router.announced_fallback_names.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn two_local_models_keep_unknown_names_not_found() {
        let config = config_from_toml(
            "[[models]]\npath = \"some/a\"\nname = \"a\"\n[[models]]\npath = \"some/b\"\nname = \"b\"\n",
        );
        let router = Router::from_config(&config, stub_engines(&["a", "b"])).unwrap();
        assert!(matches!(
            router.resolve("gpt-4o", None).await,
            Err(RouteError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn provider_or_route_disables_the_single_model_fallback() {
        for extra in [
            "[provider.remote]\nurl = \"http://127.0.0.1:9\"\n",
            "[provider.remote]\nurl = \"http://127.0.0.1:9\"\n[[routes]]\npattern = \"^r-\"\nprovider = \"remote\"\n",
            "[[routes]]\npattern = \"^r-\"\nprovider = \"higgs\"\nmodel = \"solo\"\n",
        ] {
            let config = config_from_toml(&format!(
                "[[models]]\npath = \"some/model\"\nname = \"solo\"\n{extra}"
            ));
            let router = Router::from_config(&config, stub_engines(&["solo"])).unwrap();
            assert!(
                matches!(
                    router.resolve("gpt-4o", None).await,
                    Err(RouteError::NotFound(_))
                ),
                "config: {extra}"
            );
        }
    }

    #[tokio::test]
    async fn higgs_default_errors_when_model_not_found() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        // No routes, default is "higgs", no engines loaded
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("nonexistent-model", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resolved_route_includes_provider_name() {
        let router = router_from_toml(production_toml());

        let route = router.resolve("claude-opus-4-6", None).await.unwrap();
        match route {
            ResolvedRoute::Remote { provider_name, .. } => {
                assert_eq!(provider_name, "anthropic");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote"),
        }

        let route = router
            .resolve("claude-sonnet-4-5-20250929", None)
            .await
            .unwrap();
        match route {
            ResolvedRoute::Remote { provider_name, .. } => {
                assert_eq!(provider_name, "ollama");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote"),
        }
    }

    #[test]
    fn no_routes_no_providers_uses_higgs_default() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        // default_target should be Higgs
        match &router.default_target {
            RouteTarget::Higgs { model_rewrite } => {
                assert!(model_rewrite.is_none());
            }
            RouteTarget::Remote { .. } => panic!("expected Higgs default"),
        }
    }

    #[test]
    fn pool_accessor_lists_no_models_when_none_are_supplied() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert!(router.pool().statuses().is_empty());
    }

    #[tokio::test]
    async fn auto_model_with_higgs_default_picks_first_engine() {
        // When model="auto", auto-router returns None, default is higgs
        // with no model_rewrite, it should pick the first loaded engine
        // instead of trying to look up "auto" as a model name.
        let config = config_from_toml(
            r#"
            [[models]]
            path = "test/model-a"
            "#,
        );
        let engine = crate::test_support::StubEngine::named("test/model-a").into_loaded();
        let mut engines = HashMap::new();
        engines.insert("test/model-a".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines).unwrap();
        // No auto-router configured, so "auto" falls through to default
        let result = router.resolve("auto", None).await;
        match result {
            Ok(ResolvedRoute::Higgs {
                model_name,
                routing_method,
                ..
            }) => {
                assert_eq!(model_name, "test/model-a");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            Ok(ResolvedRoute::Remote { .. }) => panic!("expected Higgs route"),
            Err(e) => panic!("should resolve to first engine, got error: {e}"),
        }
    }

    #[test]
    fn auto_router_model_resolved_by_name() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "router"
            "#,
        );
        let engine = crate::test_support::StubEngine::named("router").into_loaded();
        let mut engines = HashMap::new();
        engines.insert("router".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(router.is_ok(), "should resolve auto_router model by name");
    }

    #[test]
    fn auto_router_model_resolved_by_path() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "/Users/someone/models/Arch-Router-1.5B-4bit"
            "#,
        );
        let engine = crate::test_support::StubEngine::named("router").into_loaded();
        let mut engines = HashMap::new();
        engines.insert("router".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(router.is_ok(), "should resolve auto_router model by path");
    }

    #[test]
    fn auto_router_model_resolved_by_path_without_name() {
        // ensure_auto_router_model injects the model entry with name=None,
        // so the engine key is derived from engine.model_name() (the basename).
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"

            [auto_router]
            enabled = true
            model = "/Users/someone/models/Arch-Router-1.5B-4bit"
            "#,
        );
        let engine = crate::test_support::StubEngine::named("Arch-Router-1.5B-4bit").into_loaded();
        let mut engines = HashMap::new();
        engines.insert("Arch-Router-1.5B-4bit".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(
            router.is_ok(),
            "should resolve auto_router model by path when model has no name"
        );
    }

    #[tokio::test]
    async fn auto_model_with_remote_default_rejects_without_rewrite() {
        // model="auto" falling through to a remote default with no model_rewrite
        // should error -- "auto" is not a real model name for upstream providers.
        let router = router_from_toml(production_toml());
        let result = router.resolve("auto", None).await;
        match result {
            Err(e) => assert!(
                e.to_string().contains("cannot forward virtual model name"),
                "got: {e}"
            ),
            Ok(_) => panic!("expected error for auto with remote default"),
        }
    }

    #[tokio::test]
    async fn force_mode_falls_through_without_engine() {
        // force=true, no auto-router engine loaded, named model should
        // fall through to normal resolution (pattern/direct/default).
        let config = config_from_toml(
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [[routes]]
            pattern = "opus"
            provider = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = false
            force = true
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("claude-opus-4-6", None).await.unwrap();
        match result {
            ResolvedRoute::Remote {
                provider_name,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(routing_method, RoutingMethod::Pattern);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn force_mode_unmatched_falls_to_default() {
        // force=true, no auto-router engine, unmatched model falls to default
        let config = config_from_toml(
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = false
            force = true
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("some-unknown-model", None).await.unwrap();
        match result {
            ResolvedRoute::Remote {
                provider_name,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    // -- Lazy loading through the pool --

    use crate::model_pool::{ModelState, PoolLimits};
    use crate::state::test_support::{stub_pool_with, stub_spec};

    #[tokio::test]
    async fn direct_route_loads_an_on_demand_model_on_first_use_and_tracks_the_request() {
        let (pool, loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        let router = Router::new(&HiggsConfig::default(), pool).unwrap();
        assert_eq!(
            router.pool().status("lazy").unwrap().state,
            ModelState::Unloaded
        );

        let route = router.resolve("lazy", None).await.unwrap();

        let ResolvedRoute::Higgs { engine, .. } = route else {
            panic!("expected a local route");
        };
        assert_eq!(engine.model_name(), "lazy");
        assert_eq!(router.pool().status("lazy").unwrap().in_flight, 1);
        assert_eq!(*loader.events.lock().unwrap(), ["load:lazy"]);
        drop(engine);
        assert_eq!(router.pool().status("lazy").unwrap().in_flight, 0);
    }

    #[tokio::test]
    async fn an_alias_resolves_to_its_models_served_name() {
        let (pool, loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            name = "lazy"
            aliases = ["default", "chat"]

            [provider.remote]
            url = "http://127.0.0.1:9"
            "#,
        );
        let router = Router::new(&config, pool).unwrap();

        for requested in ["default", "chat", "lazy"] {
            let route = router.resolve(requested, None).await.unwrap();
            let ResolvedRoute::Higgs {
                model_name,
                routing_method,
                ..
            } = route
            else {
                panic!("expected a local route for {requested}");
            };
            assert_eq!(model_name, "lazy", "{requested}");
            assert_eq!(routing_method, RoutingMethod::Direct);
        }
        assert_eq!(
            loader.events.lock().unwrap().len(),
            1,
            "one load serves all names"
        );
        assert!(matches!(
            router.resolve("unknown", None).await,
            Err(RouteError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn a_route_can_rewrite_to_an_alias() {
        let (pool, _loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            name = "lazy"
            aliases = ["short"]

            [[routes]]
            match = "^gpt-"
            provider = "higgs"
            model = "short"
            "#,
        );
        let router = Router::new(&config, pool).unwrap();

        let route = router.resolve("gpt-4", None).await.unwrap();

        let ResolvedRoute::Higgs { model_name, .. } = route else {
            panic!("expected a local route");
        };
        assert_eq!(model_name, "lazy");
    }

    #[tokio::test]
    async fn pattern_route_to_higgs_loads_the_target_model() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            name = "lazy"

            [[routes]]
            pattern = "^gpt-"
            provider = "higgs"
            model = "lazy"
            "#,
        );
        let (pool, loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        let router = Router::new(&config, pool).unwrap();

        let route = router.resolve("gpt-4", None).await.unwrap();

        let ResolvedRoute::Higgs {
            model_name,
            routing_method,
            ..
        } = route
        else {
            panic!("expected a local route");
        };
        assert_eq!(model_name, "lazy");
        assert_eq!(routing_method, RoutingMethod::Pattern);
        assert_eq!(*loader.events.lock().unwrap(), ["load:lazy"]);
    }

    #[tokio::test]
    async fn load_failure_is_a_model_error_not_a_missing_model() {
        let (pool, loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        *loader.fail_with.lock().unwrap() = Some("out of memory".to_owned());
        let router = Router::new(&HiggsConfig::default(), pool).unwrap();

        let Err(error) = router.resolve("lazy", None).await else {
            panic!("expected the load to fail");
        };

        assert!(matches!(
            error,
            RouteError::Model(ModelError::LoadFailed { .. })
        ));
        assert!(error.to_string().contains("out of memory"));
    }

    #[tokio::test]
    async fn unknown_model_is_not_found_and_loads_nothing() {
        let (pool, loader) = stub_pool_with(vec![stub_spec("lazy")], PoolLimits::default());
        // A provider turns off the single-model fallback, which would serve "other".
        let config = config_from_toml("[provider.remote]\nurl = \"http://127.0.0.1:9\"\n");
        let router = Router::new(&config, pool).unwrap();

        let result = router.resolve("other", None).await;

        assert!(matches!(result, Err(RouteError::NotFound(_))));
        assert!(loader.events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn auto_model_prefers_a_model_that_is_already_loaded() {
        let (pool, _loader) = stub_pool_with(
            vec![stub_spec("first"), stub_spec("second")],
            PoolLimits::default(),
        );
        pool.load("second").await.unwrap();
        let router = Router::new(&HiggsConfig::default(), pool).unwrap();

        let route = router.resolve("auto", None).await.unwrap();

        let ResolvedRoute::Higgs { model_name, .. } = route else {
            panic!("expected a local route");
        };
        assert_eq!(model_name, "second");
    }

    #[test]
    fn auto_router_classifier_must_be_a_configured_model() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            name = "chat"

            [auto_router]
            enabled = true
            model = "router"
            "#,
        );
        let (pool, _loader) = stub_pool_with(vec![stub_spec("chat")], PoolLimits::default());

        let error = Router::new(&config, pool).err().unwrap();

        assert!(error.contains("'router' not found"), "{error}");
    }
}
