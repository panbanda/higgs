use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use regex::Regex;
use tracing::warn;

use crate::config::{ApiFormat, HiggsConfig};
use crate::state::Engine;

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
        engine: Arc<Engine>,
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
        api_key: Option<String>,
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
        api_key: Option<String>,
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
/// 2. Pattern matching (first match wins)
/// 3. Direct engine lookup by model name
/// 4. Default provider fallback
pub struct Router {
    local_engines: HashMap<String, Arc<Engine>>,
    compiled_routes: Vec<CompiledRoute>,
    auto_routes: Vec<AutoRouteEntry>,
    auto_candidates: Vec<RouteCandidate>,
    auto_router_engine: Option<Arc<Engine>>,
    auto_router_timeout_ms: u64,
    default_target: RouteTarget,
}

impl Router {
    /// Build a router from the unified config and loaded local engines.
    pub fn from_config(
        config: &HiggsConfig,
        engines: HashMap<String, Arc<Engine>>,
    ) -> Result<Self, String> {
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

        let auto_router_engine = if config.auto_router.enabled {
            if config.auto_router.model.is_empty() {
                return Err("auto_router.enabled is true but model is empty".to_owned());
            }
            if auto_candidates.is_empty() {
                warn!("auto_router is enabled but no routes have descriptions");
            }
            let engine = engines
                .get(&config.auto_router.model)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "auto_router model '{}' not found among loaded models",
                        config.auto_router.model
                    )
                })?;
            Some(engine)
        } else {
            None
        };

        let default_target = build_route_target(&config.default.provider, None, config)?;

        Ok(Self {
            local_engines: engines,
            compiled_routes,
            auto_routes,
            auto_candidates,
            auto_router_engine,
            auto_router_timeout_ms: config.auto_router.timeout_ms,
            default_target,
        })
    }

    /// Resolve a model name to a route.
    ///
    /// Pass `messages` for auto-routing support (only used when `model == "auto"`).
    pub async fn resolve(
        &self,
        model: &str,
        messages: Option<&[serde_json::Value]>,
    ) -> Result<ResolvedRoute, String> {
        if model == "auto" {
            if let Some(resolved) = self.try_auto_route(messages).await {
                return Ok(resolved);
            }
            return self.resolve_target(&self.default_target, model, RoutingMethod::Default);
        }

        // Pattern matching (first match wins)
        for route in &self.compiled_routes {
            if route.pattern.is_match(model) {
                return self.resolve_target(&route.target, model, RoutingMethod::Pattern);
            }
        }

        // Direct engine lookup
        if let Some(engine) = self.local_engines.get(model) {
            return Ok(ResolvedRoute::Higgs {
                engine: Arc::clone(engine),
                model_name: model.to_owned(),
                routing_method: RoutingMethod::Direct,
            });
        }

        // Default fallback
        self.resolve_target(&self.default_target, model, RoutingMethod::Default)
    }

    /// Returns all loaded local engines.
    pub const fn local_engines(&self) -> &HashMap<String, Arc<Engine>> {
        &self.local_engines
    }

    // -- Private helpers ---------------------------------------------------

    async fn try_auto_route(
        &self,
        messages: Option<&[serde_json::Value]>,
    ) -> Option<ResolvedRoute> {
        let auto_engine = self.auto_router_engine.as_ref()?;
        let msg_slice = messages?;
        if self.auto_candidates.is_empty() || msg_slice.is_empty() {
            return None;
        }

        let engine_clone = Arc::clone(auto_engine);
        let candidates = self.auto_candidates.clone();
        let messages_owned = msg_slice.to_vec();

        let timeout = std::time::Duration::from_millis(self.auto_router_timeout_ms);
        let timeout_result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                crate::auto_router::classify_local(&engine_clone, &candidates, &messages_owned)
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
            .ok()
    }

    fn resolve_target(
        &self,
        target: &RouteTarget,
        model: &str,
        method: RoutingMethod,
    ) -> Result<ResolvedRoute, String> {
        match target {
            RouteTarget::Higgs { model_rewrite } => {
                let lookup_name = model_rewrite.as_deref().unwrap_or(model);
                let engine = self.local_engines.get(lookup_name).ok_or_else(|| {
                    format!("model '{lookup_name}' not found among loaded local models")
                })?;
                Ok(ResolvedRoute::Higgs {
                    engine: Arc::clone(engine),
                    model_name: lookup_name.to_owned(),
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
            } => Ok(ResolvedRoute::Remote {
                provider_name: provider_name.clone(),
                provider_url: provider_url.clone(),
                provider_format: *provider_format,
                model_rewrite: model_rewrite.clone(),
                strip_auth: *strip_auth,
                api_key: api_key.clone(),
                stub_count_tokens: *stub_count_tokens,
                routing_method: method,
            }),
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

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::shadow_unrelated
)]
mod tests {
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
                assert_eq!(api_key.as_deref(), Some("ollama"));
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
                e.contains("not found among loaded local models"),
                "got: {e}"
            ),
            Ok(_) => panic!("expected error for missing local model"),
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
    fn local_engines_accessor_returns_engines() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert!(router.local_engines().is_empty());
    }
}
