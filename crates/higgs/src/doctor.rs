use std::collections::HashSet;
use std::time::Instant;

use crate::config::HiggsConfig;
use crate::model_resolver;

pub struct DoctorResult {
    pub passes: u32,
    pub warnings: u32,
    pub failures: u32,
}

#[allow(clippy::print_stderr)]
fn pass(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[32m[PASS]\x1b[0m {msg}");
    result.passes += 1;
}

#[allow(clippy::print_stderr)]
fn warn(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[33m[WARN]\x1b[0m {msg}");
    result.warnings += 1;
}

#[allow(clippy::print_stderr)]
fn fail(msg: &str, result: &mut DoctorResult) {
    eprintln!("\x1b[31m[FAIL]\x1b[0m {msg}");
    result.failures += 1;
}

#[allow(clippy::print_stderr)]
pub async fn run_doctor(config: &HiggsConfig) -> DoctorResult {
    let mut result = DoctorResult {
        passes: 0,
        warnings: 0,
        failures: 0,
    };

    eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");

    check_config_valid(&mut result);
    check_models(config, &mut result);
    check_duplicate_models(config, &mut result);
    check_providers(config, &mut result).await;
    check_route_consistency(config, &mut result);
    check_default_provider(config, &mut result);
    check_auto_router(config, &mut result);
    check_port_availability(config, &mut result);
    check_orphaned_providers(config, &mut result);

    eprintln!(
        "\n{} passed, {} warnings, {} failures",
        result.passes, result.warnings, result.failures
    );

    result
}

fn check_config_valid(result: &mut DoctorResult) {
    // If we got this far, the config parsed and validated successfully.
    pass("config file is valid", result);
}

fn check_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        match model_resolver::resolve(&model.path) {
            Ok(_) => pass(&format!("model {} resolvable", model.path), result),
            Err(err) => fail(&format!("model {} not found: {err}", model.path), result),
        }
    }
}

fn check_duplicate_models(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    for model in &config.models {
        if !seen.insert(&model.path) {
            duplicates.push(model.path.clone());
        }
    }
    if duplicates.is_empty() {
        if config.models.len() > 1 {
            pass("no duplicate model paths", result);
        }
    } else {
        for dup in &duplicates {
            warn(&format!("duplicate model path: {dup}"), result);
        }
    }
}

async fn check_providers(config: &HiggsConfig, result: &mut DoctorResult) {
    let http_client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            warn(&format!("could not create HTTP client: {err}"), result);
            return;
        }
    };

    for (name, provider) in &config.providers {
        let start = Instant::now();
        match http_client.head(&provider.url).send().await {
            Ok(response) => {
                let elapsed = start.elapsed();
                pass(
                    &format!(
                        "provider {name} reachable ({} {}ms)",
                        response.status(),
                        elapsed.as_millis()
                    ),
                    result,
                );
            }
            Err(err) => {
                warn(&format!("provider {name} unreachable: {err}"), result);
            }
        }
    }
}

fn check_route_consistency(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut all_valid = true;
    for route in &config.routes {
        if route.provider == "higgs" {
            if config.models.is_empty() {
                warn(
                    &format!(
                        "route {:?} targets \"higgs\" but no models are loaded",
                        route
                            .name
                            .as_deref()
                            .or(route.pattern.as_deref())
                            .unwrap_or("(unnamed)")
                    ),
                    result,
                );
                all_valid = false;
            }
        } else if !config.providers.contains_key(&route.provider) {
            fail(
                &format!(
                    "route {:?} references unknown provider \"{}\"",
                    route
                        .name
                        .as_deref()
                        .or(route.pattern.as_deref())
                        .unwrap_or("(unnamed)"),
                    route.provider
                ),
                result,
            );
            all_valid = false;
        }
    }
    if all_valid && !config.routes.is_empty() {
        pass("all route providers exist", result);
    }
}

fn check_default_provider(config: &HiggsConfig, result: &mut DoctorResult) {
    let provider = &config.default.provider;
    if provider == "higgs" {
        if config.models.is_empty() {
            warn(
                "default provider is \"higgs\" but no models are loaded",
                result,
            );
        } else {
            pass(&format!("default provider \"{provider}\" exists"), result);
        }
    } else if config.providers.contains_key(provider) {
        pass(&format!("default provider \"{provider}\" exists"), result);
    } else {
        fail(
            &format!("default provider \"{provider}\" not found in providers"),
            result,
        );
    }
}

fn check_auto_router(config: &HiggsConfig, result: &mut DoctorResult) {
    if !config.auto_router.enabled {
        return;
    }

    let model_name = &config.auto_router.model;
    if model_name.is_empty() {
        fail("auto_router enabled but no model specified", result);
        return;
    }

    let model_known = config.models.iter().any(|m| m.path == *model_name);
    if model_known {
        pass(
            &format!("auto_router model \"{model_name}\" found in models"),
            result,
        );
    } else {
        fail(
            &format!("auto_router model \"{model_name}\" not found in models"),
            result,
        );
    }

    match model_resolver::resolve(model_name) {
        Ok(_) => pass(
            &format!("auto_router model \"{model_name}\" downloaded"),
            result,
        ),
        Err(err) => fail(
            &format!("auto_router model \"{model_name}\" not downloaded: {err}"),
            result,
        ),
    }

    let routes_with_descriptions = config
        .routes
        .iter()
        .filter(|r| r.description.is_some())
        .count();
    if routes_with_descriptions == 0 && !config.routes.is_empty() {
        warn(
            "auto_router enabled but no routes have descriptions",
            result,
        );
    }
}

fn check_port_availability(config: &HiggsConfig, result: &mut DoctorResult) {
    let addr = format!("{}:{}", config.server.host, config.server.port);
    match std::net::TcpListener::bind(&addr) {
        Ok(_) => pass(&format!("port {} available", config.server.port), result),
        Err(err) => warn(
            &format!("port {} unavailable: {err}", config.server.port),
            result,
        ),
    }
}

fn check_orphaned_providers(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut referenced: HashSet<&str> = HashSet::new();

    if config.default.provider != "higgs" {
        referenced.insert(&config.default.provider);
    }

    for route in &config.routes {
        if route.provider != "higgs" {
            referenced.insert(&route.provider);
        }
    }

    for name in config.providers.keys() {
        if !referenced.contains(name.as_str()) {
            warn(
                &format!("provider \"{name}\" defined but not used by any route"),
                result,
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::{
        AutoRouterConfig, DefaultRoute, HiggsConfig, ModelConfig, ProviderConfig, RouteConfig,
        ServerSection,
    };
    use std::collections::HashMap;

    fn empty_result() -> DoctorResult {
        DoctorResult {
            passes: 0,
            warnings: 0,
            failures: 0,
        }
    }

    // -- Helper function counter tests --

    #[test]
    fn test_pass_increments_counter() {
        let mut result = empty_result();
        pass("test", &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_warn_increments_counter() {
        let mut result = empty_result();
        warn("test", &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_fail_increments_counter() {
        let mut result = empty_result();
        fail("test", &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 1);
    }

    // -- Duplicate model detection --

    #[test]
    fn test_no_duplicates_passes() {
        let config = HiggsConfig {
            models: vec![
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    batch: false,
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    batch: false,
                },
            ],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_duplicate_models(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_duplicate_models_warns() {
        let config = HiggsConfig {
            models: vec![
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    batch: false,
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    batch: false,
                },
            ],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_duplicate_models(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    // -- Orphaned provider detection --

    #[test]
    fn test_orphaned_provider_warns() {
        let mut providers = HashMap::new();
        providers.insert(
            "openai".to_owned(),
            ProviderConfig {
                url: "https://api.openai.com".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "higgs".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_orphaned_providers(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_referenced_provider_not_orphaned() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "anthropic".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_orphaned_providers(&config, &mut result);
        assert_eq!(result.warnings, 0);
    }

    // -- Route consistency --

    #[test]
    fn test_route_unknown_provider_fails() {
        let config = HiggsConfig {
            routes: vec![RouteConfig {
                name: Some("test".to_owned()),
                description: None,
                pattern: None,
                provider: "nonexistent".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_route_higgs_no_models_warns() {
        let config = HiggsConfig {
            routes: vec![RouteConfig {
                name: Some("local".to_owned()),
                description: None,
                pattern: None,
                provider: "higgs".to_owned(),
                model: None,
            }],
            models: vec![],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn test_route_valid_provider_passes() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            routes: vec![RouteConfig {
                name: Some("claude".to_owned()),
                description: None,
                pattern: Some("claude-.*".to_owned()),
                provider: "anthropic".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_route_consistency(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 0);
    }

    // -- Default provider --

    #[test]
    fn test_default_provider_exists() {
        let mut providers = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            ProviderConfig {
                url: "https://api.anthropic.com".to_owned(),
                format: crate::config::ApiFormat::Anthropic,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            default: DefaultRoute {
                provider: "anthropic".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn test_default_provider_missing_fails() {
        let config = HiggsConfig {
            default: DefaultRoute {
                provider: "nonexistent".to_owned(),
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_default_higgs_no_models_warns() {
        let config = HiggsConfig {
            default: DefaultRoute {
                provider: "higgs".to_owned(),
            },
            models: vec![],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_default_provider(&config, &mut result);
        assert_eq!(result.warnings, 1);
    }

    // -- Port availability --

    #[test]
    fn test_port_zero_available() {
        let config = HiggsConfig {
            server: ServerSection {
                host: "127.0.0.1".to_owned(),
                port: 0,
                ..ServerSection::default()
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_port_availability(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    // -- Auto router --

    #[test]
    fn test_auto_router_disabled_skips() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: false,
                force: false,
                model: String::new(),
                timeout_ms: 2000,
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        assert_eq!(result.passes, 0);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_auto_router_empty_model_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: String::new(),
                timeout_ms: 2000,
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn test_auto_router_unknown_model_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "nonexistent/model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/other-model".to_owned(),
                batch: false,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Fails twice: not in [[models]] and not downloaded
        assert_eq!(result.failures, 2);
    }

    #[test]
    fn test_auto_router_model_not_downloaded_fails() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "org/router-model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                batch: false,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Model is in [[models]] (pass), but not downloaded (fail)
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 1);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_auto_router_no_descriptions_warns() {
        let config = HiggsConfig {
            auto_router: AutoRouterConfig {
                enabled: true,
                force: false,
                model: "org/router-model".to_owned(),
                timeout_ms: 2000,
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                batch: false,
            }],
            routes: vec![RouteConfig {
                name: Some("test".to_owned()),
                description: None,
                pattern: None,
                provider: "higgs".to_owned(),
                model: None,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Should pass for model found, but warn for no descriptions
        assert_eq!(result.passes, 1);
        assert_eq!(result.warnings, 1);
    }

    // -- Provider reachability --

    #[tokio::test]
    async fn test_unreachable_provider_warns() {
        let mut providers = HashMap::new();
        providers.insert(
            "bad".to_owned(),
            ProviderConfig {
                url: "http://127.0.0.1:1".to_owned(),
                format: crate::config::ApiFormat::OpenAi,
                api_key: None,
                strip_auth: false,
                stub_count_tokens: false,
            },
        );
        let config = HiggsConfig {
            providers,
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_providers(&config, &mut result).await;
        assert_eq!(result.warnings, 1);
        assert_eq!(result.passes, 0);
    }
}
