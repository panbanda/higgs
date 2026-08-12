use std::collections::HashSet;
use std::time::Instant;

use higgs_engine::mlx_tuning::resolve_effective_mlx_profile;

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
pub async fn run_doctor(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
) -> DoctorResult {
    let mut result = DoctorResult {
        passes: 0,
        warnings: 0,
        failures: 0,
    };

    eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");

    check_config_valid(&mut result);
    check_config_file_permissions(config, config_path, &mut result);
    check_server_section(config, &mut result);
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

fn check_server_section(config: &crate::config::HiggsConfig, result: &mut DoctorResult) {
    let server = &config.server;

    if server.max_tokens == 0 {
        fail(
            "server.max_tokens=0 produces empty completions; set a positive value",
            result,
        );
    } else {
        pass(&format!("server.max_tokens={}", server.max_tokens), result);
    }

    if !server.timeout.is_finite() || server.timeout <= 0.0 {
        fail(
            &format!(
                "server.timeout={} must be a positive finite number of seconds",
                server.timeout
            ),
            result,
        );
    } else if server.timeout > 600.0 {
        warn(
            &format!(
                "server.timeout={}s is unusually high (>10 min); check intent",
                server.timeout
            ),
            result,
        );
    } else {
        pass(&format!("server.timeout={}s", server.timeout), result);
    }

    if server.max_body_size == 0 {
        fail("server.max_body_size=0 rejects all request bodies", result);
    } else if server.max_body_size > 1 << 30 {
        warn(
            &format!(
                "server.max_body_size={} bytes (>1 GiB); check intent",
                server.max_body_size
            ),
            result,
        );
    } else {
        pass(
            &format!("server.max_body_size={} bytes", server.max_body_size),
            result,
        );
    }

    if server.host.parse::<std::net::IpAddr>().is_ok() || server.host == "localhost" {
        pass(&format!("server.host=\"{}\"", server.host), result);
    } else {
        warn(
            &format!(
                "server.host=\"{}\" is not an IP address or \"localhost\"; bind may fail at runtime",
                server.host
            ),
            result,
        );
    }

    if server.api_key.is_some() {
        pass("server.api_key set; API key auth enabled", result);
    } else {
        pass(
            "server.api_key unset; no auth enforced (server is open)",
            result,
        );
    }

    let non_loopback = matches!(
        server.host.parse::<std::net::IpAddr>(),
        Ok(ip) if !ip.is_loopback()
    );
    if non_loopback && server.api_key.is_none() {
        warn(
            &format!(
                "server.host=\"{}\" is reachable from the network but server.api_key is unset; \
                 anyone on the network can use this server",
                server.host
            ),
            result,
        );
    }

    check_cors_origins(server, non_loopback, result);

    if server.rate_limit == 0 {
        pass("server.rate_limit=0 (disabled)", result);
    } else {
        pass(
            &format!("server.rate_limit={} req/min/client", server.rate_limit),
            result,
        );
    }
}

fn check_cors_origins(
    server: &crate::config::ServerSection,
    non_loopback: bool,
    result: &mut DoctorResult,
) {
    match &server.cors_origins {
        None => pass("server.cors_origins unset; no CORS headers sent", result),
        Some(origins) if origins.iter().any(|o| o == "*") => {
            if non_loopback {
                warn(
                    "server.cors_origins allows any origin (\"*\") on a network-reachable host; \
                     consider an explicit origin list",
                    result,
                );
            } else {
                pass("server.cors_origins=[\"*\"] (permissive)", result);
            }
        }
        Some(origins) => {
            let mut all_valid = true;
            for origin in origins {
                let parses = origin.parse::<http::HeaderValue>().is_ok();
                if !parses || !(origin.starts_with("http://") || origin.starts_with("https://")) {
                    fail(
                        &format!(
                            "server.cors_origins entry \"{origin}\" is not a valid origin \
                             (expected e.g. \"https://example.com\")"
                        ),
                        result,
                    );
                    all_valid = false;
                }
            }
            if all_valid {
                pass(
                    &format!("server.cors_origins lists {} origin(s)", origins.len()),
                    result,
                );
            }
        }
    }
}

/// Warn when the config file holding API keys is readable by other users.
#[cfg(unix)]
fn check_config_file_permissions(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
    result: &mut DoctorResult,
) {
    use std::os::unix::fs::PermissionsExt as _;

    let Some(path) = config_path else { return };
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };
    let mode = metadata.permissions().mode() & 0o777;
    let has_secrets =
        config.server.api_key.is_some() || config.providers.values().any(|p| p.api_key.is_some());
    if mode.trailing_zeros() >= 6 {
        pass(
            &format!("config file permissions are owner-only ({mode:03o})"),
            result,
        );
    } else if has_secrets {
        warn(
            &format!(
                "config file {} is group/world-accessible (mode {mode:03o}) and contains API \
                 keys; run: chmod 600 {}",
                path.display(),
                path.display()
            ),
            result,
        );
    } else {
        pass(
            &format!("config file permissions {mode:03o} (no API keys present)"),
            result,
        );
    }
}

#[cfg(not(unix))]
fn check_config_file_permissions(
    _config: &HiggsConfig,
    _config_path: Option<&std::path::Path>,
    _result: &mut DoctorResult,
) {
}

fn model_label(model: &crate::config::ModelConfig) -> String {
    model.name.as_ref().map_or_else(
        || model.path.clone(),
        |name| format!("\"{name}\" ({})", model.path),
    )
}

fn check_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        let label = model_label(model);
        match model.kv_cache_config().validate() {
            Ok(()) => {}
            Err(err) => {
                fail(
                    &format!("model {label} has invalid KV cache config: {err}"),
                    result,
                );
                continue;
            }
        }
        if model.batch && model.kv_cache_config().is_turboquant() {
            fail(
                &format!(
                    "model {label} enables unsupported combination: TurboQuant with batch=true"
                ),
                result,
            );
            continue;
        }
        match model_resolver::resolve(&model.path) {
            Ok(resolved) => {
                if model.batch {
                    match crate::config::resolved_model_supports_batch(&resolved) {
                        Ok(true) => {}
                        Ok(false) => {
                            fail(
                                &format!(
                                    "model {label} enables unsupported batch=true; only transformer models (llama, mistral, qwen2, qwen3) support true batched decode"
                                ),
                                result,
                            );
                            continue;
                        }
                        Err(err) => {
                            fail(
                                &format!("model {label} batch validation failed: {err}"),
                                result,
                            );
                            continue;
                        }
                    }
                }
                let requested_profile = model.requested_mlx_profile(&config.local);
                let profile_msg = if model.batch {
                    "batch=true; batched decode supported".to_owned()
                } else {
                    let effective_profile =
                        resolve_effective_mlx_profile(&resolved, requested_profile);
                    if effective_profile.as_str() == requested_profile.as_str() {
                        format!("mlx_profile={}", effective_profile.as_str())
                    } else {
                        format!(
                            "mlx_profile={} (requested {})",
                            effective_profile.as_str(),
                            requested_profile.as_str()
                        )
                    }
                };
                pass(&format!("model {label} resolvable ({profile_msg})"), result);
            }
            Err(err) => fail(&format!("model {label} not found: {err}"), result),
        }
    }
}

fn check_duplicate_models(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut seen_paths = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut duplicates = Vec::new();
    for model in &config.models {
        if !seen_paths.insert(&model.path) {
            duplicates.push(format!("path: {}", model.path));
        }
        if let Some(ref name) = model.name {
            if !seen_names.insert(name) {
                duplicates.push(format!("name: {name}"));
            }
        }
    }
    if duplicates.is_empty() {
        if config.models.len() > 1 {
            pass("no duplicate model paths or names", result);
        }
    } else {
        for dup in &duplicates {
            warn(&format!("duplicate model {dup}"), result);
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

    let model_ref = &config.auto_router.model;
    if model_ref.is_empty() {
        fail("auto_router enabled but no model specified", result);
        return;
    }

    // Match by name or path
    let matched = config
        .models
        .iter()
        .find(|m| m.path == *model_ref || m.name.as_deref() == Some(model_ref));

    if let Some(matched_model) = matched {
        let label = model_label(matched_model);
        pass(
            &format!("auto_router model {label} found in models"),
            result,
        );
        match model_resolver::resolve(&matched_model.path) {
            Ok(_) => pass(&format!("auto_router model {label} downloaded"), result),
            Err(err) => fail(
                &format!("auto_router model {label} not downloaded: {err}"),
                result,
            ),
        }
    } else {
        fail(
            &format!("auto_router model \"{model_ref}\" not found in models"),
            result,
        );
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
    let host = &config.server.host;
    let port = config.server.port;
    let addr = host.parse::<std::net::IpAddr>().map_or_else(
        |_| format!("{host}:{port}"),
        |ip| std::net::SocketAddr::new(ip, port).to_string(),
    );
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

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
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
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
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
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
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

    // -- Server section validation --

    fn server_with(modify: impl FnOnce(&mut ServerSection)) -> HiggsConfig {
        let mut server = ServerSection::default();
        modify(&mut server);
        HiggsConfig {
            server,
            ..HiggsConfig::default()
        }
    }

    #[test]
    fn test_server_default_passes() {
        let config = HiggsConfig::default();
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_max_tokens_zero_fails() {
        let config = server_with(|s| s.max_tokens = 0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_zero_fails() {
        let config = server_with(|s| s.timeout = 0.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_negative_fails() {
        let config = server_with(|s| s.timeout = -1.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_nan_fails() {
        let config = server_with(|s| s.timeout = f64::NAN);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_infinite_fails() {
        let config = server_with(|s| s.timeout = f64::INFINITY);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_timeout_unusually_high_warns() {
        let config = server_with(|s| s.timeout = 3600.0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_max_body_size_zero_fails() {
        let config = server_with(|s| s.max_body_size = 0);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_max_body_size_huge_warns() {
        let config = server_with(|s| s.max_body_size = (1 << 30) + 1);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_host_localhost_passes() {
        let config = server_with(|s| s.host = "localhost".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_ipv4_passes() {
        let config = server_with(|s| s.host = "127.0.0.1".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_ipv6_passes() {
        let config = server_with(|s| s.host = "::1".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_host_garbage_warns() {
        let config = server_with(|s| s.host = "not a valid host!!".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
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

    #[test]
    fn test_port_available_ipv6_localhost() {
        let config = HiggsConfig {
            server: ServerSection {
                host: "::1".to_owned(),
                port: 0,
                ..ServerSection::default()
            },
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_port_availability(&config, &mut result);
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn test_api_key_set_passes() {
        let config = server_with(|s| s.api_key = Some("secret".to_owned()));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_rate_limit_nonzero_passes() {
        let config = server_with(|s| s.rate_limit = 120);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
    }

    #[test]
    fn test_non_loopback_host_without_api_key_warns() {
        let config = server_with(|s| s.host = "0.0.0.0".to_owned());
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_non_loopback_host_with_api_key_no_warning() {
        let config = server_with(|s| {
            s.host = "0.0.0.0".to_owned();
            s.api_key = Some("sk-test".to_owned());
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_wildcard_on_loopback_passes() {
        let config = server_with(|s| s.cors_origins = Some(vec!["*".to_owned()]));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_wildcard_on_network_host_warns() {
        let config = server_with(|s| {
            s.host = "0.0.0.0".to_owned();
            s.api_key = Some("sk-test".to_owned());
            s.cors_origins = Some(vec!["*".to_owned()]);
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_valid_origin_list_passes() {
        let config = server_with(|s| {
            s.cors_origins = Some(vec![
                "https://example.com".to_owned(),
                "http://localhost:3000".to_owned(),
            ]);
        });
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert_eq!(result.warnings, 0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_cors_invalid_origin_fails() {
        let config = server_with(|s| s.cors_origins = Some(vec!["not a url".to_owned()]));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
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
                name: None,
                mlx_profile: None,
                batch: false,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_auto_router(&config, &mut result);
        // Fails once: not in [[models]] (download check skipped)
        assert_eq!(result.failures, 1);
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
                name: None,
                mlx_profile: None,
                batch: false,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
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
                name: None,
                mlx_profile: None,
                batch: false,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
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
