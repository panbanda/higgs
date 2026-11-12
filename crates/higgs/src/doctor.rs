use std::collections::HashSet;
use std::time::Instant;

use higgs_engine::{mlx_tuning::resolve_effective_mlx_profile, model_loader};

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
    check_local_section(config, &mut result);
    check_models(config, &mut result);
    check_model_lifecycle(
        config,
        crate::routes::system::physical_memory_bytes(),
        &|model| model_resolver::estimated_footprint_bytes(&model.path),
        &mut result,
    );
    check_duplicate_models(config, &mut result);
    check_providers(config, &mut result).await;
    check_route_consistency(config, &mut result);
    check_default_provider(config, &mut result);
    check_auto_router(config, &mut result);
    check_port_availability(config, &mut result);
    check_orphaned_providers(config, &mut result);
    check_tool_integrations(config, &mut result);

    eprintln!(
        "\n{} passed, {} warnings, {} failures",
        result.passes, result.warnings, result.failures
    );

    result
}

fn check_tool_integrations(config: &HiggsConfig, result: &mut DoctorResult) {
    use crate::mcp::doctor::{Level, tool_integration_findings};

    for finding in tool_integration_findings(config) {
        match finding.level {
            Level::Pass => pass(&finding.message, result),
            Level::Warn => warn(&finding.message, result),
            Level::Fail => fail(&finding.message, result),
        }
    }
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

fn check_prefill_yield_tokens(
    label: &str,
    prefill_yield_tokens: Option<u32>,
    result: &mut DoctorResult,
) -> bool {
    let Some(tokens) = prefill_yield_tokens else {
        return true;
    };
    if tokens != 0 && tokens < 128 {
        fail(
            &format!("model {label} prefill_yield_tokens={tokens} must be 0 or at least 128"),
            result,
        );
        return false;
    }
    if tokens != 0 && tokens < 512 {
        warn(
            &format!("model {label} prefill_yield_tokens={tokens} is below the recommended 512"),
            result,
        );
    }
    true
}

/// Reject a zero thinking budget and warn when the budget alone would use up
/// the server's `max_tokens`, leaving no room for the answer.
///
/// Disabling thinking is the job of reasoning effort `"none"`, not a zero budget.
fn check_thinking_budget(
    scope: &str,
    thinking_budget: Option<u32>,
    max_tokens: u32,
    result: &mut DoctorResult,
) -> bool {
    let Some(budget) = thinking_budget else {
        return true;
    };
    if budget == 0 {
        fail(
            &format!(
                "{scope} thinking_budget=0 is invalid; it must be at least 1 (disable thinking with reasoning effort \"none\")"
            ),
            result,
        );
        return false;
    }
    if budget >= max_tokens {
        warn(
            &format!(
                "{scope} thinking_budget={budget} is not below server max_tokens={max_tokens}; reasoning can consume the whole response"
            ),
            result,
        );
    }
    true
}

fn check_local_section(config: &HiggsConfig, result: &mut DoctorResult) {
    check_thinking_budget(
        "[local]",
        config.local.thinking_budget,
        config.server.max_tokens,
        result,
    );
}

/// Warn (not fail) when the *resolved* MLA decision is enabled for an adapter
/// that does not advertise MLA latent-cache support. The flag is a no-op for
/// those adapters at runtime, so this is advisory rather than a hard failure.
///
/// Uses [`higgs_models::cache::resolve_mla_latent_cache`] rather than the raw
/// `model.mla_latent_cache` field, so this matches runtime behavior: e.g.
/// `HIGGS_MLA_LATENT_CACHE=1` with `mla_latent_cache` unset in config still
/// warns (the flag is effectively on), and `HIGGS_MLA_LATENT_CACHE=0` with
/// `mla_latent_cache=true` in config does not warn (the flag is effectively
/// off).
#[cfg(test)]
fn check_mla_latent_cache_architecture(
    label: &str,
    model: &crate::config::ModelConfig,
    resolved: &std::path::Path,
    result: &mut DoctorResult,
) {
    if !higgs_models::cache::resolve_mla_latent_cache(model.kv_cache_config().mla_latent) {
        return;
    }
    match model_loader::ModelConfig::from_dir(resolved) {
        Ok(inspected) => check_mla_latent_cache_adapter(label, &inspected, result),
        Err(err) => {
            warn(
                &format!(
                    "model {label} enables mla_latent_cache=true but its architecture could not be determined: {err}"
                ),
                result,
            );
        }
    }
}

fn check_mla_latent_cache_adapter(
    label: &str,
    inspected: &model_loader::ModelConfig,
    result: &mut DoctorResult,
) {
    if inspected.capabilities.mla_latent_cache {
        pass(
            &format!(
                "model {label} mla_latent_cache=true (adapter {})",
                inspected.adapter_id
            ),
            result,
        );
    } else {
        warn(
            &format!(
                "model {label} enables mla_latent_cache=true but adapter '{}' does not support MLA latent cache; the flag is a no-op at runtime",
                inspected.adapter_id
            ),
            result,
        );
    }
}

#[allow(clippy::too_many_lines)]
fn check_models(config: &HiggsConfig, result: &mut DoctorResult) {
    for model in &config.models {
        let label = model_label(model);
        if let Err(error) = model.validate_disk_prefix_store() {
            fail(&format!("model {label} disk prefix store: {error}"), result);
        } else if model.kv_disk_dir.is_some() {
            pass(
                &format!("model {label} disk prefix store is writable"),
                result,
            );
        }
        if !check_prefill_yield_tokens(&label, model.prefill_yield_tokens, result) {
            continue;
        }
        if !check_thinking_budget(
            &format!("model {label}"),
            model.thinking_budget,
            config.server.max_tokens,
            result,
        ) {
            continue;
        }
        if model.batch && model.thinking_budget.is_some() {
            warn(
                &format!("model {label} thinking_budget has no effect with batch=true"),
                result,
            );
        }
        let kv_cache_config = model.kv_cache_config();
        match kv_cache_config.validate() {
            Ok(()) => {
                // `validate()` only rejects the MLA/TurboQuant combination
                // using the *resolved* decision (env-aware), so a
                // config-declared conflict that HIGGS_MLA_LATENT_CACHE
                // overrides away passes silently there. Surface that as an
                // advisory warning rather than staying silent.
                if model.mla_latent_cache == Some(true)
                    && kv_cache_config.is_turboquant()
                    && !higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent)
                {
                    warn(
                        &format!(
                            "model {label} sets mla_latent_cache=true with kv_cache=turboquant, but HIGGS_MLA_LATENT_CACHE overrides MLA off; the conflict is masked at runtime"
                        ),
                        result,
                    );
                }
            }
            Err(err) => {
                fail(
                    &format!("model {label} has invalid KV cache config: {err}"),
                    result,
                );
                continue;
            }
        }
        if model.batch && kv_cache_config.is_turboquant() {
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
                let inspected = match model_loader::ModelConfig::from_dir(&resolved) {
                    Ok(inspected) => inspected,
                    Err(err) => {
                        fail(
                            &format!("model {label} architecture validation failed: {err}"),
                            result,
                        );
                        continue;
                    }
                };
                if model.batch && !inspected.capabilities.batch_engine {
                    fail(
                        &format!(
                            "model {label} enables unsupported batch=true; adapter '{}' does not support true batched decode",
                            inspected.adapter_id
                        ),
                        result,
                    );
                    continue;
                }
                if higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent) {
                    check_mla_latent_cache_adapter(&label, &inspected, result);
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
                let version = inspected
                    .version
                    .map_or_else(|| "unknown".to_owned(), |version| version.to_string());
                pass(
                    &format!(
                        "model {label} resolvable (adapter={}, family={}, version={version}; {profile_msg})",
                        inspected.adapter_id, inspected.family
                    ),
                    result,
                );
            }
            Err(err) => fail(&format!("model {label} not found: {err}"), result),
        }
    }
}

fn is_auto_router_model(config: &HiggsConfig, model: &crate::config::ModelConfig) -> bool {
    config.auto_router.enabled && model.name.as_deref() == Some(config.auto_router.model.as_str())
}

struct LifecycleTotals {
    /// Models that load when the server starts.
    startup_models: usize,
    startup_bytes: u64,
    /// Models that are never evicted: pinned ones and the auto-router classifier.
    pinned_bytes: u64,
}

/// Warn about per-model lifecycle conflicts and total what loads at startup or stays resident.
fn tally_lifecycle(
    config: &HiggsConfig,
    footprint: &dyn Fn(&crate::config::ModelConfig) -> Option<u64>,
    result: &mut DoctorResult,
) -> LifecycleTotals {
    use crate::config::LoadMode;

    let local = &config.local;
    let mut totals = LifecycleTotals {
        startup_models: 0,
        startup_bytes: 0,
        pinned_bytes: 0,
    };
    for model in &config.models {
        let label = model_label(model);
        let is_classifier = is_auto_router_model(config, model);
        if is_classifier && model.effective_load(local) == LoadMode::OnDemand {
            warn(
                &format!(
                    "model {label} is the auto_router classifier; it always loads at startup and is never evicted, so load=\"on_demand\" is ignored"
                ),
                result,
            );
        }
        if model.pinned && model.idle_ttl_seconds.is_some_and(|seconds| seconds > 0) {
            warn(
                &format!("model {label} is pinned, so its idle_ttl_seconds never applies"),
                result,
            );
        }
        let bytes = footprint(model).unwrap_or(0);
        if is_classifier || model.effective_load(local) == LoadMode::Startup {
            totals.startup_models += 1;
            totals.startup_bytes = totals.startup_bytes.saturating_add(bytes);
        }
        if model.pinned || is_classifier {
            totals.pinned_bytes = totals.pinned_bytes.saturating_add(bytes);
        }
    }
    totals
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib_label(bytes: u64) -> String {
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    let gib = bytes as f64 / GIB;
    format!("{gib:.1} GiB")
}

/// Validate on-demand loading, eviction and memory-ceiling settings.
///
/// `footprint` returns a model's on-disk weight size, or `None` when it is not downloaded.
fn check_model_lifecycle(
    config: &HiggsConfig,
    physical_memory: Option<u64>,
    footprint: &dyn Fn(&crate::config::ModelConfig) -> Option<u64>,
    result: &mut DoctorResult,
) {
    use crate::config::LoadMode;

    let local = &config.local;
    if let Err(error) = local.validate_lifecycle() {
        fail(&error, result);
        return;
    }

    let ceiling = local.memory_ceiling_bytes(physical_memory);
    if let Some(ceiling_bytes) = ceiling {
        pass(
            &format!(
                "memory ceiling for loaded models is {}",
                gib_label(ceiling_bytes)
            ),
            result,
        );
    } else if local.memory_limit_gb.is_none()
        && local.memory_guard != crate::config::MemoryGuard::Off
    {
        warn(
            "local.memory_guard is set but total memory could not be read; no ceiling will be enforced",
            result,
        );
    }

    let LifecycleTotals {
        startup_models,
        startup_bytes,
        pinned_bytes,
    } = tally_lifecycle(config, footprint, result);

    if let Some(ceiling_bytes) = ceiling
        && pinned_bytes > ceiling_bytes
    {
        warn(
            &format!(
                "pinned models alone need about {} on disk, over the {} memory ceiling; loading them will be refused",
                gib_label(pinned_bytes),
                gib_label(ceiling_bytes)
            ),
            result,
        );
    } else if let Some(ceiling_bytes) = ceiling
        && startup_bytes > ceiling_bytes
    {
        warn(
            &format!(
                "models loaded at startup need about {} on disk, over the {} memory ceiling; some will be evicted as others load",
                gib_label(startup_bytes),
                gib_label(ceiling_bytes)
            ),
            result,
        );
    }

    if let Some(cap) = local.max_loaded_models
        && startup_models > cap
    {
        warn(
            &format!(
                "{startup_models} models load at startup but local.max_loaded_models={cap}; the least recently used will be evicted during startup"
            ),
            result,
        );
    }

    let on_demand_models = config
        .models
        .iter()
        .filter(|model| {
            !is_auto_router_model(config, model)
                && model.effective_load(local) == LoadMode::OnDemand
        })
        .count();
    if on_demand_models > 0 && config.models.len() == 1 {
        warn(
            "load=\"on_demand\" with a single model only delays the first request; use \"startup\" unless you rely on idle_ttl_seconds",
            result,
        );
    }

    let lifecycle_configured = ceiling.is_some()
        || local.max_loaded_models.is_some()
        || local.idle_ttl_seconds.is_some()
        || on_demand_models > 0
        || config
            .models
            .iter()
            .any(|model| model.pinned || model.idle_ttl_seconds.is_some());
    if lifecycle_configured {
        pass("model lifecycle settings are valid", result);
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

    #[test]
    fn thinking_budget_unset_is_accepted() {
        let mut result = empty_result();
        assert!(check_thinking_budget("model x", None, 4096, &mut result));
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn thinking_budget_zero_is_rejected() {
        let mut result = empty_result();
        assert!(!check_thinking_budget(
            "model x",
            Some(0),
            4096,
            &mut result
        ));
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn thinking_budget_one_is_accepted() {
        let mut result = empty_result();
        assert!(check_thinking_budget("model x", Some(1), 4096, &mut result));
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn thinking_budget_not_below_max_tokens_warns() {
        let mut result = empty_result();
        assert!(check_thinking_budget(
            "model x",
            Some(4096),
            4096,
            &mut result
        ));
        assert_eq!((result.warnings, result.failures), (1, 0));
    }

    #[test]
    fn local_section_rejects_zero_thinking_budget() {
        let mut config = HiggsConfig::default();
        config.local.thinking_budget = Some(0);
        let mut result = empty_result();
        check_local_section(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn local_section_accepts_valid_thinking_budget() {
        let mut config = HiggsConfig::default();
        config.local.thinking_budget = Some(512);
        let mut result = empty_result();
        check_local_section(&config, &mut result);
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn prefill_yield_tokens_rejects_small_nonzero_values() {
        let mut result = empty_result();
        assert!(!check_prefill_yield_tokens("test", Some(127), &mut result));
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn prefill_yield_tokens_warns_below_recommended_quantum() {
        let mut result = empty_result();
        assert!(check_prefill_yield_tokens("test", Some(128), &mut result));
        assert_eq!(result.warnings, 1);
    }

    #[test]
    fn prefill_yield_tokens_accepts_disabled_quantum() {
        let mut result = empty_result();
        assert!(check_prefill_yield_tokens("test", Some(0), &mut result));
        assert_eq!(result.failures, 0);
        assert_eq!(result.warnings, 0);
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
                    prefill_yield_tokens: None,
                    thinking_budget: None,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                    kv_disk_dir: None,
                    kv_disk_space_mb: 4096,
                    mla_latent_cache: None,
                    load: None,
                    idle_ttl_seconds: None,
                    pinned: false,
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    prefill_yield_tokens: None,
                    thinking_budget: None,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                    kv_disk_dir: None,
                    kv_disk_space_mb: 4096,
                    mla_latent_cache: None,
                    load: None,
                    idle_ttl_seconds: None,
                    pinned: false,
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
                    prefill_yield_tokens: None,
                    thinking_budget: None,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                    kv_disk_dir: None,
                    kv_disk_space_mb: 4096,
                    mla_latent_cache: None,
                    load: None,
                    idle_ttl_seconds: None,
                    pinned: false,
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    name: None,
                    mlx_profile: None,
                    batch: false,
                    prefill_yield_tokens: None,
                    thinking_budget: None,
                    kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                    kv_bits: 3,
                    kv_seed: 0,
                    kv_key_bits: None,
                    kv_value_bits: None,
                    kv_norm_correction: true,
                    kv_adaptive_dense_layers: 0,
                    kv_disk_dir: None,
                    kv_disk_space_mb: 4096,
                    mla_latent_cache: None,
                    load: None,
                    idle_ttl_seconds: None,
                    pinned: false,
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
                prefill_yield_tokens: None,
                thinking_budget: None,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
                kv_disk_dir: None,
                kv_disk_space_mb: 4096,
                mla_latent_cache: None,
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
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
                prefill_yield_tokens: None,
                thinking_budget: None,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
                kv_disk_dir: None,
                kv_disk_space_mb: 4096,
                mla_latent_cache: None,
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
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
                prefill_yield_tokens: None,
                thinking_budget: None,
                kv_cache: higgs_models::turboquant::KvCacheMode::Off,
                kv_bits: 3,
                kv_seed: 0,
                kv_key_bits: None,
                kv_value_bits: None,
                kv_norm_correction: true,
                kv_adaptive_dense_layers: 0,
                kv_disk_dir: None,
                kv_disk_space_mb: 4096,
                mla_latent_cache: None,
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
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

    // -- mla_latent_cache --

    fn model_with_path(path: String) -> ModelConfig {
        ModelConfig {
            path,
            name: None,
            mlx_profile: None,
            batch: false,
            prefill_yield_tokens: None,
            thinking_budget: None,
            kv_cache: higgs_models::turboquant::KvCacheMode::Off,
            kv_bits: 3,
            kv_seed: 0,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_norm_correction: true,
            kv_adaptive_dense_layers: 0,
            kv_disk_dir: None,
            kv_disk_space_mb: 4096,
            mla_latent_cache: None,
            load: None,
            idle_ttl_seconds: None,
            pinned: false,
        }
    }

    fn write_model_config_json(dir: &std::path::Path, model_type: &str) {
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
    }

    /// Run `f` with `HIGGS_MLA_LATENT_CACHE` set to `env_value` (or unset,
    /// for `None`), restoring the prior value afterward. Serialized via
    /// `crate::test_env_lock()` since this mutates process-global state;
    /// combined with `--test-threads=1` (the repo's mandated test-runner
    /// flag for this crate) there is no interleaving risk, but the lock
    /// keeps the guarantee explicit and independent of that flag.
    #[allow(unsafe_code)]
    fn with_mla_env<R>(env_value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var("HIGGS_MLA_LATENT_CACHE").ok();
        match env_value {
            Some(v) => unsafe { std::env::set_var("HIGGS_MLA_LATENT_CACHE", v) },
            None => unsafe { std::env::remove_var("HIGGS_MLA_LATENT_CACHE") },
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        match previous.as_deref() {
            Some(v) => unsafe { std::env::set_var("HIGGS_MLA_LATENT_CACHE", v) },
            None => unsafe { std::env::remove_var("HIGGS_MLA_LATENT_CACHE") },
        }
        result.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    }

    #[test]
    fn test_mla_latent_cache_turboquant_conflict_fails_in_check_models() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.mla_latent_cache = Some(true);
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 1);
            assert_eq!(result.warnings, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_env_off_masks_turboquant_conflict_as_warning() {
        with_mla_env(Some("0"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.mla_latent_cache = Some(true);
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(
                result.failures, 0,
                "HIGGS_MLA_LATENT_CACHE=0 should resolve the conflict away, not fail"
            );
            assert_eq!(
                result.warnings, 1,
                "the masked conflict should still surface as a warning"
            );
        });
    }

    #[test]
    fn test_mla_latent_cache_env_on_triggers_turboquant_conflict() {
        with_mla_env(Some("1"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let mut model = model_with_path(dir.path().to_str().unwrap().to_owned());
            model.kv_cache = higgs_models::turboquant::KvCacheMode::Turboquant;
            // mla_latent_cache left unset in config -- the env var alone
            // must be enough to trigger the conflict.
            let config = HiggsConfig {
                models: vec![model],
                ..HiggsConfig::default()
            };
            let mut result = empty_result();
            check_models(&config, &mut result);
            assert_eq!(result.failures, 1);
            assert_eq!(result.warnings, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_passes_for_deepseek_v2() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "deepseek_v2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 1);
            assert_eq!(result.warnings, 0);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_warns_for_non_deepseek() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 1);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_env_on_warns_for_non_deepseek() {
        with_mla_env(Some("1"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            // config leaves mla_latent_cache unset -- the env override alone
            // must be enough to trigger the architecture warning.
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 1);
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_env_off_suppresses_warning() {
        with_mla_env(Some("0"), || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = ModelConfig {
                mla_latent_cache: Some(true),
                load: None,
                idle_ttl_seconds: None,
                pinned: false,
                ..model_with_path(dir.path().to_str().unwrap().to_owned())
            };
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(
                result.warnings, 0,
                "env forcing MLA off should suppress the architecture warning"
            );
            assert_eq!(result.failures, 0);
        });
    }

    #[test]
    fn test_mla_latent_cache_architecture_noop_when_unset() {
        with_mla_env(None, || {
            let dir = tempfile::tempdir().unwrap();
            write_model_config_json(dir.path(), "qwen2");
            let model = model_with_path(dir.path().to_str().unwrap().to_owned());
            let mut result = empty_result();
            check_mla_latent_cache_architecture("test-model", &model, dir.path(), &mut result);
            assert_eq!(result.passes, 0);
            assert_eq!(result.warnings, 0);
            assert_eq!(result.failures, 0);
        });
    }

    // -- Model lifecycle --

    fn lifecycle_config(
        local: impl FnOnce(&mut crate::config::LocalConfig),
        models: Vec<ModelConfig>,
    ) -> HiggsConfig {
        let mut config = HiggsConfig {
            models,
            ..HiggsConfig::default()
        };
        local(&mut config.local);
        config
    }

    fn named_model(name: &str, modify: impl FnOnce(&mut ModelConfig)) -> ModelConfig {
        let mut model = model_with_path(format!("/models/{name}"));
        model.name = Some(name.to_owned());
        modify(&mut model);
        model
    }

    const GIB_BYTES: u64 = 1 << 30;

    fn run_lifecycle_check(
        config: &HiggsConfig,
        physical_memory: Option<u64>,
        sizes_gib: &[(&str, u64)],
    ) -> DoctorResult {
        let mut result = empty_result();
        let footprint = |model: &ModelConfig| {
            sizes_gib
                .iter()
                .find(|(name, _)| model.name.as_deref() == Some(*name))
                .map(|(_, gib)| gib * GIB_BYTES)
        };
        check_model_lifecycle(config, physical_memory, &footprint, &mut result);
        result
    }

    #[test]
    fn lifecycle_defaults_are_silent() {
        let config = lifecycle_config(
            |_| {},
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        let result = run_lifecycle_check(&config, Some(64 * GIB_BYTES), &[]);

        assert_eq!((result.passes, result.warnings, result.failures), (0, 0, 0));
    }

    #[test]
    fn lifecycle_rejects_a_non_positive_memory_limit() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let config = lifecycle_config(|local| local.memory_limit_gb = Some(bad), vec![]);
            let result = run_lifecycle_check(&config, None, &[]);
            assert_eq!(result.failures, 1, "memory_limit_gb={bad}");
        }
    }

    #[test]
    fn lifecycle_rejects_zero_max_loaded_models() {
        let config = lifecycle_config(|local| local.max_loaded_models = Some(0), vec![]);

        assert_eq!(run_lifecycle_check(&config, None, &[]).failures, 1);
    }

    #[test]
    fn lifecycle_warns_when_guard_cannot_read_physical_memory() {
        let config = lifecycle_config(
            |local| local.memory_guard = crate::config::MemoryGuard::Safe,
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        let result = run_lifecycle_check(&config, None, &[]);

        assert_eq!((result.warnings, result.failures), (1, 0));
    }

    #[test]
    fn lifecycle_warns_when_pinned_models_exceed_the_ceiling() {
        let config = lifecycle_config(
            |local| local.memory_limit_gb = Some(30.0),
            vec![
                named_model("big-a", |m| m.pinned = true),
                named_model("big-b", |m| m.pinned = true),
                named_model("small", |_| {}),
            ],
        );

        let result =
            run_lifecycle_check(&config, None, &[("big-a", 20), ("big-b", 20), ("small", 1)]);

        assert_eq!((result.warnings, result.failures), (1, 0));
    }

    #[test]
    fn lifecycle_does_not_warn_when_pinned_models_fit() {
        let config = lifecycle_config(
            |local| local.memory_limit_gb = Some(30.0),
            vec![
                named_model("a", |m| m.pinned = true),
                named_model("b", |_| {}),
            ],
        );

        let result = run_lifecycle_check(&config, None, &[("a", 10), ("b", 10)]);

        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn lifecycle_uses_the_guard_ceiling_for_the_pinned_check() {
        let config = lifecycle_config(
            |local| local.memory_guard = crate::config::MemoryGuard::Safe,
            vec![
                named_model("a", |m| m.pinned = true),
                named_model("b", |_| {}),
            ],
        );

        let fits = run_lifecycle_check(&config, Some(100 * GIB_BYTES), &[("a", 70)]);
        let too_big = run_lifecycle_check(&config, Some(100 * GIB_BYTES), &[("a", 90)]);

        assert_eq!(fits.warnings, 0);
        assert_eq!(too_big.warnings, 1);
    }

    #[test]
    fn lifecycle_warns_when_on_demand_has_a_single_model() {
        let config = lifecycle_config(
            |local| local.load = crate::config::LoadMode::OnDemand,
            vec![named_model("only", |_| {})],
        );

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }

    #[test]
    fn lifecycle_accepts_on_demand_with_several_models() {
        let config = lifecycle_config(
            |local| local.load = crate::config::LoadMode::OnDemand,
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        let result = run_lifecycle_check(&config, None, &[]);

        assert_eq!((result.warnings, result.failures), (0, 0));
        assert_eq!(result.passes, 1);
    }

    #[test]
    fn lifecycle_warns_when_startup_models_exceed_the_count_cap() {
        let config = lifecycle_config(
            |local| local.max_loaded_models = Some(1),
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }

    #[test]
    fn lifecycle_count_cap_ignores_on_demand_models() {
        let config = lifecycle_config(
            |local| {
                local.max_loaded_models = Some(1);
                local.load = crate::config::LoadMode::OnDemand;
            },
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 0);
    }

    #[test]
    fn lifecycle_warns_when_a_pinned_model_has_an_idle_ttl() {
        let config = lifecycle_config(
            |_| {},
            vec![
                named_model("a", |m| {
                    m.pinned = true;
                    m.idle_ttl_seconds = Some(60);
                }),
                named_model("b", |_| {}),
            ],
        );

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }

    #[test]
    fn lifecycle_warns_when_the_auto_router_model_is_on_demand() {
        let mut config = lifecycle_config(
            |_| {},
            vec![
                named_model("router", |m| {
                    m.load = Some(crate::config::LoadMode::OnDemand);
                }),
                named_model("other", |_| {}),
            ],
        );
        config.auto_router.enabled = true;
        config.auto_router.model = "router".to_owned();

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }
    // -- MCP integrations --

    #[test]
    fn tool_integrations_add_no_checks_when_nothing_is_configured() {
        let mut result = empty_result();
        check_tool_integrations(&HiggsConfig::default(), &mut result);
        assert_eq!((result.passes, result.warnings, result.failures), (0, 0, 0));
    }

    #[test]
    fn tool_integrations_report_mcp_servers_and_failures() {
        let mut config = HiggsConfig::default();
        config.mcp.servers.insert(
            "good".to_owned(),
            crate::mcp::config::McpServerConfig {
                command: Some("sh".to_owned()),
                ..crate::mcp::config::McpServerConfig::default()
            },
        );
        config.mcp.servers.insert(
            "broken".to_owned(),
            crate::mcp::config::McpServerConfig::default(),
        );
        let mut result = empty_result();
        check_tool_integrations(&config, &mut result);
        assert_eq!(result.passes, 1);
        assert_eq!(result.failures, 1);
    }
}
