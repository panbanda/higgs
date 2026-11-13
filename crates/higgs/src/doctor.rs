use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use higgs_engine::{mlx_tuning::resolve_effective_mlx_profile, model_loader};

use crate::config::explain::Provenance;
use crate::config::load::LoadOptions;
use crate::config::units::HumanDuration;
use crate::config::{BatchMode, Environment, HiggsConfig, ServeArgs};
use crate::decode_strategy::{DecodeFacts, DecodeStrategy, decode_strategy};
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

/// Where `higgs doctor` reads the configuration from.
#[derive(Debug, Clone)]
pub enum ConfigSource {
    File(PathBuf),
    /// No file: the models and settings come from `serve` flags.
    Flags,
}

const fn empty_result() -> DoctorResult {
    DoctorResult {
        passes: 0,
        warnings: 0,
        failures: 0,
    }
}

#[allow(clippy::print_stderr)]
fn print_summary(result: &DoctorResult) {
    eprintln!(
        "\n{} passed, {} warnings, {} failures",
        result.passes, result.warnings, result.failures
    );
}

/// Load the configuration, report every problem found, and run the checks. Returns the process
/// exit code.
#[allow(clippy::print_stderr)]
pub async fn run_doctor_command(source: ConfigSource, args: &ServeArgs) -> i32 {
    let environment = Environment::process();
    let (loaded, path) = match &source {
        ConfigSource::File(path) => {
            let options = LoadOptions {
                flags: Some(args),
                environment: environment.clone(),
            };
            (
                crate::config::load::load(path, &options),
                Some(path.clone()),
            )
        }
        ConfigSource::Flags => (crate::config::load::load_simple(args, &environment), None),
    };
    match loaded {
        Ok(found) => {
            let result = run_doctor_with_layers(
                &found.config,
                path.as_deref(),
                Some(&Provenance::new(&found, &environment, Some(args))),
            )
            .await;
            i32::from(result.failures > 0)
        }
        Err(error) => {
            eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");
            let mut result = empty_result();
            for diagnostic in &error.diagnostics {
                fail(&diagnostic.to_string(), &mut result);
            }
            print_summary(&result);
            1
        }
    }
}

pub async fn run_doctor(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
) -> DoctorResult {
    run_doctor_with_layers(config, config_path, None).await
}

#[allow(clippy::print_stderr)]
async fn run_doctor_with_layers(
    config: &HiggsConfig,
    config_path: Option<&std::path::Path>,
    provenance: Option<&Provenance<'_>>,
) -> DoctorResult {
    let mut result = empty_result();

    eprintln!("\x1b[1mhiggs doctor\x1b[0m\n");

    check_config_valid(&mut result);
    if let Some(sources) = provenance {
        check_included_files(sources.files(), &mut result);
    }
    check_deprecations(config, &mut result);
    check_environment_overrides(config, &mut result);
    check_config_file_permissions(config, config_path, &mut result);
    check_server_section(config, &mut result);
    check_observability(config, &mut result);
    check_experimental(config, &mut result);
    check_models(config, &mut result);
    check_model_lifecycle(
        config,
        crate::routes::system::physical_memory_bytes(),
        &|model| model_resolver::estimated_footprint_bytes(&model.path),
        provenance,
        &mut result,
    );
    check_duplicate_models(config, &mut result);
    check_aliases(config, &mut result);
    check_providers(config, &mut result).await;
    check_route_consistency(config, &mut result);
    check_default_provider(config, &mut result);
    check_auto_router(config, &mut result);
    check_port_availability(config, &mut result);
    check_orphaned_providers(config, &mut result);
    check_tool_integrations(config, &mut result);

    print_summary(&result);

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

    let timeout = HumanDuration::new(server.request_timeout);
    if server.request_timeout.is_zero() {
        fail(
            "server.request_timeout must be longer than zero (e.g. \"5m\")",
            result,
        );
    } else if server.request_timeout > std::time::Duration::from_secs(600) {
        warn(
            &format!("server.request_timeout={timeout} is unusually high (>10 min); check intent"),
            result,
        );
    } else {
        pass(&format!("server.request_timeout={timeout}"), result);
    }

    if server.max_body_size.bytes() == 0 {
        fail("server.max_body_size=0B rejects all request bodies", result);
    } else if server.max_body_size.bytes() > 1 << 30 {
        warn(
            &format!(
                "server.max_body_size={} (>1GB); check intent",
                server.max_body_size
            ),
            result,
        );
    } else {
        pass(
            &format!("server.max_body_size={}", server.max_body_size),
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
            &format!("model {label} prefill.yield_tokens={tokens} must be 0 or at least 128"),
            result,
        );
        return false;
    }
    if tokens != 0 && tokens < 512 {
        warn(
            &format!("model {label} prefill.yield_tokens={tokens} is below the recommended 512"),
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

/// Warn (not fail) when the *resolved* MLA decision is enabled for an adapter
/// that does not advertise MLA latent-cache support. The flag is a no-op for
/// those adapters at runtime, so this is advisory rather than a hard failure.
///
/// Uses [`higgs_models::cache::resolve_mla_latent_cache`] rather than the raw
/// `kv_cache.mla_latent` value, so this matches runtime behavior: e.g.
/// `HIGGS_MLA_LATENT_CACHE=1` with `mla_latent` unset in config still
/// warns (the flag is effectively on), and `HIGGS_MLA_LATENT_CACHE=0` with
/// `mla_latent = true` in config does not warn (the flag is effectively
/// off).
#[cfg(test)]
fn check_mla_latent_cache_architecture(
    label: &str,
    model: &crate::config::ModelConfig,
    resolved: &std::path::Path,
    result: &mut DoctorResult,
) {
    if !higgs_models::cache::resolve_mla_latent_cache(model.kv_cache.mla_latent) {
        return;
    }
    match model_loader::ModelConfig::from_dir(resolved) {
        Ok(inspected) => check_mla_latent_cache_adapter(label, &inspected, result),
        Err(err) => {
            warn(
                &format!(
                    "model {label} enables kv_cache.mla_latent=true but its architecture could not be determined: {err}"
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
                "model {label} kv_cache.mla_latent=true (adapter {})",
                inspected.adapter_id
            ),
            result,
        );
    } else {
        warn(
            &format!(
                "model {label} enables kv_cache.mla_latent=true but adapter '{}' does not support MLA latent cache; the flag is a no-op at runtime",
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
        } else if model.prefix_cache.disk_dir.is_some() {
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
            Some(model.thinking_budget),
            config.server.max_tokens,
            result,
        ) {
            continue;
        }
        let kv_cache_config = model.kv_cache;
        match kv_cache_config.validate() {
            Ok(()) => {
                // `validate()` only rejects the MLA/TurboQuant combination
                // using the *resolved* decision (env-aware), so a
                // config-declared conflict that HIGGS_MLA_LATENT_CACHE
                // overrides away passes silently there. Surface that as an
                // advisory warning rather than staying silent.
                if model.kv_cache.mla_latent
                    && kv_cache_config.is_turboquant()
                    && !higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent)
                {
                    warn(
                        &format!(
                            "model {label} sets kv_cache.mla_latent=true with kv_cache.mode=turboquant, but HIGGS_MLA_LATENT_CACHE overrides MLA off; the conflict is masked at runtime"
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
        if model.batch == BatchMode::On {
            warn(
                &format!(
                    "model {label} sets batch=true, which has no effect: every model decodes requests together whenever it can; remove it"
                ),
                result,
            );
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
                if higgs_models::cache::resolve_mla_latent_cache(kv_cache_config.mla_latent) {
                    check_mla_latent_cache_adapter(&label, &inspected, result);
                }
                let requested_profile = model.profile;
                let facts = DecodeFacts::inspect(model, &config.experimental, &resolved).ok();
                let effective_profile = resolve_effective_mlx_profile(&resolved, requested_profile);
                let profile_msg = decode_summary(
                    facts.as_ref(),
                    config.experimental.prompt_lookup,
                    model.prefix_cache.disk_dir.is_some(),
                    effective_profile.as_str(),
                    requested_profile.as_str(),
                );
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

/// How the model's requests are decoded and the settings that matter for it, as
/// the doctor reports them. `facts` is `None` when the checkpoint could not be inspected.
fn decode_summary(
    facts: Option<&DecodeFacts>,
    prompt_lookup: bool,
    disk_prefix_store: bool,
    effective_profile: &str,
    requested_profile: &str,
) -> String {
    let profile = if effective_profile == requested_profile {
        format!("profile={effective_profile}")
    } else {
        format!("profile={effective_profile} (requested {requested_profile})")
    };
    let mut parts = vec![match facts.map(decode_strategy) {
        Some(DecodeStrategy::Batched) => "decode=batched".to_owned(),
        Some(DecodeStrategy::Sequential(reason)) => {
            format!("decode=one request at a time because {reason}")
        }
        None => "decode=chosen at load".to_owned(),
    }];
    if facts.is_some_and(|found| found.mtp_speculative_decode) {
        parts.push("MTP speculative decode for a lone request".to_owned());
    }
    if prompt_lookup {
        parts.push("prompt-lookup decode for a lone request".to_owned());
    }
    if disk_prefix_store {
        parts.push("disk prefix store".to_owned());
    }
    parts.push(profile);
    parts.join("; ")
}

fn is_auto_router_model(config: &HiggsConfig, model: &crate::config::ModelConfig) -> bool {
    config.routing.auto.enabled && model.name.as_deref() == Some(config.routing.auto.model.as_str())
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
    provenance: Option<&Provenance<'_>>,
    result: &mut DoctorResult,
) -> LifecycleTotals {
    use crate::config::LoadMode;

    let mut totals = LifecycleTotals {
        startup_models: 0,
        startup_bytes: 0,
        pinned_bytes: 0,
    };
    for (index, model) in config.models.iter().enumerate() {
        let label = model_label(model);
        let is_classifier = is_auto_router_model(config, model);
        if is_classifier && model.load == LoadMode::OnDemand {
            warn(
                &format!(
                    "model {label} is the auto_router classifier; it always loads at startup and is never evicted, so load=\"on_demand\" is ignored"
                ),
                result,
            );
        }
        let sets_own_idle_ttl =
            provenance.is_none_or(|sources| sources.model_entry_sets(index, "idle_ttl"));
        if model.pinned && model.idle_ttl.is_some() && sets_own_idle_ttl {
            warn(
                &format!("model {label} is pinned, so its idle_ttl never applies"),
                result,
            );
        }
        let bytes = footprint(model).unwrap_or(0);
        if is_classifier || model.load == LoadMode::Startup {
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
    provenance: Option<&Provenance<'_>>,
    result: &mut DoctorResult,
) {
    use crate::config::LoadMode;

    let memory = &config.memory;
    if memory.limit.is_some_and(|limit| limit.bytes() == 0) {
        fail("memory.limit must be greater than zero", result);
        return;
    }
    if memory.max_loaded_models == Some(0) {
        fail("memory.max_loaded_models must be at least 1", result);
        return;
    }

    let ceiling = memory.ceiling_bytes(physical_memory);
    if let Some(ceiling_bytes) = ceiling {
        pass(
            &format!(
                "memory ceiling for loaded models is {}",
                gib_label(ceiling_bytes)
            ),
            result,
        );
    } else if memory.limit.is_none() && memory.guard != crate::config::MemoryGuard::Off {
        warn(
            "memory.guard is set but total memory could not be read; no ceiling will be enforced",
            result,
        );
    }

    let LifecycleTotals {
        startup_models,
        startup_bytes,
        pinned_bytes,
    } = tally_lifecycle(config, footprint, provenance, result);

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

    if let Some(cap) = memory.max_loaded_models
        && startup_models > cap
    {
        warn(
            &format!(
                "{startup_models} models load at startup but memory.max_loaded_models={cap}; the least recently used will be evicted during startup"
            ),
            result,
        );
    }

    let on_demand_models = config
        .models
        .iter()
        .filter(|model| !is_auto_router_model(config, model) && model.load == LoadMode::OnDemand)
        .count();
    if on_demand_models > 0 && config.models.len() == 1 {
        warn(
            "load=\"on_demand\" with a single model only delays the first request; use \"startup\" unless you rely on idle_ttl",
            result,
        );
    }

    let lifecycle_configured = ceiling.is_some()
        || memory.max_loaded_models.is_some()
        || on_demand_models > 0
        || config
            .models
            .iter()
            .any(|model| model.pinned || model.idle_ttl.is_some());
    if lifecycle_configured {
        pass("model lifecycle settings are valid", result);
    }
}

fn check_included_files(files: &[PathBuf], result: &mut DoctorResult) {
    if files.len() > 1 {
        let names: Vec<String> = files
            .iter()
            .map(|file| file.display().to_string())
            .collect();
        pass(
            &format!(
                "config assembled from {} files: {}",
                files.len(),
                names.join(", ")
            ),
            result,
        );
    }
}

fn check_deprecations(config: &HiggsConfig, result: &mut DoctorResult) {
    if config.notices.deprecations.is_empty() {
        pass("config uses only current keys", result);
        return;
    }
    for deprecation in &config.notices.deprecations {
        warn(&deprecation.to_string(), result);
    }
    warn(
        &format!(
            "{} deprecated key(s) still work; run `higgs config migrate` to rewrite the file",
            config.notices.deprecations.len()
        ),
        result,
    );
}

fn check_environment_overrides(config: &HiggsConfig, result: &mut DoctorResult) {
    for entry in &config.notices.env_overrides {
        if entry.applied {
            warn(
                &format!(
                    "{} is set in the environment and overrides config key {}",
                    entry.variable, entry.key
                ),
                result,
            );
        } else {
            warn(
                &format!(
                    "{} is set in the environment to a value config key {} cannot use; it is ignored",
                    entry.variable, entry.key
                ),
                result,
            );
        }
    }
    for variable in &config.notices.env_only {
        warn(
            &format!("{variable} is set in the environment; it has no config key"),
            result,
        );
    }
}

fn check_observability(config: &HiggsConfig, result: &mut DoctorResult) {
    let observability = &config.observability;
    match observability.retention {
        None => pass(
            "observability.retention is off; the dashboard keeps no history",
            result,
        ),
        Some(window) if window < std::time::Duration::from_secs(60) => warn(
            &format!(
                "observability.retention={} is under a minute; the dashboard will show little",
                HumanDuration::new(window)
            ),
            result,
        ),
        Some(window) => pass(
            &format!("observability.retention={}", HumanDuration::new(window)),
            result,
        ),
    }
    let log = &observability.metrics_log;
    if log.enabled {
        if log.max_size.bytes() == 0 {
            warn(
                "observability.metrics_log.max_size=0B rotates the log after every write",
                result,
            );
        } else {
            pass(
                &format!(
                    "observability.metrics_log.max_size={} (keeping {} rotated file(s))",
                    log.max_size, log.max_files
                ),
                result,
            );
        }
    }
    if config.routing.auto.enabled && config.routing.auto.timeout.is_zero() {
        fail(
            "routing.auto.timeout must be longer than zero (e.g. \"2s\")",
            result,
        );
    }
}

fn check_experimental(config: &HiggsConfig, result: &mut DoctorResult) {
    let experimental = &config.experimental;
    if experimental.mtp_draft_tokens == Some(0) {
        fail("experimental.mtp_draft_tokens must be at least 1", result);
    } else if experimental
        .mtp_draft_tokens
        .is_some_and(|tokens| tokens > 8)
    {
        warn(
            "experimental.mtp_draft_tokens is clamped to 8 by the engine",
            result,
        );
    }
    if experimental.chunked_prefill_threshold == Some(0) {
        fail(
            "experimental.chunked_prefill.threshold must be at least 1",
            result,
        );
    }
    if experimental.chunked_prefill_chunk == Some(0) {
        fail(
            "experimental.chunked_prefill.chunk must be at least 1",
            result,
        );
    }
    if experimental.mtp_prompt_lookup && experimental.mtp == crate::config::MtpMode::Off {
        warn(
            "experimental.mtp_prompt_lookup has no effect while experimental.mtp = \"off\"",
            result,
        );
    }
}

/// Names a client may send for each local model, and routes that those names make unreachable.
fn check_aliases(config: &HiggsConfig, result: &mut DoctorResult) {
    let mut local_names: Vec<(String, String)> = Vec::new();
    for model in &config.models {
        let served = crate::engine_loader::served_name(model);
        local_names.push((served.clone(), served.clone()));
        for alias in &model.aliases {
            local_names.push((alias.clone(), served.clone()));
        }
    }
    let mut shadowed = false;
    for route in &config.routes {
        let Some(pattern) = route.pattern.as_deref() else {
            continue;
        };
        let Ok(compiled) = regex::Regex::new(pattern) else {
            fail(
                &format!("route match = {pattern:?} is not a valid regular expression"),
                result,
            );
            continue;
        };
        for (name, served) in &local_names {
            if compiled.is_match(name) {
                shadowed = true;
                warn(
                    &format!(
                        "route match = {pattern:?} also matches local model name {name:?} (model {served}); local models win, so that name never reaches provider {:?}",
                        route.provider
                    ),
                    result,
                );
            }
        }
    }
    let aliases = config
        .models
        .iter()
        .map(|model| model.aliases.len())
        .sum::<usize>();
    if aliases > 0 && !shadowed {
        pass(
            &format!("{aliases} model alias(es) do not collide with any model name or route"),
            result,
        );
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
    let provider = &config.routing.default_provider;
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
            &format!("routing.default \"{provider}\" not found in providers"),
            result,
        );
    }
}

fn check_auto_router(config: &HiggsConfig, result: &mut DoctorResult) {
    if !config.routing.auto.enabled {
        return;
    }

    let model_ref = &config.routing.auto.model;
    if model_ref.is_empty() {
        fail("routing.auto enabled but no model specified", result);
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
            &format!("routing.auto model {label} found in models"),
            result,
        );
        match model_resolver::resolve(&matched_model.path) {
            Ok(_) => pass(&format!("routing.auto model {label} downloaded"), result),
            Err(err) => fail(
                &format!("routing.auto model {label} not downloaded: {err}"),
                result,
            ),
        }
    } else {
        fail(
            &format!("routing.auto model \"{model_ref}\" not found in models"),
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
            "routing.auto enabled but no routes have descriptions",
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

    if config.routing.default_provider != "higgs" {
        referenced.insert(&config.routing.default_provider);
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
        AutoRouterConfig, HiggsConfig, ModelConfig, ProviderConfig, RouteConfig, RoutingConfig,
        Secret, ServerSection,
    };
    use std::collections::BTreeMap as HashMap;
    use std::time::Duration;

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
    fn models_reject_a_zero_thinking_budget_inherited_from_defaults() {
        let config = HiggsConfig {
            models: vec![ModelConfig {
                thinking_budget: 0,
                ..ModelConfig::with_path("org/m")
            }],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        assert_eq!(result.failures, 1);
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
                    batch: BatchMode::Off,
                    ..crate::config::ModelConfig::with_path("")
                },
                ModelConfig {
                    path: "org/model-b".to_owned(),
                    batch: BatchMode::Off,
                    ..crate::config::ModelConfig::with_path("")
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
                    batch: BatchMode::Off,
                    ..crate::config::ModelConfig::with_path("")
                },
                ModelConfig {
                    path: "org/model-a".to_owned(),
                    batch: BatchMode::Off,
                    ..crate::config::ModelConfig::with_path("")
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
            routing: RoutingConfig {
                default_provider: "higgs".to_owned(),
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                default_provider: "anthropic".to_owned(),
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                default_provider: "anthropic".to_owned(),
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                default_provider: "nonexistent".to_owned(),
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                default_provider: "higgs".to_owned(),
                ..RoutingConfig::default()
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
    fn test_request_timeout_zero_fails() {
        let config = server_with(|s| s.request_timeout = Duration::ZERO);
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_request_timeout_unusually_high_warns() {
        let config = server_with(|s| s.request_timeout = Duration::from_secs(3600));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.warnings >= 1);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_max_body_size_zero_fails() {
        let config =
            server_with(|s| s.max_body_size = crate::config::units::ByteSize::from_bytes(0));
        let mut result = empty_result();
        check_server_section(&config, &mut result);
        assert!(result.failures >= 1);
    }

    #[test]
    fn test_max_body_size_huge_warns() {
        let config = server_with(|s| {
            s.max_body_size = crate::config::units::ByteSize::from_bytes((1 << 30) + 1);
        });
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
        let config = server_with(|s| s.api_key = Some(Secret::new("secret".to_owned())));
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
            s.api_key = Some(Secret::new("sk-test".to_owned()));
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
            s.api_key = Some(Secret::new("sk-test".to_owned()));
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
            routing: RoutingConfig {
                auto: AutoRouterConfig {
                    enabled: false,
                    force: false,
                    model: String::new(),
                    timeout: Duration::from_millis(2000),
                },
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                auto: AutoRouterConfig {
                    enabled: true,
                    force: false,
                    model: String::new(),
                    timeout: Duration::from_millis(2000),
                },
                ..RoutingConfig::default()
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
            routing: RoutingConfig {
                auto: AutoRouterConfig {
                    enabled: true,
                    force: false,
                    model: "nonexistent/model".to_owned(),
                    timeout: Duration::from_millis(2000),
                },
                ..RoutingConfig::default()
            },
            models: vec![ModelConfig {
                path: "org/other-model".to_owned(),
                batch: BatchMode::Off,
                ..crate::config::ModelConfig::with_path("")
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
            routing: RoutingConfig {
                auto: AutoRouterConfig {
                    enabled: true,
                    force: false,
                    model: "org/router-model".to_owned(),
                    timeout: Duration::from_millis(2000),
                },
                ..RoutingConfig::default()
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                batch: BatchMode::Off,
                ..crate::config::ModelConfig::with_path("")
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
            routing: RoutingConfig {
                auto: AutoRouterConfig {
                    enabled: true,
                    force: false,
                    model: "org/router-model".to_owned(),
                    timeout: Duration::from_millis(2000),
                },
                ..RoutingConfig::default()
            },
            models: vec![ModelConfig {
                path: "org/router-model".to_owned(),
                batch: BatchMode::Off,
                ..crate::config::ModelConfig::with_path("")
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
            batch: BatchMode::Off,
            ..ModelConfig::with_path(path)
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
            model.kv_cache.mode = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.kv_cache.mla_latent = true;
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
            model.kv_cache.mode = higgs_models::turboquant::KvCacheMode::Turboquant;
            model.kv_cache.mla_latent = true;
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
            model.kv_cache.mode = higgs_models::turboquant::KvCacheMode::Turboquant;
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
                kv_cache: higgs_models::turboquant::KvCacheConfig {
                    mla_latent: true,
                    ..ModelConfig::with_path("").kv_cache
                },
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
                kv_cache: higgs_models::turboquant::KvCacheConfig {
                    mla_latent: true,
                    ..ModelConfig::with_path("").kv_cache
                },
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
                kv_cache: higgs_models::turboquant::KvCacheConfig {
                    mla_latent: true,
                    ..ModelConfig::with_path("").kv_cache
                },
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

    /// The settings the lifecycle tests vary: the memory section and a load default.
    #[derive(Default)]
    struct Local {
        limit: Option<crate::config::units::ByteSize>,
        guard: crate::config::MemoryGuard,
        max_loaded_models: Option<usize>,
        load: Option<crate::config::LoadMode>,
    }

    fn lifecycle_config(local: impl FnOnce(&mut Local), models: Vec<ModelConfig>) -> HiggsConfig {
        let mut settings = Local::default();
        local(&mut settings);
        let mut config = HiggsConfig {
            models,
            memory: crate::config::MemoryConfig {
                guard: settings.guard,
                limit: settings.limit,
                max_loaded_models: settings.max_loaded_models,
                ..crate::config::MemoryConfig::default()
            },
            ..HiggsConfig::default()
        };
        if let Some(load) = settings.load {
            for model in &mut config.models {
                model.load = load;
            }
        }
        config
    }

    fn gib(count: u64) -> crate::config::units::ByteSize {
        crate::config::units::ByteSize::from_bytes(count * (1 << 30))
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
        check_model_lifecycle(config, physical_memory, &footprint, None, &mut result);
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
        let config = lifecycle_config(
            |local| local.limit = Some(crate::config::units::ByteSize::from_bytes(0)),
            vec![],
        );
        let result = run_lifecycle_check(&config, None, &[]);
        assert_eq!(result.failures, 1);
    }

    #[test]
    fn lifecycle_rejects_zero_max_loaded_models() {
        let config = lifecycle_config(|local| local.max_loaded_models = Some(0), vec![]);

        assert_eq!(run_lifecycle_check(&config, None, &[]).failures, 1);
    }

    #[test]
    fn lifecycle_warns_when_guard_cannot_read_physical_memory() {
        let config = lifecycle_config(
            |local| local.guard = crate::config::MemoryGuard::Safe,
            vec![named_model("a", |_| {}), named_model("b", |_| {})],
        );

        let result = run_lifecycle_check(&config, None, &[]);

        assert_eq!((result.warnings, result.failures), (1, 0));
    }

    #[test]
    fn lifecycle_warns_when_pinned_models_exceed_the_ceiling() {
        let config = lifecycle_config(
            |local| local.limit = Some(gib(30)),
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
            |local| local.limit = Some(gib(30)),
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
            |local| local.guard = crate::config::MemoryGuard::Safe,
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
            |local| local.load = Some(crate::config::LoadMode::OnDemand),
            vec![named_model("only", |_| {})],
        );

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }

    #[test]
    fn lifecycle_accepts_on_demand_with_several_models() {
        let config = lifecycle_config(
            |local| local.load = Some(crate::config::LoadMode::OnDemand),
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
                local.load = Some(crate::config::LoadMode::OnDemand);
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
                    m.idle_ttl = Some(Duration::from_secs(60));
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
                    m.load = crate::config::LoadMode::OnDemand;
                }),
                named_model("other", |_| {}),
            ],
        );
        config.routing.auto.enabled = true;
        config.routing.auto.model = "router".to_owned();

        assert_eq!(run_lifecycle_check(&config, None, &[]).warnings, 1);
    }
    // -- MCP and web integrations --

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

    #[test]
    fn tool_integrations_validate_the_web_section() {
        let mut config = HiggsConfig::default();
        config.web.enabled = true;
        config.web.search_provider = "searxng".to_owned();
        let mut result = empty_result();
        check_tool_integrations(&config, &mut result);
        assert_eq!(result.failures, 1);

        config.web.searxng_url = Some("http://localhost:8888".to_owned());
        let mut fixed = empty_result();
        check_tool_integrations(&config, &mut fixed);
        assert_eq!((fixed.passes, fixed.failures), (1, 0));
    }

    // -- decode strategy reporting --

    use crate::decode_strategy::SequentialReason;

    const BATCHABLE: DecodeFacts = DecodeFacts {
        architecture_batches: true,
        turboquant_kv: false,
        serialized_by_config: false,
        mtp_speculative_decode: false,
    };

    #[test]
    fn decode_summary_names_how_requests_are_decoded() {
        let summary = |facts: Option<&DecodeFacts>, lookup, disk| {
            decode_summary(facts, lookup, disk, "balanced", "auto")
        };
        assert_eq!(
            summary(Some(&BATCHABLE), false, false),
            "decode=batched; profile=balanced (requested auto)"
        );
        assert_eq!(
            summary(
                Some(&DecodeFacts {
                    turboquant_kv: true,
                    ..BATCHABLE
                }),
                false,
                false
            ),
            format!(
                "decode=one request at a time because {}; profile=balanced (requested auto)",
                SequentialReason::TurboQuantKv
            )
        );
        assert_eq!(
            summary(None, false, false),
            "decode=chosen at load; profile=balanced (requested auto)"
        );
    }

    #[test]
    fn decode_summary_lists_the_features_a_lone_request_uses() {
        let facts = DecodeFacts {
            mtp_speculative_decode: true,
            ..BATCHABLE
        };
        let summary = decode_summary(Some(&facts), true, true, "balanced", "balanced");
        assert_eq!(
            summary,
            "decode=batched; MTP speculative decode for a lone request; prompt-lookup decode for a lone request; disk prefix store; profile=balanced"
        );
    }

    fn check_one_model(model: ModelConfig) -> DoctorResult {
        let config = HiggsConfig {
            models: vec![model],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_models(&config, &mut result);
        result
    }

    fn llama_checkpoint() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        write_model_config_json(dir.path(), "llama");
        dir
    }

    #[test]
    fn a_thinking_budget_does_not_warn() {
        let dir = llama_checkpoint();
        let result = check_one_model(ModelConfig {
            thinking_budget: 64,
            ..model_with_path(dir.path().display().to_string())
        });
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn auto_batch_model_passes_cleanly() {
        let dir = llama_checkpoint();
        let result = check_one_model(ModelConfig {
            batch: BatchMode::Auto,
            ..model_with_path(dir.path().display().to_string())
        });
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn a_turboquant_model_passes_and_batch_true_only_warns_it_has_no_effect() {
        let dir = llama_checkpoint();
        let turboquant = |batch| ModelConfig {
            batch,
            kv_cache: higgs_models::turboquant::KvCacheConfig {
                mode: higgs_models::turboquant::KvCacheMode::Turboquant,
                ..ModelConfig::with_path("").kv_cache
            },
            ..model_with_path(dir.path().display().to_string())
        };
        let auto = check_one_model(turboquant(BatchMode::Auto));
        assert_eq!((auto.warnings, auto.failures), (0, 0));
        let forced = check_one_model(turboquant(BatchMode::On));
        assert_eq!((forced.warnings, forced.failures), (1, 0));
    }

    #[test]
    fn a_disk_prefix_store_is_not_a_warning() {
        let dir = llama_checkpoint();
        let store = tempfile::tempdir().unwrap();
        let result = check_one_model(ModelConfig {
            prefix_cache: crate::config::PrefixCacheConfig {
                disk_dir: Some(store.path().display().to_string()),
                ..crate::config::PrefixCacheConfig::default()
            },
            ..model_with_path(dir.path().display().to_string())
        });
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    #[test]
    fn an_architecture_without_batched_decode_passes() {
        let dir = tempfile::tempdir().unwrap();
        write_model_config_json(dir.path(), "deepseek_v2");
        let result = check_one_model(ModelConfig {
            batch: BatchMode::Auto,
            ..model_with_path(dir.path().display().to_string())
        });
        assert_eq!((result.warnings, result.failures), (0, 0));
    }

    // -- Deprecations, environment overrides and the new sections --

    fn load_for_doctor(toml: &str) -> HiggsConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        crate::config::load_config_file_in(&path, None, &crate::config::Environment::default())
            .unwrap()
    }

    #[test]
    fn deprecated_keys_are_reported_once_each_with_the_replacement() {
        let config = load_for_doctor(
            "[local]\nload = \"on_demand\"\n[[models]]\npath = \"org/a\"\n[[models]]\npath = \"org/b\"\n",
        );
        let mut result = empty_result();
        check_deprecations(&config, &mut result);
        assert_eq!(result.failures, 0);
        assert_eq!(
            result.warnings, 2,
            "one warning for the key plus the summary"
        );
        assert_eq!(result.passes, 0);

        let mut clean = empty_result();
        check_deprecations(&HiggsConfig::default(), &mut clean);
        assert_eq!((clean.passes, clean.warnings), (1, 0));
    }

    #[test]
    fn environment_overrides_name_the_key_they_replace() {
        let mut config = HiggsConfig::default();
        config.notices.env_overrides = vec![
            crate::config::EnvOverride {
                variable: "HIGGS_MTP",
                key: "experimental.mtp",
                applied: true,
            },
            crate::config::EnvOverride {
                variable: "HIGGS_CHUNKED_PREFILL_CHUNK_SIZE",
                key: "experimental.chunked_prefill.chunk",
                applied: false,
            },
        ];
        config.notices.env_only = vec!["HIGGS_ENABLE_THINKING"];
        let mut result = empty_result();
        check_environment_overrides(&config, &mut result);
        assert_eq!((result.warnings, result.failures), (3, 0));
    }

    #[test]
    fn observability_checks_the_retention_window_and_log_size() {
        let defaults = HiggsConfig::default();
        let mut defaults_result = empty_result();
        check_observability(&defaults, &mut defaults_result);
        assert_eq!((defaults_result.warnings, defaults_result.failures), (0, 0));

        let mut off = HiggsConfig::default();
        off.observability.retention = None;
        let mut off_result = empty_result();
        check_observability(&off, &mut off_result);
        assert_eq!(off_result.passes, 2);

        let mut tiny = HiggsConfig::default();
        tiny.observability.retention = Some(Duration::from_secs(5));
        tiny.observability.metrics_log.max_size = crate::config::units::ByteSize::from_bytes(0);
        let mut tiny_result = empty_result();
        check_observability(&tiny, &mut tiny_result);
        assert_eq!((tiny_result.warnings, tiny_result.failures), (2, 0));
    }

    #[test]
    fn a_zero_auto_router_timeout_fails_only_when_the_router_is_on() {
        let mut config = HiggsConfig::default();
        config.routing.auto.timeout = Duration::ZERO;
        let mut router_off = empty_result();
        check_observability(&config, &mut router_off);
        assert_eq!(router_off.failures, 0);
        config.routing.auto.enabled = true;
        let mut router_on = empty_result();
        check_observability(&config, &mut router_on);
        assert_eq!(router_on.failures, 1);
    }

    #[test]
    fn experimental_ranges_are_checked() {
        let check = |modify: fn(&mut crate::config::ExperimentalConfig)| {
            let mut config = HiggsConfig::default();
            modify(&mut config.experimental);
            let mut result = empty_result();
            check_experimental(&config, &mut result);
            (result.warnings, result.failures)
        };
        assert_eq!(check(|_| {}), (0, 0));
        assert_eq!(check(|e| e.mtp_draft_tokens = Some(0)), (0, 1));
        assert_eq!(check(|e| e.mtp_draft_tokens = Some(12)), (1, 0));
        assert_eq!(check(|e| e.mtp_draft_tokens = Some(4)), (0, 0));
        assert_eq!(check(|e| e.chunked_prefill_threshold = Some(0)), (0, 1));
        assert_eq!(check(|e| e.chunked_prefill_chunk = Some(0)), (0, 1));
        assert_eq!(check(|e| e.chunked_prefill_chunk = Some(512)), (0, 0));
        assert_eq!(
            check(|e| {
                e.mtp = crate::config::MtpMode::Off;
                e.mtp_prompt_lookup = true;
            }),
            (1, 0)
        );
    }

    fn aliased(name: &str, aliases: &[&str]) -> ModelConfig {
        ModelConfig {
            name: Some(name.to_owned()),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            ..ModelConfig::with_path(format!("org/{name}"))
        }
    }

    fn route(pattern: &str) -> RouteConfig {
        RouteConfig {
            name: None,
            description: None,
            pattern: Some(pattern.to_owned()),
            provider: "remote".to_owned(),
            model: None,
        }
    }

    #[test]
    fn a_route_that_matches_a_local_name_or_alias_is_flagged() {
        let config = HiggsConfig {
            models: vec![aliased("qwen", &["gpt-local"])],
            routes: vec![route("gpt-.*"), route("claude-.*")],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_aliases(&config, &mut result);
        assert_eq!((result.warnings, result.failures, result.passes), (1, 0, 0));
    }

    #[test]
    fn aliases_that_collide_with_nothing_pass() {
        let config = HiggsConfig {
            models: vec![aliased("qwen", &["default", "chat"])],
            routes: vec![route("claude-.*")],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_aliases(&config, &mut result);
        assert_eq!((result.warnings, result.failures, result.passes), (0, 0, 1));
    }

    #[test]
    fn an_invalid_route_pattern_fails() {
        let config = HiggsConfig {
            routes: vec![route("(unclosed")],
            ..HiggsConfig::default()
        };
        let mut result = empty_result();
        check_aliases(&config, &mut result);
        assert_eq!(result.failures, 1);
    }

    #[tokio::test]
    async fn the_doctor_reports_load_errors_as_failures_and_exits_non_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[server]\nprot = 1\n[defaults]\nidel_ttl = \"5m\"\n").unwrap();
        let code = run_doctor_command(ConfigSource::File(path), &ServeArgs::default()).await;
        assert_eq!(code, 1);
    }
}
