//! The resolved configuration: validated, with defaults applied, durations as `Duration`, sizes as
//! bytes and secrets expanded. The rest of the server reads only these types.

use std::collections::BTreeMap;
use std::time::Duration;

use higgs_engine::mlx_tuning::{RequestedMlxProfile, TuningOverrides};
use higgs_models::turboquant::{KvCacheConfig, KvCacheMode};

use super::environment::Environment;
use super::legacy::Deprecation;
use super::raw::{
    RawConfig, RawMcp, RawMcpServer, RawModel, RawModelSettings, RawRetention, RawWeb,
};
use super::types::{ApiFormat, BatchMode, LoadMode, MemoryGuard, MtpMode, Secret};
use super::units::ByteSize;
use crate::mcp::config::{McpConfig, McpServerConfig};
use crate::web::config::WebConfig;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8000;
pub const DEFAULT_MAX_TOKENS: u32 = 32768;
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
pub const DEFAULT_MAX_BODY_SIZE: ByteSize = ByteSize::from_bytes(10 * 1024 * 1024);
pub const DEFAULT_KV_BITS: u8 = 3;
pub const DEFAULT_PREFIX_DISK_BUDGET: ByteSize = ByteSize::from_bytes(4 * 1024 * 1024 * 1024);
pub const DEFAULT_ROUTING_PROVIDER: &str = "higgs";
pub const DEFAULT_AUTO_ROUTER_MODEL: &str = "katanemo/Arch-Router-1.5B";
pub const DEFAULT_AUTO_ROUTER_TIMEOUT: Duration = Duration::from_millis(2000);
pub const DEFAULT_RETENTION: Duration = Duration::from_secs(60 * 60);
pub const DEFAULT_METRICS_LOG_MAX_SIZE: ByteSize = ByteSize::from_bytes(50 * 1024 * 1024);
pub const DEFAULT_METRICS_LOG_MAX_FILES: u32 = 5;

/// A `HIGGS_*` environment variable that is set and replaces a configuration key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvOverride {
    pub variable: &'static str,
    pub key: &'static str,
    /// False when the variable is set to something the setting cannot use, so it has no effect.
    pub applied: bool,
}

/// Things worth telling the user about a loaded configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Notices {
    pub deprecations: Vec<Deprecation>,
    pub env_overrides: Vec<EnvOverride>,
    /// Behavior-changing `HIGGS_*` variables that are set but have no config key.
    pub env_only: Vec<&'static str>,
}

/// Documented `HIGGS_*` variables read directly by the engines that change behavior and have no
/// config key.
pub const ENV_ONLY_BEHAVIOR: &[&str] = &[
    "HIGGS_ENABLE_THINKING",
    "HIGGS_TURBOQUANT_MIN_TOKENS",
    "HIGGS_TURBOQUANT_ACTIVATE_AT",
    "HIGGS_WIRED_LIMIT_MODE",
    "HIGGS_NO_MEM_LIMIT",
    "HIGGS_MTP_PRIME_PREFILL",
    "HIGGS_MTP_MIRROR_VERIFY",
    "HIGGS_PROMPT_LOOKUP_UNCHECKED",
    "HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX",
    "HIGGS_PROMPT_LOOKUP_NGRAM_MAX",
    "HIGGS_PROMPT_LOOKUP_WINDOW",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
    pub max_tokens: u32,
    pub api_key: Option<Secret>,
    pub rate_limit: u32,
    pub request_timeout: Duration,
    pub max_body_size: ByteSize,
    /// `None` sends no CORS headers; `["*"]` allows any origin.
    pub cors_origins: Option<Vec<String>>,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_owned(),
            port: DEFAULT_PORT,
            max_tokens: DEFAULT_MAX_TOKENS,
            api_key: None,
            rate_limit: 0,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            max_body_size: DEFAULT_MAX_BODY_SIZE,
            cors_origins: None,
        }
    }
}

/// Backward-compatible alias. Route handlers access `state.config.max_tokens`.
pub type ServerConfig = ServerSection;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryConfig {
    pub guard: MemoryGuard,
    /// Explicit ceiling on MLX active memory; wins over `guard`.
    pub limit: Option<ByteSize>,
    /// Keep at most this many models loaded; the least recently used idle one is evicted first.
    pub max_loaded_models: Option<usize>,
    /// Raise MLX's wired memory limit to the device-recommended maximum.
    pub raise_wired_limit: bool,
}

impl MemoryConfig {
    /// Ceiling on MLX active memory in bytes, if any is configured.
    ///
    /// An explicit `limit` wins; otherwise `guard` reserves a share of `physical_memory_bytes`.
    /// Returns `None` when neither applies or the machine's memory is unknown.
    pub fn ceiling_bytes(&self, physical_memory_bytes: Option<u64>) -> Option<u64> {
        if let Some(limit) = self.limit {
            return Some(limit.bytes());
        }
        if self.guard == MemoryGuard::Off {
            return None;
        }
        let total = physical_memory_bytes?;
        Some(total.saturating_sub(total / 100 * self.guard.reserved_percent()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    /// Directory for durable, disk-backed prefix KV entries.
    pub disk_dir: Option<String>,
    pub disk_budget: ByteSize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            disk_dir: None,
            disk_budget: DEFAULT_PREFIX_DISK_BUDGET,
        }
    }
}

/// One local model with every setting resolved from the entry, `[defaults]` and built-in values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    /// Filesystem path or Hugging Face reference for the model.
    pub path: String,
    /// External name exposed to API clients.
    pub name: Option<String>,
    /// Other request names that resolve to this model.
    pub aliases: Vec<String>,
    pub profile: RequestedMlxProfile,
    /// `false` serves one request at a time; otherwise concurrent requests decode together
    /// whenever the model can.
    pub batch: BatchMode,
    pub load: LoadMode,
    /// Unload after this long with no in-flight request; `None` never unloads on idle.
    pub idle_ttl: Option<Duration>,
    /// Never evict this model or unload it on idle. Explicit `/unload` still works.
    pub pinned: bool,
    /// Reasoning tokens the model may spend before `</think>` is forced.
    pub thinking_budget: u32,
    pub kv_cache: KvCacheConfig,
    pub prefix_cache: PrefixCacheConfig,
    /// Maximum prompt tokens to prefill per batch-loop iteration; `None` and `0` keep prefill
    /// synchronous.
    pub prefill_yield_tokens: Option<u32>,
}

impl ModelConfig {
    /// A model at `path` with every setting at its built-in value.
    pub fn with_path<P: Into<String>>(path: P) -> Self {
        Self {
            path: path.into(),
            name: None,
            aliases: Vec::new(),
            profile: RequestedMlxProfile::Auto,
            batch: BatchMode::Auto,
            load: LoadMode::Startup,
            idle_ttl: None,
            pinned: false,
            thinking_budget: higgs_engine::engine::DEFAULT_THINKING_BUDGET,
            kv_cache: KvCacheConfig {
                mode: KvCacheMode::Off,
                bits: DEFAULT_KV_BITS,
                key_bits_override: None,
                value_bits_override: None,
                norm_correction: true,
                adaptive_dense_layers: 0,
                seed: 0,
                mla_latent: false,
            },
            prefix_cache: PrefixCacheConfig::default(),
            prefill_yield_tokens: None,
        }
    }

    /// Validate disk-prefix-store settings independently of engine startup.
    pub fn validate_disk_prefix_store(&self) -> Result<(), String> {
        let Some(dir) = self.prefix_cache.disk_dir.as_deref() else {
            return Ok(());
        };
        if self.prefix_cache.disk_budget.bytes() < 64 * 1024 * 1024 {
            return Err("prefix_cache.disk_budget must be at least 64MB".to_owned());
        }
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("prefix_cache.disk_dir {dir:?} is not creatable: {e}"))?;
        let probe = std::path::Path::new(dir).join(".higgs-write-probe");
        std::fs::write(&probe, [])
            .map_err(|e| format!("prefix_cache.disk_dir {dir:?} is not writable: {e}"))?;
        std::fs::remove_file(probe)
            .map_err(|e| format!("prefix_cache.disk_dir cleanup failed: {e}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub url: String,
    pub format: ApiFormat,
    pub api_key: Option<Secret>,
    pub strip_auth: bool,
    pub stub_count_tokens: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteConfig {
    pub name: Option<String>,
    pub description: Option<String>,
    /// Regular expression matched against the request's `model`.
    pub pattern: Option<String>,
    pub provider: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoRouterConfig {
    pub enabled: bool,
    pub force: bool,
    pub model: String,
    pub timeout: Duration,
}

impl Default for AutoRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            force: false,
            model: DEFAULT_AUTO_ROUTER_MODEL.to_owned(),
            timeout: DEFAULT_AUTO_ROUTER_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingConfig {
    /// Where a request goes when no model or route matches: `higgs` or a provider name.
    pub default_provider: String,
    pub auto: AutoRouterConfig,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_provider: DEFAULT_ROUTING_PROVIDER.to_owned(),
            auto: AutoRouterConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsLogConfig {
    pub enabled: bool,
    pub path: String,
    pub max_size: ByteSize,
    pub max_files: u32,
}

impl Default for MetricsLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: default_metrics_log_path(),
            max_size: DEFAULT_METRICS_LOG_MAX_SIZE,
            max_files: DEFAULT_METRICS_LOG_MAX_FILES,
        }
    }
}

pub fn default_metrics_log_path() -> String {
    directories::BaseDirs::new()
        .map_or_else(
            || std::path::PathBuf::from("/tmp/higgs/logs/metrics.jsonl"),
            |d| d.home_dir().join(".config/higgs/logs/metrics.jsonl"),
        )
        .to_string_lossy()
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservabilityConfig {
    /// How long metrics stay in memory for the dashboard; `None` keeps none.
    pub retention: Option<Duration>,
    pub metrics_log: MetricsLogConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            retention: Some(DEFAULT_RETENTION),
            metrics_log: MetricsLogConfig::default(),
        }
    }
}

/// Behavior toggles for the local engines. Each can also be set with a `HIGGS_*` environment
/// variable, which wins over the file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExperimentalConfig {
    pub mtp: MtpMode,
    pub mtp_draft_tokens: Option<usize>,
    pub mtp_adaptive_draft: bool,
    pub mtp_prompt_lookup: bool,
    pub prompt_lookup: bool,
    pub clear_cache_after_prefill: Option<bool>,
    pub chunked_prefill_threshold: Option<u32>,
    pub chunked_prefill_chunk: Option<u32>,
}

impl ExperimentalConfig {
    /// The engine tuning settings this section carries. Environment variables still win over
    /// them inside the engine.
    pub fn tuning_overrides(&self) -> TuningOverrides {
        TuningOverrides {
            chunked_prefill_threshold: self
                .chunked_prefill_threshold
                .and_then(|value| i32::try_from(value).ok()),
            chunked_prefill_chunk_size: self
                .chunked_prefill_chunk
                .and_then(|value| i32::try_from(value).ok()),
            clear_cache_after_prefill: self.clear_cache_after_prefill,
            enable_mtp: self.mtp.forced(),
            mtp_draft_n_max: self.mtp_draft_tokens,
            mtp_adaptive_draft: Some(self.mtp_adaptive_draft),
            mtp_prompt_lookup: Some(self.mtp_prompt_lookup),
            prompt_lookup: Some(self.prompt_lookup),
        }
    }
}

/// The whole resolved configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HiggsConfig {
    pub server: ServerSection,
    pub memory: MemoryConfig,
    pub models: Vec<ModelConfig>,
    pub providers: BTreeMap<String, ProviderConfig>,
    pub routes: Vec<RouteConfig>,
    pub routing: RoutingConfig,
    pub observability: ObservabilityConfig,
    pub experimental: ExperimentalConfig,
    pub mcp: McpConfig,
    pub web: WebConfig,
    pub notices: Notices,
}

impl HiggsConfig {
    /// The configuration without its load-time notices, for comparing what two files mean.
    #[must_use]
    pub fn without_notices(&self) -> Self {
        Self {
            notices: Notices::default(),
            ..self.clone()
        }
    }
}

// -- Resolution -------------------------------------------------------------

/// Everything resolution needs besides the file: values that sit between the file and the
/// built-in defaults in the precedence order.
#[derive(Debug, Clone, Default)]
pub struct ResolveContext {
    pub environment: Environment,
    /// `--mlx-profile`, which sits above `HIGGS_MLX_PROFILE` and `[defaults]`.
    pub profile_flag: Option<RequestedMlxProfile>,
}

pub fn resolve_model(
    entry: &RawModel,
    defaults: &RawModelSettings,
    profile_override: Option<RequestedMlxProfile>,
) -> ModelConfig {
    let own = &entry.settings;
    let fallback = ModelConfig::with_path(entry.path.clone());
    let profile = own
        .profile
        .map(super::types::MlxProfile::to_requested)
        .or(profile_override)
        .or_else(|| defaults.profile.map(super::types::MlxProfile::to_requested))
        .unwrap_or(fallback.profile);
    let idle_ttl = own
        .idle_ttl
        .or(defaults.idle_ttl)
        .map(super::units::HumanDuration::get)
        .filter(|duration| !duration.is_zero());
    let kv = &own.kv_cache;
    let kv_defaults = &defaults.kv_cache;
    let kv_cache = KvCacheConfig {
        mode: kv
            .mode
            .or(kv_defaults.mode)
            .unwrap_or(fallback.kv_cache.mode),
        bits: kv
            .bits
            .or(kv_defaults.bits)
            .unwrap_or(fallback.kv_cache.bits),
        key_bits_override: kv.key_bits.or(kv_defaults.key_bits),
        value_bits_override: kv.value_bits.or(kv_defaults.value_bits),
        norm_correction: kv
            .norm_correction
            .or(kv_defaults.norm_correction)
            .unwrap_or(fallback.kv_cache.norm_correction),
        adaptive_dense_layers: kv
            .adaptive_dense_layers
            .or(kv_defaults.adaptive_dense_layers)
            .unwrap_or(fallback.kv_cache.adaptive_dense_layers),
        seed: kv
            .seed
            .or(kv_defaults.seed)
            .unwrap_or(fallback.kv_cache.seed),
        mla_latent: kv
            .mla_latent
            .or(kv_defaults.mla_latent)
            .unwrap_or(fallback.kv_cache.mla_latent),
    };
    ModelConfig {
        path: entry.path.clone(),
        name: entry.name.clone(),
        aliases: entry.aliases.clone(),
        profile,
        batch: own.batch.or(defaults.batch).unwrap_or(fallback.batch),
        load: own.load.or(defaults.load).unwrap_or(fallback.load),
        idle_ttl,
        pinned: own.pinned.or(defaults.pinned).unwrap_or(fallback.pinned),
        thinking_budget: own
            .thinking_budget
            .or(defaults.thinking_budget)
            .unwrap_or(fallback.thinking_budget),
        kv_cache,
        prefix_cache: PrefixCacheConfig {
            disk_dir: own
                .prefix_cache
                .disk_dir
                .clone()
                .or_else(|| defaults.prefix_cache.disk_dir.clone()),
            disk_budget: own
                .prefix_cache
                .disk_budget
                .or(defaults.prefix_cache.disk_budget)
                .unwrap_or(fallback.prefix_cache.disk_budget),
        },
        prefill_yield_tokens: own.prefill.yield_tokens.or(defaults.prefill.yield_tokens),
    }
}

fn resolve_mcp_server(raw: &RawMcpServer) -> McpServerConfig {
    let base = McpServerConfig::default();
    McpServerConfig {
        enabled: raw.enabled.unwrap_or(base.enabled),
        transport: raw.transport.clone(),
        command: raw.command.clone(),
        args: raw.args.clone(),
        env: raw.env.clone(),
        cwd: raw.cwd.clone(),
        url: raw.url.clone(),
        headers: raw.headers.clone(),
        timeout: raw
            .timeout
            .map_or(base.timeout, super::units::HumanDuration::get),
    }
}

fn resolve_mcp(raw: &RawMcp) -> McpConfig {
    let base = McpConfig::default();
    McpConfig {
        servers: raw
            .servers
            .iter()
            .map(|(name, server)| (name.clone(), resolve_mcp_server(server)))
            .collect(),
        max_result_bytes: raw
            .max_result_size
            .and_then(|size| usize::try_from(size.bytes()).ok())
            .unwrap_or(base.max_result_bytes),
    }
}

fn resolve_web(raw: &RawWeb) -> WebConfig {
    let base = WebConfig::default();
    WebConfig {
        enabled: raw.enabled.unwrap_or(base.enabled),
        search_provider: raw.search_provider.clone().unwrap_or(base.search_provider),
        searxng_url: raw.searxng_url.clone(),
        allow_private_addresses: raw
            .allow_private_addresses
            .unwrap_or(base.allow_private_addresses),
        expose_as_tools: raw.expose_as_tools.unwrap_or(base.expose_as_tools),
        timeout: raw
            .timeout
            .map_or(base.timeout, super::units::HumanDuration::get),
    }
}

/// Parse an on/off environment value the way the engines do.
pub fn parse_enabled_flag(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

struct EnvToggle<'a> {
    environment: &'a Environment,
    overrides: Vec<EnvOverride>,
}

impl EnvToggle<'_> {
    fn flag(&mut self, variable: &'static str, key: &'static str) -> Option<bool> {
        let raw = self.environment.get(variable)?;
        let parsed = parse_enabled_flag(raw);
        self.overrides.push(EnvOverride {
            variable,
            key,
            applied: parsed.is_some(),
        });
        parsed
    }

    fn positive<T: std::str::FromStr + PartialOrd + Default>(
        &mut self,
        variable: &'static str,
        key: &'static str,
    ) -> Option<T> {
        let raw = self.environment.get(variable)?;
        let parsed = raw
            .trim()
            .parse::<T>()
            .ok()
            .filter(|value| *value > T::default());
        self.overrides.push(EnvOverride {
            variable,
            key,
            applied: parsed.is_some(),
        });
        parsed
    }
}

fn resolve_experimental(
    raw: &super::raw::RawExperimental,
    environment: &Environment,
    overrides: &mut Vec<EnvOverride>,
) -> ExperimentalConfig {
    let mut toggle = EnvToggle {
        environment,
        overrides: Vec::new(),
    };
    let mtp = toggle
        .flag("HIGGS_MTP", "experimental.mtp")
        .map(|on| if on { MtpMode::On } else { MtpMode::Off })
        .or(raw.mtp)
        .unwrap_or_default();
    let config = ExperimentalConfig {
        mtp,
        mtp_draft_tokens: toggle
            .positive::<usize>("HIGGS_MTP_DRAFT_N_MAX", "experimental.mtp_draft_tokens")
            .or(raw.mtp_draft_tokens),
        mtp_adaptive_draft: toggle
            .flag(
                "HIGGS_MTP_ADAPTIVE_DRAFT",
                "experimental.mtp_adaptive_draft",
            )
            .or(raw.mtp_adaptive_draft)
            .unwrap_or(false),
        mtp_prompt_lookup: toggle
            .flag("HIGGS_MTP_PROMPT_LOOKUP", "experimental.mtp_prompt_lookup")
            .or(raw.mtp_prompt_lookup)
            .unwrap_or(false),
        prompt_lookup: toggle
            .flag("HIGGS_PROMPT_LOOKUP", "experimental.prompt_lookup")
            .or(raw.prompt_lookup)
            .unwrap_or(false),
        clear_cache_after_prefill: toggle
            .flag(
                "HIGGS_CLEAR_CACHE_AFTER_PREFILL",
                "experimental.clear_cache_after_prefill",
            )
            .or(raw.clear_cache_after_prefill),
        chunked_prefill_threshold: toggle
            .positive::<u32>(
                "HIGGS_CHUNKED_PREFILL_THRESHOLD",
                "experimental.chunked_prefill.threshold",
            )
            .or(raw.chunked_prefill.threshold),
        chunked_prefill_chunk: toggle
            .positive::<u32>(
                "HIGGS_CHUNKED_PREFILL_CHUNK_SIZE",
                "experimental.chunked_prefill.chunk",
            )
            .or(raw.chunked_prefill.chunk),
    };
    overrides.extend(toggle.overrides);
    config
}

/// Turn a merged, typed file into the resolved configuration.
#[allow(clippy::too_many_lines)]
///
/// `profile_override` is the already-resolved `--mlx-profile` / `HIGGS_MLX_PROFILE` value.
pub fn resolve(raw: &RawConfig, context: &ResolveContext) -> Result<HiggsConfig, String> {
    let mut overrides = Vec::new();

    let profile_env = match context.environment.get("HIGGS_MLX_PROFILE") {
        None => None,
        Some(value) => {
            let parsed = RequestedMlxProfile::from_env_raw(Some(value))?;
            overrides.push(EnvOverride {
                variable: "HIGGS_MLX_PROFILE",
                key: "defaults.profile",
                applied: parsed.is_some(),
            });
            parsed
        }
    };
    let profile_override = context.profile_flag.or(profile_env);

    if let Some(value) = context.environment.get("HIGGS_MLA_LATENT_CACHE") {
        overrides.push(EnvOverride {
            variable: "HIGGS_MLA_LATENT_CACHE",
            key: "models[].kv_cache.mla_latent",
            applied: parse_enabled_flag(value).is_some(),
        });
    }

    let experimental =
        resolve_experimental(&raw.experimental, &context.environment, &mut overrides);

    let server = ServerSection {
        host: raw
            .server
            .host
            .clone()
            .unwrap_or_else(|| DEFAULT_HOST.to_owned()),
        port: raw.server.port.unwrap_or(DEFAULT_PORT),
        max_tokens: raw.server.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        api_key: raw.server.api_key.clone().map(Secret::new),
        rate_limit: raw.server.rate_limit.unwrap_or(0),
        request_timeout: raw
            .server
            .request_timeout
            .map_or(DEFAULT_REQUEST_TIMEOUT, super::units::HumanDuration::get),
        max_body_size: raw.server.max_body_size.unwrap_or(DEFAULT_MAX_BODY_SIZE),
        cors_origins: raw.server.cors_origins.clone(),
    };
    let memory = MemoryConfig {
        guard: raw.memory.guard.unwrap_or_default(),
        limit: raw.memory.limit,
        max_loaded_models: raw.memory.max_loaded_models,
        raise_wired_limit: raw.memory.raise_wired_limit.unwrap_or(false),
    };
    let models = raw
        .models
        .iter()
        .map(|entry| resolve_model(entry, &raw.defaults, profile_override))
        .collect();
    let providers = raw
        .providers
        .iter()
        .map(|(name, provider)| {
            let url = provider
                .url
                .clone()
                .ok_or_else(|| format!("`providers.{name}`: missing field `url`"))?;
            Ok((
                name.clone(),
                ProviderConfig {
                    url,
                    format: provider.format.unwrap_or(ApiFormat::OpenAi),
                    api_key: provider.api_key.clone().map(Secret::new),
                    strip_auth: provider.strip_auth.unwrap_or(false),
                    stub_count_tokens: provider.stub_count_tokens.unwrap_or(false),
                },
            ))
        })
        .collect::<Result<_, String>>()?;
    let routes = raw
        .routes
        .iter()
        .map(|route| RouteConfig {
            name: route.name.clone(),
            description: route.description.clone(),
            pattern: route.pattern.clone(),
            provider: route.provider.clone(),
            model: route.model.clone(),
        })
        .collect();
    let auto_defaults = AutoRouterConfig::default();
    let routing = RoutingConfig {
        default_provider: raw
            .routing
            .default
            .clone()
            .unwrap_or_else(|| DEFAULT_ROUTING_PROVIDER.to_owned()),
        auto: AutoRouterConfig {
            enabled: raw.routing.auto.enabled.unwrap_or(auto_defaults.enabled),
            force: raw.routing.auto.force.unwrap_or(auto_defaults.force),
            model: raw
                .routing
                .auto
                .model
                .clone()
                .unwrap_or(auto_defaults.model),
            timeout: raw
                .routing
                .auto
                .timeout
                .map_or(auto_defaults.timeout, super::units::HumanDuration::get),
        },
    };
    let log_defaults = MetricsLogConfig::default();
    let observability = ObservabilityConfig {
        retention: match raw.observability.retention {
            None | Some(RawRetention::Default) => Some(DEFAULT_RETENTION),
            Some(RawRetention::Disabled) => None,
            Some(RawRetention::Window(window)) => Some(window.get()),
        },
        metrics_log: MetricsLogConfig {
            enabled: raw
                .observability
                .metrics_log
                .enabled
                .unwrap_or(log_defaults.enabled),
            path: raw
                .observability
                .metrics_log
                .path
                .clone()
                .unwrap_or(log_defaults.path),
            max_size: raw
                .observability
                .metrics_log
                .max_size
                .unwrap_or(log_defaults.max_size),
            max_files: raw
                .observability
                .metrics_log
                .max_files
                .unwrap_or(log_defaults.max_files),
        },
    };

    Ok(HiggsConfig {
        server,
        memory,
        models,
        providers,
        routes,
        routing,
        observability,
        experimental,
        mcp: resolve_mcp(&raw.mcp),
        web: resolve_web(&raw.web),
        notices: Notices {
            deprecations: Vec::new(),
            env_overrides: overrides,
            env_only: ENV_ONLY_BEHAVIOR
                .iter()
                .copied()
                .filter(|name| context.environment.get(name).is_some())
                .collect(),
        },
    })
}
