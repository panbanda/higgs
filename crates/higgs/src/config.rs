use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use higgs_engine::{mlx_tuning::RequestedMlxProfile, model_loader};
use higgs_models::turboquant::{KvCacheConfig, KvCacheMode};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "higgs",
    author,
    version,
    about = "Unified AI gateway: serve local MLX models and proxy to remote providers"
)]
pub struct Cli {
    /// Path to config file (default: ~/.config/higgs/config.toml when auto-discovered).
    #[arg(
        short,
        long,
        global = true,
        value_name = "FILE",
        conflicts_with = "profile"
    )]
    pub config: Option<PathBuf>,

    /// Named config profile (resolves to ~/.config/higgs/config.<NAME>.toml).
    #[arg(long, global = true, value_name = "NAME")]
    pub profile: Option<String>,

    /// Enable debug logging.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the server in the foreground.
    Serve(ServeArgs),
    /// Start the server as a background daemon from config or profile.
    Start(StartArgs),
    /// Stop a running daemon.
    Stop(StopArgs),
    /// Open the daemon metrics dashboard.
    Attach,
    /// Create a default config file at ~/.config/higgs/config.toml.
    Init,
    /// Print shell environment variables (for eval).
    Shellenv,
    /// Set shell environment and exec a command.
    Exec {
        /// Command and arguments to execute.
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// Read or modify configuration values.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Validate config, check model paths, and probe providers.
    Doctor(ServeArgs),
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Get a configuration value (dot-separated key).
    Get { key: String },
    /// Set a configuration value (dot-separated key).
    Set { key: String, value: String },
    /// Print the resolved config file path.
    Path,
}

#[derive(Args, Debug, Default)]
pub struct StartArgs {
    /// Legacy simple-mode model flag. Use `higgs serve --model ...` instead.
    #[arg(long = "model", action = clap::ArgAction::Append, hide = true)]
    pub models: Vec<String>,
    /// Legacy serve flag. Use `higgs serve --host ...` instead.
    #[arg(long, hide = true)]
    pub host: Option<String>,
    /// Legacy serve flag. Use `higgs serve --port ...` instead.
    #[arg(long, hide = true)]
    pub port: Option<u16>,
    /// Legacy serve flag. Use `higgs serve --max-tokens ...` instead.
    #[arg(long, hide = true)]
    pub max_tokens: Option<u32>,
    /// Legacy serve flag. Use `higgs serve --api-key ...` instead.
    #[arg(long, hide = true)]
    pub api_key: Option<String>,
    /// Legacy serve flag. Use `higgs serve --rate-limit ...` instead.
    #[arg(long, hide = true)]
    pub rate_limit: Option<u32>,
    /// Legacy serve flag. Use `higgs serve --timeout ...` instead.
    #[arg(long, hide = true)]
    pub timeout: Option<f64>,
    /// Legacy serve flag. Use `higgs serve --mlx-profile ...` instead.
    #[arg(long, hide = true)]
    pub mlx_profile: Option<MlxProfile>,
    /// Legacy serve flag. Use `higgs serve --batch` instead.
    #[arg(long, hide = true)]
    pub batch: bool,
    /// Legacy serve flag. Use `higgs serve --kv-cache ...` instead.
    #[arg(long, hide = true)]
    pub kv_cache: Option<String>,
    /// Legacy serve flag. Use `higgs serve --kv-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-key-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_key_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-value-bits ...` instead.
    #[arg(long, hide = true)]
    pub kv_value_bits: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-no-norm-correction` instead.
    #[arg(long, hide = true)]
    pub kv_no_norm_correction: bool,
    /// Legacy serve flag. Use `higgs serve --kv-adaptive-dense-layers ...` instead.
    #[arg(long, hide = true)]
    pub kv_adaptive_dense_layers: Option<u8>,
    /// Legacy serve flag. Use `higgs serve --kv-seed ...` instead.
    #[arg(long, hide = true)]
    pub kv_seed: Option<u64>,
}

impl StartArgs {
    pub const fn uses_serve_flags(&self) -> bool {
        !self.models.is_empty()
            || self.host.is_some()
            || self.port.is_some()
            || self.max_tokens.is_some()
            || self.api_key.is_some()
            || self.rate_limit.is_some()
            || self.timeout.is_some()
            || self.mlx_profile.is_some()
            || self.batch
            || self.kv_cache.is_some()
            || self.kv_bits.is_some()
            || self.kv_key_bits.is_some()
            || self.kv_value_bits.is_some()
            || self.kv_no_norm_correction
            || self.kv_adaptive_dense_layers.is_some()
            || self.kv_seed.is_some()
    }
}

#[derive(Args, Debug, Default)]
pub struct StopArgs {
    /// Send `SIGKILL` if the daemon does not exit after a graceful timeout.
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser, Debug, Default)]
pub struct ServeArgs {
    /// Path to a model directory or `HuggingFace` model ID. May be repeated.
    #[arg(long = "model", action = clap::ArgAction::Append)]
    pub models: Vec<String>,

    /// Host to bind the server to.
    #[arg(long)]
    pub host: Option<String>,

    /// Port to bind the server to.
    #[arg(long)]
    pub port: Option<u16>,

    /// Default maximum tokens for generation.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// API key for authentication (if unset, no auth required).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Rate limit (requests per minute per client, 0 turns it off).
    #[arg(long)]
    pub rate_limit: Option<u32>,

    /// Default request timeout in seconds.
    #[arg(long)]
    pub timeout: Option<f64>,

    /// MLX tuning profile for local simple-engine models.
    #[arg(long, value_name = "PROFILE")]
    pub mlx_profile: Option<MlxProfile>,

    /// Use batch engine for all models (simple mode only).
    #[arg(long)]
    pub batch: bool,

    /// KV cache mode for simple mode models.
    #[arg(long, value_name = "MODE", value_parser = ["off", "turboquant"])]
    pub kv_cache: Option<String>,

    /// Bit width for `TurboQuant` KV caches.
    #[arg(long)]
    pub kv_bits: Option<u8>,

    /// Override key bit width (default: kv-bits - 1).
    #[arg(long)]
    pub kv_key_bits: Option<u8>,

    /// Override value bit width (default: kv-bits).
    #[arg(long)]
    pub kv_value_bits: Option<u8>,

    /// Disable post-quantization norm correction.
    #[arg(long)]
    pub kv_no_norm_correction: bool,

    /// Number of final layers that stay dense (0 = all TQ).
    #[arg(long)]
    pub kv_adaptive_dense_layers: Option<u8>,

    /// Seed used to generate `TurboQuant` rotation/QJL matrices.
    #[arg(long)]
    pub kv_seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// Unified configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HiggsConfig {
    #[serde(default)]
    pub server: ServerSection,
    #[serde(default)]
    pub local: LocalConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default, rename = "provider")]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub routes: Vec<RouteConfig>,
    #[serde(default)]
    pub default: DefaultRoute,
    #[serde(default)]
    pub auto_router: AutoRouterConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub retention: RetentionConfig,
}

// -- Server section ---------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSection {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub api_key: Option<String>,
    #[serde(default)]
    pub rate_limit: u32,
    #[serde(default = "default_timeout")]
    pub timeout: f64,
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
    /// CORS allow-list of origins. Unset = no CORS headers are sent;
    /// `["*"]` allows any origin (permissive).
    pub cors_origins: Option<Vec<String>>,
}

impl Default for ServerSection {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            max_tokens: default_max_tokens(),
            api_key: None,
            rate_limit: 0,
            timeout: default_timeout(),
            max_body_size: default_max_body_size(),
            cors_origins: None,
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_owned()
}

const fn default_port() -> u16 {
    8000
}

const fn default_max_tokens() -> u32 {
    32768
}

const fn default_timeout() -> f64 {
    300.0
}

const fn default_max_body_size() -> usize {
    10 * 1024 * 1024
}

// -- Local defaults ---------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum MlxProfile {
    /// Use model-aware defaults (`auto` is `balanced` for small/medium, `throughput` for large/huge).
    #[default]
    Auto,
    /// Alias for legacy env `ttft`; bias for lower TTFT and decoding startup.
    Latency,
    /// Balanced tuning target for mixed latency/throughput behavior.
    Balanced,
    /// Throughput-first tuning target.
    Throughput,
}

impl MlxProfile {
    /// Canonical profile string used for rendering and logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }

    /// Convert CLI/TOML profile into the internal requested profile.
    pub const fn to_requested(self) -> RequestedMlxProfile {
        match self {
            Self::Auto => RequestedMlxProfile::Auto,
            Self::Latency => RequestedMlxProfile::Latency,
            Self::Balanced => RequestedMlxProfile::Balanced,
            Self::Throughput => RequestedMlxProfile::Throughput,
        }
    }
}

/// Local config defaults applied before per-model overrides and env/CLI overlays.
///
/// `mlx_profile` is user-facing; `requested_mlx_profile` stores the resolved request
/// after env/CLI fallback and is internal-only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Default MLX tuning profile for local simple-engine models.
    #[serde(default)]
    pub mlx_profile: MlxProfile,
    /// Raise MLX's wired memory limit to the device-recommended maximum.
    #[serde(default)]
    pub raise_wired_limit: bool,
    /// Internal requested profile after resolving precedence (`model` > `--mlx-profile` > `HIGGS_MLX_PROFILE` > local default).
    #[serde(skip, default)]
    pub requested_mlx_profile: RequestedMlxProfile,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            mlx_profile: MlxProfile::Auto,
            raise_wired_limit: false,
            requested_mlx_profile: RequestedMlxProfile::Auto,
        }
    }
}

// -- Model config -----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Filesystem path or Hugging Face reference for the model.
    pub path: String,
    /// Optional external name exposed to API clients.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional per-model MLX tuning override for the simple engine.
    #[serde(default)]
    pub mlx_profile: Option<MlxProfile>,
    /// Enable the separate batch engine for this model.
    #[serde(default)]
    pub batch: bool,
    /// KV-cache storage mode.
    #[serde(default)]
    pub kv_cache: KvCacheMode,
    /// Default `TurboQuant` bit width when per-key/value overrides are absent.
    #[serde(default = "default_kv_bits")]
    pub kv_bits: u8,
    /// Optional override for key-code bit width.
    #[serde(default)]
    pub kv_key_bits: Option<u8>,
    /// Optional override for value-code bit width.
    #[serde(default)]
    pub kv_value_bits: Option<u8>,
    /// Enable norm correction after `TurboQuant` reconstruction.
    #[serde(default = "default_norm_correction")]
    pub kv_norm_correction: bool,
    /// Number of final layers that should remain dense when `TurboQuant` is enabled.
    #[serde(default)]
    pub kv_adaptive_dense_layers: u8,
    /// Seed used by `TurboQuant` setup.
    #[serde(default)]
    pub kv_seed: u64,
}

const fn default_norm_correction() -> bool {
    true
}

const fn default_kv_bits() -> u8 {
    3
}

impl ModelConfig {
    pub const fn kv_cache_config(&self) -> KvCacheConfig {
        KvCacheConfig {
            mode: self.kv_cache,
            bits: self.kv_bits,
            key_bits_override: self.kv_key_bits,
            value_bits_override: self.kv_value_bits,
            norm_correction: self.kv_norm_correction,
            adaptive_dense_layers: self.kv_adaptive_dense_layers,
            seed: self.kv_seed,
        }
    }

    pub const fn requested_mlx_profile(&self, local: &LocalConfig) -> RequestedMlxProfile {
        match self.mlx_profile {
            Some(profile) => profile.to_requested(),
            None => local.requested_mlx_profile,
        }
    }
}

// -- Provider config --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub url: String,
    #[serde(default = "default_api_format")]
    pub format: ApiFormat,
    pub api_key: Option<String>,
    #[serde(default)]
    pub strip_auth: bool,
    #[serde(default)]
    pub stub_count_tokens: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiFormat {
    OpenAi,
    Anthropic,
}

const fn default_api_format() -> ApiFormat {
    ApiFormat::OpenAi
}

// -- Route config -----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    pub name: Option<String>,
    pub description: Option<String>,
    pub pattern: Option<String>,
    pub provider: String,
    pub model: Option<String>,
}

// -- Default route ----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultRoute {
    #[serde(default = "default_provider_name")]
    pub provider: String,
}

impl Default for DefaultRoute {
    fn default() -> Self {
        Self {
            provider: default_provider_name(),
        }
    }
}

fn default_provider_name() -> String {
    "higgs".to_owned()
}

// -- Auto router config -----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRouterConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub force: bool,
    #[serde(default = "default_auto_router_model")]
    pub model: String,
    #[serde(default = "default_auto_router_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for AutoRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            force: false,
            model: default_auto_router_model(),
            timeout_ms: default_auto_router_timeout_ms(),
        }
    }
}

fn default_auto_router_model() -> String {
    "katanemo/Arch-Router-1.5B".to_owned()
}

const fn default_auto_router_timeout_ms() -> u64 {
    2000
}

// -- Logging config ---------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default)]
    pub metrics: MetricsLogConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsLogConfig {
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
    #[serde(default = "default_metrics_log_path")]
    pub path: String,
    #[serde(default = "default_max_size_mb")]
    pub max_size_mb: u64,
    #[serde(default = "default_max_files")]
    pub max_files: u32,
}

impl Default for MetricsLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: default_metrics_log_path(),
            max_size_mb: default_max_size_mb(),
            max_files: default_max_files(),
        }
    }
}

const fn default_metrics_enabled() -> bool {
    true
}

fn default_metrics_log_path() -> String {
    directories::BaseDirs::new()
        .map_or_else(
            || PathBuf::from("/tmp/higgs/logs/metrics.jsonl"),
            |d| d.home_dir().join(".config/higgs/logs/metrics.jsonl"),
        )
        .to_string_lossy()
        .to_string()
}

const fn default_max_size_mb() -> u64 {
    50
}

const fn default_max_files() -> u32 {
    5
}

// -- Retention config -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    #[serde(default = "default_retention_enabled")]
    pub enabled: bool,
    #[serde(default = "default_retention_minutes")]
    pub minutes: u64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            enabled: default_retention_enabled(),
            minutes: default_retention_minutes(),
        }
    }
}

const fn default_retention_enabled() -> bool {
    true
}

const fn default_retention_minutes() -> u64 {
    60
}

fn env_requested_mlx_profile() -> Result<Option<RequestedMlxProfile>, String> {
    // Env-only aliases are accepted here (baseline/default/off, ttft, tps) and
    // intentionally not exposed through CLI/TOML `MlxProfile`.
    RequestedMlxProfile::from_env_raw(std::env::var("HIGGS_MLX_PROFILE").ok().as_deref())
}

fn resolve_local_requested_mlx_profile(
    config_profile: MlxProfile,
    cli_profile: Option<MlxProfile>,
) -> Result<RequestedMlxProfile, String> {
    Ok(cli_profile
        .map(MlxProfile::to_requested)
        .or(env_requested_mlx_profile()?)
        .unwrap_or_else(|| config_profile.to_requested()))
}

fn apply_requested_mlx_profile(
    config: &mut HiggsConfig,
    cli_profile: Option<MlxProfile>,
) -> Result<(), String> {
    config.local.requested_mlx_profile =
        resolve_local_requested_mlx_profile(config.local.mlx_profile, cli_profile)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

/// Returns true if this is "simple mode" -- no config file, models come from CLI.
pub const fn is_simple_mode(cli: &Cli, serve_args: &ServeArgs) -> bool {
    cli.config.is_none() && cli.profile.is_none() && !serve_args.models.is_empty()
}

/// Build a `HiggsConfig` from CLI args only (simple mode, no config file).
pub fn build_simple_config(args: &ServeArgs) -> Result<HiggsConfig, String> {
    let kv_cache = cli_kv_cache_mode(args.kv_cache.as_deref())?;
    let models: Vec<ModelConfig> = args
        .models
        .iter()
        .map(|p| ModelConfig {
            path: p.clone(),
            name: None,
            mlx_profile: None,
            batch: args.batch,
            kv_cache,
            kv_bits: args.kv_bits.unwrap_or(default_kv_bits()),
            kv_key_bits: args.kv_key_bits,
            kv_value_bits: args.kv_value_bits,
            kv_norm_correction: !args.kv_no_norm_correction,
            kv_adaptive_dense_layers: args.kv_adaptive_dense_layers.unwrap_or(0),
            kv_seed: args.kv_seed.unwrap_or_default(),
        })
        .collect();

    let mut config = HiggsConfig {
        models,
        ..HiggsConfig::default()
    };

    // Overlay HIGGS_* env vars, then re-apply explicit CLI args on top
    let figment = Figment::new()
        .merge(Serialized::defaults(&ServerSection::default()))
        .merge(Env::prefixed("HIGGS_"));
    let mut server: ServerSection = figment
        .extract()
        .map_err(|e| format!("env overlay failed: {e}"))?;
    if let Some(ref host) = args.host {
        host.clone_into(&mut server.host);
    }
    if let Some(port) = args.port {
        server.port = port;
    }
    if let Some(max_tokens) = args.max_tokens {
        server.max_tokens = max_tokens;
    }
    if let Some(ref api_key) = args.api_key {
        server.api_key = Some(api_key.clone());
    }
    if let Some(rate_limit) = args.rate_limit {
        server.rate_limit = rate_limit;
    }
    if let Some(timeout) = args.timeout {
        server.timeout = timeout;
    }
    config.server = server;
    apply_requested_mlx_profile(&mut config, args.mlx_profile)?;

    validate_config(&config, true)?;
    ensure_auto_router_model(&mut config);
    Ok(config)
}

/// Load a `HiggsConfig` from a TOML file, with env and CLI overlays (config mode).
pub fn load_config_file(path: &Path, args: Option<&ServeArgs>) -> Result<HiggsConfig, String> {
    let mut figment = Figment::new()
        .merge(Toml::file(path))
        .merge(Env::prefixed("HIGGS_").split("__"));

    // Overlay CLI args on server section
    if let Some(serve_args) = args {
        if let Some(ref host) = serve_args.host {
            figment = figment.merge(Serialized::default("server.host", host));
        }
        if let Some(port) = serve_args.port {
            figment = figment.merge(Serialized::default("server.port", port));
        }
        if let Some(max_tokens) = serve_args.max_tokens {
            figment = figment.merge(Serialized::default("server.max_tokens", max_tokens));
        }
        if let Some(ref api_key) = serve_args.api_key {
            figment = figment.merge(Serialized::default("server.api_key", api_key));
        }
        if let Some(rate_limit) = serve_args.rate_limit {
            figment = figment.merge(Serialized::default("server.rate_limit", rate_limit));
        }
        if let Some(timeout) = serve_args.timeout {
            figment = figment.merge(Serialized::default("server.timeout", timeout));
        }
        // Additional models from CLI in config mode — append to TOML models
        // (figment.merge would replace the entire array, so we extract first,
        // concatenate, then re-merge the combined list)
        if !serve_args.models.is_empty() {
            let kv_cache = cli_kv_cache_mode(serve_args.kv_cache.as_deref())?;
            let extra: Vec<ModelConfig> = serve_args
                .models
                .iter()
                .map(|p| ModelConfig {
                    path: p.clone(),
                    name: None,
                    mlx_profile: None,
                    batch: serve_args.batch,
                    kv_cache,
                    kv_bits: serve_args.kv_bits.unwrap_or(default_kv_bits()),
                    kv_key_bits: serve_args.kv_key_bits,
                    kv_value_bits: serve_args.kv_value_bits,
                    kv_norm_correction: !serve_args.kv_no_norm_correction,
                    kv_adaptive_dense_layers: serve_args.kv_adaptive_dense_layers.unwrap_or(0),
                    kv_seed: serve_args.kv_seed.unwrap_or_default(),
                })
                .collect();
            let mut existing = figment
                .extract_inner::<Option<Vec<ModelConfig>>>("models")
                .map_err(|e| format!("failed to parse models from {}: {e}", path.display()))?
                .unwrap_or_default();
            existing.extend(extra);
            figment = figment.merge(Serialized::default("models", &existing));
        }
    }

    let mut config: HiggsConfig = figment
        .extract()
        .map_err(|e| format!("failed to load config from {}: {e}", path.display()))?;
    apply_requested_mlx_profile(
        &mut config,
        args.and_then(|serve_args| serve_args.mlx_profile),
    )?;

    validate_config(&config, false)?;
    ensure_auto_router_model(&mut config);
    Ok(config)
}

fn validate_config(config: &HiggsConfig, simple_mode: bool) -> Result<(), String> {
    if simple_mode {
        if config.models.is_empty() {
            return Err("at least one --model is required".to_owned());
        }
    } else if config.models.is_empty() && config.providers.is_empty() {
        return Err("config must define at least one [[models]] entry or [provider.*]".to_owned());
    }

    for model in &config.models {
        if model.path.trim().is_empty() {
            return Err("model path must not be empty or whitespace-only".to_owned());
        }
        if let Some(ref name) = model.name {
            if name.trim().is_empty() {
                return Err("model name must not be empty or whitespace-only".to_owned());
            }
        }
        model
            .kv_cache_config()
            .validate()
            .map_err(|err| err.to_string())?;
        if model.batch && model.kv_cache_config().is_turboquant() {
            return Err(format!(
                "TurboQuant is not supported with batch=true for model {}",
                model.path
            ));
        }
        if model.batch
            && let Some(supported) = batch_support_for_model_path(&model.path)?
            && !supported
        {
            return Err(format!(
                "batch=true is only supported for transformer models (llama, mistral, qwen2, qwen3); {} is not supported",
                model.path
            ));
        }
    }

    let mut seen_paths = std::collections::HashSet::new();
    let mut seen_names = std::collections::HashSet::new();
    for model in &config.models {
        if !seen_paths.insert(&model.path) {
            return Err(format!("duplicate model path: {}", model.path));
        }
        if let Some(ref name) = model.name {
            if !seen_names.insert(name) {
                return Err(format!("duplicate model name: {name}"));
            }
        }
    }

    for route in &config.routes {
        if route.provider != "higgs" && !config.providers.contains_key(&route.provider) {
            return Err(format!(
                "route references unknown provider '{}'",
                route.provider
            ));
        }
    }

    if config.default.provider != "higgs"
        && !config.providers.contains_key(&config.default.provider)
    {
        return Err(format!(
            "default provider '{}' not found in providers",
            config.default.provider
        ));
    }

    if !config.server.timeout.is_finite() || config.server.timeout <= 0.0 {
        return Err("timeout must be a finite, positive number".to_owned());
    }

    Ok(())
}

fn supports_batch_model_type(model_type: &str) -> bool {
    matches!(model_type, "qwen2" | "qwen3" | "llama" | "mistral")
}

fn batch_support_for_model_path(model_path: &str) -> Result<Option<bool>, String> {
    match crate::model_resolver::resolve(model_path) {
        Ok(resolved) => resolved_model_supports_batch(&resolved).map(Some),
        Err(err) if batch_support_check_can_be_deferred(model_path, &err) => Ok(None),
        Err(err) => Err(err),
    }
}

fn batch_support_check_can_be_deferred(model_path: &str, err: &str) -> bool {
    crate::model_resolver::is_hf_model_id(model_path)
        && (err
            == format!(
                "model '{model_path}' is not an existing directory and was not found in the HuggingFace cache"
            )
            || (err.starts_with(&format!("could not read HF cache ref for '{model_path}':"))
                && err.contains("No such file or directory")))
}

pub fn resolved_model_supports_batch(model_dir: &Path) -> Result<bool, String> {
    let inspected = model_loader::ModelConfig::from_dir(model_dir)
        .map_err(|e| format!("failed to inspect {}: {e}", model_dir.display()))?;
    Ok(supports_batch_model_type(&inspected.model_type))
}

/// If `auto_router` is enabled, ensure its model is present in `config.models`
/// and that `auto_router.model` is normalized to a model name (not a path).
///
/// This mirrors how `routes[].model` works: always a name reference into the
/// engines map, never a raw filesystem path.
fn ensure_auto_router_model(config: &mut HiggsConfig) {
    if !config.auto_router.enabled || config.auto_router.model.is_empty() {
        return;
    }
    let auto_ref = &config.auto_router.model;

    // Already references a model by name -- nothing to do.
    if config
        .models
        .iter()
        .any(|m| m.name.as_deref() == Some(auto_ref))
    {
        return;
    }

    // References a model by path -- normalize to its name.
    if let Some(model) = config.models.iter_mut().find(|m| m.path == *auto_ref) {
        let name = model.name.get_or_insert_with(|| path_basename(&model.path));
        config.auto_router.model = name.clone();
        return;
    }

    // Not listed at all -- inject a model entry with a derived name.
    let path = config.auto_router.model.clone();
    let name = path_basename(&path);
    config.models.push(ModelConfig {
        path,
        name: Some(name.clone()),
        mlx_profile: None,
        batch: false,
        kv_cache: KvCacheMode::Off,
        kv_bits: default_kv_bits(),
        kv_key_bits: None,
        kv_value_bits: None,
        kv_norm_correction: true,
        kv_adaptive_dense_layers: 0,
        kv_seed: 0,
    });
    config.auto_router.model = name;
}

fn cli_kv_cache_mode(mode: Option<&str>) -> Result<KvCacheMode, String> {
    match mode {
        None | Some("off") => Ok(KvCacheMode::Off),
        Some("turboquant") => Ok(KvCacheMode::Turboquant),
        Some(other) => Err(format!("unknown kv_cache mode '{other}'")),
    }
}

fn path_basename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path)
        .to_owned()
}

/// Returns the default config directory path (~/.config/higgs/).
/// Honors the `HIGGS_CONFIG_DIR` environment variable if set.
pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HIGGS_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    directories::BaseDirs::new().map_or_else(
        || PathBuf::from("/tmp/higgs"),
        |d| d.home_dir().join(".config/higgs"),
    )
}

/// Returns the default config file path (~/.config/higgs/config.toml).
pub fn default_config_path() -> PathBuf {
    config_dir().join("config.toml")
}

/// Write a file with owner-only permissions (0o600 on Unix).
///
/// Used for config files (which may contain provider API keys) and other
/// daemon-private files. The mode is applied at creation; existing files
/// keep their permissions (doctor warns about loose ones).
pub fn write_private_file(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(contents.as_bytes())
}

/// Validates that a profile name is safe for use in file paths.
pub fn validate_profile_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("profile name must not be empty".to_owned());
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!("profile name '{name}' contains invalid characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!(
            "profile name '{name}' must contain only alphanumeric characters, hyphens, or underscores"
        ));
    }
    Ok(())
}

/// Returns the config file path for a named profile (~/.config/higgs/config.<name>.toml).
pub fn profile_config_path(name: &str) -> PathBuf {
    config_dir().join(format!("config.{name}.toml"))
}

/// Returns the PID file path, optionally scoped to a profile.
pub fn pid_path(profile: Option<&str>) -> PathBuf {
    profile.map_or_else(
        || config_dir().join("higgs.pid"),
        |name| config_dir().join(format!("higgs.{name}.pid")),
    )
}

/// Returns the log file path, optionally scoped to a profile.
pub fn log_path(profile: Option<&str>) -> PathBuf {
    profile.map_or_else(
        || config_dir().join("higgs.log"),
        |name| config_dir().join(format!("higgs.{name}.log")),
    )
}

/// Returns the default metrics log path, optionally scoped to a profile.
pub fn default_metrics_log_path_for_profile(profile: &str) -> String {
    directories::BaseDirs::new()
        .map_or_else(
            || PathBuf::from(format!("/tmp/higgs/logs/metrics.{profile}.jsonl")),
            |d| {
                d.home_dir()
                    .join(format!(".config/higgs/logs/metrics.{profile}.jsonl"))
            },
        )
        .to_string_lossy()
        .to_string()
}

// ---------------------------------------------------------------------------
// Legacy compat: ServerConfig alias for existing route handler code
// ---------------------------------------------------------------------------

/// Backward-compatible alias. Route handlers access `state.config.max_tokens`.
pub type ServerConfig = ServerSection;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unsafe_code)]
    fn with_env_var<R>(key: &str, desired_value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::env::var(key).ok();
        match desired_value {
            Some(new_value) => unsafe { std::env::set_var(key, new_value) },
            None => unsafe { std::env::remove_var(key) },
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        match previous.as_deref() {
            Some(previous_value) => unsafe { std::env::set_var(key, previous_value) },
            None => unsafe { std::env::remove_var(key) },
        }

        match result {
            Ok(output) => output,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn test_default_higgs_config() {
        let config = HiggsConfig::default();
        assert!(config.models.is_empty());
        assert!(config.providers.is_empty());
        assert!(config.routes.is_empty());
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.server.max_tokens, 32768);
        assert!((config.server.timeout - 300.0).abs() < f64::EPSILON);
        assert!(config.server.api_key.is_none());
        assert_eq!(config.server.rate_limit, 0);
        assert_eq!(config.local.mlx_profile, MlxProfile::Auto);
        assert_eq!(
            config.local.requested_mlx_profile,
            RequestedMlxProfile::Auto
        );
        assert_eq!(config.default.provider, "higgs");
    }

    #[test]
    #[cfg(unix)]
    fn test_write_private_file_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = std::env::temp_dir().join(format!("higgs-cfg-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        write_private_file(&path, "[server]\n").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_simple_mode_builds_models() {
        let args = ServeArgs {
            models: vec!["org/model-a".to_owned(), "org/model-b".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: true,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        let config = build_simple_config(&args).unwrap();
        assert_eq!(config.models.len(), 2);
        assert!(config.models.iter().all(|m| m.batch));
        assert_eq!(
            config.models.first().map(|m| m.path.as_str()),
            Some("org/model-a")
        );
    }

    #[test]
    fn test_simple_mode_cli_overrides() {
        let args = ServeArgs {
            models: vec!["some/model".to_owned()],
            host: Some("127.0.0.1".to_owned()),
            port: Some(9000),
            max_tokens: Some(1024),
            api_key: Some("sk-test".to_owned()),
            rate_limit: Some(60),
            timeout: Some(60.0),
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        let config = build_simple_config(&args).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
        assert_eq!(config.server.max_tokens, 1024);
        assert_eq!(config.server.api_key, Some("sk-test".to_owned()));
        assert_eq!(config.server.rate_limit, 60);
        assert!((config.server.timeout - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simple_mode_turboquant_cli_flags() {
        let args = ServeArgs {
            models: vec!["some/model".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: false,
            kv_cache: Some("turboquant".to_owned()),
            kv_bits: Some(4),
            kv_seed: Some(99),
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        let config = build_simple_config(&args).unwrap();
        let model = config.models.first().unwrap();
        assert_eq!(model.kv_cache, KvCacheMode::Turboquant);
        assert_eq!(model.kv_bits, 4);
        assert_eq!(model.kv_seed, 99);
    }

    #[test]
    fn test_simple_mode_no_models_rejected() {
        let args = ServeArgs {
            models: vec![],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        assert!(build_simple_config(&args).is_err());
    }

    #[test]
    fn test_simple_mode_empty_model_rejected() {
        let args = ServeArgs {
            models: vec!["  ".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        assert!(build_simple_config(&args).is_err());
    }

    #[test]
    fn test_simple_mode_duplicate_models_rejected() {
        let args = ServeArgs {
            models: vec!["org/model".to_owned(), "org/model".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        assert!(build_simple_config(&args).is_err());
    }

    #[test]
    fn test_simple_mode_rejects_turboquant_batch() {
        let args = ServeArgs {
            models: vec!["org/model".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: true,
            kv_cache: Some("turboquant".to_owned()),
            kv_bits: Some(3),
            kv_seed: Some(0),
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        let error = build_simple_config(&args).unwrap_err();
        assert!(error.contains("TurboQuant"));
    }

    #[test]
    fn test_config_file_parses_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [server]
            host = "127.0.0.1"
            port = 3100

            [[models]]
            path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            batch = true

            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [[routes]]
            pattern = "claude-.*"
            provider = "anthropic"

            [default]
            provider = "anthropic"
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 3100);
        assert_eq!(config.models.len(), 1);
        assert!(config.models.first().is_some_and(|m| m.batch));
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.routes.len(), 1);
        assert_eq!(config.default.provider, "anthropic");
    }

    #[test]
    fn test_config_mode_no_models_no_providers_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [server]
            host = "127.0.0.1"
            "#,
        )
        .unwrap();

        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn test_config_mode_providers_only_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [default]
            provider = "anthropic"
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        assert!(config.models.is_empty());
        assert_eq!(config.providers.len(), 1);
    }

    #[test]
    fn test_route_references_unknown_provider_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "some/model"

            [[routes]]
            pattern = "test"
            provider = "nonexistent"
            "#,
        )
        .unwrap();

        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn test_route_to_higgs_provider_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "some/model"

            [[routes]]
            pattern = "Llama.*"
            provider = "higgs"
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        assert_eq!(config.routes.len(), 1);
    }

    #[test]
    fn test_api_format_deserialization() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [provider.openai]
            url = "https://api.openai.com"
            format = "openai"

            [provider.ollama]
            url = "http://localhost:11434"
            strip_auth = true

            [default]
            provider = "anthropic"
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        assert_eq!(
            config.providers.get("anthropic").map(|p| p.format),
            Some(ApiFormat::Anthropic)
        );
        assert_eq!(
            config.providers.get("openai").map(|p| p.format),
            Some(ApiFormat::OpenAi)
        );
        assert_eq!(
            config.providers.get("ollama").map(|p| p.format),
            Some(ApiFormat::OpenAi)
        );
    }

    #[test]
    fn test_retention_defaults() {
        let config = HiggsConfig::default();
        assert!(config.retention.enabled);
        assert_eq!(config.retention.minutes, 60);
    }

    #[test]
    fn test_auto_router_defaults() {
        let config = HiggsConfig::default();
        assert!(!config.auto_router.enabled);
        assert!(!config.auto_router.force);
        assert_eq!(config.auto_router.timeout_ms, 2000);
        assert_eq!(config.auto_router.model, "katanemo/Arch-Router-1.5B");
    }

    #[test]
    fn auto_router_model_auto_injected_when_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = true
            model = "katanemo/Arch-Router-1.5B"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        assert!(
            config
                .models
                .iter()
                .any(|m| m.path == "katanemo/Arch-Router-1.5B")
        );
    }

    #[test]
    fn auto_router_model_not_duplicated_when_already_listed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "katanemo/Arch-Router-1.5B"

            [auto_router]
            enabled = true
            model = "katanemo/Arch-Router-1.5B"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        let count = config
            .models
            .iter()
            .filter(|m| m.path == "katanemo/Arch-Router-1.5B")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn auto_router_model_not_injected_when_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = false
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        assert!(
            !config
                .models
                .iter()
                .any(|m| m.path == "katanemo/Arch-Router-1.5B")
        );
    }

    #[test]
    fn auto_router_force_deserializes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "some/model"

            [auto_router]
            enabled = true
            force = true
            model = "katanemo/Arch-Router-1.5B"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        assert!(config.auto_router.force);
        assert!(config.auto_router.enabled);
    }

    #[test]
    fn test_logging_defaults() {
        let config = HiggsConfig::default();
        assert!(config.logging.metrics.enabled);
        assert_eq!(config.logging.metrics.max_size_mb, 50);
        assert_eq!(config.logging.metrics.max_files, 5);
        assert!(config.logging.metrics.path.contains("metrics.jsonl"));
    }

    #[test]
    fn test_negative_timeout_rejected() {
        let args = ServeArgs {
            models: vec!["some/model".to_owned()],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: Some(-1.0),
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };
        assert!(build_simple_config(&args).is_err());
    }

    #[test]
    fn test_config_file_cli_overlay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [server]
            host = "127.0.0.1"
            port = 3100

            [[models]]
            path = "some/model"
            "#,
        )
        .unwrap();

        let args = ServeArgs {
            models: vec![],
            host: None,
            port: Some(9000),
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            mlx_profile: None,
            batch: false,
            kv_cache: None,
            kv_bits: None,
            kv_seed: None,
            kv_key_bits: None,
            kv_value_bits: None,
            kv_no_norm_correction: false,
            kv_adaptive_dense_layers: None,
        };

        let config = load_config_file(&path, Some(&args)).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
    }

    #[test]
    fn test_config_file_parses_turboquant_model_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "some/model"
            kv_cache = "turboquant"
            kv_bits = 4
            kv_seed = 123
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        let model = config.models.first().unwrap();
        assert_eq!(model.kv_cache, KvCacheMode::Turboquant);
        assert_eq!(model.kv_bits, 4);
        assert_eq!(model.kv_seed, 123);
    }

    #[test]
    fn test_config_file_parses_local_and_model_mlx_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [local]
            mlx_profile = "auto"

            [[models]]
            path = "some/model"
            mlx_profile = "throughput"
            "#,
        )
        .unwrap();

        let config = load_config_file(&path, None).unwrap();
        assert_eq!(config.local.mlx_profile, MlxProfile::Auto);
        assert_eq!(
            config.local.requested_mlx_profile,
            RequestedMlxProfile::Auto
        );
        let model = config.models.first().unwrap();
        assert_eq!(model.mlx_profile, Some(MlxProfile::Throughput));
        assert_eq!(
            model.requested_mlx_profile(&config.local),
            RequestedMlxProfile::Throughput
        );
    }

    #[test]
    fn test_config_file_rejects_baseline_local_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [local]
            mlx_profile = "baseline"

            [[models]]
            path = "some/model"
            "#,
        )
        .unwrap();

        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn test_config_file_rejects_baseline_model_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [local]
            mlx_profile = "auto"

            [[models]]
            path = "some/model"
            mlx_profile = "baseline"
            "#,
        )
        .unwrap();

        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn test_mlx_profile_parse_rejects_baseline() {
        assert!(MlxProfile::from_str("baseline", true).is_err());
    }

    #[test]
    fn test_mlx_profile_env_overrides_local_default() {
        with_env_var("HIGGS_MLX_PROFILE", Some("throughput"), || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("config.toml");
            std::fs::write(
                &path,
                r#"
                [local]
                mlx_profile = "balanced"

                [[models]]
                path = "some/model"
                "#,
            )
            .unwrap();

            let config = load_config_file(&path, None).unwrap();
            assert_eq!(config.local.mlx_profile, MlxProfile::Balanced);
            assert_eq!(
                config.local.requested_mlx_profile,
                RequestedMlxProfile::Throughput
            );
            assert_eq!(
                config
                    .models
                    .first()
                    .unwrap()
                    .requested_mlx_profile(&config.local),
                RequestedMlxProfile::Throughput
            );
        });
    }

    #[test]
    fn test_mlx_profile_cli_overrides_env_and_config_default() {
        with_env_var("HIGGS_MLX_PROFILE", Some("balanced"), || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("config.toml");
            std::fs::write(
                &path,
                r#"
                [local]
                mlx_profile = "throughput"

                [[models]]
                path = "some/model"
                "#,
            )
            .unwrap();

            let args = ServeArgs {
                mlx_profile: Some(MlxProfile::Latency),
                ..ServeArgs::default()
            };
            let config = load_config_file(&path, Some(&args)).unwrap();
            assert_eq!(config.local.mlx_profile, MlxProfile::Throughput);
            assert_eq!(
                config.local.requested_mlx_profile,
                RequestedMlxProfile::Latency
            );
        });
    }

    #[test]
    fn test_model_mlx_profile_override_beats_cli_env_and_local() {
        with_env_var("HIGGS_MLX_PROFILE", Some("throughput"), || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("config.toml");
            std::fs::write(
                &path,
                r#"
                [local]
                mlx_profile = "balanced"

                [[models]]
                path = "some/model"
                mlx_profile = "latency"
                "#,
            )
            .unwrap();

            let args = ServeArgs {
                mlx_profile: Some(MlxProfile::Auto),
                ..ServeArgs::default()
            };
            let config = load_config_file(&path, Some(&args)).unwrap();
            assert_eq!(
                config
                    .models
                    .first()
                    .unwrap()
                    .requested_mlx_profile(&config.local),
                RequestedMlxProfile::Latency
            );
        });
    }

    #[test]
    fn test_duplicate_model_names_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "org/model-a"
            name = "coder"

            [[models]]
            path = "org/model-b"
            name = "coder"
            "#,
        )
        .unwrap();
        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn test_empty_model_name_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "org/model-a"
            name = "  "
            "#,
        )
        .unwrap();
        assert!(load_config_file(&path, None).is_err());
    }

    #[test]
    fn auto_router_matches_model_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "/some/long/path/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "router"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        // The auto_router model should match by name, not inject a duplicate
        let count = config
            .models
            .iter()
            .filter(|m| m.name.as_deref() == Some("router"))
            .count();
        assert_eq!(count, 1);
        assert_eq!(config.models.len(), 1);
    }

    #[test]
    fn test_model_name_deserialized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "org/model-a"
            name = "coder"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        assert_eq!(
            config.models.first().unwrap().name.as_deref(),
            Some("coder")
        );
    }

    #[test]
    fn test_model_name_optional() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [[models]]
            path = "org/model-a"
            "#,
        )
        .unwrap();
        let config = load_config_file(&path, None).unwrap();
        assert!(config.models.first().unwrap().name.is_none());
    }

    #[test]
    fn test_validate_profile_name_valid() {
        assert!(validate_profile_name("dev").is_ok());
        assert!(validate_profile_name("prod-us").is_ok());
        assert!(validate_profile_name("test_1").is_ok());
    }

    #[test]
    fn test_validate_profile_name_rejects_traversal() {
        assert!(validate_profile_name("../etc").is_err());
        assert!(validate_profile_name("foo/bar").is_err());
        assert!(validate_profile_name("foo\\bar").is_err());
    }

    #[test]
    fn test_validate_profile_name_rejects_empty() {
        assert!(validate_profile_name("").is_err());
    }

    #[test]
    fn test_validate_profile_name_rejects_special_chars() {
        assert!(validate_profile_name("dev.prod").is_err());
        assert!(validate_profile_name("dev prod").is_err());
    }

    #[test]
    fn test_validate_profile_name_rejects_unicode() {
        assert!(validate_profile_name("caf\u{00e9}").is_err());
        assert!(validate_profile_name("\u{1f600}").is_err());
    }

    #[test]
    fn test_validate_profile_name_rejects_control_chars() {
        assert!(validate_profile_name("dev\0prod").is_err());
        assert!(validate_profile_name("dev\nprod").is_err());
        assert!(validate_profile_name("dev\tprod").is_err());
    }

    #[test]
    fn test_validate_profile_name_rejects_shell_metacharacters() {
        assert!(validate_profile_name("dev;rm -rf").is_err());
        assert!(validate_profile_name("$(whoami)").is_err());
        assert!(validate_profile_name("dev|prod").is_err());
        assert!(validate_profile_name("dev&prod").is_err());
    }

    #[test]
    fn test_profile_config_path_format() {
        let path = profile_config_path("dev");
        let file_name = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(file_name, "config.dev.toml");
        assert!(path.parent().unwrap().ends_with(".config/higgs") || path.starts_with("/tmp"));
    }

    #[test]
    fn test_profile_config_path_different_names() {
        let dev = profile_config_path("dev");
        let prod = profile_config_path("prod");
        assert_ne!(dev, prod);
        assert!(dev.to_str().unwrap().contains("config.dev.toml"));
        assert!(prod.to_str().unwrap().contains("config.prod.toml"));
    }

    #[test]
    fn test_profile_config_path_shares_parent_with_default() {
        let profile_path = profile_config_path("test");
        let default_path = default_config_path();
        assert_eq!(profile_path.parent(), default_path.parent());
    }

    #[test]
    fn test_default_metrics_log_path_for_profile_contains_profile_name() {
        let path = default_metrics_log_path_for_profile("dev");
        assert!(path.contains("metrics.dev.jsonl"), "path was: {path}");
    }

    #[test]
    fn test_default_metrics_log_path_for_profile_different_names() {
        let dev = default_metrics_log_path_for_profile("dev");
        let prod = default_metrics_log_path_for_profile("prod");
        assert_ne!(dev, prod);
        assert!(dev.contains("metrics.dev.jsonl"));
        assert!(prod.contains("metrics.prod.jsonl"));
    }

    #[test]
    fn test_pid_path_default_vs_profile() {
        let default = pid_path(None);
        let profiled = pid_path(Some("dev"));
        assert_ne!(default, profiled);
        assert!(default.to_str().unwrap().contains("higgs.pid"));
        assert!(profiled.to_str().unwrap().contains("higgs.dev.pid"));
    }

    #[test]
    fn test_log_path_default_vs_profile() {
        let default = log_path(None);
        let profiled = log_path(Some("dev"));
        assert_ne!(default, profiled);
        assert!(default.to_str().unwrap().contains("higgs.log"));
        assert!(profiled.to_str().unwrap().contains("higgs.dev.log"));
    }
}
