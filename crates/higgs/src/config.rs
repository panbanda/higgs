use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
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
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

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
    /// Start the server as a background daemon.
    Start(ServeArgs),
    /// Stop a running daemon.
    Stop,
    /// Attach TUI to a running daemon.
    Attach,
    /// Create a default config file at ~/.config/higgs/config.toml.
    Init,
    /// Print shell environment variables (for eval).
    Shellenv,
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

#[derive(Parser, Debug)]
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

    /// Rate limit (requests per minute per client, 0 = disabled).
    #[arg(long)]
    pub rate_limit: Option<u32>,

    /// Default request timeout in seconds.
    #[arg(long)]
    pub timeout: Option<f64>,

    /// Use batch engine for all models (simple mode only).
    #[arg(long)]
    pub batch: bool,
}

// ---------------------------------------------------------------------------
// Unified configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HiggsConfig {
    #[serde(default)]
    pub server: ServerSection,
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
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_owned()
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

// -- Model config -----------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    #[serde(default)]
    pub batch: bool,
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
    #[serde(default)]
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
            enabled: false,
            path: default_metrics_log_path(),
            max_size_mb: default_max_size_mb(),
            max_files: default_max_files(),
        }
    }
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

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

/// Returns true if this is "simple mode" -- no config file, models come from CLI.
pub const fn is_simple_mode(cli: &Cli, serve_args: &ServeArgs) -> bool {
    cli.config.is_none() && !serve_args.models.is_empty()
}

/// Build a `HiggsConfig` from CLI args only (simple mode, no config file).
pub fn build_simple_config(args: &ServeArgs) -> Result<HiggsConfig, String> {
    let models: Vec<ModelConfig> = args
        .models
        .iter()
        .map(|p| ModelConfig {
            path: p.clone(),
            batch: args.batch,
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
        // Additional models from CLI in config mode
        if !serve_args.models.is_empty() {
            let extra: Vec<ModelConfig> = serve_args
                .models
                .iter()
                .map(|p| ModelConfig {
                    path: p.clone(),
                    batch: serve_args.batch,
                })
                .collect();
            figment = figment.merge(Serialized::default("models", &extra));
        }
    }

    let mut config: HiggsConfig = figment
        .extract()
        .map_err(|e| format!("failed to load config from {}: {e}", path.display()))?;

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
    }

    let mut seen = std::collections::HashSet::new();
    for model in &config.models {
        if !seen.insert(&model.path) {
            return Err(format!("duplicate model path: {}", model.path));
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

    if !config.server.timeout.is_finite() || config.server.timeout < 0.0 {
        return Err("timeout must be a finite, non-negative number".to_owned());
    }

    Ok(())
}

/// If `auto_router` is enabled, ensure its model is present in `config.models`.
fn ensure_auto_router_model(config: &mut HiggsConfig) {
    if !config.auto_router.enabled || config.auto_router.model.is_empty() {
        return;
    }
    let already_listed = config
        .models
        .iter()
        .any(|m| m.path == config.auto_router.model);
    if !already_listed {
        config.models.push(ModelConfig {
            path: config.auto_router.model.clone(),
            batch: false,
        });
    }
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

/// Returns the PID file path (~/.config/higgs/higgs.pid).
pub fn pid_path() -> PathBuf {
    config_dir().join("higgs.pid")
}

/// Returns the log file path (~/.config/higgs/higgs.log).
pub fn log_path() -> PathBuf {
    config_dir().join("higgs.log")
}

// ---------------------------------------------------------------------------
// Legacy compat: ServerConfig alias for existing route handler code
// ---------------------------------------------------------------------------

/// Backward-compatible alias. Route handlers access `state.config.max_tokens`.
pub type ServerConfig = ServerSection;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_default_higgs_config() {
        let config = HiggsConfig::default();
        assert!(config.models.is_empty());
        assert!(config.providers.is_empty());
        assert!(config.routes.is_empty());
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.server.max_tokens, 32768);
        assert!((config.server.timeout - 300.0).abs() < f64::EPSILON);
        assert!(config.server.api_key.is_none());
        assert_eq!(config.server.rate_limit, 0);
        assert_eq!(config.default.provider, "higgs");
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
            batch: true,
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
            batch: false,
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
    fn test_simple_mode_no_models_rejected() {
        let args = ServeArgs {
            models: vec![],
            host: None,
            port: None,
            max_tokens: None,
            api_key: None,
            rate_limit: None,
            timeout: None,
            batch: false,
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
            batch: false,
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
            batch: false,
        };
        assert!(build_simple_config(&args).is_err());
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
        assert!(!config.logging.metrics.enabled);
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
            batch: false,
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
            batch: false,
        };

        let config = load_config_file(&path, Some(&args)).unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 9000);
    }
}
