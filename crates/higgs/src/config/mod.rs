//! Configuration: the file schema, its deprecated spellings, and the resolved settings the rest
//! of the server reads.
//!
//! Loading goes file -> rewrite of deprecated keys -> strict key check -> includes and
//! environment overlay -> `${VAR}` expansion -> typed structs ([`raw`]) -> flags -> resolved
//! settings ([`settings`]).

pub mod cli;
pub mod document;
pub mod env_ref;
pub mod environment;
pub mod explain;
pub mod legacy;
pub mod load;
pub mod migrate;
pub mod raw;
pub mod schema;
pub mod settings;
pub mod suggest;
pub mod types;
pub mod units;
pub mod validate;

#[cfg(test)]
mod fixture_tests;
#[cfg(test)]
mod flag_tests;
#[cfg(test)]
mod load_tests;

use std::path::{Path, PathBuf};

pub use cli::{Cli, ClientArgs, Commands, ConfigAction, ServeArgs, StartArgs, StopArgs};
pub use document::{ConfigError, Diagnostic};
pub use environment::Environment;
pub use settings::{
    AutoRouterConfig, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_ROUTING_PROVIDER, EnvOverride,
    ExperimentalConfig, HiggsConfig, MemoryConfig, MetricsLogConfig, ModelConfig, Notices,
    ObservabilityConfig, PrefixCacheConfig, ProviderConfig, RouteConfig, RoutingConfig,
    ServerConfig, ServerSection,
};
pub use types::{ApiFormat, BatchMode, LoadMode, MemoryGuard, MlxProfile, MtpMode, Secret};

/// Returns true if this is "simple mode" -- no config file, models come from CLI.
pub const fn is_simple_mode(cli: &Cli, serve_args: &ServeArgs) -> bool {
    cli.config.is_none() && cli.profile.is_none() && !serve_args.models.is_empty()
}

/// Build a `HiggsConfig` from CLI args only (simple mode, no config file).
pub fn build_simple_config(args: &ServeArgs) -> Result<HiggsConfig, String> {
    load::load_simple(args, &Environment::process())
        .map(|loaded| loaded.config)
        .map_err(Into::into)
}

/// Load a `HiggsConfig` from a TOML file, with env and CLI overlays (config mode).
pub fn load_config_file(path: &Path, args: Option<&ServeArgs>) -> Result<HiggsConfig, String> {
    load_config_file_in(path, args, &Environment::process())
}

/// [`load_config_file`] against a given environment instead of the process's.
pub fn load_config_file_in(
    path: &Path,
    args: Option<&ServeArgs>,
    environment: &Environment,
) -> Result<HiggsConfig, String> {
    let options = load::LoadOptions {
        flags: args,
        environment: environment.clone(),
    };
    load::load(path, &options)
        .map(|loaded| loaded.config)
        .map_err(Into::into)
}

/// Check a config file's text the way loading does, without environment overlays or semantic
/// validation, so users can edit the bootstrap config one field at a time.
pub fn validate_config_contents(contents: &str, source: &Path) -> Result<(), String> {
    load::check_contents(contents, source).map_err(Into::into)
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
