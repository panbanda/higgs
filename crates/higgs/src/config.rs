use clap::Parser;
use figment::{
    Figment,
    providers::{Env, Serialized},
};
use serde::{Deserialize, Serialize};

/// CLI arguments (used as the highest-priority figment layer).
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Higgs - OpenAI and Anthropic-compatible inference server for Apple Silicon"
)]
struct CliArgs {
    /// Path to a model directory or `HuggingFace` model ID. May be repeated to serve multiple models.
    #[arg(long = "model", action = clap::ArgAction::Append)]
    models: Vec<String>,

    /// Host to bind the server to.
    #[arg(long)]
    host: Option<String>,

    /// Port to bind the server to.
    #[arg(long)]
    port: Option<u16>,

    /// Default maximum tokens for generation.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// API key for authentication (if unset, no auth required).
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit (requests per minute per client, 0 = disabled).
    #[arg(long)]
    rate_limit: Option<u32>,

    /// Default request timeout in seconds.
    #[arg(long)]
    timeout: Option<f64>,

    /// Use batch engine for interleaved concurrent request processing.
    #[arg(long)]
    batch: bool,
}

/// Resolved server configuration.
///
/// Layered resolution order (later wins):
/// 1. Built-in defaults
/// 2. `HIGGS_*` environment variables
/// 3. CLI arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub models: Vec<String>,
    pub host: String,
    pub port: u16,
    pub max_tokens: u32,
    pub api_key: Option<String>,
    pub rate_limit: u32,
    pub timeout: f64,
    pub batch: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            models: vec![],
            host: "0.0.0.0".to_owned(),
            port: 8000,
            max_tokens: 32768,
            api_key: None,
            rate_limit: 0,
            timeout: 300.0,
            batch: false,
        }
    }
}

impl ServerConfig {
    /// Load configuration from defaults, environment, and CLI args.
    pub fn load() -> Result<Self, Box<figment::Error>> {
        let cli = CliArgs::parse();

        let mut figment = Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Env::prefixed("HIGGS_"));

        // Overlay CLI args (only non-empty values)
        if !cli.models.is_empty() {
            figment = figment.merge(Serialized::default("models", &cli.models));
        }
        if let Some(ref host) = cli.host {
            figment = figment.merge(Serialized::default("host", host));
        }
        if let Some(port) = cli.port {
            figment = figment.merge(Serialized::default("port", port));
        }
        if let Some(max_tokens) = cli.max_tokens {
            figment = figment.merge(Serialized::default("max_tokens", max_tokens));
        }
        if let Some(ref api_key) = cli.api_key {
            figment = figment.merge(Serialized::default("api_key", api_key));
        }
        if let Some(rate_limit) = cli.rate_limit {
            figment = figment.merge(Serialized::default("rate_limit", rate_limit));
        }
        if let Some(timeout) = cli.timeout {
            figment = figment.merge(Serialized::default("timeout", timeout));
        }
        if cli.batch {
            figment = figment.merge(Serialized::default("batch", true));
        }

        let config: Self = figment.extract().map_err(Box::new)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Box<figment::Error>> {
        if self.models.is_empty() {
            return Err(Box::new(figment::Error::from(
                "at least one --model is required".to_owned(),
            )));
        }
        for model in &self.models {
            if model.trim().is_empty() {
                return Err(Box::new(figment::Error::from(
                    "model path must not be empty or whitespace-only".to_owned(),
                )));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for model in &self.models {
            if !seen.insert(model) {
                return Err(Box::new(figment::Error::from(format!(
                    "duplicate model path: {model}"
                ))));
            }
        }
        if self.timeout < 0.0 {
            return Err(Box::new(figment::Error::from(
                "timeout must not be negative".to_owned(),
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Returns a figment with `ServerConfig` defaults already merged.
    fn test_figment() -> Figment {
        Figment::new().merge(Serialized::defaults(ServerConfig::default()))
    }

    /// Creates a `ServerConfig` with a single key overridden.
    fn config_with(key: &str, value: impl Serialize) -> ServerConfig {
        test_figment()
            .merge(Serialized::default(key, value))
            .extract()
            .unwrap()
    }

    #[test]
    fn test_default_config_has_reasonable_values() {
        let config = ServerConfig::default();
        assert!(config.models.is_empty());
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8000);
        assert_eq!(config.max_tokens, 32768);
        assert!(config.api_key.is_none());
        assert_eq!(config.rate_limit, 0);
        assert!((config.timeout - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_layered_override() {
        // Verifies that later figment layers override earlier ones,
        // which is the same mechanism used by Env::prefixed("HIGGS_").
        let config: ServerConfig = test_figment()
            .merge(Serialized::default("port", 9000_u16))
            .merge(Serialized::default("host", "127.0.0.1"))
            .extract()
            .unwrap();
        assert_eq!(config.port, 9000);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_cli_overrides_defaults() {
        let config: ServerConfig = test_figment()
            .merge(Serialized::default("max_tokens", 1024_u32))
            .merge(Serialized::default("timeout", 60.0_f64))
            .extract()
            .unwrap();
        assert_eq!(config.max_tokens, 1024);
        assert!((config.timeout - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_api_key_can_be_set() {
        let config = config_with("api_key", "sk-test-123");
        assert_eq!(config.api_key, Some("sk-test-123".to_owned()));
    }

    #[test]
    fn test_port_zero_is_accepted() {
        let config = config_with("port", 0_u16);
        assert_eq!(config.port, 0);
    }

    #[test]
    fn test_max_tokens_zero() {
        let config = config_with("max_tokens", 0_u32);
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_timeout_zero() {
        let config = config_with("timeout", 0.0_f64);
        assert!(config.timeout.abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_models_rejected() {
        let config = ServerConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_single_model_accepted() {
        let config = config_with("models", vec!["some/model"]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_multiple_models_accepted() {
        let config = config_with("models", vec!["model/a", "model/b"]);
        assert_eq!(config.models.len(), 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_negative_timeout_rejected() {
        let result: Result<ServerConfig, _> = test_figment()
            .merge(Serialized::default("models", vec!["some/model"]))
            .merge(Serialized::default("timeout", -1.0_f64))
            .extract();
        let config = result.unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_rate_limit_zero_means_disabled() {
        let config = ServerConfig::default();
        assert_eq!(config.rate_limit, 0);
    }

    #[test]
    fn test_rate_limit_nonzero_means_enabled() {
        let config = config_with("rate_limit", 60_u32);
        assert_eq!(config.rate_limit, 60);
    }

    #[test]
    fn test_api_key_empty_string_vs_none() {
        let config = ServerConfig::default();
        assert!(config.api_key.is_none());

        let config_with_empty = config_with("api_key", "");
        assert_eq!(config_with_empty.api_key, Some(String::new()));
    }

    #[test]
    fn test_default_timeout_is_300() {
        let config = ServerConfig::default();
        assert!((config.timeout - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_rate_limit_is_zero() {
        let config = ServerConfig::default();
        assert_eq!(config.rate_limit, 0);
    }

    #[test]
    fn test_empty_string_model_rejected() {
        let config = config_with("models", vec![""]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_whitespace_only_model_rejected() {
        let config = config_with("models", vec!["  "]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_duplicate_models_rejected() {
        let config = config_with("models", vec!["org/model", "org/model"]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_duplicate_models_different_paths_same_name_rejected() {
        // org-a/llama and org-b/llama both resolve to "llama" in the HashMap key
        let config = config_with("models", vec!["org-a/llama", "org-b/llama"]);
        // These are different strings, so path-level dedup doesn't catch this --
        // but the name collision is detected at load time in main.rs, not here.
        // This test documents that path-level dedup in validate() only catches
        // exact string duplicates.
        assert!(config.validate().is_ok());
    }
}
