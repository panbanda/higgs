//! `[web]` configuration: search and fetch endpoints, off by default.

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const PROVIDER_SEARXNG: &str = "searxng";
pub const PROVIDER_DUCKDUCKGO: &str = "duckduckgo";

const MAX_TIMEOUT_SECONDS: f64 = 300.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    /// Serve `/v1/web/search` and `/v1/web/fetch`. When false both answer 404.
    #[serde(default)]
    pub enabled: bool,
    /// `"duckduckgo"` (keyless, scrapes the HTML endpoint) or `"searxng"`.
    #[serde(default = "default_search_provider")]
    pub search_provider: String,
    /// Base URL of a self-hosted `SearXNG` instance with the JSON format
    /// enabled. Required for the `searxng` provider.
    #[serde(default)]
    pub searxng_url: Option<String>,
    /// Allow `/v1/web/fetch` to reach loopback, private and link-local
    /// addresses. Leave off unless every API caller is trusted.
    #[serde(default)]
    pub allow_private_addresses: bool,
    /// Offer `web_search` and `web_fetch` as tools under the built-in `higgs`
    /// server in `/v1/mcp/tools` and the server-side tool loop.
    #[serde(default)]
    pub expose_as_tools: bool,
    /// Bound on one search or one fetch, redirects included.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            search_provider: default_search_provider(),
            searxng_url: None,
            allow_private_addresses: false,
            expose_as_tools: false,
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

fn default_search_provider() -> String {
    PROVIDER_DUCKDUCKGO.to_owned()
}

const fn default_timeout_seconds() -> f64 {
    15.0
}

impl WebConfig {
    pub fn timeout(&self) -> Option<Duration> {
        (self.timeout_seconds.is_finite()
            && self.timeout_seconds > 0.0
            && self.timeout_seconds <= MAX_TIMEOUT_SECONDS)
            .then(|| Duration::from_secs_f64(self.timeout_seconds))
    }

    pub fn searxng_base(&self) -> Result<reqwest::Url, String> {
        let raw = self
            .searxng_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "search_provider \"searxng\" requires searxng_url".to_owned())?;
        let url = reqwest::Url::parse(raw)
            .map_err(|error| format!("invalid searxng_url '{raw}': {error}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(format!(
                "searxng_url '{raw}' must be an http(s) URL with a host"
            ));
        }
        Ok(url)
    }

    /// Settings that make the configuration unusable.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.timeout().is_none() {
            problems.push(format!(
                "web.timeout_seconds must be a finite number in (0, {MAX_TIMEOUT_SECONDS}], got {}",
                self.timeout_seconds
            ));
        }
        match self.search_provider.as_str() {
            PROVIDER_SEARXNG => {
                if let Err(message) = self.searxng_base() {
                    problems.push(format!("web: {message}"));
                }
            }
            PROVIDER_DUCKDUCKGO => {}
            other => problems.push(format!(
                "web.search_provider '{other}' is not supported (expected \"{PROVIDER_DUCKDUCKGO}\" or \"{PROVIDER_SEARXNG}\")"
            )),
        }
        problems
    }

    /// Settings that work but are probably not what was meant.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.expose_as_tools && !self.enabled {
            warnings.push("web.expose_as_tools has no effect while web.enabled = false".to_owned());
        }
        if self.searxng_url.is_some() && self.search_provider != PROVIDER_SEARXNG {
            warnings.push(format!(
                "web.searxng_url is ignored because search_provider is \"{}\"",
                self.search_provider
            ));
        }
        if self.allow_private_addresses && self.enabled {
            warnings.push(
                "web.allow_private_addresses = true lets API callers fetch loopback, private and link-local addresses (SSRF protection is off)"
                    .to_owned(),
            );
        }
        warnings
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use figment::providers::Format;

    use super::*;

    fn parse(toml: &str) -> WebConfig {
        figment::Figment::new()
            .merge(figment::providers::Toml::string(toml))
            .extract()
            .unwrap()
    }

    #[test]
    fn defaults_are_off_and_safe() {
        let config = WebConfig::default();
        assert!(!config.enabled);
        assert!(!config.allow_private_addresses);
        assert!(!config.expose_as_tools);
        assert_eq!(config.search_provider, "duckduckgo");
        assert!(config.problems().is_empty());
        assert!(config.warnings().is_empty());
        assert_eq!(parse("").timeout(), Some(Duration::from_secs(15)));
    }

    #[test]
    fn every_field_parses() {
        let config = parse(
            r#"
            enabled = true
            search_provider = "searxng"
            searxng_url = "http://localhost:8888"
            allow_private_addresses = true
            expose_as_tools = true
            timeout_seconds = 5.5
            "#,
        );
        assert!(config.enabled && config.allow_private_addresses && config.expose_as_tools);
        assert_eq!(
            config.searxng_base().unwrap().as_str(),
            "http://localhost:8888/"
        );
        assert_eq!(config.timeout(), Some(Duration::from_millis(5500)));
        assert!(config.problems().is_empty());
    }

    #[test]
    fn problems_cover_provider_url_and_timeout() {
        let unknown = WebConfig {
            search_provider: "bing".to_owned(),
            ..WebConfig::default()
        };
        assert!(unknown.problems()[0].contains("not supported"));

        let missing_url = WebConfig {
            search_provider: "searxng".to_owned(),
            ..WebConfig::default()
        };
        assert!(missing_url.problems()[0].contains("requires searxng_url"));

        for bad in ["not a url", "ftp://host/x", "  "] {
            let config = WebConfig {
                search_provider: "searxng".to_owned(),
                searxng_url: Some(bad.to_owned()),
                ..WebConfig::default()
            };
            assert_eq!(config.problems().len(), 1, "{bad}");
        }

        for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, 301.0] {
            let config = WebConfig {
                timeout_seconds: timeout,
                ..WebConfig::default()
            };
            assert_eq!(config.problems().len(), 1, "{timeout}");
        }
    }

    #[test]
    fn warnings_cover_inert_and_risky_settings() {
        let inert = WebConfig {
            expose_as_tools: true,
            searxng_url: Some("http://x".to_owned()),
            ..WebConfig::default()
        };
        assert_eq!(inert.warnings().len(), 2);

        let risky = WebConfig {
            enabled: true,
            allow_private_addresses: true,
            ..WebConfig::default()
        };
        assert!(risky.warnings()[0].contains("SSRF"));
    }
}
