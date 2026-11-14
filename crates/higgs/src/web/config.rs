//! Resolved `[web]` configuration: search and fetch endpoints, off by default.

use std::time::Duration;

use crate::config::units::HumanDuration;

pub const PROVIDER_SEARXNG: &str = "searxng";
pub const PROVIDER_DUCKDUCKGO: &str = "duckduckgo";

const MAX_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebConfig {
    /// Serve `/v1/web/search` and `/v1/web/fetch`. When false both answer 404.
    pub enabled: bool,
    /// `"duckduckgo"` (keyless, scrapes the HTML endpoint) or `"searxng"`.
    pub search_provider: String,
    /// Base URL of a self-hosted `SearXNG` instance with the JSON format
    /// enabled. Required for the `searxng` provider.
    pub searxng_url: Option<String>,
    /// Allow `/v1/web/fetch` to reach loopback, private and link-local
    /// addresses. Leave off unless every API caller is trusted.
    pub allow_private_addresses: bool,
    /// Offer `web_search` and `web_fetch` as tools under the built-in `higgs`
    /// server in `/v1/mcp/tools` and the server-side tool loop.
    pub expose_as_tools: bool,
    /// Bound on one search or one fetch, redirects included.
    pub timeout: Duration,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            search_provider: default_search_provider(),
            searxng_url: None,
            allow_private_addresses: false,
            expose_as_tools: false,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

fn default_search_provider() -> String {
    PROVIDER_DUCKDUCKGO.to_owned()
}

impl WebConfig {
    /// The timeout, if it is within the range the fetcher can honor.
    pub fn valid_timeout(&self) -> Option<Duration> {
        (!self.timeout.is_zero() && self.timeout <= MAX_TIMEOUT).then_some(self.timeout)
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
        if self.valid_timeout().is_none() {
            problems.push(format!(
                "web.timeout must be longer than zero and at most {}, got {}",
                HumanDuration::new(MAX_TIMEOUT),
                HumanDuration::new(self.timeout)
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
    use super::*;

    #[test]
    fn defaults_are_off_and_safe() {
        let config = WebConfig::default();
        assert!(!config.enabled);
        assert!(!config.allow_private_addresses);
        assert!(!config.expose_as_tools);
        assert_eq!(config.search_provider, "duckduckgo");
        assert!(config.problems().is_empty());
        assert!(config.warnings().is_empty());
        assert_eq!(config.valid_timeout(), Some(Duration::from_secs(15)));
    }

    #[test]
    fn a_fully_set_config_is_valid() {
        let config = WebConfig {
            enabled: true,
            search_provider: "searxng".to_owned(),
            searxng_url: Some("http://localhost:8888".to_owned()),
            allow_private_addresses: true,
            expose_as_tools: true,
            timeout: Duration::from_millis(5500),
        };
        assert_eq!(
            config.searxng_base().unwrap().as_str(),
            "http://localhost:8888/"
        );
        assert_eq!(config.valid_timeout(), Some(Duration::from_millis(5500)));
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

        for timeout in [Duration::ZERO, Duration::from_secs(301)] {
            let config = WebConfig {
                timeout,
                ..WebConfig::default()
            };
            assert_eq!(config.problems().len(), 1, "{timeout:?}");
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
