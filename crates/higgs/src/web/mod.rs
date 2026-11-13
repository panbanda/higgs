//! Web search and page fetch, as HTTP endpoints and as tools a model can call.

pub mod config;
pub mod fetch;
pub mod html;
pub mod search;
pub mod ssrf;

use std::fmt::Write;

use serde_json::{Value, json};

use crate::mcp::config::BUILTIN_SERVER_NAME;
use crate::mcp::manager::{ToolCallError, ToolInfo, ToolOutput};

use self::config::{PROVIDER_SEARXNG, WebConfig};
use self::fetch::{DEFAULT_MAX_CHARS, FetchError, FetchedPage, MAX_CHARS_LIMIT, WebFetcher};
use self::search::{SearchError, SearchProvider, SearchResult, WebSearcher};
use self::ssrf::AddressPolicy;

pub const SEARCH_TOOL: &str = "web_search";
pub const FETCH_TOOL: &str = "web_fetch";

#[derive(Debug, thiserror::Error)]
pub enum WebError {
    #[error("web search and fetch are disabled; set web.enabled = true in the config")]
    Disabled,
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error(transparent)]
    Fetch(#[from] FetchError),
}

pub struct WebService {
    enabled: bool,
    expose_as_tools: bool,
    searcher: Option<WebSearcher>,
    fetcher: Option<WebFetcher>,
}

impl WebService {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            expose_as_tools: false,
            searcher: None,
            fetcher: None,
        }
    }

    /// Build the service from validated configuration. Nothing is contacted
    /// until a request arrives.
    pub fn new(config: &WebConfig) -> Result<Self, String> {
        if !config.enabled {
            return Ok(Self::disabled());
        }
        let timeout = config
            .valid_timeout()
            .ok_or_else(|| "web.timeout is out of range".to_owned())?;
        let provider = if config.search_provider == PROVIDER_SEARXNG {
            SearchProvider::SearxNg {
                base: config.searxng_base()?,
            }
        } else {
            SearchProvider::duckduckgo()
        };
        Ok(Self {
            enabled: true,
            expose_as_tools: config.expose_as_tools,
            searcher: Some(WebSearcher::new(provider, timeout)?),
            fetcher: Some(WebFetcher::new(
                AddressPolicy {
                    allow_private: config.allow_private_addresses,
                    #[cfg(test)]
                    exempt: Vec::new(),
                },
                timeout,
            )),
        })
    }

    #[cfg(test)]
    pub const fn with_parts(
        searcher: Option<WebSearcher>,
        fetcher: Option<WebFetcher>,
        expose_as_tools: bool,
    ) -> Self {
        Self {
            enabled: true,
            expose_as_tools,
            searcher,
            fetcher,
        }
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> Result<Vec<SearchResult>, WebError> {
        let searcher = self.searcher.as_ref().ok_or(WebError::Disabled)?;
        Ok(searcher.search(query, max_results).await?)
    }

    pub async fn fetch(
        &self,
        url: &str,
        max_chars: Option<usize>,
    ) -> Result<FetchedPage, WebError> {
        let fetcher = self.fetcher.as_ref().ok_or(WebError::Disabled)?;
        let limit = max_chars
            .unwrap_or(DEFAULT_MAX_CHARS)
            .clamp(1, MAX_CHARS_LIMIT);
        Ok(fetcher.fetch(url, limit).await?)
    }

    /// The tools offered to models: only when web access and tool exposure
    /// are both on.
    pub fn tools(&self) -> Vec<ToolInfo> {
        if !(self.enabled && self.expose_as_tools) {
            return Vec::new();
        }
        vec![
            ToolInfo {
                server: BUILTIN_SERVER_NAME.to_owned(),
                name: namespaced(SEARCH_TOOL),
                description: "Search the web. Returns a ranked list of titles, URLs and snippets."
                    .to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": search::MAX_RESULTS_LIMIT,
                            "description": "How many results to return."
                        }
                    },
                    "required": ["query"]
                }),
            },
            ToolInfo {
                server: BUILTIN_SERVER_NAME.to_owned(),
                name: namespaced(FETCH_TOOL),
                description: "Fetch a web page over http(s) and return its readable text."
                    .to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The http(s) URL to fetch."},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_CHARS_LIMIT,
                            "description": "Cut the text off after this many characters."
                        }
                    },
                    "required": ["url"]
                }),
            },
        ]
    }

    pub fn owns(&self, name: &str) -> bool {
        self.tools().iter().any(|tool| tool.name == name)
    }

    /// Run a built-in tool. Failures of the search or fetch themselves come
    /// back as an error result the model can read; malformed arguments are
    /// the caller's error.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<ToolOutput, ToolCallError> {
        if name == namespaced(SEARCH_TOOL) && self.owns(name) {
            let query = string_argument(arguments, "query")?;
            let max_results = integer_argument(arguments, "max_results")?;
            Ok(match self.search(&query, max_results).await {
                Ok(results) => search_output(&query, &results),
                Err(error) => error_output(&error.to_string()),
            })
        } else if name == namespaced(FETCH_TOOL) && self.owns(name) {
            let url = string_argument(arguments, "url")?;
            let max_chars = integer_argument(arguments, "max_chars")?;
            Ok(match self.fetch(&url, max_chars).await {
                Ok(page) => fetch_output(&page),
                Err(error) => error_output(&error.to_string()),
            })
        } else {
            Err(ToolCallError::UnknownTool(name.to_owned()))
        }
    }
}

fn namespaced(tool: &str) -> String {
    format!(
        "{BUILTIN_SERVER_NAME}{}{tool}",
        crate::mcp::config::NAMESPACE_SEPARATOR
    )
}

fn string_argument(arguments: &Value, key: &str) -> Result<String, ToolCallError> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ToolCallError::InvalidArguments(format!("'{key}' must be a string")))
}

fn integer_argument(arguments: &Value, key: &str) -> Result<Option<usize>, ToolCallError> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| {
                ToolCallError::InvalidArguments(format!("'{key}' must be a positive integer"))
            }),
    }
}

fn text_output(text: String, structured: Option<Value>, is_error: bool) -> ToolOutput {
    ToolOutput {
        content: vec![json!({"type": "text", "text": text})],
        text,
        structured_content: structured,
        is_error,
        truncated: false,
    }
}

fn error_output(message: &str) -> ToolOutput {
    text_output(format!("Error: {message}"), None, true)
}

fn search_output(query: &str, results: &[SearchResult]) -> ToolOutput {
    let mut text = format!("Search results for \"{query}\":\n");
    for (index, result) in results.iter().enumerate() {
        let _ = write!(
            text,
            "\n{}. {}\n   {}\n",
            index + 1,
            result.title,
            result.url
        );
        if !result.snippet.is_empty() {
            let _ = writeln!(text, "   {}", result.snippet);
        }
    }
    if results.is_empty() {
        text.push_str("\nNo results.\n");
    }
    text_output(text, Some(json!({"results": results})), false)
}

fn fetch_output(page: &FetchedPage) -> ToolOutput {
    let mut text = String::new();
    if let Some(title) = &page.title {
        let _ = writeln!(text, "Title: {title}");
    }
    let _ = write!(text, "URL: {}\n\n{}", page.url, page.text);
    if page.truncated {
        text.push_str("\n\n[content truncated]");
    }
    text_output(text, None, false)
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn service_for(server: &MockServer, expose: bool) -> WebService {
        let searcher = WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&server.uri()).unwrap(),
            },
            Duration::from_secs(5),
        )
        .unwrap();
        let fetcher = WebFetcher::new(
            AddressPolicy {
                allow_private: true,
                exempt: Vec::new(),
            },
            Duration::from_secs(5),
        );
        WebService::with_parts(Some(searcher), Some(fetcher), expose)
    }

    #[test]
    fn disabled_service_offers_nothing() {
        let service = WebService::disabled();
        assert!(!service.enabled());
        assert!(service.tools().is_empty());
        let built = WebService::new(&WebConfig::default()).unwrap();
        assert!(!built.enabled());
    }

    #[test]
    fn tools_need_both_enabled_and_exposed() {
        let enabled_only = WebService::new(&WebConfig {
            enabled: true,
            ..WebConfig::default()
        })
        .unwrap();
        assert!(enabled_only.enabled());
        assert!(enabled_only.tools().is_empty());

        let exposed = WebService::new(&WebConfig {
            enabled: true,
            expose_as_tools: true,
            ..WebConfig::default()
        })
        .unwrap();
        let names: Vec<_> = exposed.tools().into_iter().map(|tool| tool.name).collect();
        assert_eq!(names, vec!["higgs__web_search", "higgs__web_fetch"]);
        assert!(exposed.tools().iter().all(|tool| tool.server == "higgs"));
    }

    #[test]
    fn building_from_an_invalid_config_is_an_error() {
        let result = WebService::new(&WebConfig {
            enabled: true,
            search_provider: "searxng".to_owned(),
            ..WebConfig::default()
        });
        assert!(result.err().unwrap().contains("searxng_url"));
    }

    #[tokio::test]
    async fn disabled_service_refuses_search_and_fetch() {
        let service = WebService::disabled();
        assert!(matches!(
            service.search("x", None).await,
            Err(WebError::Disabled)
        ));
        assert!(matches!(
            service.fetch("http://x", None).await,
            Err(WebError::Disabled)
        ));
        assert!(matches!(
            service
                .call_tool("higgs__web_search", &json!({"query": "x"}))
                .await,
            Err(ToolCallError::UnknownTool(_))
        ));
    }

    #[tokio::test]
    async fn search_tool_formats_results_for_a_model() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{"title": "Rust", "url": "https://rust-lang.org", "content": "A language"}]
            })))
            .mount(&server)
            .await;
        let service = service_for(&server, true);
        let output = service
            .call_tool("higgs__web_search", &json!({"query": "rust"}))
            .await
            .unwrap();
        assert!(!output.is_error);
        assert!(
            output
                .text
                .contains("1. Rust\n   https://rust-lang.org\n   A language")
        );
        assert_eq!(
            output.structured_content.unwrap()["results"][0]["url"],
            "https://rust-lang.org"
        );
    }

    #[tokio::test]
    async fn fetch_tool_returns_text_and_reports_failures_as_error_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw("<title>T</title><p>Body text</p>", "text/html"),
            )
            .mount(&server)
            .await;
        let service = service_for(&server, true);
        let output = service
            .call_tool(
                "higgs__web_fetch",
                &json!({"url": format!("{}/page", server.uri())}),
            )
            .await
            .unwrap();
        assert!(output.text.starts_with("Title: T\nURL: "));
        assert!(output.text.ends_with("Body text"));

        let failure = service
            .call_tool("higgs__web_fetch", &json!({"url": "file:///etc/passwd"}))
            .await
            .unwrap();
        assert!(failure.is_error);
        assert!(failure.text.contains("not allowed"), "{}", failure.text);
    }

    #[tokio::test]
    async fn tool_arguments_are_validated() {
        let server = MockServer::start().await;
        let service = service_for(&server, true);
        for arguments in [
            json!({}),
            json!({"query": 5}),
            json!({"query": "x", "max_results": "many"}),
            json!({"query": "x", "max_results": -1}),
        ] {
            assert!(
                matches!(
                    service.call_tool("higgs__web_search", &arguments).await,
                    Err(ToolCallError::InvalidArguments(_))
                ),
                "{arguments}"
            );
        }
        assert!(matches!(
            service.call_tool("higgs__web_fetch", &json!({})).await,
            Err(ToolCallError::InvalidArguments(_))
        ));
        assert!(matches!(
            service.call_tool("higgs__other", &json!({})).await,
            Err(ToolCallError::UnknownTool(_))
        ));
        // Not exposed as tools: the names do not resolve even though web is on.
        let hidden = service_for(&server, false);
        assert!(matches!(
            hidden
                .call_tool("higgs__web_search", &json!({"query": "x"}))
                .await,
            Err(ToolCallError::UnknownTool(_))
        ));
    }
}
