//! `web_search` providers: a self-hosted `SearXNG` instance, or the keyless
//! `DuckDuckGo` HTML endpoint.

use std::time::Duration;

use regex::Regex;
use serde::Serialize;
use serde_json::Value;

use super::html;

pub const DEFAULT_MAX_RESULTS: usize = 5;
pub const MAX_RESULTS_LIMIT: usize = 20;
const DUCKDUCKGO_ENDPOINT: &str = "https://html.duckduckgo.com/html/";
const MAX_QUERY_CHARS: usize = 400;
const MAX_SNIPPET_CHARS: usize = 500;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("search query must not be empty")]
    EmptyQuery,
    #[error("search query is longer than {MAX_QUERY_CHARS} characters")]
    QueryTooLong,
    #[error("search timed out")]
    Timeout,
    #[error("search provider failed: {0}")]
    Provider(String),
}

#[derive(Debug, Clone)]
pub enum SearchProvider {
    SearxNg { base: reqwest::Url },
    DuckDuckGo { endpoint: reqwest::Url },
}

impl SearchProvider {
    pub fn duckduckgo() -> Self {
        Self::DuckDuckGo {
            endpoint: reqwest::Url::parse(DUCKDUCKGO_ENDPOINT).unwrap_or_else(|_| {
                reqwest::Url::parse("https://html.duckduckgo.com/")
                    .unwrap_or_else(|_| unreachable_url())
            }),
        }
    }
}

fn unreachable_url() -> reqwest::Url {
    // `DUCKDUCKGO_ENDPOINT` is a literal that parses; this keeps the
    // constructor total without a panic path.
    reqwest::Url::from_directory_path("/")
        .unwrap_or_else(|()| reqwest::Url::parse("file:///").unwrap_or_else(|_| unreachable_url()))
}

pub struct WebSearcher {
    client: reqwest::Client,
    provider: SearchProvider,
}

impl WebSearcher {
    pub fn new(provider: SearchProvider, timeout: Duration) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent(concat!(
                "Mozilla/5.0 (compatible; higgs-search/",
                env!("CARGO_PKG_VERSION"),
                ")"
            ))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self { client, provider })
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(SearchError::EmptyQuery);
        }
        if trimmed.chars().count() > MAX_QUERY_CHARS {
            return Err(SearchError::QueryTooLong);
        }
        let limit = max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, MAX_RESULTS_LIMIT);
        let mut results = match &self.provider {
            SearchProvider::SearxNg { base } => self.searxng(base, trimmed).await?,
            SearchProvider::DuckDuckGo { endpoint } => self.duckduckgo(endpoint, trimmed).await?,
        };
        results.truncate(limit);
        Ok(results)
    }

    async fn get_text(&self, request: reqwest::RequestBuilder) -> Result<String, SearchError> {
        let mut response = request.send().await.map_err(|error| {
            if error.is_timeout() {
                SearchError::Timeout
            } else {
                SearchError::Provider(error.to_string())
            }
        })?;
        let status = response.status();
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| SearchError::Provider(error.to_string()))?
        {
            if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
                return Err(SearchError::Provider("response is too large".to_owned()));
            }
            body.extend_from_slice(&chunk);
        }
        let text = String::from_utf8_lossy(&body).into_owned();
        if !status.is_success() {
            return Err(SearchError::Provider(
                if status == reqwest::StatusCode::FORBIDDEN {
                    "HTTP 403; if this is SearXNG, enable the json format under search.formats in settings.yml"
                    .to_owned()
                } else {
                    format!("HTTP {status}")
                },
            ));
        }
        Ok(text)
    }

    async fn searxng(
        &self,
        base: &reqwest::Url,
        query: &str,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut directory = base.clone();
        if !directory.path().ends_with('/') {
            let with_slash = format!("{}/", directory.path());
            directory.set_path(&with_slash);
        }
        let mut endpoint = directory
            .join("search")
            .map_err(|error| SearchError::Provider(error.to_string()))?;
        endpoint
            .query_pairs_mut()
            .append_pair("q", query)
            .append_pair("format", "json");
        let body = self.get_text(self.client.get(endpoint)).await?;
        let parsed: Value = serde_json::from_str(&body)
            .map_err(|error| SearchError::Provider(format!("response is not JSON: {error}")))?;
        let entries = parsed
            .get("results")
            .and_then(Value::as_array)
            .ok_or_else(|| SearchError::Provider("response has no results array".to_owned()))?;
        Ok(entries
            .iter()
            .filter_map(|entry| {
                let url = entry.get("url").and_then(Value::as_str)?;
                Some(SearchResult {
                    title: html::collapse_whitespace(
                        entry.get("title").and_then(Value::as_str).unwrap_or(url),
                    ),
                    url: url.to_owned(),
                    snippet: clip(
                        entry
                            .get("content")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                })
            })
            .collect())
    }

    async fn duckduckgo(
        &self,
        endpoint: &reqwest::Url,
        query: &str,
    ) -> Result<Vec<SearchResult>, SearchError> {
        let mut target = endpoint.clone();
        target.query_pairs_mut().append_pair("q", query);
        let body = self.get_text(self.client.get(target)).await?;
        let results = parse_duckduckgo(&body);
        if results.is_empty() {
            return Err(SearchError::Provider(
                "no results could be read from DuckDuckGo (it may be rate limiting or have changed its page); a SearXNG instance is more reliable"
                    .to_owned(),
            ));
        }
        Ok(results)
    }
}

fn clip(text: &str) -> String {
    let collapsed = html::collapse_whitespace(text);
    collapsed.chars().take(MAX_SNIPPET_CHARS).collect()
}

/// Pull result titles, targets and snippets out of the `DuckDuckGo` HTML page.
pub fn parse_duckduckgo(page: &str) -> Vec<SearchResult> {
    let Ok(anchor) = Regex::new(r"(?is)<a\s([^>]*)>(.*?)</a>") else {
        return Vec::new();
    };
    let mut results: Vec<SearchResult> = Vec::new();
    for captures in anchor.captures_iter(page) {
        let attributes = captures.get(1).map_or("", |m| m.as_str());
        let inner = captures.get(2).map_or("", |m| m.as_str());
        let classes = html::attribute(attributes, "class").unwrap_or_default();
        if classes.split_whitespace().any(|class| class == "result__a") {
            let Some(target) =
                html::attribute(attributes, "href").and_then(|href| unwrap_redirect(&href))
            else {
                continue;
            };
            results.push(SearchResult {
                title: html::collapse_whitespace(&html::extract(inner).text),
                url: target,
                snippet: String::new(),
            });
        } else if classes
            .split_whitespace()
            .any(|class| class == "result__snippet")
        {
            if let Some(last) = results.last_mut()
                && last.snippet.is_empty()
            {
                last.snippet = clip(&html::extract(inner).text);
            }
        }
    }
    results
}

/// `DuckDuckGo` links go through `//duckduckgo.com/l/?uddg=<target>`; return
/// the target, or the href itself when it is already an absolute URL.
fn unwrap_redirect(href: &str) -> Option<String> {
    let decoded = html::decode_entities(href);
    let absolute = if decoded.starts_with("//") {
        format!("https:{decoded}")
    } else {
        decoded
    };
    let parsed = reqwest::Url::parse(&absolute).ok()?;
    if parsed
        .host_str()
        .is_some_and(|host| host.ends_with("duckduckgo.com"))
        && parsed.path() == "/l/"
    {
        return parsed
            .query_pairs()
            .find(|(key, _)| key == "uddg")
            .map(|(_, value)| value.into_owned());
    }
    matches!(parsed.scheme(), "http" | "https").then(|| parsed.to_string())
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn searxng(server: &MockServer) -> WebSearcher {
        WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&server.uri()).unwrap(),
            },
            TIMEOUT,
        )
        .unwrap()
    }

    fn duckduckgo(server: &MockServer) -> WebSearcher {
        WebSearcher::new(
            SearchProvider::DuckDuckGo {
                endpoint: reqwest::Url::parse(&format!("{}/html/", server.uri())).unwrap(),
            },
            TIMEOUT,
        )
        .unwrap()
    }

    const DUCKDUCKGO_PAGE: &str = r#"
        <div class="result results_links">
          <h2 class="result__title">
            <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone%3Fa%3D1&amp;rut=abc">First <b>Result</b></a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">A <b>snippet</b> about
            the first result.</a>
        </div>
        <div class="result">
          <a class="result__a" href="https://direct.example/two">Second</a>
          <a class="result__snippet" href="https://direct.example/two">Second snippet</a>
        </div>
        <div class="result"><a class="other" href="https://ad.example/">Ad</a></div>
    "#;

    #[tokio::test]
    async fn searxng_results_are_mapped_and_limited() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "rust async"))
            .and(query_param("format", "json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [
                    {"title": "One", "url": "https://one.example", "content": "first   snippet"},
                    {"title": "Two", "url": "https://two.example"},
                    {"title": "No url"},
                    {"title": "Three", "url": "https://three.example", "content": "c"},
                ]
            })))
            .mount(&server)
            .await;
        let searcher = searxng(&server);
        let results = searcher.search("  rust async ", Some(2)).await.unwrap();
        assert_eq!(
            results,
            vec![
                SearchResult {
                    title: "One".to_owned(),
                    url: "https://one.example".to_owned(),
                    snippet: "first snippet".to_owned()
                },
                SearchResult {
                    title: "Two".to_owned(),
                    url: "https://two.example".to_owned(),
                    snippet: String::new()
                },
            ]
        );
        assert_eq!(searcher.search("rust async", None).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn searxng_base_path_prefix_is_preserved() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/searx/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
            .mount(&server)
            .await;
        let searcher = WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&format!("{}/searx", server.uri())).unwrap(),
            },
            TIMEOUT,
        )
        .unwrap();
        assert!(searcher.search("x", None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn searxng_failures_are_explained() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(query_param("q", "forbidden"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("q", "html"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("q", "shape"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"nope": 1})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("q", "slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let searcher = WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&server.uri()).unwrap(),
            },
            Duration::from_millis(300),
        )
        .unwrap();
        let forbidden = searcher
            .search("forbidden", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(forbidden.contains("json format"), "{forbidden}");
        assert!(
            searcher
                .search("html", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("not JSON")
        );
        assert!(
            searcher
                .search("shape", None)
                .await
                .unwrap_err()
                .to_string()
                .contains("no results array")
        );
        assert!(matches!(
            searcher.search("slow", None).await.unwrap_err(),
            SearchError::Timeout
        ));
    }

    #[tokio::test]
    async fn queries_are_validated_before_any_request() {
        let server = MockServer::start().await;
        let searcher = searxng(&server);
        assert!(matches!(
            searcher.search("   ", None).await,
            Err(SearchError::EmptyQuery)
        ));
        let long = "q".repeat(MAX_QUERY_CHARS + 1);
        assert!(matches!(
            searcher.search(&long, None).await,
            Err(SearchError::QueryTooLong)
        ));
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn duckduckgo_html_is_parsed_into_results() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/html/"))
            .and(query_param("q", "hello world"))
            .respond_with(ResponseTemplate::new(200).set_body_string(DUCKDUCKGO_PAGE))
            .mount(&server)
            .await;
        let results = duckduckgo(&server)
            .search("hello world", None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "First Result");
        assert_eq!(results[0].url, "https://example.com/one?a=1");
        assert_eq!(results[0].snippet, "A snippet about the first result.");
        assert_eq!(results[1].url, "https://direct.example/two");
        assert_eq!(results[1].snippet, "Second snippet");
    }

    #[tokio::test]
    async fn duckduckgo_pages_without_results_are_reported_as_such() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>captcha</html>"))
            .mount(&server)
            .await;
        let error = duckduckgo(&server)
            .search("x", None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("no results could be read"), "{error}");
    }

    #[test]
    fn redirect_unwrapping_handles_both_forms_and_rejects_other_schemes() {
        assert_eq!(
            unwrap_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fx&rut=1")
                .as_deref(),
            Some("https://a.example/x")
        );
        assert_eq!(
            unwrap_redirect("https://b.example/y").as_deref(),
            Some("https://b.example/y")
        );
        assert_eq!(unwrap_redirect("javascript:alert(1)"), None);
        assert_eq!(unwrap_redirect("//duckduckgo.com/l/?rut=1"), None);
    }

    #[test]
    fn default_duckduckgo_provider_points_at_the_html_endpoint() {
        match SearchProvider::duckduckgo() {
            SearchProvider::DuckDuckGo { endpoint } => {
                assert_eq!(endpoint.as_str(), "https://html.duckduckgo.com/html/");
            }
            SearchProvider::SearxNg { .. } => panic!("wrong provider"),
        }
    }
}
