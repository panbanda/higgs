//! `/v1/web/*`: web search and page fetch.

use axum::{Json, extract::State, http::StatusCode};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    routes::mcp::ApiError,
    state::SharedState,
    web::{WebError, fetch::FetchError, search::SearchError},
};

impl From<WebError> for ApiError {
    fn from(error: WebError) -> Self {
        let message = error.to_string();
        let (status, error_type, code) = match &error {
            WebError::Disabled => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "web_disabled",
            ),
            WebError::Search(SearchError::EmptyQuery | SearchError::QueryTooLong) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_query",
            ),
            WebError::Search(SearchError::Timeout) | WebError::Fetch(FetchError::Timeout) => {
                (StatusCode::GATEWAY_TIMEOUT, "server_error", "web_timeout")
            }
            WebError::Search(SearchError::Provider(_)) => {
                (StatusCode::BAD_GATEWAY, "server_error", "search_failed")
            }
            WebError::Fetch(FetchError::InvalidUrl(_)) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_url",
            ),
            WebError::Fetch(FetchError::Forbidden(_)) => (
                StatusCode::FORBIDDEN,
                "invalid_request_error",
                "address_forbidden",
            ),
            WebError::Fetch(FetchError::UnsupportedContentType(_)) => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "invalid_request_error",
                "unsupported_content_type",
            ),
            WebError::Fetch(
                FetchError::TooManyRedirects | FetchError::Status(_) | FetchError::Network(_),
            ) => (StatusCode::BAD_GATEWAY, "server_error", "fetch_failed"),
        };
        Self {
            status,
            error_type,
            code,
            message,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct FetchRequest {
    url: String,
    #[serde(default)]
    max_chars: Option<usize>,
}

fn parse<T: for<'de> Deserialize<'de>>(body: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice(body)
        .map_err(|error| ApiError::bad_request(format!("Invalid request body: {error}")))
}

pub async fn search(
    State(state): State<SharedState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    if !state.tools.web.enabled() {
        return Err(WebError::Disabled.into());
    }
    let request: SearchRequest = parse(&body)?;
    let results = state
        .tools
        .web
        .search(&request.query, request.max_results)
        .await?;
    Ok(Json(json!({
        "query": request.query.trim(),
        "count": results.len(),
        "results": results,
    })))
}

pub async fn fetch(State(state): State<SharedState>, body: Bytes) -> Result<Json<Value>, ApiError> {
    if !state.tools.web.enabled() {
        return Err(WebError::Disabled.into());
    }
    let request: FetchRequest = parse(&body)?;
    let page = state
        .tools
        .web
        .fetch(&request.url, request.max_chars)
        .await?;
    Ok(Json(json!({
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "truncated": page.truncated,
        "status": page.status,
        "content_type": page.content_type,
    })))
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::io;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use futures::future::BoxFuture;
    use http::StatusCode;
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::mcp::manager::McpManager;
    use crate::test_support::{app_with_tools, send_request};
    use crate::tools::ToolHub;
    use crate::web::WebService;
    use crate::web::config::WebConfig;
    use crate::web::fetch::WebFetcher;
    use crate::web::search::{SearchProvider, WebSearcher};
    use crate::web::ssrf::{AddressPolicy, Resolver};

    struct PrivateResolver;

    impl Resolver for PrivateResolver {
        fn resolve<'a>(
            &'a self,
            _host: &'a str,
            port: u16,
        ) -> BoxFuture<'a, io::Result<Vec<SocketAddr>>> {
            Box::pin(async move { Ok(vec![SocketAddr::new("10.0.0.7".parse().unwrap(), port)]) })
        }
    }

    fn web_hub(server: &MockServer, expose: bool, timeout: Duration) -> ToolHub {
        let searcher = WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&server.uri()).unwrap(),
            },
            timeout,
        )
        .unwrap();
        let fetcher = WebFetcher::new(
            AddressPolicy {
                allow_private: true,
                exempt: Vec::new(),
            },
            timeout,
        );
        ToolHub::new(
            McpManager::empty(),
            WebService::with_parts(Some(searcher), Some(fetcher), expose),
        )
    }

    async fn mock_search(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "rust"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [
                    {"title": "Rust", "url": "https://rust-lang.org", "content": "A language"},
                    {"title": "Crates", "url": "https://crates.io", "content": "Registry"}
                ]
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn endpoints_are_404_when_web_is_disabled() {
        let app = app_with_tools(ToolHub::disabled(), None, None);
        for (path, body) in [
            ("/v1/web/search", json!({"query": "x"})),
            ("/v1/web/fetch", json!({"url": "http://example.com"})),
            ("/v1/web/search", json!("not even valid")),
        ] {
            let (status, response) = send_request(&app, "POST", path, Some(&body), None).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert_eq!(response["error"]["code"], "web_disabled");
        }
    }

    #[tokio::test]
    async fn search_returns_results_and_honors_max_results() {
        let server = MockServer::start().await;
        mock_search(&server).await;
        let app = app_with_tools(web_hub(&server, false, Duration::from_secs(5)), None, None);
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/web/search",
            Some(&json!({"query": " rust "})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["query"], "rust");
        assert_eq!(body["count"], 2);
        assert_eq!(
            body["results"][0],
            json!({"title": "Rust", "url": "https://rust-lang.org", "snippet": "A language"})
        );

        let (_, limited) = send_request(
            &app,
            "POST",
            "/v1/web/search",
            Some(&json!({"query": "rust", "max_results": 1})),
            None,
        )
        .await;
        assert_eq!(limited["count"], 1);
    }

    #[tokio::test]
    async fn search_error_statuses() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(query_param("q", "broken"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(query_param("q", "slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let app = app_with_tools(
            web_hub(&server, false, Duration::from_millis(300)),
            None,
            None,
        );
        let cases = [
            (
                json!({"query": "  "}),
                StatusCode::BAD_REQUEST,
                "invalid_query",
            ),
            (
                json!({"query": "q".repeat(500)}),
                StatusCode::BAD_REQUEST,
                "invalid_query",
            ),
            (json!({}), StatusCode::BAD_REQUEST, "invalid_request"),
            (
                json!({"query": "broken"}),
                StatusCode::BAD_GATEWAY,
                "search_failed",
            ),
            (
                json!({"query": "slow"}),
                StatusCode::GATEWAY_TIMEOUT,
                "web_timeout",
            ),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, body) =
                send_request(&app, "POST", "/v1/web/search", Some(&request), None).await;
            assert_eq!(status, expected_status, "{request}: {body}");
            assert_eq!(body["error"]["code"], expected_code);
        }
    }

    #[tokio::test]
    async fn fetch_returns_readable_text_with_title_and_final_url() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/start"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/page"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "<html><head><title>Doc</title></head><body><p>Hello <b>there</b></p></body></html>",
                "text/html",
            ))
            .mount(&server)
            .await;
        let app = app_with_tools(web_hub(&server, false, Duration::from_secs(5)), None, None);
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/web/fetch",
            Some(&json!({"url": format!("{}/start", server.uri())})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["title"], "Doc");
        assert_eq!(body["text"], "Hello there");
        assert_eq!(body["url"], format!("{}/page", server.uri()));
        assert_eq!(body["status"], 200);
        assert_eq!(body["content_type"], "text/html");
        assert_eq!(body["truncated"], false);

        let (_, short) = send_request(
            &app,
            "POST",
            "/v1/web/fetch",
            Some(&json!({"url": format!("{}/page", server.uri()), "max_chars": 4})),
            None,
        )
        .await;
        assert_eq!(short["text"], "Hell");
        assert_eq!(short["truncated"], true);
    }

    #[tokio::test]
    async fn fetch_error_statuses() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/gone"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/binary"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(vec![1_u8, 2], "application/zip"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let app = app_with_tools(
            web_hub(&server, false, Duration::from_millis(400)),
            None,
            None,
        );
        let base = server.uri();
        let cases = [
            (
                json!({"url": "not a url"}),
                StatusCode::BAD_REQUEST,
                "invalid_url",
            ),
            (
                json!({"url": "file:///etc/passwd"}),
                StatusCode::FORBIDDEN,
                "address_forbidden",
            ),
            (
                json!({"url": format!("{base}/gone")}),
                StatusCode::BAD_GATEWAY,
                "fetch_failed",
            ),
            (
                json!({"url": format!("{base}/binary")}),
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_content_type",
            ),
            (
                json!({"url": format!("{base}/slow")}),
                StatusCode::GATEWAY_TIMEOUT,
                "web_timeout",
            ),
            (json!({}), StatusCode::BAD_REQUEST, "invalid_request"),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, body) =
                send_request(&app, "POST", "/v1/web/fetch", Some(&request), None).await;
            assert_eq!(status, expected_status, "{request}: {body}");
            assert_eq!(body["error"]["code"], expected_code);
        }
    }

    #[tokio::test]
    async fn fetch_refuses_private_targets_by_default() {
        let service = WebService::new(&WebConfig {
            enabled: true,
            ..WebConfig::default()
        })
        .unwrap();
        let app = app_with_tools(ToolHub::new(McpManager::empty(), service), None, None);
        for url in [
            "http://127.0.0.1:9/",
            "http://localhost:9/",
            "http://[::1]:9/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://[::ffff:192.168.0.1]/",
        ] {
            let (status, body) = send_request(
                &app,
                "POST",
                "/v1/web/fetch",
                Some(&json!({"url": url})),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{url}: {body}");
            assert_eq!(body["error"]["code"], "address_forbidden");
        }
    }

    #[tokio::test]
    async fn fetch_refuses_names_that_resolve_to_private_addresses() {
        let fetcher = WebFetcher::with_resolver(
            Arc::new(PrivateResolver),
            AddressPolicy::default(),
            Duration::from_secs(2),
        );
        let service = WebService::with_parts(None, Some(fetcher), false);
        let app = app_with_tools(ToolHub::new(McpManager::empty(), service), None, None);
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/web/fetch",
            Some(&json!({"url": "http://totally-public.example/"})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("10.0.0.7")
        );
    }

    #[tokio::test]
    async fn web_tools_are_listed_and_executable_only_when_exposed() {
        let server = MockServer::start().await;
        mock_search(&server).await;

        let hidden = app_with_tools(web_hub(&server, false, Duration::from_secs(5)), None, None);
        let (_, hidden_tools) = send_request(&hidden, "GET", "/v1/mcp/tools", None, None).await;
        assert_eq!(hidden_tools["count"], 0);
        let (hidden_status, _) = send_request(
            &hidden,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "higgs__web_search", "arguments": {"query": "rust"}})),
            None,
        )
        .await;
        assert_eq!(hidden_status, StatusCode::NOT_FOUND);

        let exposed = app_with_tools(web_hub(&server, true, Duration::from_secs(5)), None, None);
        let (_, tools) = send_request(&exposed, "GET", "/v1/mcp/tools", None, None).await;
        let names: Vec<_> = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["higgs__web_search", "higgs__web_fetch"]);
        assert_eq!(tools["tools"][0]["server"], "higgs");
        assert_eq!(
            tools["tools"][0]["function"]["parameters"]["required"],
            json!(["query"])
        );

        let (result_status, result) = send_request(
            &exposed,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "higgs__web_search", "arguments": {"query": "rust"}})),
            None,
        )
        .await;
        assert_eq!(result_status, StatusCode::OK);
        assert_eq!(result["is_error"], false);
        assert!(
            result["text"]
                .as_str()
                .unwrap()
                .contains("https://crates.io")
        );
        assert_eq!(
            result["structured_content"]["results"][1]["title"],
            "Crates"
        );

        let (bad_status, bad) = send_request(
            &exposed,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "higgs__web_search", "arguments": {}})),
            None,
        )
        .await;
        assert_eq!(bad_status, StatusCode::BAD_REQUEST);
        assert_eq!(bad["error"]["code"], "invalid_arguments");
    }

    #[tokio::test]
    async fn web_endpoints_require_the_api_key() {
        let server = MockServer::start().await;
        mock_search(&server).await;
        let app = app_with_tools(
            web_hub(&server, false, Duration::from_secs(5)),
            Some("secret"),
            None,
        );
        let request = json!({"query": "rust"});
        let (denied, _) = send_request(&app, "POST", "/v1/web/search", Some(&request), None).await;
        assert_eq!(denied, StatusCode::UNAUTHORIZED);
        let (allowed, _) = send_request(
            &app,
            "POST",
            "/v1/web/search",
            Some(&request),
            Some("secret"),
        )
        .await;
        assert_eq!(allowed, StatusCode::OK);
    }
}
