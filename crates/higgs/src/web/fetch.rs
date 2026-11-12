//! `web_fetch`: download a page and reduce it to readable text.
//!
//! Every request goes to an address this module resolved and checked itself:
//! the connection is pinned to the vetted addresses, so a hostname cannot
//! resolve to something public for the check and something internal for the
//! connection. Redirects are followed by hand so each hop is vetted too.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{ACCEPT, CONTENT_TYPE, LOCATION};

use super::html;
use super::ssrf::{AddressPolicy, Resolver, SystemResolver};

pub const DEFAULT_MAX_CHARS: usize = 20_000;
pub const MAX_CHARS_LIMIT: usize = 200_000;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
/// Deliberately generic: no operating system, user or host details.
const USER_AGENT: &str = concat!(
    "Mozilla/5.0 (compatible; higgs-fetch/",
    env!("CARGO_PKG_VERSION"),
    ")"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedPage {
    /// Final URL after redirects.
    pub url: String,
    pub title: Option<String>,
    pub text: String,
    pub truncated: bool,
    pub status: u16,
    pub content_type: String,
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),
    #[error("blocked: {0}")]
    Forbidden(String),
    #[error("too many redirects (more than {MAX_REDIRECTS})")]
    TooManyRedirects,
    #[error("the request timed out")]
    Timeout,
    #[error("the server answered HTTP {0}")]
    Status(u16),
    #[error("unsupported content type '{0}'")]
    UnsupportedContentType(String),
    #[error("request failed: {0}")]
    Network(String),
}

pub struct WebFetcher {
    resolver: Arc<dyn Resolver>,
    policy: AddressPolicy,
    timeout: Duration,
}

impl WebFetcher {
    pub fn new(policy: AddressPolicy, timeout: Duration) -> Self {
        Self::with_resolver(Arc::new(SystemResolver), policy, timeout)
    }

    pub fn with_resolver(
        resolver: Arc<dyn Resolver>,
        policy: AddressPolicy,
        timeout: Duration,
    ) -> Self {
        Self {
            resolver,
            policy,
            timeout,
        }
    }

    pub async fn fetch(&self, url: &str, max_chars: usize) -> Result<FetchedPage, FetchError> {
        tokio::time::timeout(self.timeout, self.fetch_following_redirects(url, max_chars))
            .await
            .unwrap_or(Err(FetchError::Timeout))
    }

    async fn fetch_following_redirects(
        &self,
        url: &str,
        max_chars: usize,
    ) -> Result<FetchedPage, FetchError> {
        let mut current = reqwest::Url::parse(url.trim())
            .map_err(|error| FetchError::InvalidUrl(error.to_string()))?;
        for _ in 0..=MAX_REDIRECTS {
            let addresses = self.vet(&current).await?;
            let response = self.request(&current, &addresses).await?;
            let status = response.status();
            if status.is_redirection() {
                let location = response
                    .headers()
                    .get(LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or_else(|| {
                        FetchError::Network("redirect without a Location header".to_owned())
                    })?;
                current = current.join(location).map_err(|error| {
                    FetchError::InvalidUrl(format!("bad redirect target: {error}"))
                })?;
                continue;
            }
            if !status.is_success() {
                return Err(FetchError::Status(status.as_u16()));
            }
            return read_page(response, &current, max_chars).await;
        }
        Err(FetchError::TooManyRedirects)
    }

    /// Check scheme and credentials, resolve the host and verify every
    /// address. Returns the addresses the connection must use.
    async fn vet(&self, url: &reqwest::Url) -> Result<Vec<SocketAddr>, FetchError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(FetchError::Forbidden(format!(
                "scheme '{}' is not allowed; only http and https can be fetched",
                url.scheme()
            )));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(FetchError::Forbidden(
                "URLs with embedded credentials are not allowed".to_owned(),
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| FetchError::InvalidUrl("URL has no host".to_owned()))?;
        let port = url
            .port_or_known_default()
            .ok_or_else(|| FetchError::InvalidUrl("URL has no port".to_owned()))?;
        let literal: Option<IpAddr> = host.trim_matches(|c| c == '[' || c == ']').parse().ok();
        let addresses =
            match literal {
                Some(address) => vec![SocketAddr::new(address, port)],
                None => self.resolver.resolve(host, port).await.map_err(|error| {
                    FetchError::Network(format!("cannot resolve {host}: {error}"))
                })?,
            };
        if addresses.is_empty() {
            return Err(FetchError::Network(format!(
                "{host} did not resolve to any address"
            )));
        }
        // Every address must pass: the client may try any of them.
        for address in &addresses {
            self.policy
                .check(address.ip())
                .map_err(FetchError::Forbidden)?;
        }
        Ok(addresses)
    }

    async fn request(
        &self,
        url: &reqwest::Url,
        addresses: &[SocketAddr],
    ) -> Result<reqwest::Response, FetchError> {
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            // A proxy would resolve the name itself and bypass the pinning.
            .no_proxy()
            .user_agent(USER_AGENT)
            .timeout(self.timeout);
        if let Some(host) = url.host_str()
            && host
                .trim_matches(|c| c == '[' || c == ']')
                .parse::<IpAddr>()
                .is_err()
        {
            builder = builder.resolve_to_addrs(host, addresses);
        }
        let client = builder
            .build()
            .map_err(|error| FetchError::Network(error.to_string()))?;
        client
            .get(url.clone())
            .header(
                ACCEPT,
                "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1",
            )
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    FetchError::Timeout
                } else {
                    FetchError::Network(describe_network_error(&error))
                }
            })
    }
}

fn describe_network_error(error: &reqwest::Error) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

fn media_type(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

async fn read_page(
    mut response: reqwest::Response,
    final_url: &reqwest::Url,
    max_chars: usize,
) -> Result<FetchedPage, FetchError> {
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let media = media_type(&content_type);
    let is_html = matches!(media.as_str(), "" | "text/html" | "application/xhtml+xml");
    let is_text = media.starts_with("text/")
        || media == "application/json"
        || media.ends_with("+json")
        || media == "application/xml"
        || media.ends_with("+xml");
    if !is_html && !is_text {
        return Err(FetchError::UnsupportedContentType(media));
    }

    let mut body = Vec::new();
    let mut body_truncated = false;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| FetchError::Network(describe_network_error(&error)))?
    {
        let room = MAX_BODY_BYTES - body.len();
        if chunk.len() > room {
            body.extend_from_slice(chunk.get(..room).unwrap_or_default());
            body_truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }

    let decoded = String::from_utf8_lossy(&body);
    let (title, full_text) = if is_html {
        let extracted = html::extract(&decoded);
        (extracted.title, extracted.text)
    } else {
        (None, decoded.into_owned())
    };
    let (text, text_truncated) = truncate_chars(full_text, max_chars);
    Ok(FetchedPage {
        url: final_url.to_string(),
        title,
        text,
        truncated: body_truncated || text_truncated,
        status,
        content_type: media,
    })
}

fn truncate_chars(text: String, max_chars: usize) -> (String, bool) {
    match text.char_indices().nth(max_chars) {
        Some((index, _)) => (text.get(..index).unwrap_or_default().to_owned(), true),
        None => (text, false),
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;

    use futures::future::BoxFuture;
    use wiremock::matchers::{header_exists, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const LOOPBACK: &str = "127.0.0.1";

    struct FakeResolver {
        answers: HashMap<String, Vec<IpAddr>>,
    }

    impl Resolver for FakeResolver {
        fn resolve<'a>(
            &'a self,
            host: &'a str,
            port: u16,
        ) -> BoxFuture<'a, io::Result<Vec<SocketAddr>>> {
            Box::pin(async move {
                self.answers
                    .get(host)
                    .map(|addresses| {
                        addresses
                            .iter()
                            .map(|address| SocketAddr::new(*address, port))
                            .collect()
                    })
                    .ok_or_else(|| io::Error::other("no such host"))
            })
        }
    }

    fn resolver(entries: &[(&str, &[&str])]) -> Arc<dyn Resolver> {
        Arc::new(FakeResolver {
            answers: entries
                .iter()
                .map(|(host, addresses)| {
                    (
                        (*host).to_owned(),
                        addresses.iter().map(|a| a.parse().unwrap()).collect(),
                    )
                })
                .collect(),
        })
    }

    /// Blocks everything non-public except the mock server's loopback address.
    fn fetcher_exempting_loopback(resolver: Arc<dyn Resolver>) -> WebFetcher {
        WebFetcher::with_resolver(
            resolver,
            AddressPolicy {
                allow_private: false,
                exempt: vec![LOOPBACK.parse().unwrap()],
            },
            Duration::from_secs(5),
        )
    }

    fn permissive() -> WebFetcher {
        WebFetcher::new(
            AddressPolicy {
                allow_private: true,
                exempt: Vec::new(),
            },
            Duration::from_secs(5),
        )
    }

    fn html_response(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body, "text/html; charset=utf-8")
    }

    #[tokio::test]
    async fn fetches_and_extracts_readable_text() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/article"))
            .and(header_exists("user-agent"))
            .respond_with(html_response(
                "<html><head><title>Hello</title></head><body><p>World &amp; friends</p></body></html>",
            ))
            .mount(&server)
            .await;
        let url = format!("{}/article", server.uri());
        let page = permissive().fetch(&url, 1000).await.unwrap();
        assert_eq!(page.title.as_deref(), Some("Hello"));
        assert_eq!(page.text, "World & friends");
        assert_eq!(page.url, url);
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html");
        assert!(!page.truncated);
    }

    #[tokio::test]
    async fn sends_a_generic_user_agent_without_host_or_user_details() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(html_response("<p>x</p>"))
            .mount(&server)
            .await;
        permissive().fetch(&server.uri(), 100).await.unwrap();
        let request = &server.received_requests().await.unwrap()[0];
        let agent = request.headers.get("user-agent").unwrap().to_str().unwrap();
        assert!(agent.starts_with("Mozilla/5.0 (compatible; higgs-fetch/"));
        let hostname = std::env::var("HOSTNAME").unwrap_or_default();
        let user = std::env::var("USER").unwrap_or_default();
        for private in [hostname, user] {
            assert!(private.is_empty() || !agent.contains(&private), "{agent}");
        }
        assert!(request.headers.get("cookie").is_none());
        assert!(request.headers.get("authorization").is_none());
    }

    #[tokio::test]
    async fn text_is_truncated_to_max_chars_on_a_character_boundary() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("ééééé", "text/plain"))
            .mount(&server)
            .await;
        let page = permissive().fetch(&server.uri(), 3).await.unwrap();
        assert_eq!(page.text, "ééé");
        assert!(page.truncated);
    }

    #[tokio::test]
    async fn oversized_bodies_are_cut_off() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw("a".repeat(MAX_BODY_BYTES + 1000), "text/plain"),
            )
            .mount(&server)
            .await;
        let page = permissive().fetch(&server.uri(), usize::MAX).await.unwrap();
        assert_eq!(page.text.len(), MAX_BODY_BYTES);
        assert!(page.truncated);
    }

    #[tokio::test]
    async fn binary_content_and_http_errors_are_refused() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/image"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(vec![0_u8; 10], "image/png"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let fetcher = permissive();
        let image = fetcher
            .fetch(&format!("{}/image", server.uri()), 10)
            .await
            .unwrap_err();
        assert!(matches!(image, FetchError::UnsupportedContentType(kind) if kind == "image/png"));
        let missing = fetcher
            .fetch(&format!("{}/missing", server.uri()), 10)
            .await
            .unwrap_err();
        assert!(matches!(missing, FetchError::Status(404)));
    }

    #[tokio::test]
    async fn non_http_schemes_and_credentials_are_refused() {
        let fetcher = permissive();
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/x",
            "gopher://example.com/",
            "data:text/html,hi",
            "javascript:alert(1)",
        ] {
            let error = fetcher.fetch(url, 10).await.unwrap_err();
            assert!(
                matches!(error, FetchError::Forbidden(_)),
                "{url}: {error:?}"
            );
        }
        let credentials = fetcher
            .fetch("http://user:pass@example.com/", 10)
            .await
            .unwrap_err();
        assert!(matches!(credentials, FetchError::Forbidden(_)));
        let garbage = fetcher.fetch("not a url", 10).await.unwrap_err();
        assert!(matches!(garbage, FetchError::InvalidUrl(_)));
    }

    #[tokio::test]
    async fn private_loopback_and_link_local_literals_are_blocked_by_default() {
        let fetcher = WebFetcher::new(AddressPolicy::default(), Duration::from_secs(5));
        for url in [
            "http://127.0.0.1/",
            "http://127.0.0.1:8080/admin",
            "http://localhost:1/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.0.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:7f00:1]/",
            "http://[fe80::1]/",
            "http://[fd00::1]/",
            "http://2130706433/",
            "http://0x7f000001/",
            "http://0177.0.0.1/",
            "http://127.1/",
        ] {
            let error = fetcher.fetch(url, 10).await.unwrap_err();
            assert!(
                matches!(error, FetchError::Forbidden(_)),
                "{url}: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_name_that_resolves_to_a_private_address_is_blocked() {
        let fetcher = WebFetcher::with_resolver(
            resolver(&[
                ("internal.example", &["10.1.2.3"]),
                ("mixed.example", &["93.184.216.34", "192.168.0.9"]),
                ("v6.example", &["::ffff:10.0.0.1"]),
            ]),
            AddressPolicy::default(),
            Duration::from_secs(5),
        );
        for url in [
            "http://internal.example/",
            "http://mixed.example/",
            "http://v6.example/",
        ] {
            let error = fetcher.fetch(url, 10).await.unwrap_err();
            assert!(
                matches!(error, FetchError::Forbidden(_)),
                "{url}: {error:?}"
            );
        }
        let unresolved = fetcher
            .fetch("http://nowhere.example/", 10)
            .await
            .unwrap_err();
        assert!(matches!(unresolved, FetchError::Network(_)));
    }

    #[tokio::test]
    async fn public_addresses_pass_vetting_without_any_connection_being_made() {
        let fetcher = WebFetcher::with_resolver(
            resolver(&[("public.example", &["93.184.216.34", "2606:4700:4700::1111"])]),
            AddressPolicy::default(),
            Duration::from_secs(5),
        );
        let url = reqwest::Url::parse("http://public.example:9/").unwrap();
        let addresses = fetcher.vet(&url).await.unwrap();
        assert_eq!(
            addresses,
            vec![
                "93.184.216.34:9".parse::<SocketAddr>().unwrap(),
                "[2606:4700:4700::1111]:9".parse::<SocketAddr>().unwrap()
            ]
        );
    }

    #[tokio::test]
    async fn resolved_names_are_connected_by_the_vetted_address() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(html_response("<p>via pinned address</p>"))
            .mount(&server)
            .await;
        let port = server.address().port();
        let fetcher = fetcher_exempting_loopback(resolver(&[("pinned.example", &[LOOPBACK])]));
        let page = fetcher
            .fetch(&format!("http://pinned.example:{port}/"), 100)
            .await
            .unwrap();
        assert_eq!(page.text, "via pinned address");
    }

    #[tokio::test]
    async fn redirects_are_followed_and_the_final_url_reported() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/start"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/middle"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/middle"))
            .respond_with(ResponseTemplate::new(301).insert_header("location", "/end"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/end"))
            .respond_with(html_response("<p>arrived</p>"))
            .mount(&server)
            .await;
        let page = permissive()
            .fetch(&format!("{}/start", server.uri()), 100)
            .await
            .unwrap();
        assert_eq!(page.text, "arrived");
        assert_eq!(page.url, format!("{}/end", server.uri()));
    }

    #[tokio::test]
    async fn a_redirect_to_a_private_address_is_blocked_on_that_hop() {
        let server = MockServer::start().await;
        for target in [
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://[::1]:8080/",
            "http://internal.example/",
            "file:///etc/passwd",
        ] {
            Mock::given(method("GET"))
                .and(path("/hop"))
                .respond_with(ResponseTemplate::new(307).insert_header("location", target))
                .up_to_n_times(1)
                .mount(&server)
                .await;
            let fetcher =
                fetcher_exempting_loopback(resolver(&[("internal.example", &["10.9.9.9"])]));
            let error = fetcher
                .fetch(&format!("{}/hop", server.uri()), 10)
                .await
                .unwrap_err();
            assert!(
                matches!(error, FetchError::Forbidden(_)),
                "{target}: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn redirect_loops_stop_after_the_limit() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "/again"))
            .mount(&server)
            .await;
        let error = permissive().fetch(&server.uri(), 10).await.unwrap_err();
        assert!(matches!(error, FetchError::TooManyRedirects));
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            MAX_REDIRECTS + 1
        );
    }

    #[tokio::test]
    async fn slow_servers_time_out() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(html_response("late").set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let fetcher = WebFetcher::new(
            AddressPolicy {
                allow_private: true,
                exempt: Vec::new(),
            },
            Duration::from_millis(200),
        );
        let error = fetcher.fetch(&server.uri(), 10).await.unwrap_err();
        assert!(matches!(error, FetchError::Timeout));
    }

    #[test]
    fn truncation_counts_characters_not_bytes() {
        assert_eq!(
            truncate_chars("héllo".to_owned(), 2),
            ("hé".to_owned(), true)
        );
        assert_eq!(
            truncate_chars("héllo".to_owned(), 5),
            ("héllo".to_owned(), false)
        );
        assert_eq!(truncate_chars(String::new(), 0), (String::new(), false));
    }
}
