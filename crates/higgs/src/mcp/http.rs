//! Streamable HTTP transport (MCP spec revisions 2025-03-26 and later).
//!
//! Each client message is a POST. The server answers a request with either a
//! single JSON body or an SSE stream that carries the response and possibly
//! other messages; both are forwarded to the same incoming channel stdio
//! uses. A failure to obtain a response is delivered as a synthesized
//! JSON-RPC error for that request so the client never waits on a request
//! the transport already knows is lost.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::future::BoxFuture;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use super::config::HttpSpec;
use super::jsonrpc::{TRANSPORT_FAILURE, error_response};
use super::sse::{SseEvent, SseParser};
use super::transport::{
    Connection, INCOMING_CAPACITY, Incoming, MAX_MESSAGE_BYTES, Transport, TransportError,
};

const SESSION_HEADER: &str = "mcp-session-id";
const PROTOCOL_HEADER: &str = "mcp-protocol-version";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const ERROR_BODY_PREVIEW: usize = 300;

struct Shared {
    client: reqwest::Client,
    url: reqwest::Url,
    headers: HeaderMap,
    request_timeout: Duration,
    session: Mutex<Option<String>>,
    protocol_version: Mutex<Option<String>>,
    incoming: mpsc::Sender<Incoming>,
}

pub struct HttpTransport {
    shared: Arc<Shared>,
    tasks: Mutex<JoinSet<()>>,
}

pub fn connect(spec: &HttpSpec) -> Result<Connection, TransportError> {
    let mut headers = HeaderMap::new();
    for (name, value) in &spec.headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| TransportError::Start(format!("invalid header name '{name}'")))?;
        let mut header_value = HeaderValue::from_str(value)
            .map_err(|_| TransportError::Start(format!("invalid value for header '{name}'")))?;
        header_value.set_sensitive(true);
        headers.insert(header_name, header_value);
    }
    // Redirects are not followed: replaying a POST body and credentials to a
    // different location is never what a configured MCP endpoint wants.
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .user_agent(concat!("higgs/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|error| TransportError::Start(error.to_string()))?;
    let (sender, receiver) = mpsc::channel(INCOMING_CAPACITY);
    let transport = HttpTransport {
        shared: Arc::new(Shared {
            client,
            url: spec.url.clone(),
            headers,
            request_timeout: spec.request_timeout,
            session: Mutex::new(None),
            protocol_version: Mutex::new(None),
            incoming: sender,
        }),
        tasks: Mutex::new(JoinSet::new()),
    };
    Ok(Connection {
        transport: Arc::new(transport),
        incoming: receiver,
    })
}

#[derive(Debug)]
struct PostFailure {
    message: String,
    /// The session no longer exists; the connection cannot be reused.
    fatal: bool,
}

impl PostFailure {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn decorate(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = builder.headers(self.headers.clone());
        let session = lock(&self.session).clone();
        if let Some(session_id) = session {
            request = request.header(SESSION_HEADER, session_id);
        }
        let version = lock(&self.protocol_version).clone();
        if let Some(version_id) = version {
            request = request.header(PROTOCOL_HEADER, version_id);
        }
        request
    }

    async fn deliver(&self, value: Value) -> Vec<Value> {
        let items = match value {
            Value::Array(items) => items,
            single @ (Value::Null
            | Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Object(_)) => vec![single],
        };
        for item in &items {
            let _ = self.incoming.send(Incoming::Message(item.clone())).await;
        }
        items
    }

    async fn post(&self, message: &Value, expected_id: Option<&Value>) -> Result<(), PostFailure> {
        let response = self
            .decorate(self.client.post(self.url.clone()))
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .body(message.to_string())
            .send()
            .await
            .map_err(|error| PostFailure::new(format!("request failed: {error}")))?;

        let status = response.status();
        if let Some(session) = response
            .headers()
            .get(SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            *lock(&self.session) = Some(session.to_owned());
        }
        if !status.is_success() {
            let expired = status == reqwest::StatusCode::NOT_FOUND && lock(&self.session).is_some();
            let body = read_body(response, ERROR_BODY_PREVIEW * 4)
                .await
                .unwrap_or_default();
            let preview: String = body.chars().take(ERROR_BODY_PREVIEW).collect();
            return Err(PostFailure {
                message: if expired {
                    "the server no longer recognizes the session (HTTP 404)".to_owned()
                } else {
                    format!("HTTP {status}: {preview}")
                },
                fatal: expired,
            });
        }
        if status == reqwest::StatusCode::ACCEPTED || status == reqwest::StatusCode::NO_CONTENT {
            return Ok(());
        }

        let is_stream = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"));
        let answered = if is_stream {
            self.forward_stream(response, expected_id).await?
        } else {
            let body = read_body(response, MAX_MESSAGE_BYTES)
                .await
                .map_err(PostFailure::new)?;
            if body.trim().is_empty() {
                false
            } else {
                let value: Value = serde_json::from_str(&body)
                    .map_err(|error| PostFailure::new(format!("response is not JSON: {error}")))?;
                let delivered = self.deliver(value).await;
                answers(&delivered, expected_id)
            }
        };
        match (expected_id, answered) {
            (Some(_), false) => Err(PostFailure::new(
                "the server ended the response without answering",
            )),
            (Some(_) | None, true) | (None, false) => Ok(()),
        }
    }

    async fn forward_stream(
        &self,
        mut response: reqwest::Response,
        expected_id: Option<&Value>,
    ) -> Result<bool, PostFailure> {
        let mut parser = SseParser::default();
        let mut total = 0_usize;
        let mut answered = false;
        loop {
            let chunk = response
                .chunk()
                .await
                .map_err(|error| PostFailure::new(format!("stream failed: {error}")))?;
            let Some(bytes) = chunk else {
                if let Some(event) = parser.finish() {
                    answered |= self.forward_event(&event, expected_id).await;
                }
                return Ok(answered);
            };
            total += bytes.len();
            if total > MAX_MESSAGE_BYTES * 2 {
                return Err(PostFailure::new("response stream is too large"));
            }
            for event in parser.feed(&String::from_utf8_lossy(&bytes)) {
                answered |= self.forward_event(&event, expected_id).await;
            }
            if answered {
                return Ok(true);
            }
        }
    }

    async fn forward_event(&self, event: &SseEvent, expected_id: Option<&Value>) -> bool {
        match serde_json::from_str::<Value>(&event.data) {
            Ok(value) => {
                let delivered = self.deliver(value).await;
                answers(&delivered, expected_id)
            }
            Err(error) => {
                tracing::debug!(%error, "ignoring non-JSON SSE event from MCP server");
                false
            }
        }
    }

    async fn listen(&self) {
        let Ok(mut response) = self
            .decorate(self.client.get(self.url.clone()))
            .header(ACCEPT, "text/event-stream")
            .send()
            .await
        else {
            return;
        };
        let is_stream = response.status().is_success()
            && response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
        if !is_stream {
            return;
        }
        let mut parser = SseParser::default();
        while let Ok(Some(bytes)) = response.chunk().await {
            for event in parser.feed(&String::from_utf8_lossy(&bytes)) {
                if let Ok(value) = serde_json::from_str::<Value>(&event.data) {
                    self.deliver(value).await;
                }
            }
        }
    }
}

fn answers(delivered: &[Value], expected_id: Option<&Value>) -> bool {
    expected_id.is_some_and(|expected| {
        delivered.iter().any(|message| {
            message.get("id") == Some(expected)
                && (message.get("result").is_some() || message.get("error").is_some())
        })
    })
}

async fn read_body(mut response: reqwest::Response, limit: usize) -> Result<String, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("cannot read response: {error}"))?
    {
        if body.len() + chunk.len() > limit {
            return Err(format!("response exceeds the {limit} byte limit"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

impl Transport for HttpTransport {
    fn send(&self, message: Value) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move {
            let shared = Arc::clone(&self.shared);
            let request_id = message
                .get("method")
                .and_then(|_| message.get("id"))
                .filter(|id| !id.is_null())
                .cloned();
            lock(&self.tasks).spawn(async move {
                let outcome = tokio::time::timeout(
                    shared.request_timeout,
                    shared.post(&message, request_id.as_ref()),
                )
                .await
                .unwrap_or_else(|_| Err(PostFailure::new("request timed out")));
                let Err(failure) = outcome else { return };
                if let Some(id) = &request_id {
                    let synthesized = error_response(id, TRANSPORT_FAILURE, &failure.message);
                    let _ = shared.incoming.send(Incoming::Message(synthesized)).await;
                } else {
                    tracing::debug!(error = %failure.message, "MCP notification was not delivered");
                }
                if failure.fatal {
                    let _ = shared
                        .incoming
                        .send(Incoming::Closed(failure.message))
                        .await;
                }
            });
            Ok(())
        })
    }

    fn set_protocol_version(&self, version: &str) {
        *lock(&self.shared.protocol_version) = Some(version.to_owned());
    }

    fn start_notifications(&self) {
        let shared = Arc::clone(&self.shared);
        lock(&self.tasks).spawn(async move { shared.listen().await });
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            lock(&self.tasks).abort_all();
            let session = lock(&self.shared.session).take();
            if let Some(session_id) = session {
                let _ = tokio::time::timeout(
                    Duration::from_secs(2),
                    self.shared
                        .client
                        .delete(self.shared.url.clone())
                        .headers(self.shared.headers.clone())
                        .header(SESSION_HEADER, session_id)
                        .send(),
                )
                .await;
            }
        })
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::mcp::client::{ClientError, McpClient};

    const TIMEOUT: Duration = Duration::from_secs(5);

    fn spec(server: &MockServer, headers: Vec<(String, String)>, timeout: Duration) -> HttpSpec {
        HttpSpec {
            url: reqwest::Url::parse(&format!("{}/mcp", server.uri())).unwrap(),
            headers,
            request_timeout: timeout,
        }
    }

    fn rpc_result(id: u64, result: &Value) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    async fn mount_handshake(server: &MockServer, with_session: bool) {
        let mut reply = rpc_result(
            1,
            &json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "http-fake", "version": "2"},
            }),
        );
        if with_session {
            reply = reply.insert_header("Mcp-Session-Id", "session-1");
        }
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .and(body_partial_json(json!({"method": "initialize"})))
            .respond_with(reply)
            .mount(server)
            .await;
        Mock::given(method("POST"))
            .and(body_partial_json(
                json!({"method": "notifications/initialized"}),
            ))
            .respond_with(ResponseTemplate::new(202))
            .mount(server)
            .await;
    }

    async fn client_for(server: &MockServer, headers: Vec<(String, String)>) -> McpClient {
        let connection = connect(&spec(server, headers, TIMEOUT)).unwrap();
        McpClient::connect(connection, TIMEOUT).await.unwrap()
    }

    #[tokio::test]
    async fn handshake_sends_accept_headers_and_negotiates_the_older_version() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        let client = client_for(&server, vec![("X-Api-Key".to_owned(), "k".to_owned())]).await;
        assert_eq!(client.protocol_version(), "2025-03-26");
        assert_eq!(client.server_info().unwrap().name, "http-fake");

        let requests = server.received_requests().await.unwrap();
        let initialize = &requests[0];
        assert_eq!(initialize.headers.get("x-api-key").unwrap(), "k");
        let accept = initialize.headers.get("accept").unwrap().to_str().unwrap();
        assert!(accept.contains("application/json") && accept.contains("text/event-stream"));
    }

    #[tokio::test]
    async fn session_and_protocol_version_headers_follow_the_handshake() {
        let server = MockServer::start().await;
        mount_handshake(&server, true).await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/list"})))
            .and(header("mcp-session-id", "session-1"))
            .and(header("mcp-protocol-version", "2025-03-26"))
            .respond_with(rpc_result(
                2,
                &json!({"tools": [{"name": "t", "inputSchema": {"type": "object"}}]}),
            ))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let tools = client.list_tools(TIMEOUT).await.unwrap();
        assert_eq!(tools[0].name, "t");
    }

    #[tokio::test]
    async fn sse_responses_deliver_notifications_before_the_result() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        let body = format!(
            "event: message\ndata: {}\n\nevent: message\ndata: {}\n\n",
            json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}),
            json!({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "streamed"}]}}),
        );
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let result = client.call_tool("t", json!({}), TIMEOUT).await.unwrap();
        assert_eq!(result.content[0]["text"], "streamed");
        tokio::time::timeout(TIMEOUT, client.tools_changed())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn http_errors_fail_only_the_request_that_hit_them() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let error = client.call_tool("t", json!({}), TIMEOUT).await.unwrap_err();
        let ClientError::Transport(message) = error else {
            panic!("expected a transport error")
        };
        assert!(
            message.contains("500") && message.contains("upstream exploded"),
            "{message}"
        );
        assert!(!client.is_closed());
    }

    #[tokio::test]
    async fn an_expired_session_closes_the_connection() {
        let server = MockServer::start().await;
        mount_handshake(&server, true).await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let error = client.call_tool("t", json!({}), TIMEOUT).await.unwrap_err();
        assert!(
            matches!(error, ClientError::Transport(_) | ClientError::Closed(_)),
            "{error:?}"
        );
        assert!(
            tokio::time::timeout(TIMEOUT, client.closed())
                .await
                .unwrap()
                .contains("session")
        );
    }

    #[tokio::test]
    async fn a_response_that_never_arrives_times_out_in_the_transport() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(rpc_result(2, &json!({"content": []})).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let connection = connect(&spec(&server, Vec::new(), Duration::from_millis(300))).unwrap();
        let client = McpClient::connect(connection, TIMEOUT).await.unwrap();
        let error = client.call_tool("t", json!({}), TIMEOUT).await.unwrap_err();
        assert!(
            matches!(&error, ClientError::Transport(message) if message.contains("timed out")),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn a_stream_that_ends_without_the_answer_is_an_error() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(": just a comment\n\n", "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let error = client.call_tool("t", json!({}), TIMEOUT).await.unwrap_err();
        assert!(
            matches!(&error, ClientError::Transport(message) if message.contains("without answering")),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn the_server_to_client_stream_delivers_list_changed() {
        let server = MockServer::start().await;
        mount_handshake(&server, true).await;
        Mock::given(method("GET"))
            .and(path("/mcp"))
            .and(header("mcp-session-id", "session-1"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                format!(
                    "data: {}\n\n",
                    json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
                ),
                "text/event-stream",
            ))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        tokio::time::timeout(TIMEOUT, client.tools_changed())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_server_without_a_listening_endpoint_is_fine() {
        let server = MockServer::start().await;
        mount_handshake(&server, false).await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(405))
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!client.is_closed());
    }

    #[tokio::test]
    async fn closing_deletes_the_session_and_redirects_are_not_followed() {
        let server = MockServer::start().await;
        mount_handshake(&server, true).await;
        Mock::given(method("DELETE"))
            .and(header("mcp-session-id", "session-1"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": "tools/call"})))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("location", "http://127.0.0.1:1/elsewhere"),
            )
            .mount(&server)
            .await;
        let client = client_for(&server, Vec::new()).await;
        let error = client.call_tool("t", json!({}), TIMEOUT).await.unwrap_err();
        assert!(
            matches!(&error, ClientError::Transport(message) if message.contains("307")),
            "{error:?}"
        );
        client.close().await;
        server.verify().await;
    }

    #[test]
    fn invalid_header_values_are_rejected_up_front() {
        let spec = HttpSpec {
            url: reqwest::Url::parse("http://localhost/mcp").unwrap(),
            headers: vec![("X".to_owned(), "line\nbreak".to_owned())],
            request_timeout: TIMEOUT,
        };
        assert!(matches!(connect(&spec), Err(TransportError::Start(_))));
    }
}
