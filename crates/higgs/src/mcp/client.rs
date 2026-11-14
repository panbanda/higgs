//! MCP client: request/response matching and the protocol handshake over a
//! [`Transport`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::{Notify, oneshot, watch};
use tokio::task::JoinHandle;

use super::jsonrpc::{
    self, METHOD_NOT_FOUND, Message, RpcError, TRANSPORT_FAILURE, error_response, success_response,
};
use super::transport::{Connection, Incoming, Transport};

pub const LATEST_PROTOCOL_VERSION: &str = "2025-06-18";
/// Revisions this client can speak, newest first. The differences between
/// them do not affect `initialize`, `tools/list` or `tools/call`.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Code for requests failed because the connection closed.
pub const CONNECTION_CLOSED: i64 = -32001;
const MAX_LIST_PAGES: usize = 100;
/// Cancellation is a courtesy; a server that stopped reading must not hold
/// the caller up for it.
const CANCEL_DELIVERY_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallResult {
    pub content: Vec<Value>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClientError {
    #[error("{0}")]
    Transport(String),
    #[error("server error: {0}")]
    Rpc(RpcError),
    #[error("no response within {0:?}")]
    Timeout(Duration),
    /// The connection was already gone before the request was written, so
    /// the server cannot have seen it.
    #[error("connection closed: {0}")]
    Closed(String),
    /// The connection was lost after the request was written. The server may
    /// or may not have acted on it.
    #[error("connection lost before the reply, outcome unknown: {0}")]
    Interrupted(String),
    #[error("protocol error: {0}")]
    Protocol(String),
}

type PendingReply = oneshot::Sender<Result<Value, RpcError>>;

struct Shared {
    pending: Mutex<HashMap<u64, PendingReply>>,
    closed: watch::Sender<Option<String>>,
    tools_changed: Notify,
}

pub struct McpClient {
    transport: Arc<dyn Transport>,
    shared: Arc<Shared>,
    next_id: AtomicU64,
    reader: JoinHandle<()>,
    protocol_version: String,
    server_info: Option<ServerInfo>,
    supports_tools: bool,
}

impl McpClient {
    /// Run the `initialize` handshake over `connection`. On failure the
    /// connection is closed before the error is returned.
    pub async fn connect(connection: Connection, timeout: Duration) -> Result<Self, ClientError> {
        let shared = Arc::new(Shared {
            pending: Mutex::new(HashMap::new()),
            closed: watch::channel(None).0,
            tools_changed: Notify::new(),
        });
        let reader = tokio::spawn(read_loop(
            Arc::clone(&shared),
            Arc::clone(&connection.transport),
            connection.incoming,
        ));
        let mut client = Self {
            transport: connection.transport,
            shared,
            next_id: AtomicU64::new(1),
            reader,
            protocol_version: String::new(),
            server_info: None,
            supports_tools: false,
        };
        if let Err(error) = client.initialize(timeout).await {
            client.close().await;
            return Err(error);
        }
        Ok(client)
    }

    async fn initialize(&mut self, timeout: Duration) -> Result<(), ClientError> {
        let params = json!({
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "higgs", "version": env!("CARGO_PKG_VERSION")},
        });
        let result = self.request("initialize", Some(params), timeout).await?;
        let negotiated = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ClientError::Protocol("initialize result has no protocolVersion".to_owned())
            })?;
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&negotiated) {
            return Err(ClientError::Protocol(format!(
                "server requires protocol version {negotiated}; supported: {}",
                SUPPORTED_PROTOCOL_VERSIONS.join(", ")
            )));
        }
        negotiated.clone_into(&mut self.protocol_version);
        self.supports_tools = result
            .get("capabilities")
            .and_then(|capabilities| capabilities.get("tools"))
            .is_some();
        self.server_info = result.get("serverInfo").map(|info| ServerInfo {
            name: info
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            version: info
                .get("version")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        });
        self.transport.set_protocol_version(&self.protocol_version);
        self.transport
            .send(jsonrpc::notification("notifications/initialized", None))
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        self.transport.start_notifications();
        Ok(())
    }

    pub fn protocol_version(&self) -> &str {
        &self.protocol_version
    }

    pub const fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }

    pub const fn supports_tools(&self) -> bool {
        self.supports_tools
    }

    pub fn is_closed(&self) -> bool {
        self.shared.closed.borrow().is_some()
    }

    /// Resolves with the reason once the connection is gone.
    pub async fn closed(&self) -> String {
        let mut receiver = self.shared.closed.subscribe();
        let outcome = receiver
            .wait_for(Option::is_some)
            .await
            .map(|reason| reason.clone().unwrap_or_default());
        outcome.unwrap_or_else(|_| "client dropped".to_owned())
    }

    /// Resolves when the server announces `notifications/tools/list_changed`.
    /// A notification that arrives while nobody waits is remembered.
    pub async fn tools_changed(&self) {
        self.shared.tools_changed.notified().await;
    }

    pub async fn close(&self) {
        self.transport.close().await;
        self.reader.abort();
        fail_all(&self.shared, "closed by client");
    }

    pub async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, ClientError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply_sender, reply_receiver) = oneshot::channel();
        lock(&self.shared.pending).insert(id, reply_sender);
        let closed_reason = self.shared.closed.borrow().clone();
        if let Some(reason) = closed_reason {
            lock(&self.shared.pending).remove(&id);
            return Err(ClientError::Closed(reason));
        }
        let mut guard = PendingRequest {
            shared: Arc::clone(&self.shared),
            transport: Arc::clone(&self.transport),
            id,
            armed: true,
        };
        let exchange = async {
            self.transport
                .send(jsonrpc::request(id, method, params))
                .await
                .map_err(|error| ClientError::Transport(error.to_string()))?;
            match reply_receiver.await {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(error)) => Err(classify(error)),
                Err(_) => Err(ClientError::Interrupted("reply channel dropped".to_owned())),
            }
        };
        let Ok(outcome) = tokio::time::timeout(timeout, exchange).await else {
            guard.armed = false;
            lock(&self.shared.pending).remove(&id);
            let _ = tokio::time::timeout(
                CANCEL_DELIVERY_TIMEOUT,
                self.transport.send(cancellation(id, "timed out")),
            )
            .await;
            return Err(ClientError::Timeout(timeout));
        };
        guard.armed = false;
        if outcome.is_err() {
            lock(&self.shared.pending).remove(&id);
        }
        outcome
    }

    /// Fetch every page of `tools/list`.
    pub async fn list_tools(&self, timeout: Duration) -> Result<Vec<McpTool>, ClientError> {
        if !self.supports_tools {
            return Ok(Vec::new());
        }
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen_cursors = HashSet::new();
        for _ in 0..MAX_LIST_PAGES {
            let params = cursor.as_ref().map(|value| json!({"cursor": value}));
            let page = self.request("tools/list", params, timeout).await?;
            let entries = page.get("tools").and_then(Value::as_array).ok_or_else(|| {
                ClientError::Protocol("tools/list result has no tools array".to_owned())
            })?;
            tools.extend(entries.iter().filter_map(parse_tool));
            match page.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => {
                    if !seen_cursors.insert(next.to_owned()) {
                        return Err(ClientError::Protocol(
                            "tools/list pagination returned a cursor twice".to_owned(),
                        ));
                    }
                    cursor = Some(next.to_owned());
                }
                Some(_) | None => return Ok(tools),
            }
        }
        Err(ClientError::Protocol(format!(
            "tools/list has more than {MAX_LIST_PAGES} pages"
        )))
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        timeout: Duration,
    ) -> Result<ToolCallResult, ClientError> {
        let result = self
            .request(
                "tools/call",
                Some(json!({"name": name, "arguments": arguments})),
                timeout,
            )
            .await?;
        Ok(ToolCallResult {
            content: result
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            structured_content: result.get("structuredContent").cloned(),
            is_error: result
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// Cleans up after a request future that is dropped before it completes, for
/// example by a caller's own shorter timeout: the reply slot would otherwise
/// leak and the server would keep working on a result nobody awaits.
struct PendingRequest {
    shared: Arc<Shared>,
    transport: Arc<dyn Transport>,
    id: u64,
    armed: bool,
}

impl Drop for PendingRequest {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        lock(&self.shared.pending).remove(&self.id);
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let transport = Arc::clone(&self.transport);
        let message = cancellation(self.id, "abandoned by caller");
        runtime.spawn(async move {
            let _ = tokio::time::timeout(CANCEL_DELIVERY_TIMEOUT, transport.send(message)).await;
        });
    }
}

fn cancellation(id: u64, reason: &str) -> Value {
    jsonrpc::notification(
        "notifications/cancelled",
        Some(json!({"requestId": id, "reason": reason})),
    )
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn classify(error: RpcError) -> ClientError {
    match error.code {
        CONNECTION_CLOSED => ClientError::Interrupted(error.message),
        TRANSPORT_FAILURE => ClientError::Transport(error.message),
        _ => ClientError::Rpc(error),
    }
}

fn parse_tool(entry: &Value) -> Option<McpTool> {
    let name = entry.get("name").and_then(Value::as_str)?;
    Some(McpTool {
        name: name.to_owned(),
        description: entry
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        input_schema: entry
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    })
}

fn fail_all(shared: &Shared, reason: &str) {
    shared.closed.send_if_modified(|current| {
        if current.is_none() {
            *current = Some(reason.to_owned());
            true
        } else {
            false
        }
    });
    let drained: Vec<PendingReply> = lock(&shared.pending)
        .drain()
        .map(|(_, reply)| reply)
        .collect();
    for reply in drained {
        let _ = reply.send(Err(RpcError {
            code: CONNECTION_CLOSED,
            message: reason.to_owned(),
            data: None,
        }));
    }
}

async fn read_loop(
    shared: Arc<Shared>,
    transport: Arc<dyn Transport>,
    mut incoming: tokio::sync::mpsc::Receiver<Incoming>,
) {
    let reason = loop {
        match incoming.recv().await {
            Some(Incoming::Message(value)) => dispatch(&shared, &transport, &value),
            Some(Incoming::Closed(reason)) => break reason,
            None => break "connection ended".to_owned(),
        }
    };
    fail_all(&shared, &reason);
}

fn dispatch(shared: &Shared, transport: &Arc<dyn Transport>, value: &Value) {
    let message = match Message::parse(value) {
        Ok(message) => message,
        Err(error) => {
            tracing::debug!(%error, "ignoring malformed MCP message");
            return;
        }
    };
    match message {
        Message::Response { id, outcome } => {
            let waiting = id
                .as_u64()
                .and_then(|numeric| lock(&shared.pending).remove(&numeric));
            if let Some(reply_sender) = waiting {
                let _ = reply_sender.send(outcome);
            } else {
                tracing::debug!(%id, "MCP response for an unknown or expired request");
            }
        }
        Message::Request { id, method, .. } => {
            let answer = if method == "ping" {
                success_response(&id, &json!({}))
            } else {
                error_response(
                    &id,
                    METHOD_NOT_FOUND,
                    &format!("client does not support {method}"),
                )
            };
            let replier = Arc::clone(transport);
            tokio::spawn(async move {
                let _ = replier.send(answer).await;
            });
        }
        Message::Notification { method, .. } => {
            if method == "notifications/tools/list_changed" {
                shared.tools_changed.notify_one();
            }
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::{ServerSpec, StdioSpec};
    use crate::mcp::testing::{CallBehavior, FakeFactory, FakeMcp, text_result, tool_definition};
    use crate::mcp::transport::TransportFactory;

    const TIMEOUT: Duration = Duration::from_secs(2);

    fn spec() -> ServerSpec {
        ServerSpec::Stdio(StdioSpec {
            command: "unused".to_owned(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
        })
    }

    async fn connected(server: &Arc<FakeMcp>) -> McpClient {
        let factory = FakeFactory::single("fake", server);
        let connection = factory.connect("fake", &spec()).await.unwrap();
        McpClient::connect(connection, TIMEOUT).await.unwrap()
    }

    #[tokio::test]
    async fn handshake_negotiates_version_and_sends_initialized() {
        let server = FakeMcp::new(&["echo"]);
        let client = connected(&server).await;
        assert_eq!(client.protocol_version(), "2025-06-18");
        assert_eq!(
            client.server_info(),
            Some(&ServerInfo {
                name: "fake-mcp".to_owned(),
                version: "9.9".to_owned()
            })
        );
        assert!(client.supports_tools());
        assert_eq!(
            server.received_methods(),
            vec!["initialize", "notifications/initialized"]
        );
    }

    #[tokio::test]
    async fn older_supported_version_is_accepted() {
        let server = FakeMcp::new(&[]);
        *server.protocol_version.lock().unwrap() = "2024-11-05".to_owned();
        let client = connected(&server).await;
        assert_eq!(client.protocol_version(), "2024-11-05");
    }

    #[tokio::test]
    async fn unsupported_version_fails_the_handshake() {
        let server = FakeMcp::new(&[]);
        *server.protocol_version.lock().unwrap() = "2030-01-01".to_owned();
        let factory = FakeFactory::single("fake", &server);
        let connection = factory.connect("fake", &spec()).await.unwrap();
        let error = McpClient::connect(connection, TIMEOUT).await.err().unwrap();
        assert!(
            matches!(&error, ClientError::Protocol(message) if message.contains("2030-01-01")),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn list_tools_follows_pagination() {
        let server = FakeMcp::new(&[]);
        server.set_pages(vec![
            vec![tool_definition("a"), tool_definition("b")],
            vec![tool_definition("c")],
            vec![tool_definition("d")],
        ]);
        let client = connected(&server).await;
        let tools = client.list_tools(TIMEOUT).await.unwrap();
        let names: Vec<_> = tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c", "d"]);
        assert_eq!(tools[0].description, "a tool");
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[tokio::test]
    async fn server_without_tools_capability_lists_nothing_without_asking() {
        let server = FakeMcp::new(&["echo"]);
        *server.has_tools_capability.lock().unwrap() = false;
        let client = connected(&server).await;
        assert!(client.list_tools(TIMEOUT).await.unwrap().is_empty());
        assert!(!server.received_methods().contains(&"tools/list".to_owned()));
    }

    #[tokio::test]
    async fn call_tool_returns_content_and_error_flag() {
        let server = FakeMcp::new(&["echo", "broken"]);
        server.on_call("echo", CallBehavior::Reply(text_result("hi")));
        server.on_call(
            "broken",
            CallBehavior::Reply(json!({
                "content": [{"type": "text", "text": "bad input"}],
                "isError": true,
                "structuredContent": {"code": 7}
            })),
        );
        let client = connected(&server).await;

        let ok = client
            .call_tool("echo", json!({"text": "x"}), TIMEOUT)
            .await
            .unwrap();
        assert!(!ok.is_error);
        assert_eq!(ok.content[0]["text"], "hi");
        assert_eq!(
            server.received_calls()[0],
            json!({"name": "echo", "arguments": {"text": "x"}})
        );

        let failed = client
            .call_tool("broken", json!({}), TIMEOUT)
            .await
            .unwrap();
        assert!(failed.is_error);
        assert_eq!(failed.structured_content, Some(json!({"code": 7})));
    }

    #[tokio::test]
    async fn rpc_errors_are_surfaced() {
        let server = FakeMcp::new(&["echo"]);
        server.on_call(
            "echo",
            CallBehavior::RpcError(-32602, "bad params".to_owned()),
        );
        let client = connected(&server).await;
        let error = client
            .call_tool("echo", json!({}), TIMEOUT)
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ClientError::Rpc(RpcError {
                code: -32602,
                message: "bad params".to_owned(),
                data: None
            })
        );
    }

    #[tokio::test]
    async fn timeout_cancels_the_request_and_later_calls_still_work() {
        let server = FakeMcp::new(&["slow", "echo"]);
        server.on_call("slow", CallBehavior::Hang);
        let client = connected(&server).await;
        let error = client
            .call_tool("slow", json!({}), Duration::from_millis(50))
            .await
            .unwrap_err();
        assert_eq!(error, ClientError::Timeout(Duration::from_millis(50)));
        assert!(
            server
                .received_methods()
                .contains(&"notifications/cancelled".to_owned())
        );
        assert!(lock(&client.shared.pending).is_empty());
        assert!(client.call_tool("echo", json!({}), TIMEOUT).await.is_ok());
    }

    #[tokio::test]
    async fn dropping_a_request_early_clears_it_and_cancels_it_on_the_server() {
        let server = FakeMcp::new(&["slow"]);
        server.on_call("slow", CallBehavior::Hang);
        let client = connected(&server).await;
        let outcome = tokio::time::timeout(
            Duration::from_millis(50),
            client.call_tool("slow", json!({}), TIMEOUT),
        )
        .await;
        assert!(outcome.is_err());
        assert!(lock(&client.shared.pending).is_empty());
        for _ in 0..100 {
            if server
                .received_methods()
                .contains(&"notifications/cancelled".to_owned())
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the server never received a cancellation");
    }

    #[tokio::test]
    async fn a_stalled_write_is_bounded_by_the_request_timeout() {
        let server = FakeMcp::new(&["echo"]);
        let client = connected(&server).await;
        server.stall_sends.store(true, Ordering::SeqCst);
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            client.call_tool("echo", json!({}), Duration::from_millis(50)),
        )
        .await
        .unwrap();
        assert_eq!(
            outcome.unwrap_err(),
            ClientError::Timeout(Duration::from_millis(50))
        );
        assert!(lock(&client.shared.pending).is_empty());
    }

    #[tokio::test]
    async fn crash_fails_pending_calls_and_marks_the_client_closed() {
        let server = FakeMcp::new(&["boom", "echo"]);
        server.on_call("boom", CallBehavior::Crash);
        let client = connected(&server).await;
        let error = client
            .call_tool("boom", json!({}), TIMEOUT)
            .await
            .unwrap_err();
        assert!(matches!(error, ClientError::Interrupted(_)), "{error:?}");
        assert!(client.closed().await.contains("exited"));
        assert!(client.is_closed());
        let after = client
            .call_tool("echo", json!({}), TIMEOUT)
            .await
            .unwrap_err();
        assert!(matches!(after, ClientError::Closed(_)));
    }

    #[tokio::test]
    async fn list_changed_notification_is_observable_even_before_waiting() {
        let server = FakeMcp::new(&["a"]);
        let client = connected(&server).await;
        server.notify_tools_changed().await;
        tokio::time::timeout(TIMEOUT, client.tools_changed())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn server_pings_are_answered_and_unknown_requests_rejected() {
        let server = FakeMcp::new(&[]);
        let client = connected(&server).await;
        let transport = Arc::clone(&client.transport);
        dispatch(
            &client.shared,
            &transport,
            &json!({"jsonrpc":"2.0","id":"p1","method":"ping"}),
        );
        dispatch(
            &client.shared,
            &transport,
            &json!({"jsonrpc":"2.0","id":"s1","method":"sampling/createMessage"}),
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        let replies: Vec<Value> = server
            .received
            .lock()
            .unwrap()
            .iter()
            .filter(|message| message.get("method").is_none())
            .cloned()
            .collect();
        assert_eq!(replies.len(), 2);
        assert_eq!(replies[0]["result"], json!({}));
        assert_eq!(replies[1]["error"]["code"], METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn pagination_cursor_loops_are_rejected() {
        let server = FakeMcp::new(&["a"]);
        *server.loop_cursor.lock().unwrap() = true;
        let client = connected(&server).await;
        let error = client.list_tools(TIMEOUT).await.unwrap_err();
        assert!(
            matches!(&error, ClientError::Protocol(message) if message.contains("cursor")),
            "{error:?}"
        );
    }
}
