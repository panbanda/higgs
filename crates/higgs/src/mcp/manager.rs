//! Supervision of the configured MCP servers.
//!
//! Every enabled server gets a supervisor task that connects in the
//! background (so a slow or broken server never delays startup), publishes
//! its state, keeps the tool list fresh, and reconnects with exponential
//! backoff when the server fails or crashes. A tool call against a failed
//! server cuts the backoff short, which restarts a crashed stdio server on
//! its next use.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use super::client::{ClientError, McpClient, McpTool, ServerInfo, ToolCallResult};
use super::config::{EnvLookup, McpConfig, NAMESPACE_SEPARATOR, ResolvedServer, TransportKind};
use super::transport::{DefaultTransportFactory, TransportFactory};

#[derive(Debug, Clone, Copy)]
pub struct BackoffPolicy {
    pub initial: Duration,
    pub maximum: Duration,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            initial: Duration::from_secs(1),
            maximum: Duration::from_secs(60),
        }
    }
}

impl BackoffPolicy {
    fn delay(self, consecutive_failures: u32) -> Duration {
        let exponent = consecutive_failures.saturating_sub(1).min(16);
        self.initial
            .saturating_mul(2_u32.saturating_pow(exponent))
            .min(self.maximum)
    }
}

pub struct ReadyServer {
    client: Arc<McpClient>,
    tools: Vec<McpTool>,
    /// Connection attempts finished before this one, for noticing a newer
    /// attempt than the one a caller already saw.
    generation: u64,
}

#[derive(Clone)]
enum ServerState {
    Disabled,
    Connecting { generation: u64 },
    Ready(Arc<ReadyServer>),
    Failed { error: String, generation: u64 },
}

struct ServerHandle {
    transport: TransportKind,
    timeout: Duration,
    state: watch::Sender<ServerState>,
    kick: Notify,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub name: String,
    pub transport: &'static str,
    pub state: &'static str,
    pub error: Option<String>,
    pub tool_count: usize,
    pub protocol_version: Option<String>,
    pub server_info: Option<ServerInfo>,
}

#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub server: String,
    /// Namespaced `server__tool` name used in requests.
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: Vec<Value>,
    /// The text blocks of `content`, joined by newlines.
    pub text: String,
    pub structured_content: Option<Value>,
    pub is_error: bool,
    pub truncated: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("unknown tool '{0}'")]
    UnknownTool(String),
    #[error("MCP server '{server}' is {state}{detail}")]
    NotReady {
        server: String,
        state: &'static str,
        /// Empty, or `: <last error>`.
        detail: String,
    },
    #[error("tool call timed out after {0:?}")]
    Timeout(Duration),
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    #[error("MCP server error: {0}")]
    Upstream(String),
}

pub struct McpManager {
    servers: BTreeMap<String, Arc<ServerHandle>>,
    max_result_bytes: usize,
    shutdown: watch::Sender<bool>,
    supervisors: Mutex<Vec<JoinHandle<()>>>,
}

impl McpManager {
    /// A manager with no servers: starts nothing and reports nothing.
    pub fn empty() -> Self {
        Self {
            servers: BTreeMap::new(),
            max_result_bytes: McpConfig::default().max_result_bytes,
            shutdown: watch::channel(false).0,
            supervisors: Mutex::new(Vec::new()),
        }
    }

    /// Resolve `config` and begin connecting in the background. Must be
    /// called from within a Tokio runtime.
    pub fn start(config: &McpConfig, lookup: EnvLookup<'_>) -> Result<Self, String> {
        let servers = config.resolve(lookup)?;
        let factory: Arc<dyn TransportFactory> = Arc::new(DefaultTransportFactory);
        Ok(Self::start_with(
            servers,
            config.max_result_bytes,
            &factory,
            BackoffPolicy::default(),
        ))
    }

    pub fn start_with(
        servers: Vec<ResolvedServer>,
        max_result_bytes: usize,
        factory: &Arc<dyn TransportFactory>,
        backoff: BackoffPolicy,
    ) -> Self {
        let (shutdown, _) = watch::channel(false);
        let mut handles = BTreeMap::new();
        let mut supervisors = Vec::new();
        for server in servers {
            let initial = if server.enabled {
                ServerState::Connecting { generation: 0 }
            } else {
                ServerState::Disabled
            };
            let handle = Arc::new(ServerHandle {
                transport: server.spec.kind(),
                timeout: server.timeout,
                state: watch::channel(initial).0,
                kick: Notify::new(),
            });
            if server.enabled {
                supervisors.push(tokio::spawn(supervise(
                    Arc::clone(&handle),
                    server.clone(),
                    Arc::clone(factory),
                    backoff,
                    shutdown.subscribe(),
                )));
            }
            handles.insert(server.name, handle);
        }
        Self {
            servers: handles,
            max_result_bytes,
            shutdown,
            supervisors: Mutex::new(supervisors),
        }
    }

    pub fn statuses(&self) -> Vec<ServerStatus> {
        self.servers
            .iter()
            .map(|(name, handle)| {
                let state = handle.state.borrow().clone();
                let mut status = ServerStatus {
                    name: name.clone(),
                    transport: handle.transport.label(),
                    state: "connecting",
                    error: None,
                    tool_count: 0,
                    protocol_version: None,
                    server_info: None,
                };
                match state {
                    ServerState::Disabled => status.state = "disabled",
                    ServerState::Connecting { .. } => {}
                    ServerState::Ready(ready) => {
                        status.state = "ready";
                        status.tool_count = ready.tools.len();
                        status.protocol_version = Some(ready.client.protocol_version().to_owned());
                        status.server_info = ready.client.server_info().cloned();
                    }
                    ServerState::Failed { error, .. } => {
                        status.state = "failed";
                        status.error = Some(error);
                    }
                }
                status
            })
            .collect()
    }

    /// Tools of every server that is ready, namespaced.
    pub fn tools(&self) -> Vec<ToolInfo> {
        let mut tools = Vec::new();
        for (name, handle) in &self.servers {
            let state = handle.state.borrow().clone();
            if let ServerState::Ready(ready) = state {
                tools.extend(ready.tools.iter().map(|tool| ToolInfo {
                    server: name.clone(),
                    name: namespaced(name, &tool.name),
                    description: tool.description.clone(),
                    input_schema: tool.input_schema.clone(),
                }));
            }
        }
        tools
    }

    /// Whether `full_name` addresses a server managed here, ready or not.
    pub fn owns(&self, full_name: &str) -> bool {
        split_name(full_name).is_some_and(|(server, _)| self.servers.contains_key(server))
    }

    pub async fn call(
        &self,
        full_name: &str,
        arguments: Value,
    ) -> Result<ToolOutput, ToolCallError> {
        let unknown = || ToolCallError::UnknownTool(full_name.to_owned());
        let (server_name, tool_name) = split_name(full_name).ok_or_else(unknown)?;
        let handle = self.servers.get(server_name).ok_or_else(unknown)?;
        let call_arguments = match arguments {
            Value::Null => Value::Object(serde_json::Map::new()),
            object @ Value::Object(_) => object,
            Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
                return Err(ToolCallError::InvalidArguments(
                    "arguments must be a JSON object".to_owned(),
                ));
            }
        };

        let mut retried = false;
        loop {
            let ready = ready_server(server_name, handle).await?;
            if !ready.tools.iter().any(|tool| tool.name == tool_name) {
                return Err(unknown());
            }
            match ready
                .client
                .call_tool(tool_name, call_arguments.clone(), handle.timeout)
                .await
            {
                Ok(result) => return Ok(build_output(result, self.max_result_bytes)),
                // The server died between the state check and the call; the
                // supervisor has not reconnected yet, so give it one chance.
                Err(ClientError::Closed(_)) if !retried => retried = true,
                Err(error) => return Err(map_error(error)),
            }
        }
    }

    /// Close every connection and wait for child processes to exit.
    pub async fn shutdown(&self) {
        self.shutdown.send_replace(true);
        let supervisors: Vec<_> = self
            .supervisors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain(..)
            .collect();
        for supervisor in supervisors {
            let _ = tokio::time::timeout(Duration::from_secs(10), supervisor).await;
        }
    }
}

impl Drop for McpManager {
    fn drop(&mut self) {
        self.shutdown.send_replace(true);
    }
}

fn namespaced(server: &str, tool: &str) -> String {
    format!("{server}{NAMESPACE_SEPARATOR}{tool}")
}

fn split_name(full_name: &str) -> Option<(&str, &str)> {
    full_name.split_once(NAMESPACE_SEPARATOR)
}

fn map_error(error: ClientError) -> ToolCallError {
    match error {
        ClientError::Timeout(duration) => ToolCallError::Timeout(duration),
        ClientError::Rpc(rpc) if rpc.code == -32602 => ToolCallError::InvalidArguments(rpc.message),
        ClientError::Rpc(_)
        | ClientError::Transport(_)
        | ClientError::Closed(_)
        | ClientError::Protocol(_) => ToolCallError::Upstream(error.to_string()),
    }
}

async fn ready_server(
    name: &str,
    handle: &ServerHandle,
) -> Result<Arc<ReadyServer>, ToolCallError> {
    let current = handle.state.borrow().clone();
    let baseline = match current {
        ServerState::Ready(ready) if !ready.client.is_closed() => return Ok(ready),
        ServerState::Ready(dead) => dead.generation,
        ServerState::Failed { generation, .. } | ServerState::Connecting { generation } => {
            generation
        }
        ServerState::Disabled => return Err(ToolCallError::UnknownTool(name.to_owned())),
    };
    handle.kick.notify_one();
    let mut receiver = handle.state.subscribe();
    let waited = tokio::time::timeout(handle.timeout, async {
        receiver
            .wait_for(|state| match state {
                ServerState::Ready(ready) => {
                    !ready.client.is_closed() && ready.generation >= baseline
                }
                ServerState::Failed { generation, .. } => *generation > baseline,
                ServerState::Connecting { .. } | ServerState::Disabled => false,
            })
            .await
            .map(|state| state.clone())
    })
    .await;
    let Ok(Ok(settled)) = waited else {
        return Err(ToolCallError::NotReady {
            server: name.to_owned(),
            state: "connecting",
            detail: ": timed out waiting for the server to reconnect".to_owned(),
        });
    };
    match settled {
        ServerState::Ready(ready) => Ok(ready),
        ServerState::Failed { error, .. } => Err(ToolCallError::NotReady {
            server: name.to_owned(),
            state: "failed",
            detail: format!(": {error}"),
        }),
        ServerState::Connecting { .. } | ServerState::Disabled => Err(ToolCallError::NotReady {
            server: name.to_owned(),
            state: "connecting",
            detail: String::new(),
        }),
    }
}

fn truncate_at_boundary(text: &str, limit: usize) -> &str {
    let mut end = limit.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.get(..end).unwrap_or_default()
}

/// Convert a raw result into a [`ToolOutput`], keeping at most `cap` bytes of
/// content. Text is cut at a character boundary; any other block that does
/// not fit is dropped whole.
pub fn build_output(result: ToolCallResult, cap: usize) -> ToolOutput {
    let mut remaining = cap;
    let mut truncated = false;
    let mut content = Vec::new();
    let mut texts = Vec::new();
    for block in result.content {
        if truncated {
            break;
        }
        let block_text = block
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| *kind == "text")
            .and_then(|_| block.get("text").and_then(Value::as_str));
        if let Some(text) = block_text {
            if text.len() <= remaining {
                remaining -= text.len();
                texts.push(text.to_owned());
                content.push(block);
            } else {
                let kept = truncate_at_boundary(text, remaining);
                truncated = true;
                remaining = 0;
                texts.push(kept.to_owned());
                content.push(serde_json::json!({"type": "text", "text": kept}));
            }
        } else {
            let size = block.to_string().len();
            if size <= remaining {
                remaining -= size;
                content.push(block);
            } else {
                truncated = true;
            }
        }
    }
    let structured_content = result.structured_content.and_then(|value| {
        let size = value.to_string().len();
        if !truncated && size <= remaining {
            Some(value)
        } else {
            truncated = true;
            None
        }
    });
    ToolOutput {
        content,
        text: texts.join("\n"),
        structured_content,
        is_error: result.is_error,
        truncated,
    }
}

async fn stopped(shutdown: &mut watch::Receiver<bool>) {
    let _ = shutdown.wait_for(|stop| *stop).await;
}

async fn connect_server(
    factory: &dyn TransportFactory,
    server: &ResolvedServer,
) -> Result<(Arc<McpClient>, Vec<McpTool>), String> {
    let connection = factory
        .connect(&server.name, &server.spec)
        .await
        .map_err(|error| error.to_string())?;
    let client = McpClient::connect(connection, server.timeout)
        .await
        .map_err(|error| format!("handshake failed: {error}"))?;
    match client.list_tools(server.timeout).await {
        Ok(tools) => Ok((Arc::new(client), tools)),
        Err(error) => {
            client.close().await;
            Err(format!("tools/list failed: {error}"))
        }
    }
}

async fn supervise(
    handle: Arc<ServerHandle>,
    server: ResolvedServer,
    factory: Arc<dyn TransportFactory>,
    backoff: BackoffPolicy,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut consecutive_failures: u32 = 0;
    let mut generation: u64 = 0;
    loop {
        handle
            .state
            .send_replace(ServerState::Connecting { generation });
        let attempt = connect_server(factory.as_ref(), &server);
        let outcome = tokio::select! {
            outcome = attempt => outcome,
            () = stopped(&mut shutdown) => return,
        };
        generation += 1;
        match outcome {
            Ok((client, tools)) => {
                tracing::info!(server = %server.name, tools = tools.len(), "MCP server ready");
                handle
                    .state
                    .send_replace(ServerState::Ready(Arc::new(ReadyServer {
                        client: Arc::clone(&client),
                        tools,
                        generation,
                    })));
                let Some(reason) =
                    serve(&handle, &server, &client, generation, &mut shutdown).await
                else {
                    return;
                };
                client.close().await;
                generation += 1;
                consecutive_failures = 1;
                tracing::warn!(server = %server.name, %reason, "MCP server connection lost");
                handle.state.send_replace(ServerState::Failed {
                    error: format!("connection lost: {reason}"),
                    generation,
                });
            }
            Err(error) => {
                consecutive_failures = consecutive_failures.saturating_add(1);
                tracing::warn!(server = %server.name, %error, "MCP server unavailable");
                handle
                    .state
                    .send_replace(ServerState::Failed { error, generation });
            }
        }
        let delay = backoff.delay(consecutive_failures);
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = handle.kick.notified() => {}
            () = stopped(&mut shutdown) => return,
        }
    }
}

/// Keep a connected server's tool list current until it closes. `None` means
/// shutdown was requested and the client has been closed.
async fn serve(
    handle: &ServerHandle,
    server: &ResolvedServer,
    client: &Arc<McpClient>,
    generation: u64,
    shutdown: &mut watch::Receiver<bool>,
) -> Option<String> {
    loop {
        tokio::select! {
            reason = client.closed() => return Some(reason),
            () = client.tools_changed() => {
                match client.list_tools(server.timeout).await {
                    Ok(tools) => {
                        tracing::info!(server = %server.name, tools = tools.len(), "MCP tool list refreshed");
                        handle.state.send_replace(ServerState::Ready(Arc::new(ReadyServer {
                            client: Arc::clone(client),
                            tools,
                            generation,
                        })));
                    }
                    Err(error) => {
                        tracing::warn!(server = %server.name, %error, "failed to refresh MCP tool list");
                    }
                }
            }
            () = stopped(shutdown) => {
                client.close().await;
                return None;
            }
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use serde_json::json;

    use super::*;
    use crate::mcp::config::{ServerSpec, StdioSpec};
    use crate::mcp::testing::{CallBehavior, FakeFactory, FakeMcp, text_result, tool_definition};

    const FAST: BackoffPolicy = BackoffPolicy {
        initial: Duration::from_millis(10),
        maximum: Duration::from_millis(40),
    };

    fn resolved(name: &str, enabled: bool, timeout: Duration) -> ResolvedServer {
        ResolvedServer {
            name: name.to_owned(),
            enabled,
            timeout,
            spec: ServerSpec::Stdio(StdioSpec {
                command: "unused".to_owned(),
                args: Vec::new(),
                env: Vec::new(),
                cwd: None,
            }),
        }
    }

    fn manager_for(server: &Arc<FakeMcp>, timeout: Duration) -> McpManager {
        let factory: Arc<dyn TransportFactory> = FakeFactory::single("fake", server);
        McpManager::start_with(vec![resolved("fake", true, timeout)], 1024, &factory, FAST)
    }

    async fn wait_until(manager: &McpManager, name: &str, state: &str) -> ServerStatus {
        for _ in 0..200 {
            if let Some(status) = manager
                .statuses()
                .into_iter()
                .find(|status| status.name == name && status.state == state)
            {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "server {name} never reached {state}: {:?}",
            manager.statuses()
        );
    }

    #[tokio::test]
    async fn empty_manager_reports_nothing_and_rejects_calls() {
        let manager = McpManager::empty();
        assert!(manager.statuses().is_empty());
        assert!(manager.tools().is_empty());
        assert!(matches!(
            manager.call("any__tool", json!({})).await,
            Err(ToolCallError::UnknownTool(_))
        ));
    }

    #[tokio::test]
    async fn ready_server_reports_status_and_namespaced_tools() {
        let server = FakeMcp::new(&["echo", "add"]);
        let manager = manager_for(&server, Duration::from_secs(2));
        let status = wait_until(&manager, "fake", "ready").await;
        assert_eq!(status.transport, "stdio");
        assert_eq!(status.tool_count, 2);
        assert_eq!(status.protocol_version.as_deref(), Some("2025-06-18"));
        assert_eq!(status.server_info.unwrap().name, "fake-mcp");
        assert!(status.error.is_none());

        let tools = manager.tools();
        let names: Vec<_> = tools.iter().map(|tool| tool.name.as_str()).collect();
        assert_eq!(names, vec!["fake__echo", "fake__add"]);
        assert_eq!(tools[0].server, "fake");
        assert!(manager.owns("fake__anything"));
        assert!(!manager.owns("other__echo"));
    }

    #[tokio::test]
    async fn startup_does_not_wait_for_a_failing_server_which_is_retried_with_backoff() {
        let server = FakeMcp::new(&["echo"]);
        server.fail_next_connects(2);
        let started = std::time::Instant::now();
        let manager = manager_for(&server, Duration::from_secs(2));
        assert!(started.elapsed() < Duration::from_millis(100));

        let failed = wait_until(&manager, "fake", "failed").await;
        assert!(failed.error.unwrap().contains("scripted connect failure"));
        assert!(manager.tools().is_empty());

        wait_until(&manager, "fake", "ready").await;
        assert_eq!(server.connection_count(), 1);
    }

    #[tokio::test]
    async fn call_returns_text_and_flags_tool_errors_without_failing() {
        let server = FakeMcp::new(&["echo", "broken"]);
        server.on_call("echo", CallBehavior::Reply(text_result("pong")));
        server.on_call(
            "broken",
            CallBehavior::Reply(json!({
                "content": [{"type": "text", "text": "no such file"}],
                "isError": true
            })),
        );
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;

        let ok = manager
            .call("fake__echo", json!({"text": "ping"}))
            .await
            .unwrap();
        assert_eq!(ok.text, "pong");
        assert!(!ok.is_error);
        assert_eq!(
            server.received_calls()[0],
            json!({"name": "echo", "arguments": {"text": "ping"}})
        );

        let failed = manager.call("fake__broken", Value::Null).await.unwrap();
        assert!(failed.is_error);
        assert_eq!(failed.text, "no such file");
        assert_eq!(server.received_calls()[1]["arguments"], json!({}));
    }

    #[tokio::test]
    async fn unknown_names_and_bad_arguments_are_rejected_before_the_server_is_called() {
        let server = FakeMcp::new(&["echo"]);
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        for name in ["echo", "nope__echo", "fake__missing", ""] {
            assert!(
                matches!(
                    manager.call(name, json!({})).await,
                    Err(ToolCallError::UnknownTool(_))
                ),
                "{name}"
            );
        }
        assert!(matches!(
            manager.call("fake__echo", json!([1])).await,
            Err(ToolCallError::InvalidArguments(_))
        ));
        assert!(server.received_calls().is_empty());
    }

    #[tokio::test]
    async fn server_side_invalid_params_map_to_invalid_arguments() {
        let server = FakeMcp::new(&["echo", "other"]);
        server.on_call(
            "echo",
            CallBehavior::RpcError(-32602, "missing text".to_owned()),
        );
        server.on_call(
            "other",
            CallBehavior::RpcError(-32603, "internal".to_owned()),
        );
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        assert!(matches!(
            manager.call("fake__echo", json!({})).await,
            Err(ToolCallError::InvalidArguments(message)) if message == "missing text"
        ));
        assert!(matches!(
            manager.call("fake__other", json!({})).await,
            Err(ToolCallError::Upstream(_))
        ));
    }

    #[tokio::test]
    async fn a_slow_tool_times_out() {
        let server = FakeMcp::new(&["slow"]);
        server.on_call("slow", CallBehavior::Hang);
        let manager = manager_for(&server, Duration::from_millis(100));
        wait_until(&manager, "fake", "ready").await;
        let error = manager.call("fake__slow", json!({})).await.unwrap_err();
        assert!(matches!(error, ToolCallError::Timeout(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_crashed_server_is_restarted_on_next_use() {
        let server = FakeMcp::new(&["echo"]);
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        assert_eq!(server.connection_count(), 1);

        server.crash().await;
        let output = manager.call("fake__echo", json!({})).await.unwrap();
        assert_eq!(output.text, "called echo");
        assert_eq!(server.connection_count(), 2);
        wait_until(&manager, "fake", "ready").await;
    }

    #[tokio::test]
    async fn a_crash_during_a_call_is_retried_once_on_a_fresh_connection() {
        let server = FakeMcp::new(&["flaky"]);
        server.on_call("flaky", CallBehavior::Crash);
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        // The first attempt crashes the connection; the retry crashes again,
        // so the caller sees an upstream error rather than looping forever.
        let error = manager.call("fake__flaky", json!({})).await.unwrap_err();
        assert!(matches!(error, ToolCallError::Upstream(_)), "{error:?}");
        assert_eq!(server.connection_count(), 2);

        server.on_call("flaky", CallBehavior::Reply(text_result("recovered")));
        let output = manager.call("fake__flaky", json!({})).await.unwrap();
        assert_eq!(output.text, "recovered");
    }

    #[tokio::test]
    async fn calls_to_an_unreachable_server_report_503_state_with_the_error() {
        let server = FakeMcp::new(&["echo"]);
        server.fail_next_connects(1000);
        let manager = manager_for(&server, Duration::from_millis(500));
        wait_until(&manager, "fake", "failed").await;
        let error = manager.call("fake__echo", json!({})).await.unwrap_err();
        let ToolCallError::NotReady {
            server: named,
            state,
            detail,
        } = error
        else {
            panic!("expected NotReady")
        };
        assert_eq!(named, "fake");
        assert_eq!(state, "failed");
        assert!(detail.contains("scripted connect failure"), "{detail}");
    }

    #[tokio::test]
    async fn list_changed_refreshes_the_tool_list() {
        let server = FakeMcp::new(&["old"]);
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        assert_eq!(manager.tools()[0].name, "fake__old");

        server.set_pages(vec![vec![tool_definition("new"), tool_definition("newer")]]);
        server.notify_tools_changed().await;
        for _ in 0..200 {
            if manager.tools().len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let names: Vec<_> = manager.tools().into_iter().map(|tool| tool.name).collect();
        assert_eq!(names, vec!["fake__new", "fake__newer"]);
        assert_eq!(server.connection_count(), 1);
        assert!(manager.call("fake__old", json!({})).await.is_err());
        assert!(manager.call("fake__new", json!({})).await.is_ok());
    }

    #[tokio::test]
    async fn disabled_servers_are_listed_but_never_started() {
        let server = FakeMcp::new(&["echo"]);
        let factory: Arc<dyn TransportFactory> = FakeFactory::single("off", &server);
        let manager = McpManager::start_with(
            vec![resolved("off", false, Duration::from_secs(1))],
            1024,
            &factory,
            FAST,
        );
        let status = &manager.statuses()[0];
        assert_eq!(status.state, "disabled");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(server.connection_count(), 0);
        assert!(matches!(
            manager.call("off__echo", json!({})).await,
            Err(ToolCallError::UnknownTool(_))
        ));
    }

    #[tokio::test]
    async fn multiple_servers_keep_independent_state() {
        let good = FakeMcp::new(&["echo"]);
        let bad = FakeMcp::new(&["echo"]);
        bad.fail_next_connects(1000);
        let factory: Arc<dyn TransportFactory> =
            FakeFactory::many(vec![("good", Arc::clone(&good)), ("bad", Arc::clone(&bad))]);
        let manager = McpManager::start_with(
            vec![
                resolved("good", true, Duration::from_secs(1)),
                resolved("bad", true, Duration::from_secs(1)),
            ],
            1024,
            &factory,
            FAST,
        );
        wait_until(&manager, "good", "ready").await;
        wait_until(&manager, "bad", "failed").await;
        assert_eq!(manager.tools().len(), 1);
        assert!(manager.call("good__echo", json!({})).await.is_ok());
    }

    #[tokio::test]
    async fn shutdown_closes_every_connection() {
        let server = FakeMcp::new(&["echo"]);
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        manager.shutdown().await;
        assert_eq!(server.closes.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn backoff_doubles_up_to_the_maximum() {
        let policy = BackoffPolicy::default();
        let delays: Vec<_> = (1..=8)
            .map(|failures| policy.delay(failures).as_secs())
            .collect();
        assert_eq!(delays, vec![1, 2, 4, 8, 16, 32, 60, 60]);
        assert_eq!(policy.delay(0).as_secs(), 1);
        assert_eq!(policy.delay(u32::MAX).as_secs(), 60);
    }

    fn result(content: Vec<Value>, structured: Option<Value>) -> ToolCallResult {
        ToolCallResult {
            content,
            structured_content: structured,
            is_error: false,
        }
    }

    #[test]
    fn output_within_the_cap_is_unchanged() {
        let image = json!({"type": "image", "data": "AAAA", "mimeType": "image/png"});
        let output = build_output(
            result(
                vec![
                    json!({"type": "text", "text": "a"}),
                    image.clone(),
                    json!({"type": "text", "text": "b"}),
                ],
                Some(json!({"k": 1})),
            ),
            10_000,
        );
        assert_eq!(output.text, "a\nb");
        assert_eq!(output.content.len(), 3);
        assert_eq!(output.content[1], image);
        assert_eq!(output.structured_content, Some(json!({"k": 1})));
        assert!(!output.truncated);
    }

    #[test]
    fn oversized_text_is_cut_on_a_character_boundary() {
        let output = build_output(
            result(vec![json!({"type": "text", "text": "ééééé"})], None),
            5,
        );
        assert_eq!(output.text, "éé");
        assert_eq!(output.content[0]["text"], "éé");
        assert!(output.truncated);
    }

    #[test]
    fn blocks_after_the_cap_and_oversized_structured_content_are_dropped() {
        let output = build_output(
            result(
                vec![
                    json!({"type": "text", "text": "0123456789"}),
                    json!({"type": "text", "text": "never"}),
                ],
                Some(json!({"big": "x"})),
            ),
            4,
        );
        assert_eq!(output.text, "0123");
        assert_eq!(output.content.len(), 1);
        assert!(output.structured_content.is_none());
        assert!(output.truncated);
    }

    #[test]
    fn an_oversized_non_text_block_is_dropped_whole() {
        let image = json!({"type": "image", "data": "A".repeat(100), "mimeType": "image/png"});
        let output = build_output(result(vec![image], None), 50);
        assert!(output.content.is_empty());
        assert!(output.truncated);
    }

    #[tokio::test]
    async fn call_applies_the_configured_result_cap() {
        let server = FakeMcp::new(&["big"]);
        server.on_call("big", CallBehavior::Reply(text_result(&"x".repeat(5000))));
        let manager = manager_for(&server, Duration::from_secs(2));
        wait_until(&manager, "fake", "ready").await;
        let output = manager.call("fake__big", json!({})).await.unwrap();
        assert_eq!(output.text.len(), 1024);
        assert!(output.truncated);
    }
}
