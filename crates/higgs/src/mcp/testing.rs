//! In-memory MCP server and transport factory for tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::config::ServerSpec;
use super::transport::{
    Connection, INCOMING_CAPACITY, Incoming, Transport, TransportError, TransportFactory,
};

/// What a scripted `tools/call` does.
#[derive(Clone)]
pub enum CallBehavior {
    Reply(Value),
    RpcError(i64, String),
    /// Never answers, to exercise timeouts.
    Hang,
    /// Drops the connection without answering.
    Crash,
}

pub struct FakeMcp {
    pub protocol_version: Mutex<String>,
    pub has_tools_capability: Mutex<bool>,
    pub loop_cursor: Mutex<bool>,
    pub pages: Mutex<Vec<Vec<Value>>>,
    pub calls: Mutex<HashMap<String, CallBehavior>>,
    pub received: Mutex<Vec<Value>>,
    pub connections: AtomicUsize,
    pub connect_failures: AtomicUsize,
    pub closes: AtomicUsize,
    live: Mutex<Vec<mpsc::Sender<Incoming>>>,
}

pub fn tool_definition(name: &str) -> Value {
    json!({
        "name": name,
        "description": format!("{name} tool"),
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    })
}

pub fn text_result(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": false})
}

impl FakeMcp {
    pub fn new(tool_names: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            protocol_version: Mutex::new("2025-06-18".to_owned()),
            has_tools_capability: Mutex::new(true),
            loop_cursor: Mutex::new(false),
            pages: Mutex::new(vec![
                tool_names.iter().map(|n| tool_definition(n)).collect(),
            ]),
            calls: Mutex::new(HashMap::new()),
            received: Mutex::new(Vec::new()),
            connections: AtomicUsize::new(0),
            connect_failures: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            live: Mutex::new(Vec::new()),
        })
    }

    pub fn on_call(&self, tool: &str, behavior: CallBehavior) {
        lock(&self.calls).insert(tool.to_owned(), behavior);
    }

    pub fn set_pages(&self, pages: Vec<Vec<Value>>) {
        *lock(&self.pages) = pages;
    }

    pub fn received_methods(&self) -> Vec<String> {
        lock(&self.received)
            .iter()
            .filter_map(|message| {
                message
                    .get("method")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect()
    }

    pub fn received_calls(&self) -> Vec<Value> {
        lock(&self.received)
            .iter()
            .filter(|message| message.get("method").and_then(Value::as_str) == Some("tools/call"))
            .filter_map(|message| message.get("params").cloned())
            .collect()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    /// Fail the next `count` connection attempts.
    pub fn fail_next_connects(&self, count: usize) {
        self.connect_failures.store(count, Ordering::SeqCst);
    }

    pub async fn notify_tools_changed(&self) {
        let senders: Vec<_> = lock(&self.live).clone();
        for sender in senders {
            let _ = sender
                .send(Incoming::Message(json!({
                    "jsonrpc": "2.0", "method": "notifications/tools/list_changed"
                })))
                .await;
        }
    }

    /// Drop every live connection as a crashed server would.
    pub async fn crash(&self) {
        let senders: Vec<_> = lock(&self.live).drain(..).collect();
        for sender in senders {
            let _ = sender
                .send(Incoming::Closed(
                    "server process exited (test crash)".to_owned(),
                ))
                .await;
        }
    }

    fn reply_to(&self, message: &Value) -> Option<Reply> {
        let method = message.get("method").and_then(Value::as_str)?;
        let id = message.get("id")?.clone();
        let result = match method {
            "initialize" => {
                let capabilities = if *lock(&self.has_tools_capability) {
                    json!({"tools": {"listChanged": true}})
                } else {
                    json!({})
                };
                json!({
                    "protocolVersion": *lock(&self.protocol_version),
                    "capabilities": capabilities,
                    "serverInfo": {"name": "fake-mcp", "version": "9.9"},
                })
            }
            "tools/list" => {
                let cursor = message
                    .get("params")
                    .and_then(|params| params.get("cursor"))
                    .and_then(Value::as_str);
                let index = cursor
                    .and_then(|value| value.strip_prefix("page-"))
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
                let page_count = lock(&self.pages).len();
                let tools_on_page = lock(&self.pages).get(index).cloned().unwrap_or_default();
                let next_cursor = if *lock(&self.loop_cursor) {
                    Some("page-0".to_owned())
                } else {
                    (index + 1 < page_count).then(|| format!("page-{}", index + 1))
                };
                let mut page = json!({"tools": tools_on_page});
                if let (Some(cursor_value), Some(object)) = (next_cursor, page.as_object_mut()) {
                    object.insert("nextCursor".to_owned(), Value::String(cursor_value));
                }
                page
            }
            "tools/call" => {
                let name = message
                    .get("params")
                    .and_then(|params| params.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let behavior = lock(&self.calls).get(name).cloned();
                match behavior {
                    Some(CallBehavior::Reply(result)) => result,
                    Some(CallBehavior::RpcError(code, text)) => {
                        return Some(Reply::Message(json!({
                            "jsonrpc": "2.0", "id": id,
                            "error": {"code": code, "message": text}
                        })));
                    }
                    Some(CallBehavior::Hang) => return None,
                    Some(CallBehavior::Crash) => return Some(Reply::Crash),
                    None => text_result(&format!("called {name}")),
                }
            }
            "ping" => json!({}),
            _ => {
                return Some(Reply::Message(json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": "unknown method"}
                })));
            }
        };
        Some(Reply::Message(
            json!({"jsonrpc": "2.0", "id": id, "result": result}),
        ))
    }
}

enum Reply {
    Message(Value),
    Crash,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct FakeTransport {
    server: Arc<FakeMcp>,
    sender: mpsc::Sender<Incoming>,
    closed: std::sync::atomic::AtomicBool,
}

impl Transport for FakeTransport {
    fn send(&self, message: Value) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move {
            if self.closed.load(Ordering::SeqCst) {
                return Err(TransportError::Send("closed".to_owned()));
            }
            lock(&self.server.received).push(message.clone());
            match self.server.reply_to(&message) {
                Some(Reply::Message(reply)) => {
                    let _ = self.sender.send(Incoming::Message(reply)).await;
                }
                Some(Reply::Crash) => {
                    self.closed.store(true, Ordering::SeqCst);
                    let _ = self
                        .sender
                        .send(Incoming::Closed(
                            "server process exited (test crash)".to_owned(),
                        ))
                        .await;
                }
                None => {}
            }
            Ok(())
        })
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if !self.closed.swap(true, Ordering::SeqCst) {
                self.server.closes.fetch_add(1, Ordering::SeqCst);
            }
        })
    }
}

/// Serves the same [`FakeMcp`] for every server name, or a per-name one.
pub struct FakeFactory {
    servers: HashMap<String, Arc<FakeMcp>>,
}

impl FakeFactory {
    pub fn single(name: &str, server: &Arc<FakeMcp>) -> Arc<Self> {
        Arc::new(Self {
            servers: HashMap::from([(name.to_owned(), Arc::clone(server))]),
        })
    }

    pub fn many(servers: Vec<(&str, Arc<FakeMcp>)>) -> Arc<Self> {
        Arc::new(Self {
            servers: servers
                .into_iter()
                .map(|(name, server)| (name.to_owned(), server))
                .collect(),
        })
    }
}

impl TransportFactory for FakeFactory {
    fn connect<'a>(
        &'a self,
        name: &'a str,
        _spec: &'a ServerSpec,
    ) -> BoxFuture<'a, Result<Connection, TransportError>> {
        Box::pin(async move {
            let server = self
                .servers
                .get(name)
                .ok_or_else(|| TransportError::Start(format!("no fake server named {name}")))?;
            let remaining = server.connect_failures.load(Ordering::SeqCst);
            if remaining > 0 {
                server
                    .connect_failures
                    .store(remaining - 1, Ordering::SeqCst);
                return Err(TransportError::Start("scripted connect failure".to_owned()));
            }
            server.connections.fetch_add(1, Ordering::SeqCst);
            let (sender, receiver) = mpsc::channel(INCOMING_CAPACITY);
            lock(&server.live).push(sender.clone());
            Ok(Connection {
                transport: Arc::new(FakeTransport {
                    server: Arc::clone(server),
                    sender,
                    closed: std::sync::atomic::AtomicBool::new(false),
                }),
                incoming: receiver,
            })
        })
    }
}
