//! The seam between the MCP client and the wire.
//!
//! A transport delivers JSON-RPC messages in both directions and nothing
//! else; request/response matching, timeouts and the protocol handshake live
//! in the client. Replies to HTTP requests arrive through the same incoming
//! channel as stdio lines, so the client has a single code path.

use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value;
use tokio::sync::mpsc;

use super::config::ServerSpec;

pub const INCOMING_CAPACITY: usize = 256;
/// Largest single message accepted from a server.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub enum Incoming {
    Message(Value),
    /// The connection is gone and will deliver nothing more.
    Closed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("cannot start server: {0}")]
    Start(String),
    #[error("cannot send message: {0}")]
    Send(String),
}

pub struct Connection {
    pub transport: Arc<dyn Transport>,
    pub incoming: mpsc::Receiver<Incoming>,
}

pub trait Transport: Send + Sync {
    /// Queue one message for the server. Replies are delivered on the
    /// connection's incoming channel, never as the return value.
    fn send(&self, message: Value) -> BoxFuture<'_, Result<(), TransportError>>;

    /// Called once initialization has negotiated a protocol version.
    fn set_protocol_version(&self, _version: &str) {}

    /// Called after the handshake completes; transports with a separate
    /// server-to-client channel open it here.
    fn start_notifications(&self) {}

    /// Release the connection. Idempotent. A stdio transport does not return
    /// until the child process and its process group are gone.
    fn close(&self) -> BoxFuture<'_, ()>;
}

pub trait TransportFactory: Send + Sync {
    fn connect<'a>(
        &'a self,
        name: &'a str,
        spec: &'a ServerSpec,
    ) -> BoxFuture<'a, Result<Connection, TransportError>>;
}

pub struct DefaultTransportFactory;

impl TransportFactory for DefaultTransportFactory {
    fn connect<'a>(
        &'a self,
        name: &'a str,
        spec: &'a ServerSpec,
    ) -> BoxFuture<'a, Result<Connection, TransportError>> {
        Box::pin(async move {
            match spec {
                ServerSpec::Stdio(stdio) => super::stdio::spawn(name, stdio),
                ServerSpec::Http(http) => super::http::connect(http),
            }
        })
    }
}
