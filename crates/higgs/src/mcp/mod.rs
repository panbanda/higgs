//! Model Context Protocol client.
//!
//! `config` defines what to connect to, `transport` (with `stdio` and `http`)
//! moves JSON-RPC messages, `client` runs the protocol over a transport, and
//! `manager` supervises one client per configured server.

pub mod client;
pub mod config;
pub mod doctor;
pub mod http;
pub mod jsonrpc;
pub mod manager;
pub mod sse;
pub mod stdio;
pub mod transport;

#[cfg(test)]
pub(crate) mod testing;
