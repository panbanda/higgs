//! JSON-RPC 2.0 message model and line framing for MCP transports.

use std::fmt;

use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

pub const METHOD_NOT_FOUND: i64 = -32601;
/// Code for failures that happen below the JSON-RPC layer (transport errors,
/// closed connections); the spec reserves `-32000..=-32099` for such use.
pub const TRANSPORT_FAILURE: i64 = -32000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl fmt::Display for RpcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} (code {})", self.message, self.code)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    Request {
        id: Value,
        method: String,
        params: Option<Value>,
    },
    Notification {
        method: String,
        params: Option<Value>,
    },
    Response {
        id: Value,
        outcome: Result<Value, RpcError>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FramingError {
    #[error("message is not a JSON object")]
    NotAnObject,
    #[error("missing or unsupported \"jsonrpc\" version")]
    BadVersion,
    #[error("message is neither a request, notification nor response")]
    Unclassifiable,
    #[error("malformed error object in response")]
    MalformedError,
}

impl Message {
    pub fn parse(value: &Value) -> Result<Self, FramingError> {
        let object = value.as_object().ok_or(FramingError::NotAnObject)?;
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err(FramingError::BadVersion);
        }
        let raw_id = object.get("id").filter(|id| !id.is_null()).cloned();
        let params = object.get("params").cloned();
        if let Some(name) = object.get("method").and_then(Value::as_str) {
            let method = name.to_owned();
            return Ok(match raw_id {
                Some(id) => Self::Request { id, method, params },
                None => Self::Notification { method, params },
            });
        }
        let id = raw_id.ok_or(FramingError::Unclassifiable)?;
        if let Some(error) = object.get("error") {
            let code = error
                .get("code")
                .and_then(Value::as_i64)
                .ok_or(FramingError::MalformedError)?;
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            return Ok(Self::Response {
                id,
                outcome: Err(RpcError {
                    code,
                    message,
                    data: error.get("data").cloned(),
                }),
            });
        }
        let result = object
            .get("result")
            .cloned()
            .ok_or(FramingError::Unclassifiable)?;
        Ok(Self::Response {
            id,
            outcome: Ok(result),
        })
    }
}

pub fn request(id: u64, method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "id": id, "method": method});
    if let (Some(value), Some(object)) = (params, message.as_object_mut()) {
        object.insert("params".to_owned(), value);
    }
    message
}

pub fn notification(method: &str, params: Option<Value>) -> Value {
    let mut message = json!({"jsonrpc": "2.0", "method": method});
    if let (Some(value), Some(object)) = (params, message.as_object_mut()) {
        object.insert("params".to_owned(), value);
    }
    message
}

pub fn success_response(id: &Value, result: &Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn error_response(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Serialize one message as a newline-terminated frame (stdio transport).
/// Compact JSON never contains a raw newline, so one line is one message.
pub fn encode_line(message: &Value) -> String {
    let mut line = message.to_string();
    line.push('\n');
    line
}

#[derive(Debug, thiserror::Error)]
pub enum LineError {
    #[error("message exceeds the {limit} byte limit")]
    TooLong { limit: usize },
    #[error("read failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Read one `\n`-terminated line into `buffer` without ever holding more than
/// `limit` bytes, so a misbehaving server cannot exhaust memory. Returns
/// `Ok(false)` at end of input.
pub async fn read_bounded_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    limit: usize,
) -> Result<bool, LineError> {
    buffer.clear();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(!buffer.is_empty());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |position| position + 1);
        if buffer.len() + take > limit.saturating_add(1) {
            return Err(LineError::TooLong { limit });
        }
        buffer.extend_from_slice(available.get(..take).unwrap_or_default());
        reader.consume(take);
        if newline.is_some() {
            return Ok(true);
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_request_notification_and_responses() {
        let request = Message::parse(&json!({"jsonrpc":"2.0","id":3,"method":"ping"})).unwrap();
        assert_eq!(
            request,
            Message::Request {
                id: json!(3),
                method: "ping".to_owned(),
                params: None
            }
        );

        let notification =
            Message::parse(&json!({"jsonrpc":"2.0","method":"notifications/tools/list_changed"}))
                .unwrap();
        assert!(matches!(notification, Message::Notification { .. }));

        let success =
            Message::parse(&json!({"jsonrpc":"2.0","id":1,"result":{"ok":true}})).unwrap();
        assert_eq!(
            success,
            Message::Response {
                id: json!(1),
                outcome: Ok(json!({"ok": true}))
            }
        );

        let failure = Message::parse(
            &json!({"jsonrpc":"2.0","id":"a","error":{"code":-32602,"message":"bad","data":1}}),
        )
        .unwrap();
        assert_eq!(
            failure,
            Message::Response {
                id: json!("a"),
                outcome: Err(RpcError {
                    code: -32602,
                    message: "bad".to_owned(),
                    data: Some(json!(1))
                })
            }
        );
    }

    #[test]
    fn rejects_malformed_messages() {
        assert_eq!(
            Message::parse(&json!([1])).unwrap_err(),
            FramingError::NotAnObject
        );
        assert_eq!(
            Message::parse(&json!({"id":1,"result":{}})).unwrap_err(),
            FramingError::BadVersion
        );
        assert_eq!(
            Message::parse(&json!({"jsonrpc":"1.0","id":1,"result":{}})).unwrap_err(),
            FramingError::BadVersion
        );
        assert_eq!(
            Message::parse(&json!({"jsonrpc":"2.0","id":1})).unwrap_err(),
            FramingError::Unclassifiable
        );
        assert_eq!(
            Message::parse(&json!({"jsonrpc":"2.0","id":1,"error":{"message":"x"}})).unwrap_err(),
            FramingError::MalformedError
        );
    }

    #[test]
    fn null_id_with_method_is_a_notification() {
        let parsed = Message::parse(&json!({"jsonrpc":"2.0","id":null,"method":"x"})).unwrap();
        assert!(matches!(parsed, Message::Notification { .. }));
    }

    #[test]
    fn builders_produce_wire_shapes() {
        assert_eq!(
            request(7, "tools/list", Some(json!({"cursor": "c"}))),
            json!({"jsonrpc":"2.0","id":7,"method":"tools/list","params":{"cursor":"c"}})
        );
        assert_eq!(
            request(8, "ping", None),
            json!({"jsonrpc":"2.0","id":8,"method":"ping"})
        );
        assert_eq!(
            notification("notifications/initialized", None),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"})
        );
        assert_eq!(
            error_response(&json!(2), METHOD_NOT_FOUND, "nope"),
            json!({"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"nope"}})
        );
        assert_eq!(
            success_response(&json!(2), &json!({})),
            json!({"jsonrpc":"2.0","id":2,"result":{}})
        );
    }

    #[test]
    fn encoded_line_is_single_line_even_with_newlines_in_strings() {
        let line = encode_line(&json!({"text": "a\nb"}));
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
    }

    #[tokio::test]
    async fn bounded_reader_splits_lines_and_reports_end() {
        let mut reader = tokio::io::BufReader::new(&b"one\ntwo\nlast"[..]);
        let mut buffer = Vec::new();
        assert!(
            read_bounded_line(&mut reader, &mut buffer, 100)
                .await
                .unwrap()
        );
        assert_eq!(buffer, b"one\n");
        assert!(
            read_bounded_line(&mut reader, &mut buffer, 100)
                .await
                .unwrap()
        );
        assert_eq!(buffer, b"two\n");
        assert!(
            read_bounded_line(&mut reader, &mut buffer, 100)
                .await
                .unwrap()
        );
        assert_eq!(buffer, b"last");
        assert!(
            !read_bounded_line(&mut reader, &mut buffer, 100)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn bounded_reader_rejects_oversized_lines() {
        let data = [b'x'; 64];
        let mut reader = tokio::io::BufReader::with_capacity(8, &data[..]);
        let mut buffer = Vec::new();
        let error = read_bounded_line(&mut reader, &mut buffer, 16)
            .await
            .unwrap_err();
        assert!(matches!(error, LineError::TooLong { limit: 16 }));
    }
}
