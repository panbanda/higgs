//! `/v1/mcp/*`: inspect configured MCP servers and run their tools.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    error::{ErrorDetail, ErrorResponse},
    mcp::manager::ToolCallError,
    state::SharedState,
    tools::openai_tool,
};

/// Error in the same JSON shape as every other endpoint.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub error_type: &'static str,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub const fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            code: "invalid_request",
            message,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: self.message,
                    r#type: self.error_type.to_owned(),
                    code: Some(self.code.to_owned()),
                },
            }),
        )
            .into_response()
    }
}

impl From<ToolCallError> for ApiError {
    fn from(error: ToolCallError) -> Self {
        let message = error.to_string();
        let (status, error_type, code) = match error {
            ToolCallError::UnknownTool(_) => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "tool_not_found",
            ),
            ToolCallError::NotReady { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "mcp_server_not_ready",
            ),
            ToolCallError::Timeout(_) => {
                (StatusCode::GATEWAY_TIMEOUT, "server_error", "tool_timeout")
            }
            ToolCallError::InvalidArguments(_) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_arguments",
            ),
            ToolCallError::Upstream(_) => (StatusCode::BAD_GATEWAY, "server_error", "mcp_error"),
        };
        Self {
            status,
            error_type,
            code,
            message,
        }
    }
}

pub async fn list_servers(State(state): State<SharedState>) -> Json<Value> {
    Json(json!({"servers": state.tools.mcp.statuses()}))
}

/// Tools in the `OpenAI` function-tool shape, with the originating server as a
/// sibling `server` key. Strict clients that reject unknown keys should drop
/// it before forwarding the list.
pub async fn list_tools(State(state): State<SharedState>) -> Json<Value> {
    let tools: Vec<Value> = state
        .tools
        .list()
        .iter()
        .map(|tool| {
            let mut entry = openai_tool(tool);
            if let Some(object) = entry.as_object_mut() {
                object.insert("server".to_owned(), Value::String(tool.server.clone()));
            }
            entry
        })
        .collect();
    Json(json!({"count": tools.len(), "tools": tools}))
}

#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    #[serde(alias = "tool_name", alias = "tool")]
    name: String,
    #[serde(default)]
    arguments: Value,
}

pub async fn execute(
    State(state): State<SharedState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let request: ExecuteRequest = serde_json::from_slice(&body)
        .map_err(|error| ApiError::bad_request(format!("Invalid request body: {error}")))?;
    let output = state.tools.call(&request.name, request.arguments).await?;
    let server = request
        .name
        .split_once(crate::mcp::config::NAMESPACE_SEPARATOR)
        .map_or("", |(server, _)| server);
    let mut response = json!({
        "name": request.name,
        "server": server,
        "is_error": output.is_error,
        "text": output.text,
        "content": output.content,
        "truncated": output.truncated,
    });
    if let (Some(structured), Some(object)) = (output.structured_content, response.as_object_mut())
    {
        object.insert("structured_content".to_owned(), structured);
    }
    Ok(Json(response))
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use http::StatusCode;
    use serde_json::json;

    use crate::mcp::testing::{CallBehavior, FakeMcp, ready_manager, text_result};
    use crate::test_support::{app_with_tools, send_request};
    use crate::tools::ToolHub;
    use crate::web::WebService;

    async fn hub_with(
        servers: Vec<(&str, Arc<FakeMcp>, Duration)>,
        expected_ready: usize,
    ) -> ToolHub {
        ToolHub::new(
            ready_manager(servers, expected_ready).await,
            WebService::disabled(),
        )
    }

    async fn single_server_app() -> (axum::Router, Arc<FakeMcp>) {
        let fake = FakeMcp::new(&["echo", "broken", "slow", "rpc"]);
        fake.on_call("echo", CallBehavior::Reply(text_result("pong")));
        fake.on_call(
            "broken",
            CallBehavior::Reply(json!({
                "content": [{"type": "text", "text": "disk full"}],
                "isError": true
            })),
        );
        fake.on_call("slow", CallBehavior::Hang);
        fake.on_call("rpc", CallBehavior::RpcError(-32603, "internal".to_owned()));
        let hub = hub_with(
            vec![("files", Arc::clone(&fake), Duration::from_millis(300))],
            1,
        )
        .await;
        (app_with_tools(hub, None, None), fake)
    }

    #[tokio::test]
    async fn nothing_configured_reports_empty_lists_and_unknown_tools() {
        let app = app_with_tools(ToolHub::disabled(), None, None);
        let (servers_status, servers) =
            send_request(&app, "GET", "/v1/mcp/servers", None, None).await;
        assert_eq!(servers_status, StatusCode::OK);
        assert_eq!(servers, json!({"servers": []}));
        let (tools_status, tools) = send_request(&app, "GET", "/v1/mcp/tools", None, None).await;
        assert_eq!(tools_status, StatusCode::OK);
        assert_eq!(tools, json!({"count": 0, "tools": []}));
        let (status, error) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "files__echo"})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(error["error"]["code"], "tool_not_found");
    }

    #[tokio::test]
    async fn servers_endpoint_reports_state_error_tool_count_and_server_info() {
        let good = FakeMcp::new(&["a", "b"]);
        let bad = FakeMcp::new(&["c"]);
        bad.fail_next_connects(1000);
        let hub = hub_with(
            vec![
                ("good", good, Duration::from_secs(1)),
                ("bad", bad, Duration::from_secs(1)),
            ],
            1,
        )
        .await;
        let app = app_with_tools(hub, None, None);
        let (status, body) = send_request(&app, "GET", "/v1/mcp/servers", None, None).await;
        assert_eq!(status, StatusCode::OK);
        let servers = body["servers"].as_array().unwrap();
        let by_name = |name: &str| servers.iter().find(|s| s["name"] == name).unwrap();
        let ready = by_name("good");
        assert_eq!(ready["state"], "ready");
        assert_eq!(ready["transport"], "stdio");
        assert_eq!(ready["tool_count"], 2);
        assert_eq!(ready["protocol_version"], "2025-06-18");
        assert_eq!(
            ready["server_info"],
            json!({"name": "fake-mcp", "version": "9.9"})
        );
        assert!(ready["error"].is_null());
        let failed = by_name("bad");
        assert_eq!(failed["state"], "failed");
        assert_eq!(failed["tool_count"], 0);
        assert!(
            failed["error"]
                .as_str()
                .unwrap()
                .contains("scripted connect failure")
        );
        assert!(failed["protocol_version"].is_null());
    }

    #[tokio::test]
    async fn tools_endpoint_lists_openai_function_tools_with_their_server() {
        let (app, _) = single_server_app().await;
        let (status, body) = send_request(&app, "GET", "/v1/mcp/tools", None, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["count"], 4);
        let echo = &body["tools"][0];
        assert_eq!(echo["type"], "function");
        assert_eq!(echo["server"], "files");
        assert_eq!(echo["function"]["name"], "files__echo");
        assert_eq!(echo["function"]["description"], "echo tool");
        assert_eq!(echo["function"]["parameters"]["type"], "object");
        assert!(echo["function"]["parameters"]["properties"]["text"].is_object());
    }

    #[tokio::test]
    async fn tools_of_a_failed_server_are_not_listed() {
        let bad = FakeMcp::new(&["c"]);
        bad.fail_next_connects(1000);
        let hub = hub_with(vec![("bad", bad, Duration::from_secs(1))], 0).await;
        let app = app_with_tools(hub, None, None);
        let (_, body) = send_request(&app, "GET", "/v1/mcp/tools", None, None).await;
        assert_eq!(body["count"], 0);
    }

    #[tokio::test]
    async fn execute_returns_text_content_and_is_error() {
        let (app, fake) = single_server_app().await;
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "files__echo", "arguments": {"text": "ping"}})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["name"], "files__echo");
        assert_eq!(body["server"], "files");
        assert_eq!(body["is_error"], false);
        assert_eq!(body["text"], "pong");
        assert_eq!(body["content"], json!([{"type": "text", "text": "pong"}]));
        assert_eq!(body["truncated"], false);
        assert_eq!(
            fake.received_calls()[0],
            json!({"name": "echo", "arguments": {"text": "ping"}})
        );

        let (broken_status, broken) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "files__broken"})),
            None,
        )
        .await;
        assert_eq!(broken_status, StatusCode::OK);
        assert_eq!(broken["is_error"], true);
        assert_eq!(broken["text"], "disk full");
    }

    #[tokio::test]
    async fn execute_returns_non_text_content_structurally() {
        let fake = FakeMcp::new(&["shot"]);
        fake.on_call(
            "shot",
            CallBehavior::Reply(json!({
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    {"type": "resource", "resource": {"uri": "file:///x", "text": "body"}}
                ],
                "structuredContent": {"width": 4}
            })),
        );
        let hub = hub_with(vec![("cam", fake, Duration::from_secs(1))], 1).await;
        let app = app_with_tools(hub, None, None);
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "cam__shot"})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["text"], "caption");
        assert_eq!(body["content"][1]["type"], "image");
        assert_eq!(body["content"][1]["mimeType"], "image/png");
        assert_eq!(body["content"][2]["resource"]["uri"], "file:///x");
        assert_eq!(body["structured_content"], json!({"width": 4}));
    }

    #[tokio::test]
    async fn execute_accepts_the_tool_name_aliases() {
        let (app, _) = single_server_app().await;
        for key in ["tool_name", "tool"] {
            let (status, body) = send_request(
                &app,
                "POST",
                "/v1/mcp/execute",
                Some(&json!({key: "files__echo"})),
                None,
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{key}");
            assert_eq!(body["text"], "pong");
        }
    }

    #[tokio::test]
    async fn execute_error_statuses() {
        let (app, _) = single_server_app().await;
        let cases = [
            (
                json!({"name": "files__missing"}),
                StatusCode::NOT_FOUND,
                "tool_not_found",
            ),
            (
                json!({"name": "other__echo"}),
                StatusCode::NOT_FOUND,
                "tool_not_found",
            ),
            (
                json!({"name": "echo"}),
                StatusCode::NOT_FOUND,
                "tool_not_found",
            ),
            (
                json!({"name": "files__echo", "arguments": [1]}),
                StatusCode::BAD_REQUEST,
                "invalid_arguments",
            ),
            (
                json!({"arguments": {}}),
                StatusCode::BAD_REQUEST,
                "invalid_request",
            ),
            (
                json!({"name": "files__slow"}),
                StatusCode::GATEWAY_TIMEOUT,
                "tool_timeout",
            ),
            (
                json!({"name": "files__rpc"}),
                StatusCode::BAD_GATEWAY,
                "mcp_error",
            ),
        ];
        for (request, expected_status, expected_code) in cases {
            let (status, body) =
                send_request(&app, "POST", "/v1/mcp/execute", Some(&request), None).await;
            assert_eq!(status, expected_status, "{request}: {body}");
            assert_eq!(body["error"]["code"], expected_code, "{request}");
        }
        let response = send_request(&app, "GET", "/v1/mcp/execute", None, None).await;
        assert_eq!(response.0, StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn execute_against_a_failed_server_is_503_with_the_last_error() {
        let bad = FakeMcp::new(&["echo"]);
        bad.fail_next_connects(1000);
        let hub = hub_with(vec![("bad", bad, Duration::from_millis(500))], 0).await;
        let app = app_with_tools(hub, None, None);
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "bad__echo"})),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "mcp_server_not_ready");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("failed") && message.contains("scripted connect failure"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn malformed_bodies_are_400() {
        let (app, _) = single_server_app().await;
        let (status, body) = send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!("just a string")),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("Invalid request body")
        );
    }

    #[tokio::test]
    async fn endpoints_require_the_api_key_when_one_is_configured() {
        let fake = FakeMcp::new(&["echo"]);
        let hub = hub_with(vec![("files", fake, Duration::from_secs(1))], 1).await;
        let app = app_with_tools(hub, Some("secret"), None);
        for (method, path, body) in [
            ("GET", "/v1/mcp/servers", None),
            ("GET", "/v1/mcp/tools", None),
            (
                "POST",
                "/v1/mcp/execute",
                Some(json!({"name": "files__echo"})),
            ),
        ] {
            let (denied, _) = send_request(&app, method, path, body.as_ref(), None).await;
            assert_eq!(denied, StatusCode::UNAUTHORIZED, "{path}");
            let (wrong, _) = send_request(&app, method, path, body.as_ref(), Some("nope")).await;
            assert_eq!(wrong, StatusCode::UNAUTHORIZED, "{path}");
            let (allowed, _) =
                send_request(&app, method, path, body.as_ref(), Some("secret")).await;
            assert_eq!(allowed, StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn tool_endpoints_do_not_count_as_traffic_in_metrics() {
        let metrics = Arc::new(crate::metrics::MetricsStore::new(Duration::from_secs(60)));
        let fake = FakeMcp::new(&["echo"]);
        let hub = hub_with(vec![("files", fake, Duration::from_secs(1))], 1).await;
        let app = app_with_tools(hub, None, Some(Arc::clone(&metrics)));
        send_request(&app, "GET", "/v1/mcp/servers", None, None).await;
        send_request(&app, "GET", "/v1/mcp/tools", None, None).await;
        send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "files__echo"})),
            None,
        )
        .await;
        send_request(
            &app,
            "POST",
            "/v1/mcp/execute",
            Some(&json!({"name": "files__nope"})),
            None,
        )
        .await;
        send_request(
            &app,
            "POST",
            "/v1/web/search",
            Some(&json!({"query": "x"})),
            None,
        )
        .await;
        assert!(metrics.snapshot().is_empty());

        // The same plumbing does record real traffic.
        send_request(
            &app,
            "POST",
            "/v1/chat/completions",
            Some(&json!({"model": "missing", "messages": []})),
            None,
        )
        .await;
        assert_eq!(metrics.snapshot().len(), 1);
    }
}
