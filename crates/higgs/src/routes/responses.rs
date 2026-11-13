use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::{Extension, Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response, Sse,
        sse::{Event, KeepAlive},
    },
};
use bytes::Bytes;
use tokio_stream::{Stream, StreamExt};

use crate::{
    config::ApiFormat,
    constraint::ConstraintSpec,
    error::{ErrorDetail, ErrorResponse, ServerError},
    metrics::{RequestMetricsContext, RequestRecord, RequestTiming},
    responses::{
        store::{ResponseStore, StoredResponse},
        stream::ResponseStreamBuilder,
        translate::{
            ResponseContext, build_chat_request, outcome_from_chat, output_to_input_items,
            parse_input, reject_unsupported_fields, routing_messages,
        },
    },
    router::ResolvedRoute,
    state::{EngineLease, SharedState},
    types::responses::{
        CreateResponseRequest, DeletedResponse, InputItem, ResponseObject, ResponseStatus,
        StreamEvent,
    },
};

/// Errors a Responses handler can return: everything the shared server error
/// covers, plus the 404 shape for unknown response ids.
#[derive(Debug)]
pub enum ResponsesError {
    Server(ServerError),
    NotFound { message: String, code: &'static str },
}

impl From<ServerError> for ResponsesError {
    fn from(error: ServerError) -> Self {
        Self::Server(error)
    }
}

impl IntoResponse for ResponsesError {
    fn into_response(self) -> Response {
        match self {
            Self::Server(error) => error.into_response(),
            Self::NotFound { message, code } => (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message,
                        r#type: "invalid_request_error".to_owned(),
                        code: Some(code.to_owned()),
                    },
                }),
            )
                .into_response(),
        }
    }
}

fn response_not_found(id: &str) -> ResponsesError {
    ResponsesError::NotFound {
        message: format!("Response with id '{id}' not found."),
        code: "response_not_found",
    }
}

pub async fn create_response(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    Extension(store): Extension<Arc<ResponseStore>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ResponsesError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
    let request: CreateResponseRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&request.model);

    let messages = routing_messages(request.input.as_ref());
    let resolved = state
        .router
        .resolve(&request.model, messages.as_deref())
        .await
        .map_err(ServerError::from)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
        } => {
            let local = LocalRequest {
                state,
                request_metrics,
                store,
                engine,
                model_name,
                routing_method,
                received_at,
            };
            create_local_response(local, request).await
        }
        ResolvedRoute::Remote {
            provider_name,
            provider_url,
            provider_format,
            strip_auth,
            api_key,
            model_rewrite,
            routing_method,
            ..
        } => match provider_format {
            ApiFormat::OpenAi => {
                let metrics_model = model_rewrite
                    .as_deref()
                    .unwrap_or(&request.model)
                    .to_owned();
                let proxy_body = if let Some(ref rewrite) = model_rewrite {
                    crate::proxy::rewrite_model_in_body(&body, rewrite)?
                } else {
                    body
                };
                let start = Instant::now();
                let result = crate::proxy::proxy_request(
                    &state.http_client,
                    &provider_url,
                    "/v1/responses",
                    proxy_body,
                    &headers,
                    strip_auth,
                    api_key.as_ref().map(crate::config::Secret::expose),
                )
                .await;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(metrics_model),
                        provider: Some(provider_name),
                        routing_method: routing_method.into(),
                        status: result.as_ref().map_or(502, |resp| resp.status().as_u16()),
                        duration: start.elapsed(),
                        input_tokens: 0,
                        output_tokens: 0,
                        error_body: None,
                        timing: RequestTiming::default(),
                    });
                    request_metrics.mark_recorded();
                }
                Ok(result?)
            }
            ApiFormat::Anthropic => Err(ServerError::BadRequest(format!(
                "The Responses API is only available for local models and OpenAI-format \
                 providers; model '{}' routes to the Anthropic-format provider '{provider_name}'",
                request.model
            ))
            .into()),
        },
    }
}

struct LocalRequest {
    state: SharedState,
    request_metrics: RequestMetricsContext,
    store: Arc<ResponseStore>,
    engine: EngineLease,
    model_name: String,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
}

async fn create_local_response(
    local: LocalRequest,
    mut request: CreateResponseRequest,
) -> Result<Response, ResponsesError> {
    let LocalRequest {
        state,
        request_metrics,
        store,
        engine,
        model_name,
        routing_method,
        received_at,
    } = local;

    reject_unsupported_fields(&request)?;
    let new_items = parse_input(request.input.take(), engine.is_vlm())?;
    let mut conversation = match request.previous_response_id.as_deref() {
        Some(previous_id) => {
            store
                .get(previous_id)
                .ok_or_else(|| ResponsesError::NotFound {
                    message: format!("Previous response with id '{previous_id}' not found."),
                    code: "previous_response_not_found",
                })?
                .conversation
        }
        None => Vec::new(),
    };
    conversation.extend(new_items);
    if conversation.is_empty() {
        return Err(
            ServerError::BadRequest("input must contain at least one item".to_owned()).into(),
        );
    }

    let chat = build_chat_request(&request, &model_name, &conversation)?;
    let constraint_spec = ConstraintSpec::from_request(chat.response_format.as_ref(), None, None)?;
    let context = ResponseContext::from_request(&request, model_name);
    let store_response = request.store.unwrap_or(true).then(|| Arc::clone(&store));

    if request.stream == Some(true) {
        let chunks = crate::tool_loop::chunk_stream(
            Arc::clone(&state),
            chat,
            engine,
            constraint_spec,
            state.metrics.clone(),
            routing_method,
            received_at,
        )?;
        let events = response_event_stream(chunks, context, store_response, conversation);
        if state.metrics.is_some() {
            request_metrics.mark_recorded();
        }
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let start = Instant::now();
    let chat_response = crate::tool_loop::complete(
        Arc::clone(&state),
        chat,
        engine,
        constraint_spec,
        received_at,
    )
    .await?;
    let input_tokens = u64::from(chat_response.usage.prompt_tokens);
    let output_tokens = u64::from(chat_response.usage.completion_tokens);
    let object = context.object(outcome_from_chat(chat_response));
    if let Some(response_store) = store_response {
        store_completed(&response_store, &object, conversation);
    }
    if let Some(ref metrics) = state.metrics {
        metrics.record(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(object.model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: start.elapsed(),
            input_tokens,
            output_tokens,
            error_body: None,
            timing: RequestTiming::default(),
        });
        request_metrics.mark_recorded();
    }
    Ok(Json(object).into_response())
}

pub async fn get_response(
    Extension(store): Extension<Arc<ResponseStore>>,
    Path(id): Path<String>,
) -> Result<Json<ResponseObject>, ResponsesError> {
    store
        .get(&id)
        .map(|stored| Json(stored.response))
        .ok_or_else(|| response_not_found(&id))
}

pub async fn delete_response(
    Extension(store): Extension<Arc<ResponseStore>>,
    Path(id): Path<String>,
) -> Result<Json<DeletedResponse>, ResponsesError> {
    if store.remove(&id) {
        Ok(Json(DeletedResponse {
            id,
            object: "response.deleted",
            deleted: true,
        }))
    } else {
        Err(response_not_found(&id))
    }
}

fn store_completed(
    store: &ResponseStore,
    response: &ResponseObject,
    mut conversation: Vec<InputItem>,
) {
    if response.status == ResponseStatus::Failed {
        return;
    }
    conversation.extend(output_to_input_items(&response.output));
    store.insert(StoredResponse {
        response: response.clone(),
        conversation,
    });
}

fn sse_event(event: &StreamEvent) -> Option<Event> {
    match serde_json::to_string(event) {
        Ok(json) => Some(Event::default().event(event.body.event_name()).data(json)),
        Err(e) => {
            tracing::error!(error = %e, "Failed to serialize Responses SSE event");
            None
        }
    }
}

/// Re-frame chat chunks as Responses events, storing the finished response
/// before the terminal event goes out so a client can chain immediately.
fn response_event_stream(
    chunk_source: impl Stream<Item = String>,
    context: ResponseContext,
    store: Option<Arc<ResponseStore>>,
    conversation: Vec<InputItem>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut builder = ResponseStreamBuilder::new(context);
        for event in builder.start() {
            if let Some(frame) = sse_event(&event) {
                yield Ok(frame);
            }
        }

        let mut chunks = std::pin::pin!(chunk_source);
        while let Some(data) = chunks.next().await {
            for event in builder.ingest(&data) {
                if let Some(frame) = sse_event(&event) {
                    yield Ok(frame);
                }
            }
        }

        let finish = builder.finish();
        if let Some(ref response_store) = store {
            store_completed(response_store, &finish.response, conversation);
        }
        for event in finish.events {
            if let Some(frame) = sse_event(&event) {
                yield Ok(frame);
            }
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use axum::body::Body;
    use http::Request;
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;
    use crate::metrics::MetricsStore;
    use crate::router::Router;
    use crate::state::AppState;
    use crate::state::Engine;
    use crate::types::responses::ContentPart;

    const STUB_MODEL: &str = "stub-model";

    fn local_state(metrics: Option<Arc<MetricsStore>>) -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[[models]]\npath = \"stub/model\"\nname = \"stub-model\"\n",
        )
        .unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let mut engines = HashMap::new();
        engines.insert(
            STUB_MODEL.to_owned(),
            Arc::new(Engine::test_stub(STUB_MODEL)),
        );
        let router = Router::from_config(&config, engines).unwrap();
        Arc::new(AppState {
            router,
            config,
            http_client: reqwest::Client::new(),
            metrics,
            tools: Arc::new(crate::tools::ToolHub::disabled()),
        })
    }

    fn app(metrics: Option<Arc<MetricsStore>>) -> axum::Router {
        crate::build_router(local_state(metrics), 300.0, None, 0, 1024 * 1024, None)
    }

    async fn post(app: &axum::Router, body: &Value) -> (StatusCode, Value) {
        send(
            app,
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
    }

    async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn error_message(body: &Value) -> &str {
        body["error"]["message"].as_str().unwrap()
    }

    fn stored(id: &str) -> StoredResponse {
        let context = ResponseContext::from_request(
            &serde_json::from_value(json!({"model": STUB_MODEL, "input": "hi"})).unwrap(),
            STUB_MODEL.to_owned(),
        );
        let mut response = context.object(crate::responses::translate::Outcome::in_progress());
        response.id = id.to_owned();
        response.status = ResponseStatus::Completed;
        StoredResponse {
            response,
            conversation: Vec::new(),
        }
    }

    #[tokio::test]
    async fn get_returns_a_stored_response_and_delete_removes_it() {
        let store = Arc::new(ResponseStore::default());
        store.insert(stored("resp_a"));

        let Json(found) = get_response(Extension(Arc::clone(&store)), Path("resp_a".to_owned()))
            .await
            .unwrap();
        assert_eq!(found.id, "resp_a");

        let Json(deleted) =
            delete_response(Extension(Arc::clone(&store)), Path("resp_a".to_owned()))
                .await
                .unwrap();
        assert_eq!(
            (deleted.id.as_str(), deleted.object, deleted.deleted),
            ("resp_a", "response.deleted", true)
        );

        let missing = get_response(Extension(store), Path("resp_a".to_owned())).await;
        assert!(matches!(
            missing,
            Err(ResponsesError::NotFound {
                code: "response_not_found",
                ..
            })
        ));
    }

    #[test]
    fn store_completed_keeps_the_conversation_with_the_model_output() {
        let store = ResponseStore::default();
        let mut response = stored("resp_a").response;
        response.output = vec![crate::types::responses::OutputItem::Message {
            id: "msg_1".to_owned(),
            role: "assistant",
            content: vec![ContentPart::output_text("hello".to_owned())],
            status: ResponseStatus::Completed,
        }];
        let user_turn = InputItem::Message {
            role: "user".to_owned(),
            content: vec![crate::types::responses::InputContent::Text("hi".to_owned())],
        };
        store_completed(&store, &response, vec![user_turn]);

        let kept = store.get("resp_a").unwrap();
        assert_eq!(kept.conversation.len(), 2);
        assert!(matches!(
            kept.conversation.last(),
            Some(InputItem::Message { role, .. }) if role == "assistant"
        ));
    }

    #[test]
    fn failed_responses_are_not_stored() {
        let store = ResponseStore::default();
        let mut response = stored("resp_a").response;
        response.status = ResponseStatus::Failed;
        store_completed(&store, &response, Vec::new());
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn router_reports_unknown_ids_as_404() {
        let app = app(None);
        for method in ["GET", "DELETE"] {
            let (status, body) = send(
                &app,
                Request::builder()
                    .method(method)
                    .uri("/v1/responses/resp_nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{method}");
            assert_eq!(body["error"]["code"], "response_not_found");
        }
    }

    #[tokio::test]
    async fn unknown_previous_response_id_is_404() {
        let (status, body) = post(
            &app(None),
            &json!({"model": STUB_MODEL, "input": "hi", "previous_response_id": "resp_nope"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "previous_response_not_found");
        assert!(error_message(&body).contains("resp_nope"));
    }

    #[tokio::test]
    async fn unsupported_tool_types_are_named_in_a_400() {
        let (status, body) = post(
            &app(None),
            &json!({"model": STUB_MODEL, "input": "hi", "tools": [{"type": "web_search"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(error_message(&body).contains("'web_search'"), "{body}");
    }

    #[tokio::test]
    async fn invalid_requests_are_rejected_with_a_400() {
        let cases = [
            (json!({"model": STUB_MODEL}), "input is required"),
            (
                json!({"model": STUB_MODEL, "input": []}),
                "at least one item",
            ),
            (
                json!({"model": STUB_MODEL, "input": [{"type": "computer_call"}]}),
                "'computer_call'",
            ),
            (
                json!({"model": STUB_MODEL, "input": [{"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AA"}
                ]}]}),
                "not a vision model",
            ),
            (
                json!({"model": STUB_MODEL, "input": "hi", "tool_choice": "sometimes"}),
                "\"sometimes\"",
            ),
            (
                json!({"model": STUB_MODEL, "input": "hi", "tool_choice": {"type": "web_search"}}),
                "web_search",
            ),
            (
                json!({"model": STUB_MODEL, "input": "hi", "tool_choice": "required"}),
                "no tools",
            ),
            (
                json!({"model": STUB_MODEL, "input": "hi", "background": true}),
                "background",
            ),
            (
                json!({"model": STUB_MODEL, "input": "hi", "text": {"format": {"type": "json_schema"}}}),
                "requires a schema",
            ),
            (
                json!({"model": STUB_MODEL, "input": 42}),
                "Invalid request body",
            ),
        ];
        let app = app(None);
        for (request, expected) in cases {
            let (status, body) = post(&app, &request).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{request}");
            assert!(error_message(&body).contains(expected), "{request}: {body}");
        }
    }

    #[tokio::test]
    async fn malformed_json_is_a_400() {
        let (status, body) = send(
            &app(None),
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error_message(&body).contains("Invalid request body"));
    }

    #[tokio::test]
    async fn unknown_model_is_404_model_not_found() {
        let (status, body) = post(
            &app(None),
            &json!({"model": "no-such-model", "input": "hi"}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "model_not_found");
    }

    #[tokio::test]
    async fn rejected_requests_are_recorded_in_metrics_with_the_model() {
        let metrics = Arc::new(MetricsStore::new(Duration::from_secs(60)));
        let app = app(Some(Arc::clone(&metrics)));
        let (status, _) = post(
            &app,
            &json!({"model": STUB_MODEL, "input": "hi", "tools": [{"type": "web_search"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let records = metrics.snapshot();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, 400);
        assert_eq!(records[0].model.as_deref(), Some(STUB_MODEL));
    }
}
