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
    backend::EngineLease,
    config::ApiFormat,
    error::{ErrorDetail, ErrorResponse, ServerError},
    generation::{EventStream, GenerationEvent, StreamMetrics, ToolLoopEvent},
    metrics::{RequestMetricsContext, RequestRecord, RequestTiming},
    responses::{
        store::{ResponseStore, StoredResponse},
        stream::ResponseStreamBuilder,
        translate::{
            ResponseContext, build_generation_request, loop_messages_to_input_items,
            outcome_from_completed, output_to_input_items, parse_input, reject_unsupported_fields,
            routing_messages,
        },
    },
    router::ResolvedRoute,
    state::SharedState,
    types::responses::{
        CreateResponseRequest, DeletedResponse, InputItem, OutputItem, ResponseObject,
        ResponseStatus, StreamEvent,
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

    let generation = build_generation_request(
        &request,
        &model_name,
        &conversation,
        state.config.server.max_tokens,
    )?;
    let context = ResponseContext::from_request(&request, model_name);
    let store_response = request.store.unwrap_or(true).then(|| Arc::clone(&store));

    if request.stream == Some(true) {
        let events = crate::generation::stream(
            Arc::clone(&state),
            generation,
            engine,
            state.metrics.clone().map(|metrics_store| StreamMetrics {
                store: metrics_store,
                routing_method,
            }),
            received_at,
        )?;
        let responses = response_event_stream(events, context, store_response, conversation);
        if state.metrics.is_some() {
            request_metrics.mark_recorded();
        }
        return Ok(Sse::new(responses)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let start = Instant::now();
    let mut completed =
        crate::generation::complete(Arc::clone(&state), generation, engine, received_at).await?;
    if let Some(ref metrics) = state.metrics {
        crate::generation::record_completion(
            metrics,
            &context.model,
            routing_method,
            start,
            &completed.usage,
        );
        request_metrics.mark_recorded();
    }
    let loop_messages = std::mem::take(&mut completed.loop_messages);
    let object = context.object(outcome_from_completed(completed));
    if let Some(response_store) = store_response {
        conversation.extend(loop_messages_to_input_items(&loop_messages));
        store_completed(&response_store, &object, &object.output, conversation);
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

/// Record `response` and the conversation behind it. `output` is what to
/// append to the conversation: the whole response output, except when a tool
/// loop already recorded the earlier rounds' text in `conversation`.
fn store_completed(
    store: &ResponseStore,
    response: &ResponseObject,
    output: &[OutputItem],
    mut conversation: Vec<InputItem>,
) {
    if response.status == ResponseStatus::Failed {
        return;
    }
    conversation.extend(output_to_input_items(output));
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

/// The output items of the last tool-loop round alone. The stream builder
/// merges every round's text into one message, but the loop's recorded
/// messages already carry the earlier rounds' text.
fn final_round_output(context: ResponseContext, events: Vec<GenerationEvent>) -> Vec<OutputItem> {
    let mut builder = ResponseStreamBuilder::new(context);
    for event in events {
        builder.ingest(event);
    }
    builder.finish().response.output
}

/// Render generation events as Responses events, storing the finished
/// response before the terminal event goes out so a client can chain
/// immediately.
fn response_event_stream(
    mut generated: EventStream,
    context: ResponseContext,
    store: Option<Arc<ResponseStore>>,
    mut conversation: Vec<InputItem>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let round_context = context.clone();
        let mut builder = ResponseStreamBuilder::new(context);
        let mut loop_messages = Vec::new();
        let mut final_round_events = Vec::new();
        for event in builder.start() {
            if let Some(frame) = sse_event(&event) {
                yield Ok(frame);
            }
        }

        while let Some(generation_event) = generated.next().await {
            if let GenerationEvent::ToolLoop(ToolLoopEvent::Messages(messages)) = generation_event {
                loop_messages = messages;
                continue;
            }
            match &generation_event {
                GenerationEvent::ToolLoop(ToolLoopEvent::Call { .. } | ToolLoopEvent::Result { .. }) => {
                    final_round_events.clear();
                }
                GenerationEvent::ToolLoop(_) => {}
                GenerationEvent::Started { .. }
                | GenerationEvent::PrefillProgress(_)
                | GenerationEvent::ReasoningDelta { .. }
                | GenerationEvent::TextDelta { .. }
                | GenerationEvent::ToolCall { .. }
                | GenerationEvent::Finish { .. }
                | GenerationEvent::Usage(_)
                | GenerationEvent::Error(_) => final_round_events.push(generation_event.clone()),
            }
            for event in builder.ingest(generation_event) {
                if let Some(frame) = sse_event(&event) {
                    yield Ok(frame);
                }
            }
        }

        let finish = builder.finish();
        if let Some(ref response_store) = store {
            conversation.extend(loop_messages_to_input_items(&loop_messages));
            let output = if loop_messages.is_empty() {
                finish.response.output.clone()
            } else {
                final_round_output(round_context, final_round_events)
            };
            store_completed(response_store, &finish.response, &output, conversation);
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
    use crate::test_support::StubEngine;
    use crate::types::responses::ContentPart;

    const STUB_MODEL: &str = "stub-model";

    fn local_state(metrics: Option<Arc<MetricsStore>>) -> SharedState {
        local_state_with_config(metrics, "")
    }

    fn local_state_with_config(
        metrics: Option<Arc<MetricsStore>>,
        extra_config: &str,
    ) -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            format!("[[models]]\npath = \"stub/model\"\nname = \"stub-model\"\n{extra_config}"),
        )
        .unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let mut engines = HashMap::new();
        engines.insert(
            STUB_MODEL.to_owned(),
            Arc::new(StubEngine::named(STUB_MODEL).into_loaded()),
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
        store_completed(&store, &response, &response.output, vec![user_turn]);

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
        store_completed(&store, &response, &response.output, Vec::new());
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
        // A lone local model would serve any name, so a provider keeps names strict.
        let state =
            local_state_with_config(None, "\n[provider.remote]\nurl = \"http://127.0.0.1:1\"\n");
        let strict_app = crate::build_router(state, 300.0, None, 0, 1024 * 1024, None);
        let (status, body) = post(
            &strict_app,
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

    #[tokio::test]
    async fn a_failed_stream_ends_with_response_failed_and_is_not_stored() {
        use crate::test_support::{failure_event, post_json, sse_json_events, token_event};

        let mut stub = StubEngine::named("stub");
        stub.script.stream = vec![token_event("par", 1, None), failure_event()];
        let (app, _engine) = crate::test_support::app_with_stub(stub);

        let (status, text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "hi", "stream": true}),
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{text}");
        let events = sse_json_events(&text);
        let last = events.last().unwrap();
        assert_eq!(last["type"], "response.failed", "{text}");
        assert_eq!(last["response"]["status"], "failed");
        assert_eq!(
            last["response"]["error"]["message"],
            "generation failed in the engine"
        );
        let id = last["response"]["id"].as_str().unwrap();
        let (follow_up, _) = post(
            &app,
            &json!({"model": "stub", "input": "again", "previous_response_id": id}),
        )
        .await;
        assert_eq!(follow_up, StatusCode::NOT_FOUND);
    }

    const CALL_ECHO: &str =
        r#"<tool_call>{"name": "files__echo", "arguments": {"text": "hi"}}</tool_call>"#;

    async fn tool_loop_app(configure: impl FnOnce(&mut StubEngine)) -> (axum::Router, StubEngine) {
        use crate::mcp::testing::{CallBehavior, FakeMcp, ready_manager, text_result};

        let fake = FakeMcp::new(&["echo"]);
        fake.on_call("echo", CallBehavior::Reply(text_result("pong")));
        let hub = crate::tools::ToolHub::new(
            ready_manager(vec![("files", fake, Duration::from_secs(2))], 1).await,
            crate::web::WebService::disabled(),
        );
        let mut stub = StubEngine::named("stub");
        stub.script.prompt_tokens = vec![1, 2, 3];
        configure(&mut stub);
        crate::test_support::app_with_stub_config_and_tools(stub, "", hub)
    }

    fn last_prompt_has_tool_result(engine: &StubEngine, expected: &str) -> bool {
        engine
            .recorded_prompt_messages()
            .last()
            .is_some_and(|messages| {
                messages
                    .iter()
                    .any(|(role, content)| role == "tool" && content.contains(expected))
            })
    }

    #[tokio::test]
    async fn a_follow_up_sees_tool_results_from_the_stored_non_streaming_response() {
        let (app, engine) = tool_loop_app(|stub| {
            stub.script.generations = vec![
                crate::test_support::generation_output(CALL_ECHO, 5, 3),
                crate::test_support::generation_output("The answer is pong", 5, 3),
                crate::test_support::generation_output("noted", 5, 3),
            ];
        })
        .await;
        let (first_status, first) = post(
            &app,
            &json!({"model": "stub", "input": "use the tool", "mcp": {"enabled": true}}),
        )
        .await;
        assert_eq!(first_status, StatusCode::OK, "{first}");
        let id = first["id"].as_str().unwrap();

        let (second_status, second) = post(
            &app,
            &json!({"model": "stub", "input": "thanks", "previous_response_id": id}),
        )
        .await;
        assert_eq!(second_status, StatusCode::OK, "{second}");
        assert!(
            last_prompt_has_tool_result(&engine, "pong"),
            "{:?}",
            engine.recorded_prompt_messages()
        );
    }

    #[tokio::test]
    async fn a_streamed_follow_up_replays_each_round_s_text_once_and_in_order() {
        use crate::test_support::{post_json, sse_json_events, token_event};

        let (app, engine) = tool_loop_app(|stub| {
            stub.script.streams = vec![
                vec![
                    token_event("Let me check. ", 3, None),
                    token_event(CALL_ECHO, 12, Some("stop")),
                ],
                vec![token_event("Final answer", 5, Some("stop"))],
                vec![token_event("noted", 2, Some("stop"))],
            ];
        })
        .await;
        let (_, first_text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "use the tool", "stream": true,
                    "mcp": {"enabled": true}}),
        )
        .await;
        let id = sse_json_events(&first_text)
            .iter()
            .find_map(|event| event.pointer("/response/id")?.as_str().map(str::to_owned))
            .unwrap();
        post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "thanks", "stream": true,
                    "previous_response_id": id}),
        )
        .await;

        let prompts = engine.recorded_prompt_messages();
        let follow_up = prompts.last().unwrap();
        let position = |needle: &str| -> Vec<usize> {
            follow_up
                .iter()
                .enumerate()
                .filter(|(_, (_, content))| content.contains(needle))
                .map(|(index, _)| index)
                .collect()
        };
        let intermediate = position("Let me check.");
        let result = position("pong");
        let last_text = position("Final answer");
        assert_eq!(intermediate.len(), 1, "{follow_up:?}");
        assert_eq!(result.len(), 1, "{follow_up:?}");
        assert_eq!(last_text.len(), 1, "{follow_up:?}");
        assert!(
            intermediate[0] < result[0] && result[0] < last_text[0],
            "{follow_up:?}"
        );
    }

    #[tokio::test]
    async fn a_follow_up_sees_tool_results_from_the_stored_streaming_response() {
        use crate::test_support::{post_json, sse_json_events, token_event};

        let (app, engine) = tool_loop_app(|stub| {
            stub.script.streams = vec![
                vec![
                    token_event("Let me check. ", 3, None),
                    token_event(CALL_ECHO, 12, Some("stop")),
                ],
                vec![token_event("Final answer", 5, Some("stop"))],
                vec![token_event("noted", 2, Some("stop"))],
            ];
        })
        .await;
        let (first_status, first_text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "use the tool", "stream": true,
                    "mcp": {"enabled": true}}),
        )
        .await;
        assert_eq!(first_status, StatusCode::OK, "{first_text}");
        let id = sse_json_events(&first_text)
            .iter()
            .find_map(|event| event.pointer("/response/id")?.as_str().map(str::to_owned))
            .unwrap();

        let (second_status, second_text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "thanks", "stream": true,
                    "previous_response_id": id}),
        )
        .await;
        assert_eq!(second_status, StatusCode::OK, "{second_text}");
        assert!(
            last_prompt_has_tool_result(&engine, "pong"),
            "{:?}",
            engine.recorded_prompt_messages()
        );
    }
}
