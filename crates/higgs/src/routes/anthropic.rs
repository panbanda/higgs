use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use tokio_stream::Stream;

use crate::{
    anthropic_adapter::{anthropic_messages_to_engine, openai_finish_to_anthropic_stop},
    error::ServerError,
    state::SharedState,
    types::anthropic::{
        AnthropicUsage, ContentBlockDeltaEvent, ContentBlockResponse, ContentBlockStartEvent,
        ContentBlockStartPayload, ContentBlockStopEvent, CountTokensRequest, CountTokensResponse,
        CreateMessageRequest, CreateMessageResponse, MessageDelta, MessageDeltaEvent,
        MessageStartEvent, MessageStartPayload, MessageStopEvent, TextDelta,
    },
};
use higgs_models::SamplingParams;

pub async fn create_message(
    State(state): State<SharedState>,
    Json(req): Json<CreateMessageRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".to_owned(),
        ));
    }

    if req.stream == Some(true) {
        let stream = create_message_stream(state, req)?;
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        let response = create_message_non_streaming(state, req).await?;
        Ok(Json(response).into_response())
    }
}

async fn create_message_non_streaming(
    state: SharedState,
    req: CreateMessageRequest,
) -> Result<CreateMessageResponse, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_deref());
    let tools = req.tools.as_deref();

    let prompt_tokens = engine
        .prepare_chat_prompt(&engine_messages, tools)
        .map_err(ServerError::Engine)?;

    let output = tokio::task::spawn_blocking(move || {
        engine.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            None,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let stop_reason = openai_finish_to_anthropic_stop(&output.finish_reason);
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    Ok(CreateMessageResponse {
        id: msg_id,
        message_type: "message",
        role: "assistant",
        content: vec![ContentBlockResponse {
            block_type: "text",
            text: output.text,
        }],
        model: req.model,
        stop_reason: Some(stop_reason),
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
        },
    })
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
fn create_message_stream(
    state: SharedState,
    req: CreateMessageRequest,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_deref());
    let tools = req.tools.as_deref();

    let prompt_tokens = engine
        .prepare_chat_prompt(&engine_messages, tools)
        .map_err(ServerError::Engine)?;

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = req.model;
    let prompt_token_count = u32::try_from(prompt_tokens.len())
        .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

    // Spawn generation before creating the stream so prefill starts immediately
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            &tx,
            None,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during Anthropic streaming");
        }
    });

    let stream = async_stream::stream! {
        // 1. message_start
        let start_event = MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: msg_id.clone(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens: prompt_token_count,
                    output_tokens: 0,
                },
            },
        };
        match serde_json::to_string(&start_event) {
            Ok(json) => yield Ok(Event::default().event("message_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 2. content_block_start
        let block_start = ContentBlockStartEvent {
            event_type: "content_block_start",
            index: 0,
            content_block: ContentBlockStartPayload {
                block_type: "text",
                text: String::new(),
            },
        };
        match serde_json::to_string(&block_start) {
            Ok(json) => yield Ok(Event::default().event("content_block_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 3. content_block_delta events (one per token)
        let mut final_stop_reason = None;
        let mut total_output_tokens: u32 = 0;

        while let Some(output) = rx.recv().await {
            if !output.new_text.is_empty() {
                let delta_event = ContentBlockDeltaEvent {
                    event_type: "content_block_delta",
                    index: 0,
                    delta: TextDelta {
                        delta_type: "text_delta",
                        text: output.new_text,
                    },
                };
                match serde_json::to_string(&delta_event) {
                    Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }
            total_output_tokens = output.completion_tokens;
            if let Some(reason) = output.finish_reason {
                final_stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
            }
        }

        // 4. content_block_stop
        let block_stop = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        match serde_json::to_string(&block_stop) {
            Ok(json) => yield Ok(Event::default().event("content_block_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 5. message_delta
        let msg_delta = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta {
                stop_reason: final_stop_reason,
            },
            usage: AnthropicUsage {
                input_tokens: prompt_token_count,
                output_tokens: total_output_tokens,
            },
        };
        match serde_json::to_string(&msg_delta) {
            Ok(json) => yield Ok(Event::default().event("message_delta").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 6. message_stop
        let msg_stop = MessageStopEvent {
            event_type: "message_stop",
        };
        match serde_json::to_string(&msg_stop) {
            Ok(json) => yield Ok(Event::default().event("message_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }
    };

    Ok(stream)
}

pub async fn count_tokens(
    State(state): State<SharedState>,
    Json(req): Json<CountTokensRequest>,
) -> Result<Json<CountTokensResponse>, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_deref());
    let tools = req.tools.as_deref();

    let tokens = engine
        .prepare_chat_prompt(&engine_messages, tools)
        .map_err(ServerError::Engine)?;

    let count = u32::try_from(tokens.len())
        .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

    Ok(Json(CountTokensResponse {
        input_tokens: count,
    }))
}
