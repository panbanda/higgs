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
    error::ServerError,
    state::SharedState,
    types::openai::{
        ChoiceLogprobs, CompletionChoice, CompletionChunk, CompletionChunkChoice,
        CompletionRequest, CompletionResponse, CompletionUsage, StopSequence, TokenLogprob,
        TopLogprob,
    },
};
use higgs_models::SamplingParams;

pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.prompt.is_empty() {
        return Err(ServerError::BadRequest(
            "prompt must not be empty".to_owned(),
        ));
    }

    if req.stream == Some(true) {
        let stream = completions_stream(state, req)?;
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        let response = completions_non_streaming(state, req).await?;
        Ok(Json(response).into_response())
    }
}

async fn completions_non_streaming(
    state: SharedState,
    req: CompletionRequest,
) -> Result<CompletionResponse, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), false)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    let tokenizer = engine.tokenizer().clone();
    let output = tokio::task::spawn_blocking(move || {
        engine.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            None,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let logprobs_response = output
        .token_logprobs
        .as_ref()
        .map(|lps| logprobs_to_response(lps, &tokenizer));

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.text,
            finish_reason: output.finish_reason,
            logprobs: logprobs_response,
        }],
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
}

#[allow(clippy::needless_pass_by_value)]
fn completions_stream(
    state: SharedState,
    req: CompletionRequest,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), false)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = req.model;

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            &tx,
            None,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let stream = async_stream::stream! {
        while let Some(output) = rx.recv().await {
            let chunk = CompletionChunk {
                id: request_id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: output.new_text,
                    finish_reason: output.finish_reason,
                }],
            };
            match serde_json::to_string(&chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(stream)
}

fn build_sampling_params(req: &CompletionRequest) -> SamplingParams {
    SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    }
}

fn logprobs_to_response(
    infos: &[higgs_models::TokenLogprobInfo],
    tokenizer: &higgs_engine::tokenizers::Tokenizer,
) -> ChoiceLogprobs {
    let content = infos
        .iter()
        .map(|info| {
            let token_str = tokenizer
                .decode(&[info.token_id], false)
                .unwrap_or_default();
            let top = info
                .top_logprobs
                .iter()
                .map(|e| {
                    let t = tokenizer.decode(&[e.token_id], false).unwrap_or_default();
                    TopLogprob {
                        token: t,
                        logprob: e.logprob,
                    }
                })
                .collect();
            TokenLogprob {
                token: token_str,
                logprob: info.logprob,
                top_logprobs: top,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}
