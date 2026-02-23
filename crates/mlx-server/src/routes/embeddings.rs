use axum::{Json, extract::State};

use crate::{
    error::ServerError,
    state::SharedState,
    types::openai::{
        EmbeddingInput, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    },
};

/// POST /v1/embeddings
///
/// Generates embeddings by running a model forward pass to get hidden states,
/// then mean-pooling across the sequence dimension and L2-normalizing.
pub async fn embeddings(
    State(state): State<SharedState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let inputs = match &req.input {
        EmbeddingInput::Single(s) => vec![s.clone()],
        EmbeddingInput::Multiple(v) => v.clone(),
    };

    if inputs.is_empty() {
        return Err(ServerError::BadRequest(
            "input must not be empty".to_owned(),
        ));
    }

    let mut data = Vec::new();
    let mut total_tokens: u32 = 0;

    for (idx, text) in inputs.iter().enumerate() {
        let encoding = engine
            .tokenizer()
            .encode(text.as_str(), false)
            .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;

        let token_ids = encoding.get_ids();
        let token_count: u32 = token_ids
            .len()
            .try_into()
            .map_err(|_| ServerError::BadRequest("Input too long".to_owned()))?;
        total_tokens = total_tokens.saturating_add(token_count);

        let embedding = engine
            .embed(token_ids)
            .map_err(|e| ServerError::InternalError(format!("Embedding error: {e}")))?;

        let index: u32 = idx
            .try_into()
            .map_err(|_| ServerError::BadRequest("Too many inputs".to_owned()))?;

        data.push(EmbeddingObject {
            object: "embedding",
            embedding,
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}
