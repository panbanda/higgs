use higgs_engine::constrained::{ConstrainedGenerator, build_vocabulary};
use higgs_engine::engine::ThinkingMode;
use higgs_engine::tool_format::ToolCallParser;

use super::request::{ChatPrompt, GenerationRequest, Prompt};
use crate::backend::{Engine, EngineJob};
use crate::constraint::ConstraintSpec;
use crate::error::ServerError;
use crate::tool_policy::combine_constraints;

/// How the model's output is searched for reasoning.
#[derive(Debug, Clone, Copy)]
pub(super) enum ReasoningHandling {
    /// The output is the text, whatever it contains.
    Raw,
    Parsed {
        /// The prompt opened a `<think>` block, so the output starts inside it.
        thinking: bool,
        /// A finished answer is parsed for `<think>` blocks even when `thinking` is off.
        parse_without_thinking: bool,
    },
}

/// A request turned into what the engine runs, plus how to read its output.
pub(super) struct Prepared {
    pub job: EngineJob,
    pub prompt_token_count: u32,
    /// Sequence positions that image features add beyond the prompt tokens: each image
    /// sentinel becomes several positions once the model embeds the image.
    pub image_extra_positions: u32,
    pub reasoning: ReasoningHandling,
    /// Present when the request offered tools.
    pub tool_parser: Option<ToolCallParser>,
}

pub(super) fn prepare(
    engine: &dyn Engine,
    request: &GenerationRequest,
) -> Result<Prepared, ServerError> {
    let mut prepared = match &request.prompt {
        Prompt::Chat(chat) => prepare_chat(engine, request, chat),
        Prompt::Text(text) => prepare_text(engine, request, text),
    }?;
    fit_context(engine.context_length(), &mut prepared)?;
    Ok(prepared)
}

/// Keep the request inside the model's context. A prompt that fills it is refused before
/// prefill, since there is no room for even one token. Otherwise generation is capped at the
/// end of the context, so a client that asks for a large `max_tokens` still gets an answer that
/// ends with finish reason `length`. Every API and every engine path goes through here.
fn fit_context(limit: Option<u32>, prepared: &mut Prepared) -> Result<(), ServerError> {
    let Some(context_length) = limit else {
        return Ok(());
    };
    let prompt_tokens = prepared
        .prompt_token_count
        .saturating_add(prepared.image_extra_positions);
    if prompt_tokens >= context_length {
        return Err(ServerError::ContextLengthExceeded {
            prompt_tokens,
            context_length,
        });
    }
    prepared.job.max_tokens = prepared.job.max_tokens.min(context_length - prompt_tokens);
    Ok(())
}

/// The positions that images add to a prompt. Each image sentinel (`IMAGE_TOKEN_INDEX`) takes one
/// token in the prompt, and `image_positions` once the model replaces it with the image features.
fn image_extra_positions(prompt_tokens: &[u32], image_positions: Option<i32>) -> u32 {
    let Some(positions) = image_positions.and_then(|count| u32::try_from(count).ok()) else {
        return 0;
    };
    let sentinel = higgs_models::llava_qwen2::IMAGE_TOKEN_INDEX.cast_unsigned();
    let images = prompt_tokens
        .iter()
        .filter(|token| **token == sentinel)
        .count();
    u32::try_from(images)
        .unwrap_or(u32::MAX)
        .saturating_mul(positions.saturating_sub(1))
}

fn prepare_chat(
    engine: &dyn Engine,
    request: &GenerationRequest,
    chat: &ChatPrompt,
) -> Result<Prepared, ServerError> {
    let tool_format = engine.tool_call_format();
    let tools = (!chat.tools.is_empty()).then_some(chat.tools.as_slice());
    let forced = chat
        .tool_policy
        .forced_constraint(&chat.tools, tool_format)?;
    let constraint_spec = combine_constraints(request.constraint.clone(), forced)?;
    // The engine applies a constraint from the first generated token, which a
    // thinking prompt has already opened a `<think>` block for: the constrained
    // output would land in the reasoning and the answer would be empty.
    let thinking_enabled = constraint_spec.is_none()
        && crate::reasoning::thinking_enabled_for_request(
            engine,
            &chat.tool_policy,
            &request.model,
            chat.reasoning.as_ref(),
        );

    let mut prompt_tokens = engine
        .prepare_chat_prompt(&chat.messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let pixel_values = if !chat.images.is_empty() && engine.is_vlm() {
        engine.replace_image_tokens(&mut prompt_tokens);
        let image_size = engine.vlm_image_size().unwrap_or(384);
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let size = image_size as u32;
        let first_image = chat
            .images
            .first()
            .ok_or_else(|| ServerError::BadRequest("Image data is empty".to_owned()))?;
        let pixels = higgs_models::siglip::preprocess_image(first_image, size)
            .map_err(|e| ServerError::InternalError(format!("Image preprocessing failed: {e}")))?;
        Some(pixels)
    } else {
        None
    };
    let constraint = build_constraint(constraint_spec.as_ref(), engine)?;

    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
    let image_extra_positions = image_extra_positions(&prompt_tokens, engine.vlm_image_positions());
    Ok(Prepared {
        image_extra_positions,
        job: EngineJob {
            prompt_tokens,
            max_tokens: request.max_tokens,
            sampling: request.sampling.clone(),
            stop_sequences: request.stop_sequences.clone(),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            thinking: ThinkingMode::new(thinking_enabled, chat.thinking_budget),
            constraint,
            pixel_values,
        },
        prompt_token_count,
        reasoning: ReasoningHandling::Parsed {
            thinking: thinking_enabled,
            parse_without_thinking: chat.parse_reasoning_without_thinking,
        },
        tool_parser: tools.map(|offered| chat.tool_policy.parser(tool_format, Some(offered))),
    })
}

fn prepare_text(
    engine: &dyn Engine,
    request: &GenerationRequest,
    text: &str,
) -> Result<Prepared, ServerError> {
    let encoding = engine
        .tokenizer()
        .encode(text, false)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
    let constraint = build_constraint(request.constraint.as_ref(), engine)?;
    Ok(Prepared {
        image_extra_positions: 0,
        job: EngineJob {
            prompt_tokens,
            max_tokens: request.max_tokens,
            sampling: request.sampling.clone(),
            stop_sequences: request.stop_sequences.clone(),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            thinking: ThinkingMode::new(engine.enable_thinking(), None),
            constraint,
            pixel_values: None,
        },
        prompt_token_count,
        reasoning: ReasoningHandling::Raw,
        tool_parser: None,
    })
}

/// Compile the requested constraint against the model's vocabulary.
///
/// A pattern or schema the grammar compiler rejects is the client's fault, so
/// those failures are reported as `400`; failing to build the vocabulary is a
/// server error.
fn build_constraint(
    requested: Option<&ConstraintSpec>,
    engine: &dyn Engine,
) -> Result<Option<ConstrainedGenerator>, ServerError> {
    let Some(spec) = requested else {
        return Ok(None);
    };
    let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
    let vocab = build_vocabulary(engine.tokenizer(), eos_id).map_err(ServerError::Engine)?;
    let constraint = match spec {
        ConstraintSpec::JsonObject => ConstrainedGenerator::for_json_object(&vocab),
        ConstraintSpec::JsonSchema(schema) => {
            ConstrainedGenerator::from_json_schema(&schema.to_string(), &vocab)
        }
        ConstraintSpec::Regex(pattern) => ConstrainedGenerator::from_regex(pattern, &vocab),
    }
    .map_err(|error| ServerError::BadRequest(error.to_string()))?;
    Ok(Some(constraint))
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(prompt_tokens: Vec<u32>, image_extra_positions: u32, max_tokens: u32) -> Prepared {
        Prepared {
            prompt_token_count: u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX),
            image_extra_positions,
            job: EngineJob {
                prompt_tokens,
                max_tokens,
                sampling: higgs_models::SamplingParams::default(),
                stop_sequences: Vec::new(),
                logprobs: false,
                top_logprobs: None,
                thinking: ThinkingMode::Off,
                constraint: None,
                pixel_values: None,
            },
            reasoning: ReasoningHandling::Raw,
            tool_parser: None,
        }
    }

    #[test]
    fn each_image_sentinel_adds_the_positions_of_its_features() {
        let sentinel = higgs_models::llava_qwen2::IMAGE_TOKEN_INDEX.cast_unsigned();
        assert_eq!(image_extra_positions(&[1, sentinel, 2], Some(729)), 728);
        assert_eq!(image_extra_positions(&[1, 2, 3], Some(729)), 0);
        assert_eq!(image_extra_positions(&[1, sentinel], None), 0);
        assert_eq!(image_extra_positions(&[sentinel], Some(1)), 0);
    }

    #[test]
    fn the_prompt_length_counts_the_expanded_image_positions() {
        let mut fits = prepared(vec![1; 100], 0, 50);
        assert!(fit_context(Some(1000), &mut fits).is_ok());

        let mut expanded = prepared(vec![1; 100], 728, 50);
        let error = fit_context(Some(800), &mut expanded).unwrap_err();
        assert!(matches!(
            error,
            ServerError::ContextLengthExceeded {
                prompt_tokens: 828,
                context_length: 800
            }
        ));

        let mut capped = prepared(vec![1; 100], 728, 5000);
        fit_context(Some(1000), &mut capped).unwrap();
        assert_eq!(capped.job.max_tokens, 172);
    }
}
