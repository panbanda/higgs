use std::time::Instant;

use higgs_engine::engine::GenerationOutput;
use higgs_engine::reasoning_parser::parse_reasoning;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::ToolCallParser;
use higgs_models::TokenLogprobInfo;

use super::event::{Completed, GeneratedToolCall};
use super::prepare::{Prepared, ReasoningHandling, prepare};
use crate::backend::EngineLease;
use crate::error::ServerError;
use crate::types::openai::{ChoiceLogprobs, TokenLogprob, TopLogprob};
use crate::usage::{TokenCounts, build_usage, count_text_tokens};

use super::GenerationRequest;

/// One non-streamed generation, without any tool loop.
pub(super) async fn complete_once(
    engine: EngineLease,
    request: &GenerationRequest,
    received_at: Instant,
) -> Result<Completed, ServerError> {
    let Prepared {
        job,
        reasoning,
        tool_parser,
        ..
    } = prepare(&**engine, request)?;
    let tokenizer = engine.tokenizer().clone();
    let output = tokio::task::spawn_blocking(move || engine.generate(job))
        .await
        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
        .map_err(ServerError::Engine)?;
    Ok(assemble(
        output,
        reasoning,
        tool_parser.as_ref(),
        &tokenizer,
        received_at,
    ))
}

fn assemble(
    output: GenerationOutput,
    reasoning: ReasoningHandling,
    tool_parser: Option<&ToolCallParser>,
    tokenizer: &Tokenizer,
    received_at: Instant,
) -> Completed {
    let logprobs = output
        .token_logprobs
        .as_deref()
        .map(|infos| decode_logprobs(infos, tokenizer));
    let (visible, reasoning_text) = split_reasoning(output.text, reasoning);

    let with_calls = tool_parser
        .map(|parser| parser.parse(&visible))
        .filter(|parsed| !parsed.tool_calls.is_empty());
    let (answer, tool_calls, finish_reason) = match with_calls {
        Some(parsed) => {
            let calls = parsed
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    GeneratedToolCall::new(
                        u32::try_from(index).unwrap_or(u32::MAX),
                        call.name,
                        call.arguments,
                    )
                })
                .collect();
            (parsed.text, calls, "tool_calls".to_owned())
        }
        None => (visible, Vec::new(), output.finish_reason),
    };

    let reasoning_tokens = reasoning_text
        .as_deref()
        .and_then(|block| count_text_tokens(tokenizer, block));
    Completed {
        text: answer,
        reasoning: reasoning_text,
        tool_calls,
        finish_reason,
        usage: build_usage(
            TokenCounts {
                prompt: output.prompt_tokens,
                completion: output.completion_tokens,
                reasoning: reasoning_tokens,
            },
            &output.stats,
            received_at,
        ),
        logprobs,
        tool_loop: None,
        loop_messages: Vec::new(),
    }
}

/// Separate `<think>` blocks from the answer. A prompt that opened a block
/// makes the output start inside it, so the opener is put back before parsing.
fn split_reasoning(text: String, handling: ReasoningHandling) -> (String, Option<String>) {
    let ReasoningHandling::Parsed {
        thinking,
        parse_without_thinking,
    } = handling
    else {
        return (text, None);
    };
    if !thinking && !parse_without_thinking {
        return (text, None);
    }
    let parse_input = if !thinking {
        text.clone()
    } else if text.contains("</think>") {
        format!("<think>{text}")
    } else {
        // Length-stopped mid-thinking: close the block so the parser extracts
        // the reasoning instead of leaking a raw `<think>`.
        format!("<think>{text}</think>")
    };
    let parsed = parse_reasoning(&parse_input);
    if parsed.reasoning.is_some() {
        (parsed.text, parsed.reasoning)
    } else {
        (text, None)
    }
}

pub(super) fn decode_logprobs(infos: &[TokenLogprobInfo], tokenizer: &Tokenizer) -> ChoiceLogprobs {
    let decode = |token_id: u32| tokenizer.decode(&[token_id], false).unwrap_or_default();
    let content = infos
        .iter()
        .map(|info| TokenLogprob {
            token: decode(info.token_id),
            logprob: info.logprob,
            top_logprobs: info
                .top_logprobs
                .iter()
                .map(|entry| TopLogprob {
                    token: decode(entry.token_id),
                    logprob: entry.logprob,
                })
                .collect(),
        })
        .collect();
    ChoiceLogprobs { content }
}
