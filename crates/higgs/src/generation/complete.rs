use std::time::Instant;

use higgs_engine::engine::{GenerationOutput, LoggedToken};
use higgs_engine::reasoning_parser::parse_reasoning;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::ToolCallParser;

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
        .map(|tokens| decode_logprobs(tokens, tokenizer));
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

pub(super) fn decode_logprobs(tokens: &[LoggedToken], tokenizer: &Tokenizer) -> ChoiceLogprobs {
    let decode = |token_id: u32| tokenizer.decode(&[token_id], false).unwrap_or_default();
    let content = tokens
        .iter()
        .map(|logged| TokenLogprob {
            token: decode(logged.info.token_id),
            logprob: logged.info.logprob,
            bytes: logged.text.clone().into_bytes(),
            top_logprobs: logged
                .info
                .top_logprobs
                .iter()
                .map(|entry| TopLogprob {
                    token: decode(entry.token_id),
                    logprob: entry.logprob,
                    bytes: None,
                })
                .collect(),
        })
        .collect();
    ChoiceLogprobs { content }
}

#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod logprob_bytes_tests {
    use higgs_models::{TokenLogprobInfo, TopLogprobEntry};

    use super::*;

    fn tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": {"a": 0, "b": 1}, "unk_token": "a" }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    fn logged(token_id: u32, text: &str) -> LoggedToken {
        LoggedToken {
            info: TokenLogprobInfo {
                token_id,
                logprob: -0.1,
                top_logprobs: vec![TopLogprobEntry {
                    token_id,
                    logprob: -0.1,
                }],
            },
            text: text.to_owned(),
        }
    }

    #[test]
    fn bytes_are_the_text_each_token_added_and_alternatives_have_none() {
        let tokens = [logged(0, "Hello"), logged(1, ""), logged(0, " \u{1F600}")];
        let logprobs = decode_logprobs(&tokens, &tokenizer());
        let joined: Vec<u8> = logprobs
            .content
            .iter()
            .flat_map(|entry| entry.bytes.iter().copied())
            .collect();
        assert_eq!(String::from_utf8(joined).unwrap(), "Hello \u{1F600}");
        assert!(logprobs.content[1].bytes.is_empty());
        assert!(
            logprobs
                .content
                .iter()
                .flat_map(|entry| &entry.top_logprobs)
                .all(|alternative| alternative.bytes.is_none())
        );
        let json = serde_json::to_value(&logprobs).unwrap();
        assert!(json["content"][0]["top_logprobs"][0]["bytes"].is_null());
    }
}
