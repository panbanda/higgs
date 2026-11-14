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

/// The text each token adds to the output, decoded in context. Decoding a token alone drops the
/// leading space a `SentencePiece` tokenizer marks and turns half of a multi-byte character into
/// U+FFFD; decoding a window ending at the token and subtracting the window before it (as the
/// streaming detokenizer does) gives the text the client actually receives. A token whose text
/// is incomplete adds nothing; the token that completes it adds the whole character.
fn token_deltas(token_ids: &[u32], tokenizer: &Tokenizer) -> Vec<String> {
    let decode = |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default();
    let mut deltas = vec![String::new(); token_ids.len()];
    let (mut prefix_offset, mut read_offset) = (0, 0);
    for (index, delta) in deltas.iter_mut().enumerate() {
        let before = decode(
            token_ids
                .get(prefix_offset..read_offset)
                .unwrap_or_default(),
        );
        let window = decode(token_ids.get(prefix_offset..=index).unwrap_or_default());
        if window.len() > before.len() && !window.ends_with('\u{FFFD}') {
            if let Some(added) = window.get(before.len()..) {
                added.clone_into(delta);
            }
            prefix_offset = read_offset;
            read_offset = index + 1;
        }
    }
    deltas
}

pub(super) fn decode_logprobs(infos: &[TokenLogprobInfo], tokenizer: &Tokenizer) -> ChoiceLogprobs {
    let decode = |token_id: u32| tokenizer.decode(&[token_id], false).unwrap_or_default();
    let token_ids: Vec<u32> = infos.iter().map(|info| info.token_id).collect();
    let deltas = token_deltas(&token_ids, tokenizer);
    let content = infos
        .iter()
        .zip(deltas)
        .map(|(info, delta)| {
            let token = decode(info.token_id);
            TokenLogprob {
                bytes: delta.into_bytes(),
                token,
                logprob: info.logprob,
                top_logprobs: info
                    .top_logprobs
                    .iter()
                    .map(|entry| {
                        let alternative = decode(entry.token_id);
                        TopLogprob {
                            bytes: alternative.clone().into_bytes(),
                            token: alternative,
                            logprob: entry.logprob,
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    ChoiceLogprobs { content }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod logprob_bytes_tests {
    use super::*;

    fn word_tokenizer(decoder: &str, vocab: &str) -> Tokenizer {
        let json = format!(
            r#"{{
                "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
                "normalizer": null, "pre_tokenizer": null, "post_processor": null,
                "decoder": {decoder},
                "model": {{ "type": "WordLevel", "vocab": {vocab}, "unk_token": "a" }}
            }}"#
        );
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    fn joined(deltas: &[String]) -> String {
        deltas.concat()
    }

    #[test]
    fn a_character_split_across_tokens_is_credited_to_the_token_that_completes_it() {
        let tokenizer = word_tokenizer(
            r#"{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#,
            r#"{"Hello": 0, "Ġworld": 1, "ðŁĺ": 3, "Ģ": 4}"#,
        );
        let deltas = token_deltas(&[0, 1, 3, 4], &tokenizer);
        assert_eq!(deltas, ["Hello", " world", "", "\u{1F600}"]);
    }

    #[test]
    fn a_sentencepiece_space_survives_in_context() {
        let tokenizer = word_tokenizer(
            r#"{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true}"#,
            r#"{"\u2581Hello": 0, "\u2581world": 1}"#,
        );
        let alone = tokenizer.decode(&[1], false).unwrap();
        assert_eq!(alone, "world", "a lone token loses its space");
        let deltas = token_deltas(&[0, 1], &tokenizer);
        assert_eq!(deltas, ["Hello", " world"]);
        assert_eq!(joined(&deltas), tokenizer.decode(&[0, 1], false).unwrap());
    }
}
