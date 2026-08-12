//! Pre-serialized SSE chunk builder for `/v1/chat/completions` and
//! `/v1/completions` streaming.
//!
//! For a streaming response, fields like `id`, `object`, `created`, `model`,
//! and the per-choice `index` are constant for the entire stream. Re-running
//! `serde_json::to_string` on the full chunk struct for every token re-walks
//! and re-encodes those constants. This module pre-serializes the constant
//! prefix once and only encodes the variable fields per chunk.
//!
//! Output bytes must match `serde_json::to_string(&ChatCompletionChunk)` and
//! `serde_json::to_string(&CompletionChunk)` byte-for-byte. The unit tests
//! below assert this for a representative set of inputs; do not change the
//! field order or whitespace here without updating both the chunk structs and
//! those tests.

#![allow(clippy::redundant_pub_crate)]

use crate::types::anthropic::TextDelta;
use crate::types::openai::{
    ChatCompletionDelta, ChoiceLogprobs, CompletionChunkChoice, CompletionUsage,
};

/// Pre-serialized prefix + reusable buffer for chat-completion SSE chunks.
///
/// `pub` so the criterion bench in `benches/sse_serialize.rs` can call the
/// production code path directly (kept `#[doc(hidden)]` at the module level
/// to avoid expanding the public API surface).
pub struct ChatChunkWriter {
    /// `{"id":"...","object":"chat.completion.chunk","created":N,"model":"..."`
    /// (no trailing comma; the variable part picks up the comma).
    head: String,
    /// `head` + `,"choices":[{"index":0,"delta":` — used for chunks that
    /// carry a choice.
    full_prefix: String,
    /// Reusable per-chunk scratch buffer.
    buf: String,
}

impl ChatChunkWriter {
    pub fn new(id: &str, created: i64, model: &str) -> Self {
        let mut head = String::with_capacity(64 + id.len() + model.len());
        head.push_str(r#"{"id":"#);
        push_json_string(&mut head, id);
        head.push_str(r#","object":"chat.completion.chunk","created":"#);
        head.push_str(&created.to_string());
        head.push_str(r#","model":"#);
        push_json_string(&mut head, model);

        let mut full_prefix = String::with_capacity(head.len() + 32);
        full_prefix.push_str(&head);
        full_prefix.push_str(r#","choices":[{"index":0,"delta":"#);

        Self {
            head,
            full_prefix,
            buf: String::with_capacity(256),
        }
    }

    /// Build a chunk carrying a `delta` and an optional `finish_reason` /
    /// `logprobs`, with `usage = None`.
    pub fn write_delta(
        &mut self,
        delta: &ChatCompletionDelta,
        finish_reason: Option<&str>,
        logprobs: Option<&ChoiceLogprobs>,
    ) -> Result<&str, serde_json::Error> {
        self.buf.clear();
        self.buf.push_str(&self.full_prefix);
        let delta_json = serde_json::to_string(delta)?;
        self.buf.push_str(&delta_json);
        self.buf.push_str(r#","finish_reason":"#);
        match finish_reason {
            Some(reason) => push_json_string(&mut self.buf, reason),
            None => self.buf.push_str("null"),
        }
        if let Some(lp) = logprobs {
            self.buf.push_str(r#","logprobs":"#);
            let lp_json = serde_json::to_string(lp)?;
            self.buf.push_str(&lp_json);
        }
        self.buf.push_str("}]}");
        Ok(&self.buf)
    }

    /// Build a terminal chunk with `choices: []` and a `usage` block.
    pub(crate) fn write_usage(
        &mut self,
        usage: &CompletionUsage,
    ) -> Result<&str, serde_json::Error> {
        self.buf.clear();
        self.buf.push_str(&self.head);
        self.buf.push_str(r#","choices":[],"usage":"#);
        let usage_json = serde_json::to_string(usage)?;
        self.buf.push_str(&usage_json);
        self.buf.push('}');
        Ok(&self.buf)
    }

    /// Build a prefill-progress chunk with `choices: []` and a
    /// `prompt_progress` block (llama.cpp-compatible shape). Emitted only
    /// when the request set `return_progress: true`.
    pub(crate) fn write_prompt_progress(
        &mut self,
        total: u32,
        cache: u32,
        processed: u32,
        time_ms: u64,
    ) -> &str {
        use std::fmt::Write as _;
        self.buf.clear();
        self.buf.push_str(&self.head);
        let _ = write!(
            self.buf,
            r#","choices":[],"prompt_progress":{{"total":{total},"cache":{cache},"processed":{processed},"time_ms":{time_ms}}}}}"#
        );
        &self.buf
    }
}

/// Pre-serialized prefix + reusable buffer for `/v1/completions` SSE chunks.
pub(crate) struct CompletionChunkWriter {
    prefix: String,
    buf: String,
}

impl CompletionChunkWriter {
    pub(crate) fn new(id: &str, created: i64, model: &str) -> Self {
        let mut prefix = String::with_capacity(64 + id.len() + model.len());
        prefix.push_str(r#"{"id":"#);
        push_json_string(&mut prefix, id);
        prefix.push_str(r#","object":"text_completion","created":"#);
        prefix.push_str(&created.to_string());
        prefix.push_str(r#","model":"#);
        push_json_string(&mut prefix, model);
        prefix.push_str(r#","choices":["#);
        Self {
            prefix,
            buf: String::with_capacity(192),
        }
    }

    pub(crate) fn write(
        &mut self,
        choice: &CompletionChunkChoice,
    ) -> Result<&str, serde_json::Error> {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        let choice_json = serde_json::to_string(choice)?;
        self.buf.push_str(&choice_json);
        self.buf.push_str("]}");
        Ok(&self.buf)
    }
}

/// Pre-serialized prefix for Anthropic `content_block_delta` events.
pub(crate) struct AnthropicDeltaWriter {
    prefix: &'static str,
    buf: String,
}

impl AnthropicDeltaWriter {
    pub(crate) fn new() -> Self {
        Self {
            prefix: r#"{"type":"content_block_delta","index":0,"delta":"#,
            buf: String::with_capacity(192),
        }
    }

    pub(crate) fn write(&mut self, delta: &TextDelta) -> Result<&str, serde_json::Error> {
        self.buf.clear();
        self.buf.push_str(self.prefix);
        let dj = serde_json::to_string(delta)?;
        self.buf.push_str(&dj);
        self.buf.push('}');
        Ok(&self.buf)
    }
}

/// Append `s` as a JSON string literal (with surrounding quotes) using
/// `serde_json` so escaping matches exactly.
fn push_json_string(out: &mut String, s: &str) {
    // `serde_json::to_string` on a `&str` produces a JSON-encoded quoted
    // string and cannot fail.
    if let Ok(encoded) = serde_json::to_string(s) {
        out.push_str(&encoded);
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{
        ChatCompletionChunk, ChatCompletionChunkChoice, CompletionChunk, TokenLogprob, TopLogprob,
    };

    fn delta(content: &str) -> ChatCompletionDelta {
        ChatCompletionDelta {
            role: None,
            content: Some(content.to_owned()),
            reasoning_content: None,
            tool_calls: None,
        }
    }

    #[test]
    fn chat_chunk_matches_serde_for_content_delta() {
        let id = "chatcmpl-abc";
        let created = 1_700_000_000_i64;
        let model = "qwen3-1.7B";
        let mut w = ChatChunkWriter::new(id, created, model);
        for content in ["hello", " world", "\"quoted\"", "tab\there", ""] {
            let d = delta(content);
            let got = w.write_delta(&d, None, None).unwrap().to_owned();
            let expected = serde_json::to_string(&ChatCompletionChunk {
                id: id.to_owned(),
                object: "chat.completion.chunk",
                created,
                model: model.to_owned(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: d,
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
            })
            .unwrap();
            assert_eq!(got, expected, "content={content:?}");
        }
    }

    #[test]
    fn chat_chunk_matches_serde_with_finish_reason() {
        let id = "chatcmpl-1";
        let model = "m";
        let mut w = ChatChunkWriter::new(id, 1, model);
        let d = ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        let got = w.write_delta(&d, Some("stop"), None).unwrap().to_owned();
        let expected = serde_json::to_string(&ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 1,
            model: model.to_owned(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: d,
                finish_reason: Some("stop".to_owned()),
                logprobs: None,
            }],
            usage: None,
        })
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn chat_chunk_matches_serde_with_role() {
        let id = "chatcmpl-role";
        let model = "qwen3";
        let mut w = ChatChunkWriter::new(id, 42, model);
        let d = ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        let got = w.write_delta(&d, None, None).unwrap().to_owned();
        let expected = serde_json::to_string(&ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 42,
            model: model.to_owned(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: d,
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        })
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn chat_chunk_matches_serde_with_logprobs() {
        let id = "chatcmpl-lp";
        let model = "m";
        let mut w = ChatChunkWriter::new(id, 100, model);
        let d = ChatCompletionDelta {
            role: None,
            content: Some("x".to_owned()),
            reasoning_content: None,
            tool_calls: None,
        };
        let lp = ChoiceLogprobs {
            content: vec![TokenLogprob {
                token: "x".to_owned(),
                logprob: -0.5,
                top_logprobs: vec![TopLogprob {
                    token: "x".to_owned(),
                    logprob: -0.5,
                }],
            }],
        };
        let got = w.write_delta(&d, None, Some(&lp)).unwrap().to_owned();
        let expected = serde_json::to_string(&ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 100,
            model: model.to_owned(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: d,
                finish_reason: None,
                logprobs: Some(lp),
            }],
            usage: None,
        })
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn chat_usage_chunk_matches_serde() {
        let id = "chatcmpl-x";
        let model = "m";
        let mut w = ChatChunkWriter::new(id, 7, model);
        let usage = CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 5,
            total_tokens: 8,
        };
        let got = w.write_usage(&usage).unwrap().to_owned();
        let expected = serde_json::to_string(&ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 7,
            model: model.to_owned(),
            choices: vec![],
            usage: Some(usage),
        })
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn chat_prompt_progress_chunk_is_valid_json_with_expected_shape() {
        let id = "chatcmpl-p";
        let model = "m";
        let mut w = ChatChunkWriter::new(id, 11, model);
        let got = w.write_prompt_progress(100, 30, 64, 250).to_owned();

        // The buffer is concatenated by hand, so the failure mode is malformed
        // JSON — assert it parses, then pin the exact llama.cpp-compatible wire
        // shape (`choices: []` + a `prompt_progress` block).
        serde_json::from_str::<serde_json::Value>(&got).unwrap();
        let expected = format!(
            r#"{{"id":"{id}","object":"chat.completion.chunk","created":11,"model":"{model}","choices":[],"prompt_progress":{{"total":100,"cache":30,"processed":64,"time_ms":250}}}}"#
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn completion_chunk_matches_serde() {
        let id = "cmpl-1";
        let model = "m";
        let mut w = CompletionChunkWriter::new(id, 99, model);
        let choice = CompletionChunkChoice {
            index: 0,
            text: "hi".to_owned(),
            finish_reason: Some("stop".to_owned()),
        };
        let got = w.write(&choice).unwrap().to_owned();
        let expected = serde_json::to_string(&CompletionChunk {
            id: id.to_owned(),
            object: "text_completion",
            created: 99,
            model: model.to_owned(),
            choices: vec![choice],
        })
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn anthropic_delta_matches_serde() {
        use crate::types::anthropic::ContentBlockDeltaEvent;
        let mut w = AnthropicDeltaWriter::new();
        for text in ["Hello", " world", "\"q\"", ""] {
            let d = TextDelta {
                delta_type: "text_delta",
                text: text.to_owned(),
            };
            let got = w.write(&d).unwrap().to_owned();
            let expected = serde_json::to_string(&ContentBlockDeltaEvent {
                event_type: "content_block_delta",
                index: 0,
                delta: d,
            })
            .unwrap();
            assert_eq!(got, expected, "text={text:?}");
        }
    }

    #[test]
    fn snapshot_byte_stream_matches_full_serde() {
        // Simulate the full stream a chat completion would produce: role
        // chunk, three content chunks, a finish-reason chunk, and a usage
        // chunk. Compare the full byte stream against the path that re-uses
        // `serde_json::to_string(&full_chunk)` for every chunk.
        let id = "chatcmpl-snap";
        let created = 1_700_000_001_i64;
        let model = "qwen3-1.7B-4bit";

        let role = ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        let chunks = ["Hello", ", ", "world!"];
        let usage = CompletionUsage {
            prompt_tokens: 7,
            completion_tokens: 3,
            total_tokens: 10,
        };

        // New path.
        let mut new_path = String::new();
        let mut w = ChatChunkWriter::new(id, created, model);
        new_path.push_str(w.write_delta(&role, None, None).unwrap());
        new_path.push('\n');
        for c in chunks {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(c.to_owned()),
                reasoning_content: None,
                tool_calls: None,
            };
            new_path.push_str(w.write_delta(&d, None, None).unwrap());
            new_path.push('\n');
        }
        let stop_delta = ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        new_path.push_str(w.write_delta(&stop_delta, Some("stop"), None).unwrap());
        new_path.push('\n');
        new_path.push_str(w.write_usage(&usage).unwrap());

        // Baseline path (current code).
        let mk = |delta: ChatCompletionDelta,
                  finish_reason: Option<String>,
                  chunk_usage: Option<CompletionUsage>,
                  empty_choices: bool|
         -> ChatCompletionChunk {
            ChatCompletionChunk {
                id: id.to_owned(),
                object: "chat.completion.chunk",
                created,
                model: model.to_owned(),
                choices: if empty_choices {
                    vec![]
                } else {
                    vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta,
                        finish_reason,
                        logprobs: None,
                    }]
                },
                usage: chunk_usage,
            }
        };
        let mut baseline = String::new();
        baseline.push_str(&serde_json::to_string(&mk(role, None, None, false)).unwrap());
        baseline.push('\n');
        for c in chunks {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(c.to_owned()),
                reasoning_content: None,
                tool_calls: None,
            };
            baseline.push_str(&serde_json::to_string(&mk(d, None, None, false)).unwrap());
            baseline.push('\n');
        }
        baseline.push_str(
            &serde_json::to_string(&mk(stop_delta, Some("stop".to_owned()), None, false)).unwrap(),
        );
        baseline.push('\n');
        baseline.push_str(
            &serde_json::to_string(&mk(
                ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                None,
                Some(usage),
                true,
            ))
            .unwrap(),
        );

        assert_eq!(new_path, baseline);
    }
}
