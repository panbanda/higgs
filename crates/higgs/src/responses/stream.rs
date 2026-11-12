//! Converts the chat chunk stream into the Responses event protocol.

use serde::Deserialize;

use super::translate::{Outcome, ResponseContext, generate_item_id, status_for_finish_reason};
use crate::types::responses::{
    ContentPart, InputTokensDetails, OutputItem, ResponseError, ResponseObject, ResponseStatus,
    ResponseUsage, StreamEvent, StreamEventBody,
};

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
    #[serde(default)]
    prompt_progress: Option<PromptProgress>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ToolFunctionDelta>,
}

#[derive(Deserialize)]
struct ToolFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct PromptProgress {
    cache: u32,
}

/// The item currently receiving text deltas.
enum OpenItem {
    Reasoning { id: String, text: String },
    Message { id: String, text: String },
}

/// Result of finishing a stream: the closing events and the final object.
pub struct StreamFinish {
    pub events: Vec<StreamEvent>,
    pub response: ResponseObject,
}

/// Stateful translator from chat chunks to ordered Responses events.
///
/// Text and reasoning stream as deltas into an open item; a tool call
/// arrives complete from the chat path, so it is emitted as a whole item.
/// Switching kinds closes the open item first, so `output_index` always
/// equals the number of items closed before it.
pub struct ResponseStreamBuilder {
    context: ResponseContext,
    sequence_number: u64,
    items: Vec<OutputItem>,
    open: Option<OpenItem>,
    finish_reason: Option<String>,
    usage: Option<(u32, u32)>,
    cached_tokens: Option<u32>,
}

impl ResponseStreamBuilder {
    pub const fn new(context: ResponseContext) -> Self {
        Self {
            context,
            sequence_number: 0,
            items: Vec::new(),
            open: None,
            finish_reason: None,
            usage: None,
            cached_tokens: None,
        }
    }

    fn push(&mut self, events: &mut Vec<StreamEvent>, body: StreamEventBody) {
        events.push(StreamEvent {
            sequence_number: self.sequence_number,
            body,
        });
        self.sequence_number += 1;
    }

    fn output_index(&self) -> u32 {
        u32::try_from(self.items.len()).unwrap_or(u32::MAX)
    }

    pub fn start(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::with_capacity(2);
        let response = Box::new(self.context.object(Outcome::in_progress()));
        self.push(
            &mut events,
            StreamEventBody::Created {
                response: response.clone(),
            },
        );
        self.push(&mut events, StreamEventBody::InProgress { response });
        events
    }

    /// Feed one chat SSE `data` payload. Payloads that are not chunk JSON
    /// (the `[DONE]` sentinel) produce no events.
    pub fn ingest(&mut self, data: &str) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let Ok(chunk) = serde_json::from_str::<Chunk>(data) else {
            return events;
        };
        if let Some(progress) = chunk.prompt_progress {
            self.cached_tokens = Some(progress.cache);
        }
        if let Some(usage) = chunk.usage {
            self.usage = Some((usage.prompt_tokens, usage.completion_tokens));
        }
        for choice in chunk.choices {
            if let Some(text) = choice
                .delta
                .reasoning_content
                .filter(|text| !text.is_empty())
            {
                self.reasoning_delta(&mut events, text);
            }
            for call in choice.delta.tool_calls.unwrap_or_default() {
                self.tool_call(&mut events, call);
            }
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                self.message_delta(&mut events, text);
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
        }
        events
    }

    fn reasoning_delta(&mut self, events: &mut Vec<StreamEvent>, delta: String) {
        if !matches!(self.open, Some(OpenItem::Reasoning { .. })) {
            self.close_open(events, ResponseStatus::Completed);
            let id = generate_item_id("rs");
            let output_index = self.output_index();
            self.push(
                events,
                StreamEventBody::OutputItemAdded {
                    output_index,
                    item: OutputItem::Reasoning {
                        id: id.clone(),
                        summary: Vec::new(),
                        content: Vec::new(),
                        status: ResponseStatus::InProgress,
                    },
                },
            );
            self.push(
                events,
                StreamEventBody::ContentPartAdded {
                    item_id: id.clone(),
                    output_index,
                    content_index: 0,
                    part: ContentPart::ReasoningText {
                        text: String::new(),
                    },
                },
            );
            self.open = Some(OpenItem::Reasoning {
                id,
                text: String::new(),
            });
        }
        let output_index = self.output_index();
        if let Some(OpenItem::Reasoning { id, text }) = &mut self.open {
            text.push_str(&delta);
            let item_id = id.clone();
            self.push(
                events,
                StreamEventBody::ReasoningTextDelta {
                    item_id,
                    output_index,
                    content_index: 0,
                    delta,
                },
            );
        }
    }

    fn message_delta(&mut self, events: &mut Vec<StreamEvent>, delta: String) {
        if !matches!(self.open, Some(OpenItem::Message { .. })) {
            self.close_open(events, ResponseStatus::Completed);
            self.open_message(events);
        }
        let output_index = self.output_index();
        if let Some(OpenItem::Message { id, text }) = &mut self.open {
            text.push_str(&delta);
            let item_id = id.clone();
            self.push(
                events,
                StreamEventBody::OutputTextDelta {
                    item_id,
                    output_index,
                    content_index: 0,
                    delta,
                    logprobs: Vec::new(),
                },
            );
        }
    }

    fn open_message(&mut self, events: &mut Vec<StreamEvent>) {
        let id = generate_item_id("msg");
        let output_index = self.output_index();
        self.push(
            events,
            StreamEventBody::OutputItemAdded {
                output_index,
                item: OutputItem::Message {
                    id: id.clone(),
                    role: "assistant",
                    content: Vec::new(),
                    status: ResponseStatus::InProgress,
                },
            },
        );
        self.push(
            events,
            StreamEventBody::ContentPartAdded {
                item_id: id.clone(),
                output_index,
                content_index: 0,
                part: ContentPart::output_text(String::new()),
            },
        );
        self.open = Some(OpenItem::Message {
            id,
            text: String::new(),
        });
    }

    fn tool_call(&mut self, events: &mut Vec<StreamEvent>, call: ToolCallDelta) {
        let Some(function) = call.function else {
            return;
        };
        self.close_open(events, ResponseStatus::Completed);
        let id = generate_item_id("fc");
        let call_id = call.id.unwrap_or_else(|| generate_item_id("call"));
        let name = function.name.unwrap_or_default();
        let arguments = function.arguments.unwrap_or_default();
        let output_index = self.output_index();

        self.push(
            events,
            StreamEventBody::OutputItemAdded {
                output_index,
                item: OutputItem::FunctionCall {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                    status: ResponseStatus::InProgress,
                },
            },
        );
        self.push(
            events,
            StreamEventBody::FunctionCallArgumentsDelta {
                item_id: id.clone(),
                output_index,
                delta: arguments.clone(),
            },
        );
        self.push(
            events,
            StreamEventBody::FunctionCallArgumentsDone {
                item_id: id.clone(),
                output_index,
                name: name.clone(),
                arguments: arguments.clone(),
            },
        );
        let item = OutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            status: ResponseStatus::Completed,
        };
        self.push(
            events,
            StreamEventBody::OutputItemDone {
                output_index,
                item: item.clone(),
            },
        );
        self.items.push(item);
    }

    /// Close the open reasoning or message item, if any, with `status`.
    fn close_open(&mut self, events: &mut Vec<StreamEvent>, status: ResponseStatus) {
        let output_index = self.output_index();
        let item = match self.open.take() {
            None => return,
            Some(OpenItem::Reasoning { id, text }) => {
                self.push(
                    events,
                    StreamEventBody::ReasoningTextDone {
                        item_id: id.clone(),
                        output_index,
                        content_index: 0,
                        text: text.clone(),
                    },
                );
                let part = ContentPart::ReasoningText { text };
                self.push(
                    events,
                    StreamEventBody::ContentPartDone {
                        item_id: id.clone(),
                        output_index,
                        content_index: 0,
                        part: part.clone(),
                    },
                );
                OutputItem::Reasoning {
                    id,
                    summary: Vec::new(),
                    content: vec![part],
                    status,
                }
            }
            Some(OpenItem::Message { id, text }) => {
                self.push(
                    events,
                    StreamEventBody::OutputTextDone {
                        item_id: id.clone(),
                        output_index,
                        content_index: 0,
                        text: text.clone(),
                        logprobs: Vec::new(),
                    },
                );
                let part = ContentPart::output_text(text);
                self.push(
                    events,
                    StreamEventBody::ContentPartDone {
                        item_id: id.clone(),
                        output_index,
                        content_index: 0,
                        part: part.clone(),
                    },
                );
                OutputItem::Message {
                    id,
                    role: "assistant",
                    content: vec![part],
                    status,
                }
            }
        };
        self.push(
            events,
            StreamEventBody::OutputItemDone {
                output_index,
                item: item.clone(),
            },
        );
        self.items.push(item);
    }

    /// End the stream. A missing finish reason means generation died
    /// mid-stream, which is reported as `response.failed`.
    pub fn finish(mut self) -> StreamFinish {
        let mut events = Vec::new();
        let (status, incomplete_reason) = self
            .finish_reason
            .as_deref()
            .map_or((ResponseStatus::Failed, None), status_for_finish_reason);
        let item_status = if status == ResponseStatus::Incomplete {
            ResponseStatus::Incomplete
        } else {
            ResponseStatus::Completed
        };

        self.close_open(&mut events, item_status);
        if self.items.is_empty() && status != ResponseStatus::Failed {
            self.open_message(&mut events);
            self.close_open(&mut events, item_status);
        }

        let usage = self.usage.map(|(prompt, completion)| ResponseUsage {
            input_tokens: prompt,
            output_tokens: completion,
            total_tokens: prompt + completion,
            input_tokens_details: self
                .cached_tokens
                .map(|cached_tokens| InputTokensDetails { cached_tokens }),
            output_tokens_details: None,
        });
        let error = (status == ResponseStatus::Failed).then(|| ResponseError {
            code: "server_error".to_owned(),
            message: "generation ended before the response completed".to_owned(),
        });
        let response = self.context.object(Outcome {
            status,
            output: std::mem::take(&mut self.items),
            usage,
            incomplete_reason,
            error,
        });

        let boxed = Box::new(response.clone());
        let body = match status {
            ResponseStatus::Completed | ResponseStatus::InProgress => {
                StreamEventBody::Completed { response: boxed }
            }
            ResponseStatus::Incomplete => StreamEventBody::Incomplete { response: boxed },
            ResponseStatus::Failed => StreamEventBody::Failed { response: boxed },
        };
        self.push(&mut events, body);
        StreamFinish { events, response }
    }
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn context() -> ResponseContext {
        ResponseContext::from_request(
            &serde_json::from_value(json!({"model": "m", "input": "hi"})).unwrap(),
            "m".to_owned(),
        )
    }

    fn delta_chunk(delta: &serde_json::Value, finish: Option<&str>) -> String {
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        })
        .to_string()
    }

    fn usage_chunk(prompt: u32, completion: u32) -> String {
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        })
        .to_string()
    }

    fn progress_chunk(cache: u32) -> String {
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [],
            "prompt_progress": {"total": 20, "cache": cache, "processed": 20, "time_ms": 3}
        })
        .to_string()
    }

    fn run(chunks: &[String]) -> (Vec<StreamEvent>, ResponseObject) {
        let mut builder = ResponseStreamBuilder::new(context());
        let mut events = builder.start();
        for chunk in chunks {
            events.extend(builder.ingest(chunk));
        }
        let finish = builder.finish();
        events.extend(finish.events);
        (events, finish.response)
    }

    fn names(events: &[StreamEvent]) -> Vec<&'static str> {
        events.iter().map(|e| e.body.event_name()).collect()
    }

    fn assert_monotonic(events: &[StreamEvent]) {
        for (index, event) in events.iter().enumerate() {
            assert_eq!(
                event.sequence_number,
                u64::try_from(index).unwrap(),
                "sequence_number must count 0,1,2,... without gaps"
            );
        }
    }

    #[test]
    fn text_only_stream_has_the_exact_event_order() {
        let (events, response) = run(&[
            progress_chunk(4),
            delta_chunk(&json!({"role": "assistant"}), None),
            delta_chunk(&json!({"content": "Hel"}), None),
            delta_chunk(&json!({"content": "lo"}), None),
            delta_chunk(&json!({}), Some("stop")),
            usage_chunk(12, 2),
            "[DONE]".to_owned(),
        ]);

        assert_eq!(
            names(&events),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_monotonic(&events);

        match &events[9].body {
            StreamEventBody::Completed { response: carried } => {
                assert_eq!(**carried, response);
            }
            other => panic!("unexpected final event {other:?}"),
        }
        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(response.output_text, "Hello");
        let usage = response.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens, usage.total_tokens),
            (12, 2, 14)
        );
        assert_eq!(usage.input_tokens_details.unwrap().cached_tokens, 4);
        assert!(usage.output_tokens_details.is_none());
    }

    #[test]
    fn text_deltas_carry_item_ids_and_the_done_event_the_full_text() {
        let (events, _) = run(&[
            delta_chunk(&json!({"content": "Hi"}), None),
            delta_chunk(&json!({}), Some("stop")),
        ]);
        let item_id = match &events[2].body {
            StreamEventBody::OutputItemAdded { item, output_index } => {
                assert_eq!(*output_index, 0);
                item.id().to_owned()
            }
            other => panic!("unexpected {other:?}"),
        };
        match &events[4].body {
            StreamEventBody::OutputTextDelta {
                item_id: delta_item,
                delta,
                ..
            } => {
                assert_eq!(delta_item, &item_id);
                assert_eq!(delta, "Hi");
            }
            other => panic!("unexpected {other:?}"),
        }
        match &events[5].body {
            StreamEventBody::OutputTextDone { text, .. } => assert_eq!(text, "Hi"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn reasoning_then_tool_call_stream_has_the_exact_event_order() {
        let (events, response) = run(&[
            delta_chunk(&json!({"role": "assistant"}), None),
            delta_chunk(&json!({"reasoning_content": "need "}), None),
            delta_chunk(&json!({"reasoning_content": "weather"}), None),
            delta_chunk(
                &json!({"tool_calls": [{
                    "index": 0, "id": "call_0_abc", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]}),
                None,
            ),
            delta_chunk(&json!({}), Some("tool_calls")),
            usage_chunk(30, 9),
        ]);

        assert_eq!(
            names(&events),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_monotonic(&events);

        match &events[9].body {
            StreamEventBody::OutputItemAdded { output_index, item } => {
                assert_eq!(*output_index, 1);
                assert!(
                    matches!(item, OutputItem::FunctionCall { arguments, .. } if arguments.is_empty())
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        match &events[11].body {
            StreamEventBody::FunctionCallArgumentsDone {
                name, arguments, ..
            } => {
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, "{\"city\":\"Paris\"}");
            }
            other => panic!("unexpected {other:?}"),
        }

        assert_eq!(response.output.len(), 2);
        match &response.output[1] {
            OutputItem::FunctionCall {
                call_id, status, ..
            } => {
                assert_eq!(call_id, "call_0_abc");
                assert_eq!(*status, ResponseStatus::Completed);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(response.output_text, "");
    }

    #[test]
    fn text_then_tool_call_closes_the_message_before_the_call() {
        let (events, response) = run(&[
            delta_chunk(&json!({"content": "Checking."}), None),
            delta_chunk(
                &json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": "{}"}}]}),
                None,
            ),
            delta_chunk(&json!({}), Some("tool_calls")),
        ]);
        assert_eq!(
            names(&events)[2..],
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_monotonic(&events);
        assert_eq!(response.output.len(), 2);
    }

    #[test]
    fn two_tool_calls_get_consecutive_output_indices() {
        let call = |id: &str| {
            delta_chunk(
                &json!({"tool_calls": [{"index": 0, "id": id, "type": "function",
                    "function": {"name": "f", "arguments": "{}"}}]}),
                None,
            )
        };
        let (events, response) = run(&[
            call("c1"),
            call("c2"),
            delta_chunk(&json!({}), Some("tool_calls")),
        ]);
        let indices: Vec<u32> = events
            .iter()
            .filter_map(|event| match &event.body {
                StreamEventBody::OutputItemAdded { output_index, .. } => Some(*output_index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(response.output.len(), 2);
    }

    #[test]
    fn length_finish_ends_with_response_incomplete() {
        let (events, response) = run(&[
            delta_chunk(&json!({"content": "cut off"}), None),
            delta_chunk(&json!({}), Some("length")),
            usage_chunk(5, 7),
        ]);
        assert_eq!(*names(&events).last().unwrap(), "response.incomplete");
        assert_eq!(response.status, ResponseStatus::Incomplete);
        assert_eq!(
            response.incomplete_details.unwrap().reason,
            "max_output_tokens"
        );
        match &response.output[0] {
            OutputItem::Message { status, .. } => assert_eq!(*status, ResponseStatus::Incomplete),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn stream_without_a_finish_reason_fails() {
        let (events, response) = run(&[delta_chunk(&json!({"content": "partial"}), None)]);
        assert_eq!(*names(&events).last().unwrap(), "response.failed");
        assert_eq!(response.status, ResponseStatus::Failed);
        assert_eq!(response.error.unwrap().code, "server_error");
        assert_monotonic(&events);
    }

    #[test]
    fn empty_generation_still_emits_one_empty_message() {
        let (events, response) = run(&[delta_chunk(&json!({}), Some("stop"))]);
        assert_eq!(
            names(&events),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(response.output.len(), 1);
    }

    #[test]
    fn malformed_and_sentinel_chunks_are_ignored() {
        let mut builder = ResponseStreamBuilder::new(context());
        builder.start();
        assert!(builder.ingest("[DONE]").is_empty());
        assert!(builder.ingest("not json").is_empty());
    }

    #[test]
    fn created_and_in_progress_carry_an_in_progress_response() {
        let (events, _) = run(&[delta_chunk(&json!({}), Some("stop"))]);
        for event in &events[..2] {
            match &event.body {
                StreamEventBody::Created { response }
                | StreamEventBody::InProgress { response } => {
                    assert_eq!(response.status, ResponseStatus::InProgress);
                    assert!(response.output.is_empty());
                    assert!(response.usage.is_none());
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}
