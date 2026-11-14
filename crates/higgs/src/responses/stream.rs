//! Converts generation events into the Responses event protocol.

use super::translate::{Outcome, ResponseContext, generate_item_id, status_for_finish_reason};
use crate::generation::{GeneratedToolCall, GenerationEvent};
use crate::types::responses::{
    ContentPart, InputTokensDetails, OutputItem, OutputTokensDetails, ResponseError,
    ResponseObject, ResponseStatus, ResponseUsage, StreamEvent, StreamEventBody,
};

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

/// Stateful translator from generation events to ordered Responses events.
///
/// Text and reasoning stream as deltas into an open item; a tool call
/// arrives complete from the engine, so it is emitted as a whole item.
/// Switching kinds closes the open item first, so `output_index` always
/// equals the number of items closed before it.
pub struct ResponseStreamBuilder {
    context: ResponseContext,
    sequence_number: u64,
    items: Vec<OutputItem>,
    open: Option<OpenItem>,
    finish_reason: Option<String>,
    usage: Option<(u32, u32)>,
    /// From prefill progress; superseded by the usage figure.
    cached_tokens: Option<u32>,
    reasoning_tokens: Option<u32>,
    /// Why generation failed, when it did.
    failure: Option<String>,
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
            reasoning_tokens: None,
            failure: None,
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

    /// Feed one generation event.
    pub fn ingest(&mut self, event: GenerationEvent) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match event {
            GenerationEvent::PrefillProgress(progress) => {
                self.cached_tokens = Some(progress.cached);
            }
            GenerationEvent::Usage(usage) => {
                self.usage = Some((usage.prompt_tokens, usage.completion_tokens));
                if let Some(details) = usage.prompt_tokens_details {
                    self.cached_tokens = Some(details.cached_tokens);
                }
                self.reasoning_tokens = usage
                    .completion_tokens_details
                    .map(|details| details.reasoning_tokens);
            }
            GenerationEvent::ReasoningDelta { text, .. } => {
                if !text.is_empty() {
                    self.reasoning_delta(&mut events, text);
                }
            }
            GenerationEvent::ToolCall { call, .. } => self.tool_call(&mut events, &call),
            GenerationEvent::TextDelta { text, .. } => {
                if !text.is_empty() {
                    self.message_delta(&mut events, text);
                }
            }
            GenerationEvent::Finish { reason, .. } => self.finish_reason = Some(reason),
            GenerationEvent::Error(message) => self.failure = Some(message),
            GenerationEvent::Started { .. } | GenerationEvent::ToolLoop(_) => {}
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

    fn tool_call(&mut self, events: &mut Vec<StreamEvent>, call: &GeneratedToolCall) {
        self.close_open(events, ResponseStatus::Completed);
        let id = generate_item_id("fc");
        let call_id = call.id.clone();
        let name = call.name.clone();
        let arguments = call.arguments.to_string();
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
        let (status, incomplete_reason) = if self.failure.is_some() {
            (ResponseStatus::Failed, None)
        } else {
            self.finish_reason
                .as_deref()
                .map_or((ResponseStatus::Failed, None), status_for_finish_reason)
        };
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
            output_tokens_details: self
                .reasoning_tokens
                .map(|reasoning_tokens| OutputTokensDetails { reasoning_tokens }),
        });
        let error = (status == ResponseStatus::Failed).then(|| ResponseError {
            code: "server_error".to_owned(),
            message: self
                .failure
                .take()
                .unwrap_or_else(|| "generation ended before the response completed".to_owned()),
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
    use crate::types::openai::{CompletionTokensDetails, CompletionUsage, PromptTokensDetails};
    use higgs_engine::engine::PrefillProgress;
    use serde_json::json;

    fn context() -> ResponseContext {
        ResponseContext::from_request(
            &serde_json::from_value(json!({"model": "m", "input": "hi"})).unwrap(),
            "m".to_owned(),
        )
    }

    fn text(text: &str) -> GenerationEvent {
        GenerationEvent::TextDelta {
            text: text.to_owned(),
            logprobs: None,
        }
    }

    fn reasoning(text: &str) -> GenerationEvent {
        GenerationEvent::ReasoningDelta {
            text: text.to_owned(),
            logprobs: None,
        }
    }

    fn finish(reason: &str) -> GenerationEvent {
        GenerationEvent::Finish {
            reason: reason.to_owned(),
            logprobs: None,
        }
    }

    fn usage(prompt: u32, completion: u32) -> GenerationEvent {
        GenerationEvent::Usage(CompletionUsage::new(prompt, completion))
    }

    fn progress(cached: u32) -> GenerationEvent {
        GenerationEvent::PrefillProgress(PrefillProgress {
            processed: 20,
            cached,
            total: 20,
        })
    }

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> GenerationEvent {
        let mut generated = GeneratedToolCall::new(0, name.to_owned(), arguments);
        generated.id = id.to_owned();
        GenerationEvent::ToolCall {
            call: generated,
            logprobs: None,
        }
    }

    fn run(events: Vec<GenerationEvent>) -> (Vec<StreamEvent>, ResponseObject) {
        let mut builder = ResponseStreamBuilder::new(context());
        let mut emitted = builder.start();
        for event in events {
            emitted.extend(builder.ingest(event));
        }
        let finished = builder.finish();
        emitted.extend(finished.events);
        (emitted, finished.response)
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
    fn usage_details_set_cached_and_reasoning_tokens() {
        let mut detailed = CompletionUsage::new(20, 9);
        detailed.prompt_tokens_details = Some(PromptTokensDetails { cached_tokens: 16 });
        detailed.completion_tokens_details = Some(CompletionTokensDetails {
            reasoning_tokens: 4,
        });

        let (_events, response) = run(vec![progress(2), GenerationEvent::Usage(detailed)]);

        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens_details.unwrap().cached_tokens, 16);
        assert_eq!(usage.output_tokens_details.unwrap().reasoning_tokens, 4);
    }

    #[test]
    fn text_only_stream_has_the_exact_event_order() {
        let (events, response) = run(vec![
            progress(4),
            text("Hel"),
            text("lo"),
            finish("stop"),
            usage(12, 2),
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
        let (events, _) = run(vec![text("Hi"), finish("stop")]);
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
        let (events, response) = run(vec![
            reasoning("need "),
            reasoning("weather"),
            call("call_0_abc", "get_weather", json!({"city": "Paris"})),
            finish("tool_calls"),
            usage(30, 9),
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
        let (events, response) = run(vec![
            text("Checking."),
            call("c1", "f", json!({})),
            finish("tool_calls"),
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
        let (events, response) = run(vec![
            call("c1", "f", json!({})),
            call("c2", "f", json!({})),
            finish("tool_calls"),
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
        let (events, response) = run(vec![text("cut off"), finish("length"), usage(5, 7)]);
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
        let (events, response) = run(vec![text("partial")]);
        assert_eq!(*names(&events).last().unwrap(), "response.failed");
        assert_eq!(response.status, ResponseStatus::Failed);
        assert_eq!(response.error.unwrap().code, "server_error");
        assert_monotonic(&events);
    }

    #[test]
    fn empty_generation_still_emits_one_empty_message() {
        let (events, response) = run(vec![finish("stop")]);
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
    fn events_without_output_are_ignored() {
        let mut builder = ResponseStreamBuilder::new(context());
        builder.start();
        assert!(
            builder
                .ingest(GenerationEvent::Started { prompt_tokens: 3 })
                .is_empty()
        );
        assert!(
            builder
                .ingest(GenerationEvent::Error("tool loop failed".to_owned()))
                .is_empty()
        );
    }

    #[test]
    fn created_and_in_progress_carry_an_in_progress_response() {
        let (events, _) = run(vec![finish("stop")]);
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
