use std::sync::Arc;
use std::time::{Duration, Instant};

use higgs_engine::engine::{GenerationStats, StreamingOutput};
use higgs_engine::reasoning_parser::StreamingReasoningTracker;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::{ToolCallParser, ToolCallTracker};
use higgs_engine::tool_parser::StreamingToolOutput;

use super::GenerationRequest;
use super::complete::decode_logprobs;
use super::event::{EventStream, GeneratedToolCall, GenerationEvent};
use super::prepare::{Prepared, ReasoningHandling, prepare};
use crate::backend::EngineLease;
use crate::error::ServerError;
use crate::metrics::{MetricsStore, RequestRecord, RequestTiming};
use crate::router::RoutingMethod;
use crate::types::openai::ChoiceLogprobs;
use crate::usage::{TokenCounts, build_usage, count_text_tokens};

/// Where a streamed generation reports its progress for the metrics page.
#[derive(Clone)]
pub struct StreamMetrics {
    pub store: Arc<MetricsStore>,
    pub routing_method: RoutingMethod,
}

/// One streamed generation, without any tool loop. Preparation errors are
/// returned here; the engine starts working before the first event is polled.
pub(super) fn stream_once(
    engine: EngineLease,
    request: &GenerationRequest,
    received_at: Instant,
    metrics: Option<StreamMetrics>,
) -> Result<EventStream, ServerError> {
    let Prepared {
        job,
        prompt_token_count,
        reasoning,
        tool_parser,
    } = prepare(&**engine, request)?;
    let tokenizer = engine.tokenizer().clone();
    let pending_metrics = metrics.map(|handle| {
        let id = handle.store.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(request.model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: handle.routing_method.into(),
            status: 200,
            duration: Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
            timing: RequestTiming::default(),
        });
        (handle.store, id)
    });

    let (sender, mut receiver) = tokio::sync::mpsc::channel(32);
    tokio::task::spawn_blocking(move || {
        if let Err(error) = engine.generate_streaming(job, &sender) {
            tracing::error!(%error, "Generation error during streaming");
        }
    });

    Ok(Box::pin(async_stream::stream! {
        let mut assembler = Assembler::new(reasoning, tool_parser.as_ref(), tokenizer);
        let mut output_tokens: u32 = 0;
        let mut timing = RequestTiming::default();
        let mut stats = GenerationStats::default();

        yield GenerationEvent::Started { prompt_tokens: prompt_token_count };
        while let Some(output) = receiver.recv().await {
            stats.observe(&output, Instant::now());
            if let Some(progress) = output.prefill_progress {
                timing.cached_tokens = Some(u64::from(progress.cached));
                yield GenerationEvent::PrefillProgress(progress);
                continue;
            }
            if timing.ttft_ms.is_none() {
                timing.ttft_ms =
                    Some(u64::try_from(received_at.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            output_tokens = output.completion_tokens;
            for event in assembler.push(&output) {
                yield event;
            }
        }
        for event in assembler.finish() {
            yield event;
        }

        let usage = build_usage(
            TokenCounts {
                prompt: prompt_token_count,
                completion: output_tokens,
                reasoning: assembler.reasoning_tokens(),
            },
            &stats,
            received_at,
        );
        if let Some((store, id)) = pending_metrics {
            store.finalize_stream(id, u64::from(output_tokens), received_at.elapsed(), timing);
        }
        yield GenerationEvent::Usage(usage);
    }))
}

/// Turns the engine's pieces into events: splits reasoning from the answer
/// and tool calls from the visible text, holding back text that might still
/// become either.
struct Assembler {
    /// `None` when the output is raw text.
    reasoning: Option<StreamingReasoningTracker>,
    tools: Option<ToolCallTracker>,
    tokenizer: Tokenizer,
    reasoning_text: String,
    /// Held until the trackers have drained, since a tool call changes the reason.
    pending_finish: Option<(String, Option<ChoiceLogprobs>)>,
}

impl Assembler {
    fn new(
        handling: ReasoningHandling,
        tool_parser: Option<&ToolCallParser>,
        tokenizer: Tokenizer,
    ) -> Self {
        let reasoning = match handling {
            ReasoningHandling::Raw => None,
            ReasoningHandling::Parsed { thinking: true, .. } => {
                Some(StreamingReasoningTracker::new_inside_think())
            }
            ReasoningHandling::Parsed {
                thinking: false, ..
            } => Some(StreamingReasoningTracker::new()),
        };
        Self {
            reasoning,
            tools: tool_parser.map(|parser| parser.tracker(true)),
            tokenizer,
            reasoning_text: String::new(),
            pending_finish: None,
        }
    }

    fn push(&mut self, output: &StreamingOutput) -> Vec<GenerationEvent> {
        let mut events = Vec::new();
        let Some(tracker) = self.reasoning.as_mut() else {
            events.push(GenerationEvent::TextDelta {
                text: output.new_text.clone(),
                logprobs: None,
            });
            if let Some(reason) = &output.finish_reason {
                self.pending_finish = Some((reason.clone(), None));
            }
            return events;
        };

        let chunk_logprobs = output
            .token_logprob
            .as_ref()
            .map(|info| decode_logprobs(std::slice::from_ref(info), &self.tokenizer));
        let (visible, reasoning) = tracker.process(&output.new_text);
        self.emit_reasoning(reasoning, &mut events);
        let tool_output = match self.tools.as_mut() {
            Some(tools) => tools.process(&visible),
            None => StreamingToolOutput {
                visible,
                new_tool_calls: Vec::new(),
            },
        };
        let visible_was_empty = tool_output.visible.is_empty();
        self.emit_tool_output(tool_output, chunk_logprobs.clone(), &mut events);
        if let Some(reason) = &output.finish_reason {
            let logprobs = if visible_was_empty {
                chunk_logprobs
            } else {
                None
            };
            self.pending_finish = Some((reason.clone(), logprobs));
        }
        events
    }

    /// Drain everything the trackers hold back and close the stream with its
    /// finish reason. An unclosed tool call comes out as visible text rather
    /// than being dropped.
    fn finish(&mut self) -> Vec<GenerationEvent> {
        let mut events = Vec::new();
        if let Some(tracker) = self.reasoning.as_mut() {
            let (visible, reasoning) = tracker.flush();
            self.emit_reasoning(reasoning, &mut events);
            let tool_output = match self.tools.as_mut() {
                Some(tools) => tools.process(&visible),
                None => StreamingToolOutput {
                    visible,
                    new_tool_calls: Vec::new(),
                },
            };
            self.emit_tool_output(tool_output, None, &mut events);
        }
        if let Some(tools) = self.tools.as_mut() {
            let tail = tools.flush();
            self.emit_tool_output(tail, None, &mut events);
        }
        if let Some((reason, logprobs)) = self.pending_finish.take() {
            let called_tools = self
                .tools
                .as_ref()
                .is_some_and(ToolCallTracker::has_tool_calls);
            events.push(GenerationEvent::Finish {
                reason: if called_tools {
                    "tool_calls".to_owned()
                } else {
                    reason
                },
                logprobs,
            });
        }
        events
    }

    fn reasoning_tokens(&self) -> Option<u32> {
        count_text_tokens(&self.tokenizer, &self.reasoning_text)
    }

    fn emit_reasoning(&mut self, reasoning: String, events: &mut Vec<GenerationEvent>) {
        if reasoning.is_empty() {
            return;
        }
        self.reasoning_text.push_str(&reasoning);
        events.push(GenerationEvent::ReasoningDelta(reasoning));
    }

    fn emit_tool_output(
        &self,
        output: StreamingToolOutput,
        logprobs: Option<ChoiceLogprobs>,
        events: &mut Vec<GenerationEvent>,
    ) {
        // Calls are numbered across the whole response; this batch covers the
        // indices just before the tracker's running total.
        let completed = self
            .tools
            .as_ref()
            .map_or(0, ToolCallTracker::completed_count);
        let first_index = completed.saturating_sub(output.new_tool_calls.len());
        for (offset, call) in output.new_tool_calls.into_iter().enumerate() {
            events.push(GenerationEvent::ToolCall(GeneratedToolCall::new(
                u32::try_from(first_index + offset).unwrap_or(u32::MAX),
                call.name,
                call.arguments,
            )));
        }
        if !output.visible.is_empty() {
            events.push(GenerationEvent::TextDelta {
                text: output.visible,
                logprobs,
            });
        }
    }
}
