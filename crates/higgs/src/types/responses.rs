//! Wire types for the `OpenAI` Responses API (`/v1/responses`).

use serde::{Deserialize, Serialize};

use super::openai::ReasoningConfig;

pub type Metadata = serde_json::Map<String, serde_json::Value>;

/// POST /v1/responses request body.
///
/// Fields Higgs does not act on (`user`, `truncation`, `include`, ...) are
/// ignored by serde. Fields whose silent omission would change behavior
/// (`background`, `conversation`) are captured so the route can reject them.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateResponseRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<ResponseInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub text: Option<TextConfig>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<Metadata>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub conversation: Option<serde_json::Value>,
    /// Higgs extension: server-side MCP tool loop, as on chat completions.
    #[serde(default)]
    pub mcp: Option<crate::generation::tool_loop::McpOptions>,
}

/// The `input` field: a bare prompt string or a list of input items.
///
/// Items stay as raw JSON here so that translation can name exactly which
/// item or content part is unsupported.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<serde_json::Value>),
}

/// The `text` request block controlling output formatting.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default)]
    pub format: Option<TextFormat>,
}

/// `text.format`: unlike chat's `response_format`, the schema fields are
/// siblings of `type`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextFormat {
    pub r#type: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

/// An input item after validation, in the form the translator consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputItem {
    Message {
        role: String,
        content: Vec<InputContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

/// A content part of an input message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputContent {
    Text(String),
    ImageUrl(String),
}

/// Lifecycle state of a response or of one of its output items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}

/// A content part inside an output item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        annotations: Vec<serde_json::Value>,
    },
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

impl ContentPart {
    pub const fn output_text(text: String) -> Self {
        Self::OutputText {
            text,
            annotations: Vec::new(),
        }
    }
}

/// One element of a response's `output` array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<serde_json::Value>,
        content: Vec<ContentPart>,
        status: ResponseStatus,
    },
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        content: Vec<ContentPart>,
        status: ResponseStatus,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: ResponseStatus,
    },
}

impl OutputItem {
    pub fn id(&self) -> &str {
        match self {
            Self::Reasoning { id, .. }
            | Self::Message { id, .. }
            | Self::FunctionCall { id, .. } => id,
        }
    }
}

/// Token accounting in the Responses shape.
///
/// The detail blocks are present only when the server actually measured them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

/// The Response object returned by every `/v1/responses` endpoint.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: ResponseStatus,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub output_text: String,
    pub usage: Option<ResponseUsage>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub error: Option<ResponseError>,
    pub instructions: Option<String>,
    pub metadata: Metadata,
    pub previous_response_id: Option<String>,
    pub tools: Vec<serde_json::Value>,
    pub tool_choice: serde_json::Value,
    pub parallel_tool_calls: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

/// DELETE /v1/responses/{id} result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeletedResponse {
    pub id: String,
    pub object: &'static str,
    pub deleted: bool,
}

/// One server-sent event of a streamed response.
///
/// `sequence_number` is assigned by the stream builder so it can never repeat
/// or go backwards.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamEvent {
    pub sequence_number: u64,
    #[serde(flatten)]
    pub body: StreamEventBody,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum StreamEventBody {
    #[serde(rename = "response.created")]
    Created { response: Box<ResponseObject> },
    #[serde(rename = "response.in_progress")]
    InProgress { response: Box<ResponseObject> },
    #[serde(rename = "response.completed")]
    Completed { response: Box<ResponseObject> },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: Box<ResponseObject> },
    #[serde(rename = "response.failed")]
    Failed { response: Box<ResponseObject> },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { output_index: u32, item: OutputItem },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { output_index: u32, item: OutputItem },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ContentPart,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        logprobs: Vec<serde_json::Value>,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        logprobs: Vec<serde_json::Value>,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u32,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: u32,
        name: String,
        arguments: String,
    },
}

impl StreamEventBody {
    /// The value of the `type` field, used as the SSE `event:` name.
    pub const fn event_name(&self) -> &'static str {
        match self {
            Self::Created { .. } => "response.created",
            Self::InProgress { .. } => "response.in_progress",
            Self::Completed { .. } => "response.completed",
            Self::Incomplete { .. } => "response.incomplete",
            Self::Failed { .. } => "response.failed",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            Self::ReasoningTextDone { .. } => "response.reasoning_text.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_response() -> ResponseObject {
        ResponseObject {
            id: "resp_1".to_owned(),
            object: "response",
            created_at: 1_700_000_000,
            status: ResponseStatus::Completed,
            model: "m".to_owned(),
            output: vec![
                OutputItem::Reasoning {
                    id: "rs_1".to_owned(),
                    summary: vec![],
                    content: vec![ContentPart::ReasoningText {
                        text: "think".to_owned(),
                    }],
                    status: ResponseStatus::Completed,
                },
                OutputItem::Message {
                    id: "msg_1".to_owned(),
                    role: "assistant",
                    content: vec![ContentPart::output_text("hi".to_owned())],
                    status: ResponseStatus::Completed,
                },
                OutputItem::FunctionCall {
                    id: "fc_1".to_owned(),
                    call_id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: "{\"city\":\"Paris\"}".to_owned(),
                    status: ResponseStatus::Completed,
                },
            ],
            output_text: "hi".to_owned(),
            usage: Some(ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            incomplete_details: None,
            error: None,
            instructions: Some("be brief".to_owned()),
            metadata: Metadata::new(),
            previous_response_id: Some("resp_0".to_owned()),
            tools: vec![],
            tool_choice: json!("auto"),
            parallel_tool_calls: true,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
        }
    }

    #[test]
    fn response_object_serializes_in_the_responses_shape() {
        let value = serde_json::to_value(sample_response()).unwrap();
        assert_eq!(value["object"], "response");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output_text"], "hi");
        assert_eq!(value["instructions"], "be brief");
        assert_eq!(value["previous_response_id"], "resp_0");
        assert_eq!(value["metadata"], json!({}));
        assert!(value["incomplete_details"].is_null());
        assert!(value["error"].is_null());

        assert_eq!(value["output"][0]["type"], "reasoning");
        assert_eq!(value["output"][0]["content"][0]["type"], "reasoning_text");
        assert_eq!(value["output"][1]["type"], "message");
        assert_eq!(value["output"][1]["role"], "assistant");
        assert_eq!(value["output"][1]["content"][0]["type"], "output_text");
        assert_eq!(value["output"][1]["content"][0]["annotations"], json!([]));
        assert_eq!(value["output"][2]["type"], "function_call");
        assert_eq!(value["output"][2]["call_id"], "call_1");
        assert_eq!(value["output"][2]["name"], "get_weather");
        assert_eq!(value["output"][2]["arguments"], "{\"city\":\"Paris\"}");
    }

    #[test]
    fn usage_omits_detail_blocks_that_were_not_measured() {
        let value = serde_json::to_value(sample_response()).unwrap();
        assert_eq!(
            value["usage"],
            json!({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        );
    }

    #[test]
    fn usage_includes_detail_blocks_when_measured() {
        let usage = ResponseUsage {
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 8 }),
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 2,
            }),
        };
        let value = serde_json::to_value(usage).unwrap();
        assert_eq!(value["input_tokens_details"]["cached_tokens"], 8);
        assert_eq!(value["output_tokens_details"]["reasoning_tokens"], 2);
    }

    #[test]
    fn stream_event_flattens_sequence_number_beside_type() {
        let event = StreamEvent {
            sequence_number: 7,
            body: StreamEventBody::OutputTextDelta {
                item_id: "msg_1".to_owned(),
                output_index: 0,
                content_index: 0,
                delta: "hel".to_owned(),
                logprobs: vec![],
            },
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "response.output_text.delta");
        assert_eq!(value["sequence_number"], 7);
        assert_eq!(value["delta"], "hel");
    }

    #[test]
    fn event_name_matches_serialized_type_for_every_variant() {
        let response = Box::new(sample_response());
        let item = sample_response().output.remove(1);
        let part = ContentPart::output_text(String::new());
        let bodies = vec![
            StreamEventBody::Created {
                response: response.clone(),
            },
            StreamEventBody::InProgress {
                response: response.clone(),
            },
            StreamEventBody::Completed {
                response: response.clone(),
            },
            StreamEventBody::Incomplete {
                response: response.clone(),
            },
            StreamEventBody::Failed { response },
            StreamEventBody::OutputItemAdded {
                output_index: 0,
                item: item.clone(),
            },
            StreamEventBody::OutputItemDone {
                output_index: 0,
                item,
            },
            StreamEventBody::ContentPartAdded {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                part: part.clone(),
            },
            StreamEventBody::ContentPartDone {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                part,
            },
            StreamEventBody::OutputTextDelta {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                delta: String::new(),
                logprobs: vec![],
            },
            StreamEventBody::OutputTextDone {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                text: String::new(),
                logprobs: vec![],
            },
            StreamEventBody::ReasoningTextDelta {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                delta: String::new(),
            },
            StreamEventBody::ReasoningTextDone {
                item_id: String::new(),
                output_index: 0,
                content_index: 0,
                text: String::new(),
            },
            StreamEventBody::FunctionCallArgumentsDelta {
                item_id: String::new(),
                output_index: 0,
                delta: String::new(),
            },
            StreamEventBody::FunctionCallArgumentsDone {
                item_id: String::new(),
                output_index: 0,
                name: String::new(),
                arguments: String::new(),
            },
        ];
        for body in bodies {
            let expected = body.event_name();
            let value = serde_json::to_value(StreamEvent {
                sequence_number: 0,
                body,
            })
            .unwrap();
            assert_eq!(value["type"], expected);
        }
    }

    #[test]
    fn request_accepts_string_and_item_inputs() {
        let text: CreateResponseRequest =
            serde_json::from_value(json!({"model": "m", "input": "hello"})).unwrap();
        assert!(matches!(text.input, Some(ResponseInput::Text(ref s)) if s == "hello"));

        let items: CreateResponseRequest = serde_json::from_value(
            json!({"model": "m", "input": [{"role": "user", "content": "hi"}]}),
        )
        .unwrap();
        assert!(matches!(items.input, Some(ResponseInput::Items(ref v)) if v.len() == 1));
    }

    #[test]
    fn request_ignores_unknown_fields() {
        let request: CreateResponseRequest = serde_json::from_value(
            json!({"model": "m", "input": "x", "user": "u", "truncation": "auto"}),
        )
        .unwrap();
        assert_eq!(request.model, "m");
    }
}
