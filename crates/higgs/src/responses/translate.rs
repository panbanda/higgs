//! Translation between the Responses API and the shared generation request
//! and result.

use serde_json::{Value, json};

use higgs_models::SamplingParams;

use crate::constraint::ConstraintSpec;
use crate::error::ServerError;
use crate::generation::messages::prompt_messages;
use crate::generation::{ChatPrompt, Completed, GenerationRequest, Prompt};
use crate::tool_choice::ToolChoice;
use crate::tool_policy::ToolPolicy;
use crate::types::openai::{
    ChatCompletionMessage, ContentPart as ChatPart, ImageUrl, MessageContent, ResponseFormat,
    ToolCall, ToolCallFunction,
};
use crate::types::responses::{
    ContentPart, CreateResponseRequest, IncompleteDetails, InputContent, InputItem,
    InputTokensDetails, Metadata, OutputItem, OutputTokensDetails, ResponseError, ResponseInput,
    ResponseObject, ResponseStatus, ResponseUsage, TextConfig,
};

const SUPPORTED_ROLES: &[&str] = &["user", "assistant", "system", "developer"];

pub fn generate_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

pub fn generate_item_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

const fn bad_request(message: String) -> ServerError {
    ServerError::BadRequest(message)
}

/// Validate the `input` field into [`InputItem`]s.
///
/// `allow_images` is true only for vision models; otherwise an `input_image`
/// part is rejected instead of being dropped.
pub fn parse_input(
    input: Option<ResponseInput>,
    allow_images: bool,
) -> Result<Vec<InputItem>, ServerError> {
    match input {
        None => Err(bad_request("input is required".to_owned())),
        Some(ResponseInput::Text(text)) => Ok(vec![InputItem::Message {
            role: "user".to_owned(),
            content: vec![InputContent::Text(text)],
        }]),
        Some(ResponseInput::Items(values)) => {
            let mut items = Vec::with_capacity(values.len());
            for (index, value) in values.iter().enumerate() {
                if let Some(item) = parse_item(index, value, allow_images)? {
                    items.push(item);
                }
            }
            Ok(items)
        }
    }
}

/// `Ok(None)` marks an item that is valid but intentionally not replayed to
/// the model: prior `reasoning` items, which clients echo back verbatim.
fn parse_item(
    index: usize,
    value: &Value,
    allow_images: bool,
) -> Result<Option<InputItem>, ServerError> {
    let object = value
        .as_object()
        .ok_or_else(|| bad_request(format!("input[{index}] must be an object")))?;
    let required_string = |field: &str| -> Result<String, ServerError> {
        object
            .get(field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| bad_request(format!("input[{index}].{field} must be a string")))
    };

    match object.get("type").and_then(Value::as_str) {
        None | Some("message") => parse_message(index, object, allow_images).map(Some),
        Some("function_call") => {
            let arguments = match object.get("arguments") {
                Some(Value::String(text)) => text.clone(),
                Some(other) if !other.is_null() => other.to_string(),
                Some(_) | None => "{}".to_owned(),
            };
            Ok(Some(InputItem::FunctionCall {
                call_id: required_string("call_id")?,
                name: required_string("name")?,
                arguments,
            }))
        }
        Some("function_call_output") => Ok(Some(InputItem::FunctionCallOutput {
            call_id: required_string("call_id")?,
            output: function_output_text(object.get("output")),
        })),
        Some("reasoning") => Ok(None),
        Some(other) => Err(bad_request(format!(
            "Unsupported input item type '{other}' at input[{index}]; supported types are \
             message, function_call, function_call_output and reasoning"
        ))),
    }
}

fn function_output_text(output: Option<&Value>) -> String {
    match output {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            if texts.len() == parts.len() {
                texts.join("")
            } else {
                Value::Array(parts.clone()).to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

fn parse_message(
    index: usize,
    object: &serde_json::Map<String, Value>,
    allow_images: bool,
) -> Result<InputItem, ServerError> {
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request(format!("input[{index}].role must be a string")))?;
    if !SUPPORTED_ROLES.contains(&role) {
        return Err(bad_request(format!(
            "Unsupported role '{role}' at input[{index}]; supported roles are user, assistant, \
             system and developer"
        )));
    }

    let content = match object.get("content") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![InputContent::Text(text.clone())],
        Some(Value::Array(parts)) => {
            let mut content = Vec::with_capacity(parts.len());
            for (part_index, part) in parts.iter().enumerate() {
                content.push(parse_content_part(index, part_index, part, allow_images)?);
            }
            content
        }
        Some(_) => {
            return Err(bad_request(format!(
                "input[{index}].content must be a string or an array of content parts"
            )));
        }
    };

    Ok(InputItem::Message {
        role: role.to_owned(),
        content,
    })
}

fn parse_content_part(
    index: usize,
    part_index: usize,
    part: &Value,
    allow_images: bool,
) -> Result<InputContent, ServerError> {
    let location = format!("input[{index}].content[{part_index}]");
    let part_type = part
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request(format!("{location}.type must be a string")))?;
    match part_type {
        "input_text" | "output_text" => part
            .get("text")
            .and_then(Value::as_str)
            .map(|text| InputContent::Text(text.to_owned()))
            .ok_or_else(|| bad_request(format!("{location}.text must be a string"))),
        "input_image" => {
            if !allow_images {
                return Err(bad_request(format!(
                    "Unsupported content part type 'input_image' at {location}: the model is not \
                     a vision model"
                )));
            }
            let url = part
                .get("image_url")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    bad_request(format!(
                        "{location}.image_url must be a string (file_id images are not supported)"
                    ))
                })?;
            if !url.starts_with("data:") {
                return Err(bad_request(format!(
                    "{location}.image_url must be a base64 data URI; remote image URLs are not \
                     supported"
                )));
            }
            Ok(InputContent::ImageUrl(url.to_owned()))
        }
        other => Err(bad_request(format!(
            "Unsupported content part type '{other}' at {location}; supported types are \
             input_text, output_text and input_image"
        ))),
    }
}

/// Convert Responses function tools to the chat `tools` shape.
pub fn convert_tools(tools: &[Value]) -> Result<Vec<Value>, ServerError> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let tool_type = tool
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request(format!("tools[{index}].type must be a string")))?;
            if tool_type != "function" {
                return Err(bad_request(format!(
                    "Unsupported tool type '{tool_type}' at tools[{index}]; only function tools \
                     are supported"
                )));
            }
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request(format!("tools[{index}].name must be a string")))?;
            let mut function = json!({ "name": name });
            for field in ["description", "parameters"] {
                if let Some(value) = tool.get(field).filter(|value| !value.is_null())
                    && let Some(target) = function.as_object_mut()
                {
                    target.insert(field.to_owned(), value.clone());
                }
            }
            Ok(json!({ "type": "function", "function": function }))
        })
        .collect()
}

pub fn convert_text_format(
    text: Option<&TextConfig>,
) -> Result<Option<ResponseFormat>, ServerError> {
    let Some(format) = text.and_then(|config| config.format.as_ref()) else {
        return Ok(None);
    };
    match format.r#type.as_str() {
        "text" => Ok(None),
        "json_object" => Ok(Some(ResponseFormat {
            r#type: "json_object".to_owned(),
            json_schema: None,
            regex: None,
        })),
        "json_schema" => {
            let schema = format.schema.clone().ok_or_else(|| {
                bad_request("text.format of type 'json_schema' requires a schema".to_owned())
            })?;
            Ok(Some(ResponseFormat {
                r#type: "json_schema".to_owned(),
                json_schema: Some(json!({
                    "name": format.name.as_deref().unwrap_or("response"),
                    "schema": schema,
                })),
                regex: None,
            }))
        }
        other => Err(bad_request(format!(
            "Unsupported text.format type '{other}'; supported types are text, json_object and \
             json_schema"
        ))),
    }
}

/// Reject request fields that would change behavior if silently ignored.
pub fn reject_unsupported_fields(request: &CreateResponseRequest) -> Result<(), ServerError> {
    if request.background == Some(true) {
        return Err(bad_request(
            "background responses are not supported".to_owned(),
        ));
    }
    if request
        .conversation
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(bad_request(
            "conversation is not supported; use previous_response_id to chain responses".to_owned(),
        ));
    }
    Ok(())
}

fn text_of(content: &[InputContent]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            InputContent::Text(text) => Some(text.as_str()),
            InputContent::ImageUrl(_) => None,
        })
        .collect()
}

fn chat_content(content: &[InputContent]) -> Option<MessageContent> {
    if content.is_empty() {
        return None;
    }
    let has_image = content
        .iter()
        .any(|part| matches!(part, InputContent::ImageUrl(_)));
    if !has_image {
        return Some(MessageContent::Text(text_of(content)));
    }
    Some(MessageContent::Parts(
        content
            .iter()
            .map(|part| match part {
                InputContent::Text(text) => ChatPart::Text { text: text.clone() },
                InputContent::ImageUrl(url) => ChatPart::ImageUrl {
                    image_url: ImageUrl { url: url.clone() },
                },
            })
            .collect(),
    ))
}

fn chat_message(role: &str) -> ChatCompletionMessage {
    ChatCompletionMessage {
        role: role.to_owned(),
        content: None,
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Flatten instructions plus the conversation into chat messages.
///
/// Consecutive `function_call` items collapse into one assistant message
/// (parallel tool calls), and a call that directly follows assistant text
/// joins that message, matching how the model produced them.
pub fn items_to_messages(
    instructions: Option<&str>,
    items: &[InputItem],
) -> Vec<ChatCompletionMessage> {
    let mut messages = Vec::with_capacity(items.len() + 1);
    if let Some(text) = instructions.filter(|text| !text.is_empty()) {
        messages.push(ChatCompletionMessage {
            content: Some(MessageContent::Text(text.to_owned())),
            ..chat_message("system")
        });
    }
    for item in items {
        match item {
            InputItem::Message { role, content } => {
                let chat_role = if role == "developer" { "system" } else { role };
                messages.push(ChatCompletionMessage {
                    content: chat_content(content),
                    ..chat_message(chat_role)
                });
            }
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let call = ToolCall {
                    id: call_id.clone(),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                };
                match messages.last_mut() {
                    Some(last) if last.role == "assistant" => {
                        last.tool_calls.get_or_insert_with(Vec::new).push(call);
                    }
                    Some(_) | None => messages.push(ChatCompletionMessage {
                        tool_calls: Some(vec![call]),
                        ..chat_message("assistant")
                    }),
                }
            }
            InputItem::FunctionCallOutput { call_id, output } => {
                messages.push(ChatCompletionMessage {
                    content: Some(MessageContent::Text(output.clone())),
                    tool_call_id: Some(call_id.clone()),
                    ..chat_message("tool")
                });
            }
        }
    }
    messages
}

/// Translate a Responses request, with the conversation so far, into the
/// shared request.
pub fn build_generation_request(
    request: &CreateResponseRequest,
    model: &str,
    conversation: &[InputItem],
    default_max_tokens: u32,
) -> Result<GenerationRequest, ServerError> {
    let tool_choice = ToolChoice::from_responses(request.tool_choice.as_ref())?;
    let tools = if tool_choice == ToolChoice::Disabled {
        Vec::new()
    } else {
        match request.tools.as_deref() {
            Some(tools) if !tools.is_empty() => convert_tools(tools)?,
            Some(_) | None => Vec::new(),
        }
    };
    let constraint = ConstraintSpec::from_request(
        convert_text_format(request.text.as_ref())?.as_ref(),
        None,
        None,
    )?;
    let thinking_budget = crate::reasoning::requested_thinking_budget(request.reasoning.as_ref())?;
    let (messages, images) = prompt_messages(&items_to_messages(
        request.instructions.as_deref(),
        conversation,
    ));
    Ok(GenerationRequest {
        model: model.to_owned(),
        prompt: Prompt::Chat(ChatPrompt {
            messages,
            images,
            tools,
            tool_policy: ToolPolicy::new(tool_choice, request.parallel_tool_calls.unwrap_or(true)),
            reasoning: request.reasoning.clone(),
            thinking_budget,
            parse_reasoning_without_thinking: false,
        }),
        max_tokens: request.max_output_tokens.unwrap_or(default_max_tokens),
        sampling: SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            ..SamplingParams::default()
        },
        stop_sequences: Vec::new(),
        logprobs: false,
        top_logprobs: None,
        constraint,
        mcp: request.mcp.clone(),
    })
}

/// Per-request values echoed in every Response object.
#[derive(Debug, Clone)]
pub struct ResponseContext {
    pub id: String,
    pub created_at: i64,
    pub model: String,
    pub instructions: Option<String>,
    pub metadata: Metadata,
    pub previous_response_id: Option<String>,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel_tool_calls: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

/// How a response ended, plus what it produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub status: ResponseStatus,
    pub output: Vec<OutputItem>,
    pub usage: Option<ResponseUsage>,
    pub incomplete_reason: Option<String>,
    pub error: Option<ResponseError>,
}

impl Outcome {
    pub const fn in_progress() -> Self {
        Self {
            status: ResponseStatus::InProgress,
            output: Vec::new(),
            usage: None,
            incomplete_reason: None,
            error: None,
        }
    }
}

impl ResponseContext {
    pub fn from_request(request: &CreateResponseRequest, model: String) -> Self {
        Self {
            id: generate_response_id(),
            created_at: chrono::Utc::now().timestamp(),
            model,
            instructions: request.instructions.clone(),
            metadata: request.metadata.clone().unwrap_or_default(),
            previous_response_id: request.previous_response_id.clone(),
            tools: request.tools.clone().unwrap_or_default(),
            tool_choice: request
                .tool_choice
                .clone()
                .unwrap_or_else(|| Value::String("auto".to_owned())),
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_output_tokens,
        }
    }

    pub fn object(&self, outcome: Outcome) -> ResponseObject {
        let output_text = outcome
            .output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Message { content, .. } => Some(content),
                OutputItem::Reasoning { .. } | OutputItem::FunctionCall { .. } => None,
            })
            .flatten()
            .filter_map(|part| match part {
                ContentPart::OutputText { text, .. } => Some(text.as_str()),
                ContentPart::ReasoningText { .. } => None,
            })
            .collect();
        ResponseObject {
            id: self.id.clone(),
            object: "response",
            created_at: self.created_at,
            status: outcome.status,
            model: self.model.clone(),
            output: outcome.output,
            output_text,
            usage: outcome.usage,
            incomplete_details: outcome
                .incomplete_reason
                .map(|reason| IncompleteDetails { reason }),
            error: outcome.error,
            instructions: self.instructions.clone(),
            metadata: self.metadata.clone(),
            previous_response_id: self.previous_response_id.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
        }
    }
}

/// Map a chat `finish_reason` to a response status and incomplete reason.
pub fn status_for_finish_reason(finish_reason: &str) -> (ResponseStatus, Option<String>) {
    match finish_reason {
        "length" => (
            ResponseStatus::Incomplete,
            Some("max_output_tokens".to_owned()),
        ),
        "content_filter" => (
            ResponseStatus::Incomplete,
            Some("content_filter".to_owned()),
        ),
        _ => (ResponseStatus::Completed, None),
    }
}

/// Build the output items of a non-streaming generation.
pub fn outcome_from_completed(completed: Completed) -> Outcome {
    let (status, incomplete_reason) = status_for_finish_reason(&completed.finish_reason);

    let mut output = Vec::new();
    let has_reasoning = completed
        .reasoning
        .as_deref()
        .is_some_and(|text| !text.is_empty());
    if let Some(reasoning) = completed.reasoning.filter(|text| !text.is_empty()) {
        output.push(OutputItem::Reasoning {
            id: generate_item_id("rs"),
            summary: Vec::new(),
            content: vec![ContentPart::ReasoningText { text: reasoning }],
            status: ResponseStatus::Completed,
        });
    }
    if !completed.text.is_empty() || (!has_reasoning && completed.tool_calls.is_empty()) {
        output.push(OutputItem::Message {
            id: generate_item_id("msg"),
            role: "assistant",
            content: vec![ContentPart::output_text(completed.text)],
            status: ResponseStatus::Completed,
        });
    }
    for call in completed.tool_calls {
        output.push(OutputItem::FunctionCall {
            id: generate_item_id("fc"),
            call_id: call.id,
            name: call.name,
            arguments: call.arguments.to_string(),
            status: ResponseStatus::Completed,
        });
    }
    if status == ResponseStatus::Incomplete
        && let Some(last) = output.last_mut()
    {
        set_item_status(last, ResponseStatus::Incomplete);
    }

    let usage = completed.usage;
    Outcome {
        status,
        output,
        usage: Some(ResponseUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: usage
                .prompt_tokens_details
                .map(|details| InputTokensDetails {
                    cached_tokens: details.cached_tokens,
                }),
            output_tokens_details: usage.completion_tokens_details.map(|details| {
                OutputTokensDetails {
                    reasoning_tokens: details.reasoning_tokens,
                }
            }),
        }),
        incomplete_reason,
        error: None,
    }
}

pub const fn set_item_status(item: &mut OutputItem, new_status: ResponseStatus) {
    match item {
        OutputItem::Reasoning { status, .. }
        | OutputItem::Message { status, .. }
        | OutputItem::FunctionCall { status, .. } => *status = new_status,
    }
}

/// Convert a finished turn's output back into input items so a later request
/// can chain from it. Reasoning is not replayed.
pub fn output_to_input_items(output: &[OutputItem]) -> Vec<InputItem> {
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message { content, .. } => {
                let text: String = content
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::OutputText { text, .. } => Some(text.as_str()),
                        ContentPart::ReasoningText { .. } => None,
                    })
                    .collect();
                (!text.is_empty()).then(|| InputItem::Message {
                    role: "assistant".to_owned(),
                    content: vec![InputContent::Text(text)],
                })
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => Some(InputItem::FunctionCall {
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            OutputItem::Reasoning { .. } => None,
        })
        .collect()
}

/// The conversation items for messages a server-side tool loop appended:
/// assistant turns (text, then each tool call) and tool results.
pub fn chat_messages_to_input_items(messages: &[ChatCompletionMessage]) -> Vec<InputItem> {
    let mut items = Vec::new();
    for message in messages {
        match message.role.as_str() {
            "assistant" => {
                let text = message
                    .content
                    .as_ref()
                    .map(MessageContent::text)
                    .unwrap_or_default();
                if !text.is_empty() {
                    items.push(InputItem::Message {
                        role: "assistant".to_owned(),
                        content: vec![InputContent::Text(text)],
                    });
                }
                for call in message.tool_calls.iter().flatten() {
                    items.push(InputItem::FunctionCall {
                        call_id: call.id.clone(),
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    });
                }
            }
            "tool" => {
                if let Some(call_id) = &message.tool_call_id {
                    items.push(InputItem::FunctionCallOutput {
                        call_id: call_id.clone(),
                        output: message
                            .content
                            .as_ref()
                            .map(MessageContent::text)
                            .unwrap_or_default(),
                    });
                }
            }
            _ => {}
        }
    }
    items
}

/// Best-effort `role`/`content` view of the input for model auto-routing,
/// which runs before strict validation (a remote provider may understand
/// items that local translation rejects).
pub fn routing_messages(input: Option<&ResponseInput>) -> Option<Vec<Value>> {
    let messages: Vec<Value> = match input? {
        ResponseInput::Text(text) => vec![json!({ "role": "user", "content": text })],
        ResponseInput::Items(values) => values
            .iter()
            .filter_map(|value| {
                let role = value.get("role")?.as_str()?;
                let content = match value.get("content")? {
                    Value::String(text) => text.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect(),
                    Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_) => {
                        return None;
                    }
                };
                Some(json!({ "role": role, "content": content }))
            })
            .collect(),
    };
    Some(messages)
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
    use crate::generation::GeneratedToolCall;
    use crate::types::openai::CompletionUsage;

    fn request(value: Value) -> CreateResponseRequest {
        serde_json::from_value(value).unwrap()
    }

    fn parse(value: &Value) -> Vec<InputItem> {
        let parsed = request(json!({"model": "m", "input": value}));
        parse_input(parsed.input, false).unwrap()
    }

    fn parse_error(value: &Value, allow_images: bool) -> String {
        let parsed = request(json!({"model": "m", "input": value}));
        match parse_input(parsed.input, allow_images) {
            Err(ServerError::BadRequest(message)) => message,
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn string_input_becomes_one_user_message() {
        let items = parse(&json!("hello"));
        assert_eq!(
            items,
            vec![InputItem::Message {
                role: "user".to_owned(),
                content: vec![InputContent::Text("hello".to_owned())],
            }]
        );
    }

    #[test]
    fn missing_input_is_rejected() {
        let parsed = request(json!({"model": "m"}));
        assert!(matches!(
            parse_input(parsed.input, false),
            Err(ServerError::BadRequest(message)) if message == "input is required"
        ));
    }

    #[test]
    fn mixed_items_translate_to_chat_messages() {
        let items = parse(&json!([
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": [{"type": "input_text", "text": "weather in "}, {"type": "input_text", "text": "Paris?"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "output_text", "text": "noon"}]},
            {"type": "reasoning", "id": "rs_1", "summary": []}
        ]));
        let messages = items_to_messages(Some("You are helpful."), &items);

        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            vec!["system", "system", "user", "assistant", "tool", "tool"]
        );
        assert_eq!(
            messages[0].content.as_ref().unwrap().text(),
            "You are helpful."
        );
        assert_eq!(messages[1].content.as_ref().unwrap().text(), "be terse");
        assert_eq!(
            messages[2].content.as_ref().unwrap().text(),
            "weather in Paris?"
        );

        let assistant = &messages[3];
        assert_eq!(assistant.content.as_ref().unwrap().text(), "Checking.");
        let calls = assistant.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].id, "call_2");

        assert_eq!(messages[4].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[4].content.as_ref().unwrap().text(), "sunny");
        assert_eq!(messages[5].content.as_ref().unwrap().text(), "noon");
    }

    #[test]
    fn function_call_without_preceding_assistant_gets_its_own_message() {
        let items = parse(&json!([
            {"role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
        ]));
        let messages = items_to_messages(None, &items);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].content.is_none());
        assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn empty_instructions_add_no_system_message() {
        let items = parse(&json!("hi"));
        assert_eq!(items_to_messages(Some(""), &items).len(), 1);
    }

    #[test]
    fn previous_turn_is_prepended_to_new_input() {
        let history = vec![
            InputItem::Message {
                role: "user".to_owned(),
                content: vec![InputContent::Text("first".to_owned())],
            },
            InputItem::Message {
                role: "assistant".to_owned(),
                content: vec![InputContent::Text("answer".to_owned())],
            },
        ];
        let mut conversation = history;
        conversation.extend(parse(&json!("second")));
        let generation = build_generation_request(
            &request(json!({"model": "m", "input": "second", "instructions": "sys"})),
            "m",
            &conversation,
            100,
        )
        .unwrap();
        let texts: Vec<(String, String)> = generation
            .chat()
            .unwrap()
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        assert_eq!(
            texts,
            vec![
                ("system".to_owned(), "sys".to_owned()),
                ("user".to_owned(), "first".to_owned()),
                ("assistant".to_owned(), "answer".to_owned()),
                ("user".to_owned(), "second".to_owned()),
            ]
        );
    }

    #[test]
    fn output_items_round_trip_into_input_items() {
        let output = vec![
            OutputItem::Reasoning {
                id: "rs_1".to_owned(),
                summary: vec![],
                content: vec![ContentPart::ReasoningText {
                    text: "hmm".to_owned(),
                }],
                status: ResponseStatus::Completed,
            },
            OutputItem::Message {
                id: "msg_1".to_owned(),
                role: "assistant",
                content: vec![ContentPart::output_text("hello".to_owned())],
                status: ResponseStatus::Completed,
            },
            OutputItem::FunctionCall {
                id: "fc_1".to_owned(),
                call_id: "call_1".to_owned(),
                name: "f".to_owned(),
                arguments: "{}".to_owned(),
                status: ResponseStatus::Completed,
            },
        ];
        assert_eq!(
            output_to_input_items(&output),
            vec![
                InputItem::Message {
                    role: "assistant".to_owned(),
                    content: vec![InputContent::Text("hello".to_owned())],
                },
                InputItem::FunctionCall {
                    call_id: "call_1".to_owned(),
                    name: "f".to_owned(),
                    arguments: "{}".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_item_type_is_named() {
        let message = parse_error(&json!([{"type": "web_search_call", "id": "x"}]), false);
        assert!(message.contains("'web_search_call'"), "{message}");
        assert!(message.contains("input[0]"), "{message}");
    }

    #[test]
    fn unsupported_role_is_named() {
        let message = parse_error(&json!([{"role": "critic", "content": "x"}]), false);
        assert!(message.contains("'critic'"), "{message}");
    }

    #[test]
    fn malformed_items_are_rejected() {
        assert!(parse_error(&json!(["text"]), false).contains("must be an object"));
        assert!(parse_error(&json!([{"content": "x"}]), false).contains("role"));
        assert!(parse_error(&json!([{"role": "user", "content": 5}]), false).contains("content"));
        assert!(
            parse_error(&json!([{"type": "function_call", "name": "f"}]), false)
                .contains("call_id")
        );
    }

    #[test]
    fn unsupported_content_part_is_named() {
        let message = parse_error(
            &json!([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]),
            false,
        );
        assert!(message.contains("'input_file'"), "{message}");
        assert!(message.contains("input[0].content[0]"), "{message}");
    }

    #[test]
    fn images_require_a_vision_model() {
        let input = json!([{"role": "user", "content": [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]);
        let message = parse_error(&input, false);
        assert!(message.contains("input_image"), "{message}");
        assert!(message.contains("not a vision model"), "{message}");

        let parsed = request(json!({"model": "m", "input": input}));
        let items = parse_input(parsed.input, true).unwrap();
        let messages = items_to_messages(None, &items);
        let content = messages[0].content.as_ref().unwrap();
        assert!(content.has_images());
        assert_eq!(content.text(), "what is this");
    }

    #[test]
    fn remote_image_urls_are_rejected_for_vision_models() {
        let parsed = request(json!({"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/cat.png"}
        ]}]}));
        match parse_input(parsed.input, true) {
            Err(ServerError::BadRequest(message)) => {
                assert!(message.contains("data URI"), "{message}");
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn function_tools_convert_to_the_chat_shape() {
        let tools = convert_tools(&[json!({
            "type": "function",
            "name": "get_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "strict": true
        })])
        .unwrap();
        assert_eq!(
            tools[0],
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            })
        );
    }

    #[test]
    fn non_function_tools_are_rejected_by_name() {
        for tool_type in ["web_search", "file_search", "code_interpreter", "mcp"] {
            let error = convert_tools(&[json!({"type": tool_type})]).unwrap_err();
            match error {
                ServerError::BadRequest(message) => {
                    assert!(message.contains(&format!("'{tool_type}'")), "{message}");
                }
                other => panic!("unexpected error {other:?}"),
            }
        }
    }

    fn chat_for(extra: &Value) -> Result<ChatPrompt, ServerError> {
        let mut body = json!({
            "model": "m", "input": "hi",
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}]
        });
        if let (Some(target), Some(fields)) = (body.as_object_mut(), extra.as_object()) {
            target.extend(fields.clone());
        }
        build_generation_request(&request(body), "m", &[], 100)
            .map(|generation| generation.chat().unwrap().clone())
    }

    #[test]
    fn tool_choice_becomes_the_tool_policy() {
        let cases = [
            (json!({}), ToolChoice::Auto),
            (json!({"tool_choice": "auto"}), ToolChoice::Auto),
            (json!({"tool_choice": "required"}), ToolChoice::Required),
            (
                json!({"tool_choice": {"type": "function", "name": "f"}}),
                ToolChoice::Function("f".to_owned()),
            ),
        ];
        for (extra, expected) in cases {
            let chat = chat_for(&extra).unwrap();
            assert_eq!(chat.tool_policy, ToolPolicy::new(expected, true), "{extra}");
            assert_eq!(chat.tools.len(), 1, "{extra}");
        }
    }

    #[test]
    fn parallel_tool_calls_is_forwarded() {
        let chat = chat_for(&json!({"parallel_tool_calls": false})).unwrap();
        assert_eq!(chat.tool_policy, ToolPolicy::new(ToolChoice::Auto, false));
        let default = chat_for(&json!({})).unwrap();
        assert_eq!(default.tool_policy, ToolPolicy::new(ToolChoice::Auto, true));
    }

    #[test]
    fn invalid_tool_choices_are_rejected() {
        for bad in [
            json!("bogus"),
            json!({"type": "function"}),
            json!({"type": "web_search"}),
        ] {
            assert!(chat_for(&json!({"tool_choice": bad})).is_err(), "{bad}");
        }
    }

    #[test]
    fn tool_choice_none_drops_tools_from_the_request() {
        let generation = build_generation_request(
            &request(json!({
                "model": "m", "input": "hi", "tool_choice": "none",
                "tools": [{"type": "function", "name": "f"}]
            })),
            "m",
            &[],
            100,
        )
        .unwrap();
        assert!(generation.chat().unwrap().tools.is_empty());
    }

    #[test]
    fn text_format_maps_to_response_format() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let format = convert_text_format(
            request(json!({"model": "m", "input": "x", "text": {"format": {
                "type": "json_schema", "name": "thing", "schema": schema
            }}}))
            .text
            .as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(format.r#type, "json_schema");
        assert_eq!(
            format.json_schema.unwrap(),
            json!({"name": "thing", "schema": schema})
        );

        let object = convert_text_format(
            request(
                json!({"model": "m", "input": "x", "text": {"format": {"type": "json_object"}}}),
            )
            .text
            .as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(object.r#type, "json_object");

        assert!(
            convert_text_format(
                request(json!({"model": "m", "input": "x", "text": {"format": {"type": "text"}}}))
                    .text
                    .as_ref()
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn invalid_text_formats_are_rejected() {
        for format in [json!({"type": "json_schema"}), json!({"type": "xml"})] {
            let parsed = request(json!({"model": "m", "input": "x", "text": {"format": format}}));
            assert!(convert_text_format(parsed.text.as_ref()).is_err());
        }
    }

    #[test]
    fn unsupported_request_fields_are_rejected() {
        assert!(
            reject_unsupported_fields(&request(json!({"model": "m", "background": true}))).is_err()
        );
        assert!(
            reject_unsupported_fields(&request(json!({"model": "m", "conversation": "conv_1"})))
                .is_err()
        );
        assert!(
            reject_unsupported_fields(&request(json!({"model": "m", "background": false}))).is_ok()
        );
    }

    #[test]
    fn generation_request_carries_sampling_and_limits() {
        let generation = build_generation_request(
            &request(json!({
                "model": "m", "input": "hi", "stream": true, "max_output_tokens": 64,
                "temperature": 0.2, "top_p": 0.9, "reasoning": {"effort": "low"}
            })),
            "resolved",
            &[],
            100,
        )
        .unwrap();
        assert_eq!(generation.model, "resolved");
        assert_eq!(generation.max_tokens, 64);
        assert!((generation.sampling.temperature - 0.2).abs() < f32::EPSILON);
        assert!((generation.sampling.top_p - 0.9).abs() < f32::EPSILON);
        let reasoning = generation.chat().unwrap().reasoning.clone().unwrap();
        assert_eq!(reasoning.effort.as_deref(), Some("low"));
        let defaulted = build_generation_request(
            &request(json!({"model": "m", "input": "hi"})),
            "resolved",
            &[],
            100,
        )
        .unwrap();
        assert_eq!(defaulted.max_tokens, 100);
    }

    fn completed(text: &str, finish_reason: &str, usage: (u32, u32)) -> Completed {
        Completed {
            text: text.to_owned(),
            reasoning: None,
            tool_calls: Vec::new(),
            finish_reason: finish_reason.to_owned(),
            usage: CompletionUsage::new(usage.0, usage.1),
            logprobs: None,
            tool_loop: None,
        }
    }

    fn context() -> ResponseContext {
        ResponseContext::from_request(
            &request(json!({
                "model": "m", "input": "hi", "instructions": "sys",
                "metadata": {"k": "v"}, "previous_response_id": "resp_prev"
            })),
            "m".to_owned(),
        )
    }

    #[test]
    fn text_response_builds_message_item_with_usage_and_echoes() {
        let outcome = outcome_from_completed(completed("Hello!", "stop", (7, 3)));
        let object = context().object(outcome);

        assert!(object.id.starts_with("resp_"));
        assert_eq!(object.status, ResponseStatus::Completed);
        assert_eq!(object.output_text, "Hello!");
        assert_eq!(object.output.len(), 1);
        let usage = object.usage.as_ref().unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens, usage.total_tokens),
            (7, 3, 10)
        );
        assert!(usage.input_tokens_details.is_none());
        assert!(usage.output_tokens_details.is_none());
        assert_eq!(object.instructions.as_deref(), Some("sys"));
        assert_eq!(object.metadata["k"], "v");
        assert_eq!(object.previous_response_id.as_deref(), Some("resp_prev"));
        assert!(object.incomplete_details.is_none());
    }

    #[test]
    fn measured_usage_details_are_carried_into_the_response() {
        let mut chat = completed("Hello!", "stop", (7, 3));
        chat.usage.prompt_tokens_details =
            Some(crate::types::openai::PromptTokensDetails { cached_tokens: 5 });
        chat.usage.completion_tokens_details =
            Some(crate::types::openai::CompletionTokensDetails {
                reasoning_tokens: 2,
            });

        let object = context().object(outcome_from_completed(chat));

        let usage = object.usage.as_ref().unwrap();
        assert_eq!(
            usage.input_tokens_details.as_ref().unwrap().cached_tokens,
            5
        );
        assert_eq!(
            usage
                .output_tokens_details
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            2
        );
    }

    #[test]
    fn reasoning_and_tool_calls_become_separate_items_in_order() {
        let mut generation = completed("Let me check.", "tool_calls", (1, 1));
        generation.reasoning = Some("need weather".to_owned());
        let mut call =
            GeneratedToolCall::new(0, "get_weather".to_owned(), json!({"city": "Paris"}));
        call.id = "call_0_x".to_owned();
        generation.tool_calls = vec![call];
        let outcome = outcome_from_completed(generation);
        let kinds: Vec<&str> = outcome
            .output
            .iter()
            .map(|item| match item {
                OutputItem::Reasoning { .. } => "reasoning",
                OutputItem::Message { .. } => "message",
                OutputItem::FunctionCall { .. } => "function_call",
            })
            .collect();
        assert_eq!(kinds, vec!["reasoning", "message", "function_call"]);
        assert_eq!(outcome.status, ResponseStatus::Completed);
        match &outcome.output[2] {
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                assert_eq!(call_id, "call_0_x");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, "{\"city\":\"Paris\"}");
            }
            other => panic!("unexpected item {other:?}"),
        }
    }

    #[test]
    fn tool_call_only_response_has_no_empty_message() {
        let mut generation = completed("", "tool_calls", (1, 1));
        generation.tool_calls = vec![GeneratedToolCall::new(0, "f".to_owned(), json!({}))];
        let outcome = outcome_from_completed(generation);
        assert_eq!(outcome.output.len(), 1);
        assert!(matches!(outcome.output[0], OutputItem::FunctionCall { .. }));
    }

    #[test]
    fn length_stop_is_incomplete_with_details() {
        let outcome = outcome_from_completed(completed("cut", "length", (1, 1)));
        let object = context().object(outcome);
        assert_eq!(object.status, ResponseStatus::Incomplete);
        assert_eq!(
            object.incomplete_details.unwrap().reason,
            "max_output_tokens"
        );
        match &object.output[0] {
            OutputItem::Message { status, .. } => assert_eq!(*status, ResponseStatus::Incomplete),
            other => panic!("unexpected item {other:?}"),
        }
    }

    #[test]
    fn empty_text_still_yields_a_message_item() {
        let outcome = outcome_from_completed(completed("", "stop", (1, 0)));
        assert_eq!(outcome.output.len(), 1);
    }

    #[test]
    fn routing_messages_flattens_text_content() {
        let input = ResponseInput::Items(vec![
            json!({"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}),
            json!({"type": "function_call", "call_id": "c"}),
        ]);
        let messages = routing_messages(Some(&input)).unwrap();
        assert_eq!(messages, vec![json!({"role": "user", "content": "ab"})]);
    }
}
