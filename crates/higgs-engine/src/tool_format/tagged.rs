//! XML-tagged calls whose arguments are raw strings coerced by the declared
//! parameter types: GLM and `MiniMax`.
//!
//! ```text
//! <tool_call>get_weather
//! <arg_key>city</arg_key>
//! <arg_value>Paris</arg_value>
//! </tool_call>
//!
//! <minimax:tool_call>
//! <invoke name="get_weather">
//! <parameter name="city">Paris</parameter>
//! </invoke>
//! </minimax:tool_call>
//! ```

use super::{ScanContext, Unit};
use crate::tool_parser::{
    ParsedToolCall, TOOL_CALL_CLOSE, TOOL_CALL_OPEN, ToolSchema, coerce_param_value,
    parse_tool_call_block, strip_one_wrapping_newline,
};

const ARGUMENT_KEY_OPEN: &str = "<arg_key>";
const ARGUMENT_KEY_CLOSE: &str = "</arg_key>";
const ARGUMENT_VALUE_OPEN: &str = "<arg_value>";
const ARGUMENT_VALUE_CLOSE: &str = "</arg_value>";

pub(super) const MINIMAX_OPEN: &str = "<minimax:tool_call>";
const MINIMAX_CLOSE: &str = "</minimax:tool_call>";
const INVOKE_OPEN: &str = "<invoke name=\"";
const INVOKE_CLOSE: &str = "</invoke>";
const PARAMETER_OPEN: &str = "<parameter name=\"";
const PARAMETER_CLOSE: &str = "</parameter>";

fn is_function_name(candidate: &str) -> bool {
    !candidate.is_empty()
        && candidate.chars().all(|character| {
            character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        })
}

/// Parse the body of a GLM `<tool_call>`: the function name followed by
/// `<arg_key>`/`<arg_value>` pairs.
pub(super) fn parse_glm_body(body: &str, schema: Option<&ToolSchema>) -> Option<ParsedToolCall> {
    let trimmed = body.trim();
    let name_end = trimmed.find(ARGUMENT_KEY_OPEN).unwrap_or(trimmed.len());
    let name = trimmed.get(..name_end)?.trim();
    if !is_function_name(name) {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    let mut rest = trimmed.get(name_end..)?;
    while !rest.trim().is_empty() {
        let after_key_open = rest.trim_start().strip_prefix(ARGUMENT_KEY_OPEN)?;
        let key_end = after_key_open.find(ARGUMENT_KEY_CLOSE)?;
        let key = after_key_open.get(..key_end)?.trim();
        let after_key = after_key_open
            .get(key_end + ARGUMENT_KEY_CLOSE.len()..)?
            .trim_start();
        let after_value_open = after_key.strip_prefix(ARGUMENT_VALUE_OPEN)?;
        let value_end = after_value_open.find(ARGUMENT_VALUE_CLOSE)?;
        let raw_value = after_value_open.get(..value_end)?;
        if key.is_empty() {
            return None;
        }
        let declared = schema.and_then(|declared_types| declared_types.param_type(name, key));
        arguments.insert(
            key.to_owned(),
            coerce_param_value(strip_one_wrapping_newline(raw_value), declared),
        );
        rest = after_value_open.get(value_end + ARGUMENT_VALUE_CLOSE.len()..)?;
    }

    Some(ParsedToolCall {
        name: name.to_owned(),
        arguments: serde_json::Value::Object(arguments),
    })
}

/// Scan a `<tool_call>…</tool_call>` block as GLM's argument-tag form, falling
/// back to the Qwen inner forms when `accept_qwen_bodies` is set.
pub(super) fn scan_tool_call(
    text: &str,
    context: &ScanContext<'_>,
    accept_qwen_bodies: bool,
) -> Unit {
    let after_open = text.get(TOOL_CALL_OPEN.len()..).unwrap_or_default();
    let Some(body_end) = after_open.find(TOOL_CALL_CLOSE) else {
        return Unit::Incomplete;
    };
    let body = after_open.get(..body_end).unwrap_or_default();
    let consumed = TOOL_CALL_OPEN.len() + body_end + TOOL_CALL_CLOSE.len();

    let qwen_call = if accept_qwen_bodies {
        parse_tool_call_block(body.trim(), context.schema)
    } else {
        None
    };
    qwen_call
        .or_else(|| parse_glm_body(body, context.schema))
        .map_or(Unit::Text { consumed }, |call| Unit::Calls {
            consumed,
            calls: vec![call],
        })
}

fn parse_invoke_parameters(
    function: &str,
    mut rest: &str,
    schema: Option<&ToolSchema>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut arguments = serde_json::Map::new();
    while let Some(open) = rest.find(PARAMETER_OPEN) {
        let after_open = rest.get(open + PARAMETER_OPEN.len()..)?;
        let key_end = after_open.find("\">")?;
        let key = after_open.get(..key_end)?;
        let value_region = after_open.get(key_end + 2..)?;
        let value_end = value_region.find(PARAMETER_CLOSE)?;
        let raw_value = value_region.get(..value_end)?.trim();
        if key.is_empty() {
            return None;
        }
        let declared = schema.and_then(|declared_types| declared_types.param_type(function, key));
        arguments.insert(key.to_owned(), coerce_param_value(raw_value, declared));
        rest = value_region.get(value_end + PARAMETER_CLOSE.len()..)?;
    }
    Some(arguments)
}

fn parse_minimax_body(body: &str, schema: Option<&ToolSchema>) -> Option<Vec<ParsedToolCall>> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(open) = rest.find(INVOKE_OPEN) {
        let after_open = rest.get(open + INVOKE_OPEN.len()..)?;
        let name_end = after_open.find("\">")?;
        let name = after_open.get(..name_end)?;
        if !is_function_name(name) {
            return None;
        }
        let invoke_region = after_open.get(name_end + 2..)?;
        let invoke_end = invoke_region.find(INVOKE_CLOSE)?;
        let parameters = invoke_region.get(..invoke_end)?;
        calls.push(ParsedToolCall {
            name: name.to_owned(),
            arguments: serde_json::Value::Object(parse_invoke_parameters(
                name, parameters, schema,
            )?),
        });
        rest = invoke_region.get(invoke_end + INVOKE_CLOSE.len()..)?;
    }
    (!calls.is_empty()).then_some(calls)
}

/// Scan a `<minimax:tool_call>` block. `text` starts at the opening tag.
pub(super) fn scan_minimax(text: &str, context: &ScanContext<'_>) -> Unit {
    let after_open = text.get(MINIMAX_OPEN.len()..).unwrap_or_default();
    let Some(body_end) = after_open.find(MINIMAX_CLOSE) else {
        return Unit::Incomplete;
    };
    let body = after_open.get(..body_end).unwrap_or_default();
    let consumed = MINIMAX_OPEN.len() + body_end + MINIMAX_CLOSE.len();
    parse_minimax_body(body, context.schema).map_or(Unit::Text { consumed }, |calls| Unit::Calls {
        consumed,
        calls,
    })
}
