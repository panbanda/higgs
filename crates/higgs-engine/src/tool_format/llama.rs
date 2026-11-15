//! Llama 3.x tool calls.
//!
//! The JSON convention of the 3.1/3.2 chat templates has the model answer with
//! a bare object, optionally preceded by the `<|python_tag|>` token:
//!
//! ```text
//! {"name": "get_weather", "parameters": {"city": "Paris"}}
//! <|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}
//! ```
//!
//! Llama 3.1 can also emit `<function=get_weather>{"city": "Paris"}</function>`.
//! Several JSON calls may be separated by `;`.
//!
//! A bare object is only a call when it opens the response (or follows
//! another call) and names a declared tool; anything else is ordinary text.

use super::mistral::call_from_members;
use super::span::{Span, json_span, leading_whitespace, parse_json_object};
use super::{ScanContext, Unit};
use crate::tool_parser::ParsedToolCall;

pub(super) const PYTHON_TAG: &str = "<|python_tag|>";
pub(super) const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";

/// `Some(call)` when `members` is a declared-tool call in Llama's shape.
fn call_from_llama_members(
    members: &serde_json::Map<String, serde_json::Value>,
    context: &ScanContext<'_>,
) -> Option<ParsedToolCall> {
    if !members.contains_key("parameters") && !members.contains_key("arguments") {
        return None;
    }
    let call = call_from_members(members, &["parameters", "arguments"])?;
    let name = context.canonical_name(&call.name)?;
    Some(ParsedToolCall { name, ..call })
}

/// Scan a JSON object that must be a call. `text` starts at `{`.
///
/// `rejected_length` is how much of `text` to hand back as plain text when the
/// object turns out not to be a call.
fn scan_json_call(text: &str, context: &ScanContext<'_>, rejected_length: usize) -> Unit {
    match json_span(text) {
        Span::Incomplete => Unit::Incomplete,
        Span::Invalid => Unit::Text {
            consumed: rejected_length,
        },
        Span::Complete(end) => {
            let call = text
                .get(..end)
                .and_then(parse_json_object)
                .and_then(|members| call_from_llama_members(&members, context));
            call.map_or(
                Unit::Text {
                    consumed: rejected_length.max(end),
                },
                |parsed| Unit::Calls {
                    consumed: end,
                    calls: vec![parsed],
                },
            )
        }
    }
}

/// Scan a bare JSON call. `text` starts at the opening `{`.
pub(super) fn scan_bare_json(text: &str, context: &ScanContext<'_>) -> Unit {
    let after_brace = text.get(1..).unwrap_or_default();
    let key_start = after_brace.get(leading_whitespace(after_brace)..);
    match key_start.and_then(|rest| rest.chars().next()) {
        None => return Unit::Incomplete,
        Some('"') => {}
        Some(_) => return Unit::Text { consumed: 1 },
    }
    scan_json_call(text, context, 1)
}

/// Scan a `<|python_tag|>` unit. `text` starts at the tag.
pub(super) fn scan_python_tag(text: &str, context: &ScanContext<'_>) -> Unit {
    let after_tag = text.get(PYTHON_TAG.len()..).unwrap_or_default();
    let skipped = leading_whitespace(after_tag);
    let payload = after_tag.get(skipped..).unwrap_or_default();
    match payload.chars().next() {
        None => Unit::Incomplete,
        Some('{') => match scan_json_call(payload, context, PYTHON_TAG.len()) {
            Unit::Calls { consumed, calls } => Unit::Calls {
                consumed: PYTHON_TAG.len() + skipped + consumed,
                calls,
            },
            Unit::Text { .. } => Unit::Text {
                consumed: PYTHON_TAG.len(),
            },
            Unit::Incomplete => Unit::Incomplete,
        },
        Some(_) => Unit::Text {
            consumed: PYTHON_TAG.len(),
        },
    }
}

/// Scan a `<function=name>{…}</function>` unit. `text` starts at `<function=`.
pub(super) fn scan_function(text: &str, _context: &ScanContext<'_>) -> Unit {
    let rejected = Unit::Text {
        consumed: FUNCTION_OPEN.len(),
    };
    let after_open = text.get(FUNCTION_OPEN.len()..).unwrap_or_default();
    let Some(name_end) = after_open.find('>') else {
        let name_so_far = after_open.chars().all(|character| {
            character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        });
        return if name_so_far {
            Unit::Incomplete
        } else {
            rejected
        };
    };
    let name = after_open.get(..name_end).unwrap_or_default();
    if name.is_empty()
        || !name.chars().all(|character| {
            character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        })
    {
        return rejected;
    }
    let body_start = FUNCTION_OPEN.len() + name_end + 1;
    let body = text.get(body_start..).unwrap_or_default();
    let skipped = leading_whitespace(body);
    let arguments_text = body.get(skipped..).unwrap_or_default();

    let (arguments, arguments_end) = if arguments_text.starts_with('{') {
        match json_span(arguments_text) {
            Span::Incomplete => return Unit::Incomplete,
            Span::Invalid => return rejected,
            Span::Complete(end) => {
                let Some(members) = arguments_text.get(..end).and_then(parse_json_object) else {
                    return rejected;
                };
                (serde_json::Value::Object(members), end)
            }
        }
    } else {
        (serde_json::Value::Object(serde_json::Map::new()), 0)
    };

    let after_arguments = arguments_text.get(arguments_end..).unwrap_or_default();
    let closer_offset = leading_whitespace(after_arguments);
    let closer_text = after_arguments.get(closer_offset..).unwrap_or_default();
    if closer_text.starts_with(FUNCTION_CLOSE) {
        Unit::Calls {
            consumed: body_start + skipped + arguments_end + closer_offset + FUNCTION_CLOSE.len(),
            calls: vec![ParsedToolCall {
                name: name.to_owned(),
                arguments,
            }],
        }
    } else if FUNCTION_CLOSE.starts_with(closer_text) {
        Unit::Incomplete
    } else {
        rejected
    }
}
