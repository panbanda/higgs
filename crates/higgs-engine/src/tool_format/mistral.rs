//! Mistral tool calls: a `[TOOL_CALLS]` marker followed by either a JSON array
//! of calls or `name[ARGS]{…}` / `name{…}` pairs.
//!
//! ```text
//! [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]
//! [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
//! [TOOL_CALLS]get_weather{"city": "Paris"}
//! ```

use super::span::{Span, json_span, leading_whitespace, parse_json_object};
use super::{ScanContext, Unit};
use crate::tool_parser::ParsedToolCall;

pub(super) const MARKER: &str = "[TOOL_CALLS]";
pub(super) const ARGUMENTS_MARKER: &str = "[ARGS]";

fn is_function_name_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
}

/// Convert one `{"name": …, "arguments": …}` member list into a call.
///
/// Arguments may be an object, a JSON-encoded object string (some Mistral
/// checkpoints double-encode), or absent.
pub(super) fn call_from_members(
    members: &serde_json::Map<String, serde_json::Value>,
    arguments_keys: &[&str],
) -> Option<ParsedToolCall> {
    let name = members.get("name")?.as_str()?;
    if name.is_empty() {
        return None;
    }
    let raw_arguments = arguments_keys.iter().find_map(|key| members.get(*key));
    let arguments = match raw_arguments {
        None | Some(serde_json::Value::Null) => serde_json::Value::Object(serde_json::Map::new()),
        Some(serde_json::Value::String(encoded)) => {
            serde_json::Value::Object(parse_json_object(encoded)?)
        }
        Some(object @ serde_json::Value::Object(_)) => object.clone(),
        Some(
            serde_json::Value::Bool(_) | serde_json::Value::Number(_) | serde_json::Value::Array(_),
        ) => return None,
    };
    Some(ParsedToolCall {
        name: name.to_owned(),
        arguments,
    })
}

fn calls_from_array(text: &str) -> Option<Vec<ParsedToolCall>> {
    let serde_json::Value::Array(items) = serde_json::from_str(text).ok()? else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    items
        .iter()
        .map(|item| {
            let members = item.as_object()?;
            call_from_members(members, &["arguments", "parameters"])
        })
        .collect()
}

/// Scan one `[TOOL_CALLS]` unit. `text` starts at the marker.
pub(super) fn scan(text: &str, _context: &ScanContext<'_>) -> Unit {
    let after_marker = text.get(MARKER.len()..).unwrap_or_default();
    let skipped = leading_whitespace(after_marker);
    let payload = after_marker.get(skipped..).unwrap_or_default();
    let payload_start = MARKER.len() + skipped;
    let rejected = Unit::Text {
        consumed: MARKER.len(),
    };

    let Some(first) = payload.chars().next() else {
        return Unit::Incomplete;
    };
    if first == '[' {
        return match json_span(payload) {
            Span::Incomplete => Unit::Incomplete,
            Span::Invalid => rejected,
            Span::Complete(end) => {
                let array = payload.get(..end).unwrap_or_default();
                calls_from_array(array).map_or(rejected, |calls| Unit::Calls {
                    consumed: payload_start + end,
                    calls,
                })
            }
        };
    }

    let name_length = payload
        .find(|character: char| !is_function_name_character(character))
        .unwrap_or(payload.len());
    let name = payload.get(..name_length).unwrap_or_default();
    let after_name = payload.get(name_length..).unwrap_or_default();
    if name.is_empty() {
        return rejected;
    }
    let arguments_offset = if let Some(rest) = after_name.strip_prefix(ARGUMENTS_MARKER) {
        after_name.len() - rest.len()
    } else if after_name.starts_with('{') {
        0
    } else if after_name.is_empty() || ARGUMENTS_MARKER.starts_with(after_name) {
        return Unit::Incomplete;
    } else {
        return rejected;
    };
    let arguments_text = after_name.get(arguments_offset..).unwrap_or_default();
    if arguments_text.is_empty() {
        return Unit::Incomplete;
    }
    match json_span(arguments_text) {
        Span::Incomplete => Unit::Incomplete,
        Span::Invalid => rejected,
        Span::Complete(end) => {
            let Some(members) = arguments_text.get(..end).and_then(parse_json_object) else {
                return rejected;
            };
            Unit::Calls {
                consumed: payload_start + name_length + arguments_offset + end,
                calls: vec![ParsedToolCall {
                    name: name.to_owned(),
                    arguments: serde_json::Value::Object(members),
                }],
            }
        }
    }
}
