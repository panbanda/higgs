//! Locating the extent of a bracketed value inside a buffer that is still
//! growing, without parsing it.

/// Where a bracketed value ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Span {
    /// The value is closed; the offset is one past its final bracket.
    Complete(usize),
    /// More input is needed before the value closes.
    Incomplete,
    /// The text cannot be a bracketed value: it does not start with an
    /// opening bracket, or the brackets are mismatched.
    Invalid,
}

/// Find the end of the bracketed value (`{…}`, `[…]` or `(…)`) that `text`
/// starts with.
///
/// Brackets inside quoted strings do not count; `quotes` lists the characters
/// that delimit a string (JSON uses `"`, Python also `'`). A backslash escapes
/// the next character inside a string.
pub(super) fn bracketed_span(text: &str, quotes: &[char]) -> Span {
    let mut expected_closers: Vec<char> = Vec::new();
    let mut open_quote: Option<char> = None;
    let mut escaped = false;

    for (offset, character) in text.char_indices() {
        if let Some(quote) = open_quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == quote {
                open_quote = None;
            }
            continue;
        }
        if quotes.contains(&character) {
            if expected_closers.is_empty() {
                return Span::Invalid;
            }
            open_quote = Some(character);
            continue;
        }
        let closer = match character {
            '{' => Some('}'),
            '[' => Some(']'),
            '(' => Some(')'),
            _ => None,
        };
        if let Some(closer_character) = closer {
            expected_closers.push(closer_character);
            continue;
        }
        if matches!(character, '}' | ']' | ')') {
            if expected_closers.pop() != Some(character) {
                return Span::Invalid;
            }
            if expected_closers.is_empty() {
                return Span::Complete(offset + character.len_utf8());
            }
            continue;
        }
        if expected_closers.is_empty() {
            return Span::Invalid;
        }
    }
    Span::Incomplete
}

/// [`bracketed_span`] for JSON text.
pub(super) fn json_span(text: &str) -> Span {
    bracketed_span(text, &['"'])
}

/// Parse `text` as a JSON object, returning its members.
pub(super) fn parse_json_object(text: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    match serde_json::from_str::<serde_json::Value>(text).ok()? {
        serde_json::Value::Object(members) => Some(members),
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_)
        | serde_json::Value::Array(_) => None,
    }
}

/// Number of leading whitespace bytes in `text`.
pub(super) fn leading_whitespace(text: &str) -> usize {
    text.len() - text.trim_start().len()
}

#[allow(clippy::unwrap_used, clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_object_reports_its_end() {
        assert_eq!(json_span(r#"{"a": 1} tail"#), Span::Complete(8));
    }

    #[test]
    fn nested_and_string_brackets_are_balanced() {
        let text = r#"{"a": "}]", "b": [1, {"c": "\"}"}]}"#;
        assert_eq!(json_span(text), Span::Complete(text.len()));
    }

    #[test]
    fn unfinished_value_is_incomplete() {
        assert_eq!(json_span(r#"{"a": [1, 2"#), Span::Incomplete);
        assert_eq!(json_span(r#"{"a": "unterminated"#), Span::Incomplete);
    }

    #[test]
    fn mismatched_or_unopened_text_is_invalid() {
        assert_eq!(json_span("{]"), Span::Invalid);
        assert_eq!(json_span("abc"), Span::Invalid);
        assert_eq!(json_span("}"), Span::Invalid);
    }

    #[test]
    fn python_quotes_protect_brackets() {
        let text = "[f(a='x)]', b=\"it's\")] rest";
        assert_eq!(
            bracketed_span(text, &['"', '\'']),
            Span::Complete(text.len() - " rest".len())
        );
    }

    #[test]
    fn json_object_parsing_rejects_other_values() {
        assert!(parse_json_object(r#"{"a": 1}"#).is_some());
        assert!(parse_json_object("[1]").is_none());
        assert!(parse_json_object("nope").is_none());
    }
}
