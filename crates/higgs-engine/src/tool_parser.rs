//! Parse tool calls from model-generated text.
//!
//! Qwen models emit tool calls in a specific XML-like format:
//! ```text
//! <tool_call>
//! {"name": "function_name", "arguments": {"arg1": "value1"}}
//! </tool_call>
//! ```
//!
//! This module extracts those structured tool calls from the raw text.

/// A parsed tool call extracted from model output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of parsing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolParseResult {
    /// Text content before/outside any tool calls.
    pub text: String,
    /// Extracted tool calls (empty if none found).
    pub tool_calls: Vec<ParsedToolCall>,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Parse model output text for Qwen-format tool calls.
///
/// Returns the non-tool-call text and any extracted tool calls.
pub fn parse_tool_calls(text: &str) -> ToolParseResult {
    let mut result_text = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    loop {
        if let Some(start_pos) = remaining.find(TOOL_CALL_OPEN) {
            result_text.push_str(remaining.get(..start_pos).unwrap_or_default());

            let after_open = remaining
                .get(start_pos + TOOL_CALL_OPEN.len()..)
                .unwrap_or_default();

            if let Some(end_pos) = after_open.find(TOOL_CALL_CLOSE) {
                let raw_block = after_open.get(..end_pos).unwrap_or_default();
                let call_content = raw_block.trim();

                if let Some(parsed) = try_parse_tool_call(call_content) {
                    tool_calls.push(parsed);
                } else {
                    result_text.push_str(TOOL_CALL_OPEN);
                    result_text.push_str(raw_block);
                    result_text.push_str(TOOL_CALL_CLOSE);
                }

                remaining = after_open
                    .get(end_pos + TOOL_CALL_CLOSE.len()..)
                    .unwrap_or_default();
            } else {
                result_text.push_str(remaining.get(start_pos..).unwrap_or_default());
                break;
            }
        } else {
            result_text.push_str(remaining);
            break;
        }
    }

    ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    }
}

/// Try to parse a single tool call JSON block.
fn try_parse_tool_call(content: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let obj = value.as_object()?;

    let name = obj.get("name").and_then(|v| v.as_str())?.to_owned();

    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    Some(ParsedToolCall { name, arguments })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Parse input and assert expected tool call count and optional text fragment.
    fn assert_parse(
        input: &str,
        expected_tools: usize,
        text_contains: Option<&str>,
    ) -> ToolParseResult {
        let result = parse_tool_calls(input);
        assert_eq!(
            result.tool_calls.len(),
            expected_tools,
            "expected {expected_tools} tool calls, got {}",
            result.tool_calls.len()
        );
        if let Some(fragment) = text_contains {
            assert!(
                result.text.contains(fragment),
                "expected text to contain {fragment:?}, got {:?}",
                result.text
            );
        }
        result
    }

    /// Assert the parsed result has no tool calls and preserves the raw tags in text.
    fn assert_raw_preserved(input: &str) {
        let result = assert_parse(input, 0, Some("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
    }

    /// Get the name of the first parsed tool call.
    fn first_tool_name(result: &ToolParseResult) -> &str {
        &result.tool_calls.first().unwrap().name
    }

    #[test]
    fn test_no_tool_calls() {
        let result = assert_parse(
            "Hello, how can I help you?",
            0,
            Some("Hello, how can I help you?"),
        );
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_single_tool_call() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert!(result.text.is_empty());
        assert_eq!(first_tool_name(&result), "get_weather");
    }

    #[test]
    fn test_tool_call_with_surrounding_text() {
        let input = r#"Let me check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
I've requested the weather."#;
        let result = assert_parse(input, 1, Some("Let me check"));
        assert!(result.text.contains("I've requested"));
    }

    #[test]
    fn test_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expression": "2+2"}}
</tool_call>"#;
        let result = assert_parse(input, 2, None);
        assert_eq!(first_tool_name(&result), "search");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "calculate");
    }

    #[test]
    fn test_invalid_json_in_tool_call() {
        assert_parse(
            "<tool_call>\nnot valid json\n</tool_call>",
            0,
            Some("not valid json"),
        );
    }

    #[test]
    fn test_unclosed_tool_call_tag() {
        assert_parse(
            "Text before <tool_call>\n{\"name\": \"test\"}",
            0,
            Some("<tool_call>"),
        );
    }

    #[test]
    fn test_tool_call_missing_arguments() {
        let input = r#"<tool_call>
{"name": "no_args_tool"}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "no_args_tool");
        assert!(result.tool_calls.first().unwrap().arguments.is_object());
    }

    #[test]
    fn test_tool_call_missing_name() {
        let input = r#"<tool_call>
{"arguments": {"key": "value"}}
</tool_call>"#;
        assert_parse(input, 0, None);
    }

    #[test]
    fn test_empty_text() {
        let result = assert_parse("", 0, None);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_invalid_json_preserves_original_tags() {
        let input = "<tool_call>\nnot valid json\n</tool_call>";
        let result = assert_parse(input, 0, Some("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
        assert!(result.text.contains("not valid json"));
    }

    #[test]
    fn test_mix_of_valid_and_invalid_tool_calls() {
        let input = r#"<tool_call>
{"name": "good_tool", "arguments": {"key": "value"}}
</tool_call>
<tool_call>
this is not json
</tool_call>
<tool_call>
{"name": "another_good", "arguments": {}}
</tool_call>"#;
        let result = assert_parse(input, 2, Some("this is not json"));
        assert_eq!(first_tool_name(&result), "good_tool");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "another_good");
    }

    #[test]
    fn test_valid_json_but_missing_name_preserved_as_raw() {
        let input = r#"<tool_call>
{"arguments": {"key": "value"}, "description": "no name field"}
</tool_call>"#;
        assert_raw_preserved(input);
        let result = parse_tool_calls(input);
        assert!(result.text.contains("no name field"));
    }

    #[test]
    fn test_valid_json_array_not_object_preserved_as_raw() {
        let input = "<tool_call>\n[1, 2, 3]\n</tool_call>";
        assert_raw_preserved(input);
        let result = parse_tool_calls(input);
        assert!(result.text.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_valid_json_name_is_not_string_preserved_as_raw() {
        let input = r#"<tool_call>
{"name": 42, "arguments": {}}
</tool_call>"#;
        assert_raw_preserved(input);
    }

    #[test]
    fn test_text_between_multiple_tool_calls() {
        let input = r#"Before first.
<tool_call>
{"name": "tool_a", "arguments": {}}
</tool_call>
Middle text.
<tool_call>
{"name": "tool_b", "arguments": {}}
</tool_call>
After last."#;
        let result = assert_parse(input, 2, Some("Before first."));
        assert!(result.text.contains("Middle text."));
        assert!(result.text.contains("After last."));
    }

    #[test]
    fn test_nested_tool_call_tags() {
        // A <tool_call> tag nested inside another -- the inner one becomes
        // part of the content between the first open and first close.
        let input = r#"<tool_call>
<tool_call>
{"name": "inner", "arguments": {}}
</tool_call>
</tool_call>"#;
        let result = parse_tool_calls(input);
        // The parser finds the first <tool_call>, then looks for first </tool_call>.
        // Content between them: "\n<tool_call>\n{\"name\": \"inner\", \"arguments\": {}}\n"
        // This is not valid JSON (starts with <tool_call>), so it's preserved as raw text.
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
    }

    #[test]
    fn test_arguments_as_json_array() {
        let input = r#"<tool_call>
{"name": "batch_op", "arguments": [1, 2, 3]}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "batch_op");
        let first = result.tool_calls.first().unwrap();
        assert!(first.arguments.is_array());
        assert_eq!(first.arguments, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_arguments_with_special_chars_and_unicode() {
        let input = r#"<tool_call>
{"name": "translate", "arguments": {"text": "Caf\u00e9 \"quotes\" \\backslash", "emoji": "\ud83d\ude00"}}
</tool_call>"#;
        let result = assert_parse(input, 1, None);
        assert_eq!(first_tool_name(&result), "translate");
        let text_val = result
            .tool_calls
            .first()
            .unwrap()
            .arguments
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(text_val.contains("Caf\u{00e9}"));
        assert!(text_val.contains("\"quotes\""));
        assert!(text_val.contains("\\backslash"));
    }

    #[test]
    fn test_whitespace_only_content_between_tags() {
        let input = "<tool_call>\n   \n  \t  \n</tool_call>";
        assert_parse(input, 0, Some("<tool_call>"));
    }
}
