/// Map an `OpenAI` `finish_reason` to an Anthropic `stop_reason`.
pub fn openai_finish_to_anthropic_stop(finish_reason: &str) -> String {
    match finish_reason {
        "stop" => "end_turn".to_owned(),
        "length" => "max_tokens".to_owned(),
        "tool_calls" => "tool_use".to_owned(),
        other => other.to_owned(),
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finish_reason_mapping() {
        let cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("other", "other"),
            ("", ""),
            ("content_filter", "content_filter"),
            ("something_new", "something_new"),
        ];
        for (input, expected) in cases {
            assert_eq!(openai_finish_to_anthropic_stop(input), expected);
        }
    }
}
