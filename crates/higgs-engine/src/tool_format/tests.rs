#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines
)]

use serde_json::{Value, json};

use super::*;

fn tools() -> Vec<Value> {
    vec![
        json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}
            }}
        }}),
        json!({"type": "function", "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}}
        }}),
        json!({"type": "function", "function": {
            "name": "search",
            "parameters": {"type": "object", "properties": {
                "q": {"type": "string"}, "limit": {"type": "integer"},
                "filters": {"type": "object"}, "tags": {"type": "array"}
            }}
        }}),
    ]
}

struct Case {
    label: &'static str,
    input: String,
    text: String,
    calls: Vec<(&'static str, Value)>,
}

fn case(label: &'static str, input: &str, text: &str, calls: Vec<(&'static str, Value)>) -> Case {
    Case {
        label,
        input: input.to_owned(),
        text: text.to_owned(),
        calls,
    }
}

fn call_pairs(calls: &[ParsedToolCall]) -> Vec<(String, Value)> {
    calls
        .iter()
        .map(|call| (call.name.clone(), call.arguments.clone()))
        .collect()
}

fn expected_pairs(case: &Case) -> Vec<(String, Value)> {
    case.calls
        .iter()
        .map(|(name, arguments)| ((*name).to_owned(), arguments.clone()))
        .collect()
}

/// Every way of cutting `text` into two chunks, plus one chunk per character.
fn chunkings(text: &str) -> Vec<Vec<String>> {
    let mut all: Vec<Vec<String>> = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .map(|cut| {
            vec![
                text.get(..cut).unwrap().to_owned(),
                text.get(cut..).unwrap().to_owned(),
            ]
        })
        .collect();
    all.push(text.chars().map(String::from).collect());
    all
}

fn stream(parser: &ToolCallParser, chunks: &[String]) -> (String, Vec<ParsedToolCall>) {
    let mut tracker = parser.tracker(true);
    let mut visible = String::new();
    let mut calls = Vec::new();
    for chunk in chunks {
        let output = tracker.process(chunk);
        visible.push_str(&output.visible);
        calls.extend(output.new_tool_calls);
    }
    let tail = tracker.flush();
    visible.push_str(&tail.visible);
    calls.extend(tail.new_tool_calls);
    assert_eq!(tracker.completed_count(), calls.len());
    (visible, calls)
}

fn check_cases(format: ToolCallFormat, cases: &[Case]) {
    let tool_list = tools();
    let parser = ToolCallParser::new(format, Some(&tool_list));
    for case in cases {
        let parsed = parser.parse(&case.input);
        assert_eq!(
            parsed.text, case.text,
            "{format:?} / {}: visible text",
            case.label
        );
        assert_eq!(
            call_pairs(&parsed.tool_calls),
            expected_pairs(case),
            "{format:?} / {}: calls",
            case.label
        );

        for chunks in chunkings(&case.input) {
            let (visible, calls) = stream(&parser, &chunks);
            assert_eq!(
                visible.trim(),
                case.text,
                "{format:?} / {}: streamed text for chunks {chunks:?}",
                case.label
            );
            assert_eq!(
                call_pairs(&calls),
                expected_pairs(case),
                "{format:?} / {}: streamed calls for chunks {chunks:?}",
                case.label
            );
        }
    }
}

fn mistral_cases() -> Vec<Case> {
    let nested = json!({
        "q": "say \"hi\" }{ [ ]",
        "filters": {"tags": ["a", "b"], "n": 1.5, "deep": {"x": [1, {"y": null}]}}
    });
    vec![
        case(
            "json array single",
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]"#,
            "",
            vec![("get_weather", json!({"city": "Paris"}))],
        ),
        case(
            "json array without space",
            r#"[TOOL_CALLS][{"name": "get_time", "arguments": {"zone": "CET"}}]"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "json array multiple",
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "get_time", "arguments": {"zone": "B"}}]"#,
            "",
            vec![
                ("get_weather", json!({"city": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "json array with call id and string arguments",
            r#"[TOOL_CALLS] [{"name": "get_time", "id": "abc123xyz", "arguments": "{\"zone\": \"UTC\"}"}]"#,
            "",
            vec![("get_time", json!({"zone": "UTC"}))],
        ),
        case(
            "name and ARGS",
            r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "days": 3}"#,
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "name and braces",
            r#"[TOOL_CALLS]get_time{"zone": "CET"}"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "name and ARGS multiple",
            r#"[TOOL_CALLS]get_weather[ARGS]{"city": "A"}[TOOL_CALLS]get_time[ARGS]{"zone": "B"}"#,
            "",
            vec![
                ("get_weather", json!({"city": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "nested and escaped arguments",
            &format!(
                "[TOOL_CALLS]search[ARGS]{}",
                serde_json::to_string(&nested).unwrap()
            ),
            "",
            vec![("search", nested)],
        ),
        case(
            "unicode",
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Zürich 東京 🌍"}}]"#,
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍"}))],
        ),
        case(
            "text before and after",
            "Let me check. [TOOL_CALLS]get_time[ARGS]{\"zone\": \"CET\"} Done.",
            "Let me check.  Done.",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "arguments missing",
            r#"[TOOL_CALLS] [{"name": "get_time"}]"#,
            "",
            vec![("get_time", json!({}))],
        ),
        case("no markup", "Just an answer.", "Just an answer.", vec![]),
        case(
            "truncated array stays text",
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": "#,
            r#"[TOOL_CALLS] [{"name": "get_weather", "arguments":"#,
            vec![],
        ),
        case(
            "marker alone stays text",
            "[TOOL_CALLS]",
            "[TOOL_CALLS]",
            vec![],
        ),
        case(
            "name without arguments stays text",
            "[TOOL_CALLS]bad_input",
            "[TOOL_CALLS]bad_input",
            vec![],
        ),
        case(
            "invalid json stays text",
            "[TOOL_CALLS] not json",
            "[TOOL_CALLS] not json",
            vec![],
        ),
        case(
            "array of non calls stays text",
            "[TOOL_CALLS] [1, 2, 3]",
            "[TOOL_CALLS] [1, 2, 3]",
            vec![],
        ),
        case(
            "bad call does not hide the next one",
            "[TOOL_CALLS]oops [TOOL_CALLS]get_time[ARGS]{\"zone\": \"X\"}",
            "[TOOL_CALLS]oops",
            vec![("get_time", json!({"zone": "X"}))],
        ),
    ]
}

fn llama_cases() -> Vec<Case> {
    let nested = json!({"q": "a \"quoted\" } brace", "filters": {"k": [1, {"z": false}]}});
    vec![
        case(
            "bare json",
            r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#,
            "",
            vec![("get_weather", json!({"city": "Paris"}))],
        ),
        case(
            "bare json with leading whitespace",
            "\n  {\"name\": \"get_time\", \"parameters\": {\"zone\": \"CET\"}}",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "arguments spelled arguments",
            r#"{"name": "get_time", "arguments": {"zone": "CET"}}"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "type function wrapper key",
            r#"{"type": "function", "name": "get_time", "parameters": {"zone": "CET"}}"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "python tag",
            r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}"#,
            "",
            vec![("get_weather", json!({"city": "Paris"}))],
        ),
        case(
            "function tags",
            r#"<function=get_weather>{"city": "Paris", "days": 2}</function>"#,
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 2}))],
        ),
        case(
            "function tags without arguments",
            "<function=get_time></function>",
            "",
            vec![("get_time", json!({}))],
        ),
        case(
            "semicolon separated",
            r#"{"name": "get_weather", "parameters": {"city": "A"}}; {"name": "get_time", "parameters": {"zone": "B"}}"#,
            "",
            vec![
                ("get_weather", json!({"city": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "multiple function tags",
            r#"<function=get_weather>{"city": "A"}</function><function=get_time>{"zone": "B"}</function>"#,
            "",
            vec![
                ("get_weather", json!({"city": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "text after a bare call",
            r#"{"name": "get_time", "parameters": {"zone": "CET"}} Hope that helps"#,
            "Hope that helps",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "text around function tags",
            "Calling now <function=get_time>{\"zone\": \"UTC\"}</function> done",
            "Calling now  done",
            vec![("get_time", json!({"zone": "UTC"}))],
        ),
        case(
            "nested and escaped",
            &format!(
                r#"{{"name": "search", "parameters": {}}}"#,
                serde_json::to_string(&nested).unwrap()
            ),
            "",
            vec![("search", nested)],
        ),
        case(
            "unicode",
            r#"{"name": "get_weather", "parameters": {"city": "Zürich 東京 🌍"}}"#,
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍"}))],
        ),
        case(
            "bare json after prose is text",
            r#"Sure! {"name": "get_time", "parameters": {"zone": "CET"}}"#,
            r#"Sure! {"name": "get_time", "parameters": {"zone": "CET"}}"#,
            vec![],
        ),
        case(
            "undeclared tool is text",
            r#"{"name": "launch_missiles", "parameters": {}}"#,
            r#"{"name": "launch_missiles", "parameters": {}}"#,
            vec![],
        ),
        case(
            "json answer is text",
            r#"{"answer": 42}"#,
            r#"{"answer": 42}"#,
            vec![],
        ),
        case(
            "name without parameters is text",
            r#"{"name": "get_time"}"#,
            r#"{"name": "get_time"}"#,
            vec![],
        ),
        case(
            "braces in prose",
            "{ not json at all",
            "{ not json at all",
            vec![],
        ),
        case(
            "truncated json is text",
            r#"{"name": "get_weather", "parameters": {"city": "#,
            r#"{"name": "get_weather", "parameters": {"city":"#,
            vec![],
        ),
        case(
            "bad function body is text",
            "<function=get_time>{oops}</function>",
            "<function=get_time>{oops}</function>",
            vec![],
        ),
        case(
            "unclosed function tag is text",
            "<function=get_time>{\"zone\": \"CET\"}",
            "<function=get_time>{\"zone\": \"CET\"}",
            vec![],
        ),
        case("plain answer", "It is sunny.", "It is sunny.", vec![]),
    ]
}

fn gemma_pythonic_cases() -> Vec<Case> {
    vec![
        case(
            "fenced single",
            "```tool_code\nget_weather(city=\"Paris\", days=3)\n```",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "fenced multiple lines",
            "```tool_code\nget_weather(city='A')\nget_time(zone=\"B\")\n```",
            "",
            vec![
                ("get_weather", json!({"city": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "fenced no arguments",
            "```tool_code\nget_time()\n```",
            "",
            vec![("get_time", json!({}))],
        ),
        case(
            "bare list",
            "[get_weather(city=\"Paris\"), get_time(zone=\"CET\")]",
            "",
            vec![
                ("get_weather", json!({"city": "Paris"})),
                ("get_time", json!({"zone": "CET"})),
            ],
        ),
        case(
            "nested python literals",
            "```tool_code\nsearch(q=\"say \\\"hi\\\")\", filters={\"tags\": [\"a\", 'b'], \"ok\": True, \"n\": None}, limit=-2, tags=(1, 2.5))\n```",
            "",
            vec![(
                "search",
                json!({
                    "q": "say \"hi\")",
                    "filters": {"tags": ["a", "b"], "ok": true, "n": null},
                    "limit": -2,
                    "tags": [1, 2.5]
                }),
            )],
        ),
        case(
            "unicode and escapes",
            "```tool_code\nget_weather(city=\"Zürich 東京 🌍\\n\\u00e9\")\n```",
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍\n\u{e9}"}))],
        ),
        case(
            "namespaced name",
            "```tool_code\ndefault_api.get_time(zone=\"UTC\")\n```",
            "",
            vec![("get_time", json!({"zone": "UTC"}))],
        ),
        case(
            "text before and after a fence",
            "Checking.\n```tool_code\nget_time(zone=\"UTC\")\n```\nOne moment.",
            "Checking.\n\nOne moment.",
            vec![("get_time", json!({"zone": "UTC"}))],
        ),
        case(
            "positional arguments are text",
            "```tool_code\nget_weather(\"Paris\")\n```",
            "```tool_code\nget_weather(\"Paris\")\n```",
            vec![],
        ),
        case(
            "unclosed fence is text",
            "```tool_code\nget_time(zone=\"UTC\")",
            "```tool_code\nget_time(zone=\"UTC\")",
            vec![],
        ),
        case(
            "undeclared fenced tool is text",
            "```tool_code\nlaunch(x=1)\n```",
            "```tool_code\nlaunch(x=1)\n```",
            vec![],
        ),
        case(
            "bare list after prose is text",
            "Sure [get_time(zone=\"UTC\")]",
            "Sure [get_time(zone=\"UTC\")]",
            vec![],
        ),
        case(
            "bracketed note is text",
            "[Note] this is fine",
            "[Note] this is fine",
            vec![],
        ),
        case(
            "undeclared bare list is text",
            "[launch(x=1)]",
            "[launch(x=1)]",
            vec![],
        ),
        case(
            "malformed bare list is text",
            "[get_time(zone=)]",
            "[get_time(zone=)]",
            vec![],
        ),
        case(
            "other code fence is text",
            "```python\nprint(1)\n```",
            "```python\nprint(1)\n```",
            vec![],
        ),
    ]
}

fn function_gemma_cases() -> Vec<Case> {
    vec![
        case(
            "single",
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>,days:3}<end_function_call>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "multiple with text",
            "Sure.<start_function_call>call:get_time{zone:<escape>A<escape>}<end_function_call><start_function_call>call:get_time{zone:<escape>B<escape>}<end_function_call> Done",
            "Sure. Done",
            vec![
                ("get_time", json!({"zone": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "separators inside strings",
            "<start_function_call>call:search{q:<escape>a, b} c ] d<escape>,limit:2}<end_function_call>",
            "",
            vec![("search", json!({"q": "a, b} c ] d", "limit": 2}))],
        ),
        case(
            "unterminated call is text",
            "<start_function_call>call:get_time{zone:<escape>CE",
            "<start_function_call>call:get_time{zone:<escape>CE",
            vec![],
        ),
    ]
}

fn gemma4_cases() -> Vec<Case> {
    vec![
        case(
            "single",
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "nested object and array",
            "<|tool_call>call:search{q:<|\"|>x<|\"|>,filters:{tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],ok:true,none:null,ratio:1.5},tags:[]}<tool_call|>",
            "",
            vec![(
                "search",
                json!({"q": "x", "filters": {"tags": ["a", "b"], "ok": true, "none": null, "ratio": 1.5}, "tags": []}),
            )],
        ),
        case(
            "separators and quotes inside delimited strings",
            "<|tool_call>call:search{q:<|\"|>he said \"hi\", {ok}]<|\"|>}<tool_call|>",
            "",
            vec![("search", json!({"q": "he said \"hi\", {ok}]"}))],
        ),
        case(
            "unicode",
            "<|tool_call>call:get_weather{city:<|\"|>Zürich 東京 🌍<|\"|>}<tool_call|>",
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍"}))],
        ),
        case(
            "bare string values when delimiters were stripped",
            "<|tool_call>call:get_weather{city:Paris,days:2}<tool_call|>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 2}))],
        ),
        case(
            "multiple with whitespace and text",
            "I will check.\n<|tool_call>\ncall:get_time{zone:<|\"|>A<|\"|>}\n<tool_call|><|tool_call>call:get_time{zone:<|\"|>B<|\"|>}<tool_call|>\nAll set.",
            "I will check.\n\nAll set.",
            vec![
                ("get_time", json!({"zone": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "missing closing marker is accepted at the end of the stream",
            "<|tool_call>call:get_time{zone:<|\"|>CET<|\"|>}",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "truncated is text",
            "<|tool_call>call:get_weather{city:<|\"|>Par",
            "<|tool_call>call:get_weather{city:<|\"|>Par",
            vec![],
        ),
        case(
            "garbage body is text",
            "<|tool_call>garbage<tool_call|>",
            "<|tool_call>garbage<tool_call|>",
            vec![],
        ),
        case(
            "missing colon after key is text",
            "<|tool_call>call:get_time{zone}<tool_call|>",
            "<|tool_call>call:get_time{zone}<tool_call|>",
            vec![],
        ),
    ]
}

fn glm_cases() -> Vec<Case> {
    vec![
        case(
            "single typed",
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "no arguments",
            "<tool_call>get_time</tool_call>",
            "",
            vec![("get_time", json!({}))],
        ),
        case(
            "multiple blocks with text",
            "Checking both.<tool_call>get_time<arg_key>zone</arg_key><arg_value>A</arg_value></tool_call>\n<tool_call>get_time<arg_key>zone</arg_key><arg_value>B</arg_value></tool_call>Bye",
            "Checking both.\nBye",
            vec![
                ("get_time", json!({"zone": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "structured and escaped values",
            "<tool_call>search<arg_key>q</arg_key><arg_value>say \"hi\" <b>bold</b></arg_value><arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value><arg_key>filters</arg_key><arg_value>{\"k\": [1, 2]}</arg_value></tool_call>",
            "",
            vec![(
                "search",
                json!({"q": "say \"hi\" <b>bold</b>", "tags": ["a", "b"], "filters": {"k": [1, 2]}}),
            )],
        ),
        case(
            "unicode",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Zürich 東京 🌍</arg_value></tool_call>",
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍"}))],
        ),
        case(
            "missing value is text",
            "<tool_call>get_weather<arg_key>city</arg_key></tool_call>",
            "<tool_call>get_weather<arg_key>city</arg_key></tool_call>",
            vec![],
        ),
        case(
            "unterminated is text",
            "<tool_call>get_time<arg_key>zone</arg_key><arg_value>CE",
            "<tool_call>get_time<arg_key>zone</arg_key><arg_value>CE",
            vec![],
        ),
        case(
            "json body is not glm",
            r#"<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"#,
            r#"<tool_call>{"name": "get_time", "arguments": {}}</tool_call>"#,
            vec![],
        ),
    ]
}

fn minimax_cases() -> Vec<Case> {
    vec![
        case(
            "single typed",
            "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Paris</parameter>\n<parameter name=\"days\">3</parameter>\n</invoke>\n</minimax:tool_call>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 3}))],
        ),
        case(
            "several invokes in one block",
            "<minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">A</parameter></invoke><invoke name=\"get_time\"><parameter name=\"zone\">B</parameter></invoke></minimax:tool_call>",
            "",
            vec![
                ("get_time", json!({"zone": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "two blocks with text",
            "Hold on.<minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">A</parameter></invoke></minimax:tool_call> and <minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">B</parameter></invoke></minimax:tool_call>.",
            "Hold on. and .",
            vec![
                ("get_time", json!({"zone": "A"})),
                ("get_time", json!({"zone": "B"})),
            ],
        ),
        case(
            "structured values",
            "<minimax:tool_call><invoke name=\"search\"><parameter name=\"q\">a \"b\" {c}</parameter><parameter name=\"filters\">{\"k\": [1, {\"z\": null}]}</parameter><parameter name=\"tags\">[\"x\", \"y\"]</parameter></invoke></minimax:tool_call>",
            "",
            vec![(
                "search",
                json!({"q": "a \"b\" {c}", "filters": {"k": [1, {"z": null}]}, "tags": ["x", "y"]}),
            )],
        ),
        case(
            "unicode",
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"city\">Zürich 東京 🌍</parameter></invoke></minimax:tool_call>",
            "",
            vec![("get_weather", json!({"city": "Zürich 東京 🌍"}))],
        ),
        case(
            "no invoke is text",
            "<minimax:tool_call>nothing here</minimax:tool_call>",
            "<minimax:tool_call>nothing here</minimax:tool_call>",
            vec![],
        ),
        case(
            "unterminated is text",
            "<minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">A",
            "<minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">A",
            vec![],
        ),
    ]
}

#[test]
fn mistral_parsing_and_streaming() {
    check_cases(ToolCallFormat::MistralJsonArray, &mistral_cases());
    check_cases(ToolCallFormat::MistralNameArguments, &mistral_cases());
}

#[test]
fn llama_parsing_and_streaming() {
    check_cases(ToolCallFormat::Llama, &llama_cases());
}

#[test]
fn gemma_pythonic_parsing_and_streaming() {
    check_cases(ToolCallFormat::GemmaPythonic, &gemma_pythonic_cases());
}

#[test]
fn function_gemma_parsing_and_streaming() {
    check_cases(ToolCallFormat::FunctionGemma, &function_gemma_cases());
}

#[test]
fn gemma4_parsing_and_streaming() {
    check_cases(ToolCallFormat::Gemma4, &gemma4_cases());
}

#[test]
fn glm_parsing_and_streaming() {
    check_cases(ToolCallFormat::Glm, &glm_cases());
}

#[test]
fn minimax_parsing_and_streaming() {
    check_cases(ToolCallFormat::MiniMax, &minimax_cases());
}

#[test]
fn auto_recognizes_formats_with_unique_markers() {
    let auto_cases = vec![
        case(
            "mistral",
            r#"[TOOL_CALLS]get_time[ARGS]{"zone": "CET"}"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "qwen json",
            r#"<tool_call>{"name": "get_time", "arguments": {"zone": "CET"}}</tool_call>"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "qwen xml",
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            "",
            vec![("get_weather", json!({"city": "Paris", "days": 2}))],
        ),
        case(
            "glm",
            "<tool_call>get_time<arg_key>zone</arg_key><arg_value>CET</arg_value></tool_call>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "minicpm",
            "<function name=\"get_time\"><param name=\"zone\">CET</param></function>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "python tag",
            r#"<|python_tag|>{"name": "get_time", "parameters": {"zone": "CET"}}"#,
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "llama function tags",
            "<function=get_time>{\"zone\": \"CET\"}</function>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "tool_code",
            "```tool_code\nget_time(zone=\"CET\")\n```",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "gemma 4",
            "<|tool_call>call:get_time{zone:<|\"|>CET<|\"|>}<tool_call|>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "function gemma",
            "<start_function_call>call:get_time{zone:<escape>CET<escape>}<end_function_call>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "minimax",
            "<minimax:tool_call><invoke name=\"get_time\"><parameter name=\"zone\">CET</parameter></invoke></minimax:tool_call>",
            "",
            vec![("get_time", json!({"zone": "CET"}))],
        ),
        case(
            "bare json is never a call",
            r#"{"name": "get_time", "parameters": {"zone": "CET"}}"#,
            r#"{"name": "get_time", "parameters": {"zone": "CET"}}"#,
            vec![],
        ),
        case(
            "bare list is never a call",
            "[get_time(zone=\"CET\")]",
            "[get_time(zone=\"CET\")]",
            vec![],
        ),
        case("plain text", "Nothing to see.", "Nothing to see.", vec![]),
    ];
    check_cases(ToolCallFormat::Auto, &auto_cases);
}

#[test]
fn qwen_format_is_the_original_parser() {
    let tool_list = tools();
    let schema = ToolSchema::from_tools(Some(&tool_list));
    let parser = ToolCallParser::new(ToolCallFormat::Qwen, Some(&tool_list));
    for input in [
        r#"hi <tool_call>{"name": "get_time", "arguments": {"zone": "X"}}</tool_call> bye"#,
        "<tool_call>\n<function=get_weather>\n<parameter=days>\n4\n</parameter>\n</function>\n</tool_call>",
        "<function name=\"get_time\"><param name=\"zone\">Z</param></function>",
        "<tool_call>not a call</tool_call>",
        "<tool_call>unterminated",
        "plain",
    ] {
        let expected = parse_tool_calls(input, schema.as_ref());
        assert_eq!(parser.parse(input), expected, "{input}");
    }
}

#[test]
fn qwen_streaming_holds_back_partial_markup() {
    let tool_list = tools();
    let parser = ToolCallParser::new(ToolCallFormat::Qwen, Some(&tool_list));
    let input = r#"Let me look. <tool_call>{"name": "get_time", "arguments": {"zone": "X"}}</tool_call> Done."#;
    for chunks in chunkings(input) {
        let (visible, calls) = stream(&parser, &chunks);
        assert_eq!(visible.trim(), "Let me look.  Done.", "{chunks:?}");
        assert_eq!(calls.len(), 1, "{chunks:?}");
    }
}

#[test]
fn inactive_tracker_passes_text_through() {
    let parser = ToolCallParser::new(ToolCallFormat::MistralJsonArray, None);
    let mut tracker = parser.tracker(false);
    let text = r#"[TOOL_CALLS]get_time[ARGS]{"zone": "X"}"#;
    assert_eq!(tracker.process(text).visible, text);
    assert!(tracker.flush().visible.is_empty());
    assert!(!tracker.has_tool_calls());
}

#[test]
fn streaming_releases_ordinary_text_promptly() {
    let tool_list = tools();
    for format in [
        ToolCallFormat::MistralJsonArray,
        ToolCallFormat::Llama,
        ToolCallFormat::GemmaPythonic,
        ToolCallFormat::Gemma4,
        ToolCallFormat::Glm,
        ToolCallFormat::MiniMax,
        ToolCallFormat::Auto,
    ] {
        let parser = ToolCallParser::new(format, Some(&tool_list));
        let mut tracker = parser.tracker(true);
        let released = tracker
            .process("The weather is sunny and warm today.")
            .visible;
        assert!(
            released.len() >= 20,
            "{format:?} held back too much: {released:?}"
        );
    }
}

#[test]
fn max_calls_keeps_only_the_first_call() {
    let tool_list = tools();
    let input = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "get_time", "arguments": {"zone": "B"}}]"#;
    let parser = ToolCallParser::new(ToolCallFormat::MistralJsonArray, Some(&tool_list))
        .with_max_calls(Some(1));
    let parsed = parser.parse(input);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].name, "get_weather");
    for chunks in chunkings(input) {
        let (_, calls) = stream(&parser, &chunks);
        assert_eq!(calls.len(), 1, "{chunks:?}");
    }

    let qwen = ToolCallParser::new(ToolCallFormat::Qwen, Some(&tool_list)).with_max_calls(Some(1));
    let two = r#"<tool_call>{"name": "get_time", "arguments": {}}</tool_call><tool_call>{"name": "get_time", "arguments": {"zone": "B"}}</tool_call>"#;
    assert_eq!(qwen.parse(two).tool_calls.len(), 1);
    let (_, calls) = stream(&qwen, &[two.to_owned()]);
    assert_eq!(calls.len(), 1);
}

#[test]
fn oversized_unfinished_call_is_released_as_text() {
    let tool_list = tools();
    let parser = ToolCallParser::new(ToolCallFormat::MistralJsonArray, Some(&tool_list));
    let mut tracker = parser.tracker(true);
    let filler = "x".repeat(MAX_INSIDE_TOOL_CALL_BYTES + 10);
    let mut visible = tracker.process("[TOOL_CALLS] [{\"name\": \"").visible;
    visible.push_str(&tracker.process(&filler).visible);
    visible.push_str(&tracker.flush().visible);
    assert!(visible.starts_with("[TOOL_CALLS] [{\"name\": \"x"));
    assert_eq!(
        visible.len(),
        "[TOOL_CALLS] [{\"name\": \"".len() + filler.len()
    );
    assert!(!tracker.has_tool_calls());
}

#[test]
fn detection_prefers_the_template_over_the_model_type() {
    let llama_template = "<|start_header_id|>system<|end_header_id|> Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.";
    let cases: [(Option<&str>, Option<&str>, ToolCallFormat); 16] = [
        (
            Some("qwen3"),
            Some("{{ '<tool_call>' }}"),
            ToolCallFormat::Qwen,
        ),
        (
            Some("qwen3_5"),
            Some("<tool_call>\n<function="),
            ToolCallFormat::Qwen,
        ),
        (Some("llama"), Some(llama_template), ToolCallFormat::Llama),
        (Some("llama"), Some("<|python_tag|>"), ToolCallFormat::Llama),
        (Some("llama"), Some("{{ messages }}"), ToolCallFormat::Llama),
        (
            Some("mistral"),
            Some("[TOOL_CALLS] [{{ x }}]"),
            ToolCallFormat::MistralJsonArray,
        ),
        (
            Some("mistral"),
            Some("[TOOL_CALLS]name[ARGS]"),
            ToolCallFormat::MistralNameArguments,
        ),
        (
            Some("glm4_moe"),
            Some("<tool_call><arg_key>"),
            ToolCallFormat::Glm,
        ),
        (
            Some("minimax_m2"),
            Some("<minimax:tool_call>"),
            ToolCallFormat::MiniMax,
        ),
        (
            Some("gemma3"),
            Some("<start_function_call>"),
            ToolCallFormat::FunctionGemma,
        ),
        (
            Some("gemma4"),
            Some("<|tool_call>call:"),
            ToolCallFormat::Gemma4,
        ),
        (Some("gemma3"), None, ToolCallFormat::GemmaPythonic),
        (Some("gemma2"), None, ToolCallFormat::Auto),
        (Some("phi3"), Some("{{ messages }}"), ToolCallFormat::Auto),
        (None, None, ToolCallFormat::Auto),
        (None, Some("```tool_code"), ToolCallFormat::GemmaPythonic),
    ];
    for (model_type, template, expected) in cases {
        assert_eq!(
            ToolCallFormat::detect(model_type, template),
            expected,
            "{model_type:?} {template:?}"
        );
    }
}
