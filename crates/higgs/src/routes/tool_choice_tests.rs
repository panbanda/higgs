//! `tool_choice` and `parallel_tool_calls` across the Chat Completions,
//! Responses and Anthropic Messages routes, driven through the scriptable
//! stub engine.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use higgs_engine::tool_format::ToolCallFormat;
use serde_json::{Value, json};

use crate::test_support::{
    StubCall, StubEngine, app_with_stub, char_tokenizer, generation_output, post_json,
    progress_event, sse_json_events, token_event,
};

const QWEN_WEATHER: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
const QWEN_TIME: &str =
    "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>";
const MISTRAL_WEATHER: &str =
    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]";
const LLAMA_WEATHER: &str = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Chat,
    Responses,
    Anthropic,
}

const ROUTES: [Route; 3] = [Route::Chat, Route::Responses, Route::Anthropic];

fn parameters(property: &str) -> Value {
    json!({
        "type": "object",
        "properties": {property: {"type": "string"}},
        "required": [property]
    })
}

impl Route {
    const fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
            Self::Anthropic => "/v1/messages",
        }
    }

    fn tools(self) -> Value {
        match self {
            Self::Chat => json!([
                {"type": "function", "function": {"name": "get_weather", "parameters": parameters("city")}},
                {"type": "function", "function": {"name": "get_time", "parameters": parameters("zone")}},
            ]),
            Self::Responses => json!([
                {"type": "function", "name": "get_weather", "parameters": parameters("city")},
                {"type": "function", "name": "get_time", "parameters": parameters("zone")},
            ]),
            Self::Anthropic => json!([
                {"name": "get_weather", "description": "w", "input_schema": parameters("city")},
                {"name": "get_time", "description": "t", "input_schema": parameters("zone")},
            ]),
        }
    }

    /// The route's spelling of "the model must call some tool".
    fn required(self) -> Value {
        match self {
            Self::Chat | Self::Responses => json!("required"),
            Self::Anthropic => json!({"type": "any"}),
        }
    }

    /// The route's spelling of "the model must call this tool".
    fn named(self, name: &str) -> Value {
        match self {
            Self::Chat => json!({"type": "function", "function": {"name": name}}),
            Self::Responses => json!({"type": "function", "name": name}),
            Self::Anthropic => json!({"type": "tool", "name": name}),
        }
    }

    fn none(self) -> Value {
        match self {
            Self::Chat | Self::Responses => json!("none"),
            Self::Anthropic => json!({"type": "none"}),
        }
    }

    fn auto(self) -> Value {
        match self {
            Self::Chat | Self::Responses => json!("auto"),
            Self::Anthropic => json!({"type": "auto"}),
        }
    }

    fn request(self, tool_choice: Option<Value>, stream: bool) -> Value {
        let mut body = match self {
            Self::Chat => json!({
                "model": "stub",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.tools(),
            }),
            Self::Responses => json!({"model": "stub", "input": "weather?", "tools": self.tools()}),
            Self::Anthropic => json!({
                "model": "stub",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": self.tools(),
            }),
        };
        if let Some(choice) = tool_choice {
            body["tool_choice"] = choice;
        }
        if stream {
            body["stream"] = json!(true);
        }
        body
    }

    fn without_tools(self, tool_choice: Value) -> Value {
        let mut body = self.request(Some(tool_choice), false);
        if let Some(object) = body.as_object_mut() {
            object.remove("tools");
        }
        body
    }

    /// Allow at most one call, in the route's spelling.
    fn single_call_only(self, body: &mut Value) {
        match self {
            Self::Chat | Self::Responses => body["parallel_tool_calls"] = json!(false),
            Self::Anthropic => {
                let mut choice = body
                    .get("tool_choice")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "auto"}));
                choice["disable_parallel_tool_use"] = json!(true);
                body["tool_choice"] = choice;
            }
        }
    }

    /// `(name, arguments)` of each call in a non-streaming response.
    fn calls(self, response: &Value) -> Vec<(String, Value)> {
        match self {
            Self::Chat => response["choices"][0]["message"]["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|call| {
                            let arguments: Value = serde_json::from_str(
                                call["function"]["arguments"].as_str().unwrap(),
                            )
                            .unwrap();
                            (
                                call["function"]["name"].as_str().unwrap().to_owned(),
                                arguments,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Self::Responses => response["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "function_call")
                .map(|item| {
                    let arguments: Value =
                        serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
                    (item["name"].as_str().unwrap().to_owned(), arguments)
                })
                .collect(),
            Self::Anthropic => response["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|block| block["type"] == "tool_use")
                .map(|block| {
                    (
                        block["name"].as_str().unwrap().to_owned(),
                        block["input"].clone(),
                    )
                })
                .collect(),
        }
    }

    /// Whether a non-streaming response reports that it ended on tool calls.
    fn ended_on_tool_calls(self, response: &Value) -> bool {
        match self {
            Self::Chat => response["choices"][0]["finish_reason"] == "tool_calls",
            Self::Responses => {
                response["status"] == "completed" && !self.calls(response).is_empty()
            }
            Self::Anthropic => response["stop_reason"] == "tool_use",
        }
    }

    /// `(name, arguments)` of each call in a streamed response body.
    fn streamed_calls(self, body: &str) -> Vec<(String, Value)> {
        let events = sse_json_events(body);
        match self {
            Self::Chat => events
                .iter()
                .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
                .flatten()
                .map(|call| {
                    let arguments: Value =
                        serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                            .unwrap();
                    (
                        call["function"]["name"].as_str().unwrap().to_owned(),
                        arguments,
                    )
                })
                .collect(),
            Self::Responses => events
                .iter()
                .filter(|event| {
                    event["type"] == "response.output_item.done"
                        && event["item"]["type"] == "function_call"
                })
                .map(|event| {
                    let arguments: Value =
                        serde_json::from_str(event["item"]["arguments"].as_str().unwrap()).unwrap();
                    (
                        event["item"]["name"].as_str().unwrap().to_owned(),
                        arguments,
                    )
                })
                .collect(),
            Self::Anthropic => {
                let starts = events
                    .iter()
                    .filter(|event| event["content_block"]["type"] == "tool_use");
                let deltas: Vec<&Value> = events
                    .iter()
                    .filter(|event| event["delta"]["type"] == "input_json_delta")
                    .collect();
                starts
                    .zip(deltas)
                    .map(|(start, delta)| {
                        let arguments: Value =
                            serde_json::from_str(delta["delta"]["partial_json"].as_str().unwrap())
                                .unwrap();
                        (
                            start["content_block"]["name"].as_str().unwrap().to_owned(),
                            arguments,
                        )
                    })
                    .collect()
            }
        }
    }

    fn streamed_end_on_tool_calls(self, body: &str) -> bool {
        let events = sse_json_events(body);
        match self {
            Self::Chat => events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == "tool_calls"),
            Self::Responses => events
                .iter()
                .any(|event| event["type"] == "response.completed"),
            Self::Anthropic => events.iter().any(|event| {
                event["type"] == "message_delta" && event["delta"]["stop_reason"] == "tool_use"
            }),
        }
    }
}

fn stub(format: ToolCallFormat, text: &str) -> StubEngine {
    let mut stub = StubEngine::named("stub");
    stub.tokenizer = char_tokenizer();
    stub.script.prompt_tokens = vec![1, 2, 3];
    stub.script.stamp_times = true;
    stub.script.tool_call_format = format;
    stub.script.generation = Some(generation_output(text, 3, 20));
    stub.script.stream = vec![progress_event(0, 3), token_event(text, 20, Some("stop"))];
    stub
}

async fn send(
    stub: StubEngine,
    route: Route,
    body: &Value,
) -> (http::StatusCode, String, StubEngine) {
    let (app, engine) = app_with_stub(stub);
    let (status, text) = post_json(&app, route.path(), body).await;
    (status, text, engine)
}

fn recorded(engine: &StubEngine) -> Vec<StubCall> {
    engine.recorded_calls()
}

fn weather_call() -> (String, Value) {
    ("get_weather".to_owned(), json!({"city": "Paris"}))
}

const FORCING_FORMATS: [(ToolCallFormat, &str); 3] = [
    (ToolCallFormat::Qwen, QWEN_WEATHER),
    (ToolCallFormat::MistralJsonArray, MISTRAL_WEATHER),
    (ToolCallFormat::Llama, LLAMA_WEATHER),
];

#[tokio::test]
async fn required_and_named_choices_force_a_valid_call_on_every_route() {
    for route in ROUTES {
        for (format, text) in FORCING_FORMATS {
            for choice in [route.required(), route.named("get_weather")] {
                for stream in [false, true] {
                    let context = format!("{route:?} {format:?} {choice} stream={stream}");
                    let (status, body, engine) = send(
                        stub(format, text),
                        route,
                        &route.request(Some(choice.clone()), stream),
                    )
                    .await;
                    assert_eq!(status, http::StatusCode::OK, "{context}: {body}");

                    let calls = recorded(&engine);
                    assert_eq!(calls.len(), 1, "{context}");
                    assert!(calls[0].has_constraint, "{context}");
                    assert_eq!(
                        calls[0].constraint_accepts_output,
                        Some(true),
                        "{context}: constraint must accept a valid call"
                    );

                    if stream {
                        assert_eq!(
                            route.streamed_calls(&body),
                            vec![weather_call()],
                            "{context}"
                        );
                        assert!(route.streamed_end_on_tool_calls(&body), "{context}");
                    } else {
                        let json: Value = serde_json::from_str(&body).unwrap();
                        assert_eq!(route.calls(&json), vec![weather_call()], "{context}");
                        assert!(route.ended_on_tool_calls(&json), "{context}");
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn named_choice_rejects_a_call_to_another_tool() {
    for route in ROUTES {
        let (status, body, engine) = send(
            stub(ToolCallFormat::Qwen, QWEN_TIME),
            route,
            &route.request(Some(route.named("get_weather")), false),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK, "{route:?}: {body}");
        assert_eq!(
            recorded(&engine)[0].constraint_accepts_output,
            Some(false),
            "{route:?}"
        );

        let (_, _, any_engine) = send(
            stub(ToolCallFormat::Qwen, QWEN_TIME),
            route,
            &route.request(Some(route.required()), false),
        )
        .await;
        assert_eq!(
            recorded(&any_engine)[0].constraint_accepts_output,
            Some(true),
            "{route:?}"
        );
    }
}

#[tokio::test]
async fn invalid_arguments_for_the_named_tool_are_rejected_by_the_constraint() {
    let wrong_type =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": 7}}\n</tool_call>";
    let (_, _, engine) = send(
        stub(ToolCallFormat::Qwen, wrong_type),
        Route::Chat,
        &Route::Chat.request(Some(Route::Chat.named("get_weather")), false),
    )
    .await;
    assert_eq!(recorded(&engine)[0].constraint_accepts_output, Some(false));
}

#[tokio::test]
async fn auto_and_absent_choices_do_not_constrain() {
    for route in ROUTES {
        for choice in [None, Some(route.auto())] {
            let (status, body, engine) = send(
                stub(ToolCallFormat::Qwen, QWEN_WEATHER),
                route,
                &route.request(choice.clone(), false),
            )
            .await;
            assert_eq!(status, http::StatusCode::OK, "{route:?}: {body}");
            assert!(!recorded(&engine)[0].has_constraint, "{route:?} {choice:?}");
            assert_eq!(engine.recorded_prompt_tools(), vec![Some(2)], "{route:?}");
            let json: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(route.calls(&json), vec![weather_call()], "{route:?}");
        }
    }
}

#[tokio::test]
async fn none_hides_tools_and_leaves_the_text_alone() {
    for route in ROUTES {
        let (status, body, engine) = send(
            stub(ToolCallFormat::Qwen, QWEN_WEATHER),
            route,
            &route.request(Some(route.none()), false),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK, "{route:?}: {body}");
        assert!(!recorded(&engine)[0].has_constraint, "{route:?}");
        assert_eq!(engine.recorded_prompt_tools(), vec![None], "{route:?}");
        let json: Value = serde_json::from_str(&body).unwrap();
        assert!(route.calls(&json).is_empty(), "{route:?}: {body}");
        assert!(
            body.contains("get_weather"),
            "{route:?}: text stays visible"
        );
    }
}

#[tokio::test]
async fn forcing_a_call_without_tools_is_a_client_error() {
    for route in ROUTES {
        for choice in [route.required(), route.named("get_weather")] {
            let (status, body, engine) = send(
                stub(ToolCallFormat::Qwen, QWEN_WEATHER),
                route,
                &route.without_tools(choice.clone()),
            )
            .await;
            assert_eq!(
                status,
                http::StatusCode::BAD_REQUEST,
                "{route:?} {choice}: {body}"
            );
            assert!(recorded(&engine).is_empty(), "{route:?} {choice}");
        }
    }
}

#[tokio::test]
async fn naming_an_unknown_function_is_a_client_error() {
    for route in ROUTES {
        let (status, body, engine) = send(
            stub(ToolCallFormat::Qwen, QWEN_WEATHER),
            route,
            &route.request(Some(route.named("missing_tool")), false),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{route:?}: {body}");
        assert!(body.contains("missing_tool"), "{route:?}: {body}");
        assert!(recorded(&engine).is_empty(), "{route:?}");
    }
}

#[tokio::test]
async fn formats_that_cannot_be_constrained_say_so_and_still_parse_when_not_forced() {
    let formats = [
        (
            ToolCallFormat::Gemma4,
            "gemma 4",
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        ),
        (
            ToolCallFormat::GemmaPythonic,
            "gemma (pythonic)",
            "```tool_code\nget_weather(city=\"Paris\")\n```",
        ),
        (
            ToolCallFormat::Glm,
            "glm",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ),
        (ToolCallFormat::Auto, "auto-detected", QWEN_WEATHER),
    ];
    for route in ROUTES {
        for (format, label, text) in formats {
            let (status, body, engine) = send(
                stub(format, text),
                route,
                &route.request(Some(route.required()), false),
            )
            .await;
            assert_eq!(
                status,
                http::StatusCode::BAD_REQUEST,
                "{route:?} {format:?}: {body}"
            );
            assert!(body.contains(label), "{route:?} {format:?}: {body}");
            assert!(recorded(&engine).is_empty());

            let (plain_status, plain_body, _) =
                send(stub(format, text), route, &route.request(None, false)).await;
            assert_eq!(
                plain_status,
                http::StatusCode::OK,
                "{route:?} {format:?}: {plain_body}"
            );
            let json: Value = serde_json::from_str(&plain_body).unwrap();
            assert_eq!(
                route.calls(&json),
                vec![weather_call()],
                "{route:?} {format:?}"
            );
        }
    }
}

#[tokio::test]
async fn forced_choice_cannot_be_combined_with_another_constraint() {
    let mut body = Route::Chat.request(Some(json!("required")), false);
    body["guided_regex"] = json!("hello");
    let (status, text, engine) =
        send(stub(ToolCallFormat::Qwen, QWEN_WEATHER), Route::Chat, &body).await;
    assert_eq!(status, http::StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("cannot be combined"), "{text}");
    assert!(recorded(&engine).is_empty());
}

#[tokio::test]
async fn forcing_a_call_turns_thinking_off() {
    for route in [Route::Chat, Route::Anthropic] {
        for stream in [false, true] {
            let mut forced = stub(ToolCallFormat::Qwen, QWEN_WEATHER);
            forced.script.enable_thinking = true;
            let (_, _, engine) = send(
                forced,
                route,
                &route.request(Some(route.required()), stream),
            )
            .await;
            assert!(
                !recorded(&engine)[0].thinking_enabled,
                "{route:?} stream={stream}"
            );

            let mut open = stub(ToolCallFormat::Qwen, QWEN_WEATHER);
            open.script.enable_thinking = true;
            let (_, _, open_engine) = send(open, route, &route.request(None, stream)).await;
            assert!(
                recorded(&open_engine)[0].thinking_enabled,
                "{route:?} stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn parallel_tool_calls_false_returns_at_most_one_call() {
    let two_calls = format!("{QWEN_WEATHER}\n{QWEN_TIME}");
    for route in ROUTES {
        for stream in [false, true] {
            let context = format!("{route:?} stream={stream}");

            let (_, body, _) = send(
                stub(ToolCallFormat::Qwen, &two_calls),
                route,
                &route.request(None, stream),
            )
            .await;
            let all = if stream {
                route.streamed_calls(&body)
            } else {
                route.calls(&serde_json::from_str(&body).unwrap())
            };
            assert_eq!(all.len(), 2, "{context}: parallel by default");

            let mut limited = route.request(None, stream);
            route.single_call_only(&mut limited);
            let (limited_status, limited_body, _) =
                send(stub(ToolCallFormat::Qwen, &two_calls), route, &limited).await;
            assert_eq!(
                limited_status,
                http::StatusCode::OK,
                "{context}: {limited_body}"
            );
            let one = if stream {
                route.streamed_calls(&limited_body)
            } else {
                route.calls(&serde_json::from_str(&limited_body).unwrap())
            };
            assert_eq!(one, vec![weather_call()], "{context}");
        }
    }
}

#[tokio::test]
async fn parallel_tool_calls_false_with_a_forced_choice_constrains_to_exactly_one() {
    let two_calls = format!("{QWEN_WEATHER}\n{QWEN_TIME}");
    for route in ROUTES {
        let mut single = route.request(Some(route.required()), false);
        route.single_call_only(&mut single);
        let (status, body, engine) =
            send(stub(ToolCallFormat::Qwen, &two_calls), route, &single).await;
        assert_eq!(status, http::StatusCode::OK, "{route:?}: {body}");
        assert_eq!(
            recorded(&engine)[0].constraint_accepts_output,
            Some(false),
            "{route:?}: two calls must not satisfy a single-call constraint"
        );
        let json: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(route.calls(&json).len(), 1, "{route:?}");

        let (_, _, single_engine) =
            send(stub(ToolCallFormat::Qwen, QWEN_WEATHER), route, &single).await;
        assert_eq!(
            recorded(&single_engine)[0].constraint_accepts_output,
            Some(true),
            "{route:?}"
        );
    }
}

#[tokio::test]
async fn parallel_forced_choice_accepts_several_calls() {
    let (_, _, engine) = send(
        stub(ToolCallFormat::MistralJsonArray, "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}]"),
        Route::Chat,
        &Route::Chat.request(Some(json!("required")), false),
    )
    .await;
    assert_eq!(recorded(&engine)[0].constraint_accepts_output, Some(true));
}

#[tokio::test]
async fn each_format_is_parsed_from_the_model_output_when_streaming_and_not() {
    let cases = [
        (
            ToolCallFormat::MistralNameArguments,
            "Checking. [TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}",
        ),
        (
            ToolCallFormat::Llama,
            "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
        ),
        (
            ToolCallFormat::FunctionGemma,
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>}<end_function_call>",
        ),
        (
            ToolCallFormat::MiniMax,
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"city\">Paris</parameter></invoke></minimax:tool_call>",
        ),
    ];
    for route in ROUTES {
        for (format, text) in cases {
            for stream in [false, true] {
                let (status, body, _) =
                    send(stub(format, text), route, &route.request(None, stream)).await;
                assert_eq!(status, http::StatusCode::OK, "{route:?} {format:?}: {body}");
                let found = if stream {
                    route.streamed_calls(&body)
                } else {
                    route.calls(&serde_json::from_str(&body).unwrap())
                };
                assert_eq!(
                    found,
                    vec![weather_call()],
                    "{route:?} {format:?} stream={stream}"
                );
            }
        }
    }
}

#[tokio::test]
async fn malformed_tool_choices_are_client_errors() {
    for route in ROUTES {
        for choice in [json!("sometimes"), json!(3), json!({"type": "mystery"})] {
            let (status, body, engine) = send(
                stub(ToolCallFormat::Qwen, QWEN_WEATHER),
                route,
                &route.request(Some(choice.clone()), false),
            )
            .await;
            assert_eq!(
                status,
                http::StatusCode::BAD_REQUEST,
                "{route:?} {choice}: {body}"
            );
            assert!(recorded(&engine).is_empty(), "{route:?} {choice}");
        }
    }
}

#[tokio::test]
async fn anthropic_text_alongside_a_call_is_kept_as_a_text_block() {
    let text = format!("Let me check. {QWEN_WEATHER}");
    let (status, body, _) = send(
        stub(ToolCallFormat::Qwen, &text),
        Route::Anthropic,
        &Route::Anthropic.request(None, false),
    )
    .await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "Let me check.");
    assert_eq!(json["content"][1]["type"], "tool_use");
    assert!(
        json["content"][1]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_")
    );
    assert_eq!(json["stop_reason"], "tool_use");
}

#[tokio::test]
async fn anthropic_without_a_call_keeps_the_original_text_response() {
    let (status, body, _) = send(
        stub(ToolCallFormat::Qwen, "plain answer"),
        Route::Anthropic,
        &Route::Anthropic.request(None, false),
    )
    .await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let json: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["content"].as_array().unwrap().len(), 1);
    assert_eq!(json["content"][0]["text"], "plain answer");
    assert_eq!(json["stop_reason"], "end_turn");
}

#[tokio::test]
async fn anthropic_stream_orders_text_then_tool_use_blocks_then_stop() {
    let text = format!("Let me check. {QWEN_WEATHER}");
    let (status, body, _) = send(
        stub(ToolCallFormat::Qwen, &text),
        Route::Anthropic,
        &Route::Anthropic.request(None, true),
    )
    .await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let events = sse_json_events(&body);
    let sequence: Vec<(String, Option<u64>, Option<String>)> = events
        .iter()
        .map(|event| {
            let detail = event["content_block"]["type"]
                .as_str()
                .or_else(|| event["delta"]["type"].as_str())
                .or_else(|| event["delta"]["stop_reason"].as_str())
                .map(ToOwned::to_owned);
            (
                event["type"].as_str().unwrap().to_owned(),
                event["index"].as_u64(),
                detail,
            )
        })
        .collect();
    let expected = [
        ("message_start", None, None),
        ("content_block_start", Some(0), Some("text")),
        ("content_block_delta", Some(0), Some("text_delta")),
        ("content_block_stop", Some(0), None),
        ("content_block_start", Some(1), Some("tool_use")),
        ("content_block_delta", Some(1), Some("input_json_delta")),
        ("content_block_stop", Some(1), None),
        ("message_delta", None, Some("tool_use")),
        ("message_stop", None, None),
    ];
    let compact: Vec<(&str, Option<u64>, Option<&str>)> = sequence
        .iter()
        .map(|(kind, index, detail)| (kind.as_str(), *index, detail.as_deref()))
        .collect();
    assert_eq!(compact, expected);
    let start = events
        .iter()
        .find(|e| e["content_block"]["type"] == "tool_use")
        .unwrap();
    assert!(
        start["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_")
    );
    assert_eq!(start["content_block"]["input"], json!({}));
}

#[tokio::test]
async fn anthropic_accepts_tool_result_history() {
    let mut body = Route::Anthropic.request(None, false);
    body["messages"] = json!([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}
        ]}
    ]);
    let (status, text, _) = send(
        stub(ToolCallFormat::Qwen, "It is sunny."),
        Route::Anthropic,
        &body,
    )
    .await;
    assert_eq!(status, http::StatusCode::OK, "{text}");
    let json: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["content"][0]["text"], "It is sunny.");
    assert_eq!(json["stop_reason"], "end_turn");
}
