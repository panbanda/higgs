use std::sync::LazyLock;

use regex::Regex;
use tracing::{info, warn};

use crate::router::RouteCandidate;
use crate::state::Engine;

#[allow(clippy::expect_used)]
static ROUTE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\{"route"\s*:\s*"([^"]+)"\}"#).expect("hardcoded regex is valid")
});

const TASK_INSTRUCTION: &str = "\
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
";

const FORMAT_PROMPT: &str = "\
Your task is to decide which route is best suit with user intent on the conversation \
in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, \
response with other route {\"route\": \"other\"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, \
use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats \
if you decide to match any route:
{\"route\": \"route_name\"}
";

fn build_prompt(routes: &[RouteCandidate], messages: &[serde_json::Value]) -> String {
    let route_defs: Vec<serde_json::Value> = routes
        .iter()
        .map(|r| serde_json::json!({"name": &r.name, "description": &r.description}))
        .collect();

    let non_system: Vec<&serde_json::Value> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) != Some("system"))
        .collect();

    #[allow(clippy::literal_string_with_formatting_args)]
    let prompt = TASK_INSTRUCTION
        .replace(
            "{routes}",
            &serde_json::to_string(&route_defs).unwrap_or_default(),
        )
        .replace(
            "{conversation}",
            &serde_json::to_string(&non_system).unwrap_or_default(),
        );

    format!("{prompt}{FORMAT_PROMPT}")
}

fn parse_route_name(text: &str, valid_names: &[&str]) -> Option<String> {
    // Try full JSON parse first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        if let Some(name) = v.get("route").and_then(|r| r.as_str()) {
            if name != "other" && valid_names.contains(&name) {
                return Some(name.to_owned());
            }
            return None;
        }
    }

    // Fallback: regex extraction from surrounding text
    let captures = ROUTE_REGEX.captures(text)?;
    let name = captures.get(1)?.as_str();
    (name != "other" && valid_names.contains(&name)).then(|| name.to_owned())
}

/// Classify a request using a local MLX engine.
///
/// Returns the name of the best-matching route, or `None` if classification
/// fails or returns "other".
pub fn classify_local(
    engine: &Engine,
    routes: &[RouteCandidate],
    messages: &[serde_json::Value],
) -> Option<String> {
    if routes.is_empty() || messages.is_empty() {
        return None;
    }

    let prompt = build_prompt(routes, messages);
    let valid_names: Vec<&str> = routes.iter().map(|r| r.name.as_str()).collect();

    info!(
        route_count = routes.len(),
        "auto-routing request via local engine"
    );

    let chat_messages = vec![higgs_engine::chat_template::ChatMessage {
        role: "user".to_owned(),
        content: prompt,
        tool_calls: None,
    }];

    let prompt_tokens = match engine.prepare_chat_prompt(&chat_messages, None) {
        Ok(tokens) => tokens,
        Err(e) => {
            warn!(error = %e, "auto-router prompt preparation failed");
            return None;
        }
    };

    let sampling = higgs_models::SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        min_p: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    let output = match engine.generate(&prompt_tokens, 64, &sampling, &[], false, None, None, None)
    {
        Ok(o) => o,
        Err(e) => {
            warn!(error = %e, "auto-router generation failed");
            return None;
        }
    };

    let result = parse_route_name(&output.text, &valid_names);

    if let Some(name) = &result {
        info!(route = %name, "auto-router selected route");
    } else {
        let truncated: String = output.text.chars().take(64).collect();
        warn!(
            response = %truncated,
            "auto-router returned no match, falling through to default"
        );
    }

    result
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn candidates() -> Vec<RouteCandidate> {
        vec![
            RouteCandidate {
                name: "code_gen".to_owned(),
                description: "code generation".to_owned(),
            },
            RouteCandidate {
                name: "summarize".to_owned(),
                description: "summarization".to_owned(),
            },
        ]
    }

    #[test]
    fn parse_clean_json() {
        let names = vec!["code_gen", "summarize"];
        assert_eq!(
            parse_route_name(r#"{"route": "code_gen"}"#, &names),
            Some("code_gen".to_owned())
        );
    }

    #[test]
    fn parse_other_returns_none() {
        let names = vec!["code_gen"];
        assert_eq!(parse_route_name(r#"{"route": "other"}"#, &names), None);
    }

    #[test]
    fn parse_unknown_name_returns_none() {
        let names = vec!["code_gen"];
        assert_eq!(parse_route_name(r#"{"route": "unknown"}"#, &names), None);
    }

    #[test]
    fn parse_with_preamble() {
        let names = vec!["code_gen", "summarize"];
        let text = "Based on the analysis, the best route is:\n{\"route\": \"summarize\"}";
        assert_eq!(parse_route_name(text, &names), Some("summarize".to_owned()));
    }

    #[test]
    fn parse_garbage_returns_none() {
        let names = vec!["code_gen"];
        assert_eq!(parse_route_name("not json at all", &names), None);
    }

    #[test]
    fn parse_empty_returns_none() {
        let names = vec!["code_gen"];
        assert_eq!(parse_route_name("", &names), None);
    }

    #[test]
    fn build_prompt_filters_system_messages() {
        let routes = candidates();
        let messages = vec![
            serde_json::json!({"role": "system", "content": "you are helpful"}),
            serde_json::json!({"role": "user", "content": "write code"}),
        ];
        let prompt = build_prompt(&routes, &messages);
        assert!(prompt.contains("write code"));
        assert!(!prompt.contains("you are helpful"));
        assert!(prompt.contains("code_gen"));
        assert!(prompt.contains("summarize"));
    }

    #[test]
    fn build_prompt_includes_all_routes() {
        let routes = candidates();
        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let prompt = build_prompt(&routes, &messages);
        assert!(prompt.contains("code generation"));
        assert!(prompt.contains("summarization"));
    }

    #[test]
    fn build_prompt_includes_conversation() {
        let routes = candidates();
        let messages = vec![
            serde_json::json!({"role": "user", "content": "fix this bug"}),
            serde_json::json!({"role": "assistant", "content": "sure"}),
            serde_json::json!({"role": "user", "content": "now optimize it"}),
        ];
        let prompt = build_prompt(&routes, &messages);
        assert!(prompt.contains("fix this bug"));
        assert!(prompt.contains("now optimize it"));
    }
}
