use crate::types::openai::ReasoningConfig;

fn model_defaults_to_non_thinking(model_names: &[&str]) -> bool {
    model_names
        .iter()
        .any(|name| name.to_ascii_lowercase().contains("qwen3.6"))
}

pub(crate) fn effective_thinking_enabled(
    engine_default: bool,
    model_names: &[&str],
    reasoning: Option<&ReasoningConfig>,
) -> bool {
    if !engine_default {
        return false;
    }

    match reasoning.and_then(|r| r.effort.as_deref()) {
        Some(effort) if effort.eq_ignore_ascii_case("none") => false,
        Some(_) => true,
        None => !model_defaults_to_non_thinking(model_names),
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_qwen35_on() {
        assert!(effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            None,
        ));
    }

    #[test]
    fn defaults_qwen36_off_from_route_name() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.6-35B-A3B-4bit"],
            None,
        ));
    }

    #[test]
    fn defaults_qwen36_off_from_engine_name_even_when_aliased() {
        assert!(!effective_thinking_enabled(
            true,
            &["qwen", "mlx-community/Qwen3.6-35B-A3B-4bit"],
            None,
        ));
    }

    #[test]
    fn honors_reasoning_none() {
        assert!(!effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.5-foo"],
            Some(&ReasoningConfig {
                effort: Some("none".to_owned()),
            }),
        ));
    }

    #[test]
    fn honors_explicit_reasoning_request() {
        assert!(effective_thinking_enabled(
            true,
            &["mlx-community/Qwen3.6-35B-A3B-4bit"],
            Some(&ReasoningConfig {
                effort: Some("low".to_owned()),
            }),
        ));
    }
}
