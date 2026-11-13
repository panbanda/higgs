//! Regular expressions that force a model to answer with a tool call.
//!
//! The pattern spells out a complete call in the model's own output format,
//! with the arguments constrained by each tool's JSON schema, so the text the
//! constrained model produces is exactly what [`super::ToolCallParser`]
//! extracts.
//!
//! Only formats whose arguments are JSON objects can be expressed this way:
//! the Qwen/Hermes JSON envelope, both Mistral spellings and Llama's JSON
//! convention. The Pythonic, brace-delimited and XML-tagged formats carry
//! arguments in a non-JSON syntax with no schema-derived pattern, so
//! [`forced_call_pattern`] reports them as unsupported.

use outlines_core::json_schema;

use super::ToolCallFormat;

/// Which tool the model must call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedChoice<'a> {
    /// Any one of the provided tools.
    AnyTool,
    /// The named tool.
    Function(&'a str),
}

/// Why a forcing pattern could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ForcedCallError {
    #[error(
        "forcing a tool call is not supported for the {} tool-call format; \
         it works with the qwen, mistral and llama formats",
        .0.name()
    )]
    UnsupportedFormat(ToolCallFormat),
    #[error("a tool call was required but the request provides no tools")]
    NoTools,
    #[error("tool_choice names function \"{0}\", which is not among the provided tools")]
    UnknownFunction(String),
    #[error("invalid parameters schema for tool \"{name}\": {reason}")]
    InvalidSchema { name: String, reason: String },
}

impl ToolCallFormat {
    /// Whether [`forced_call_pattern`] can force this format.
    pub const fn supports_forced_choice(self) -> bool {
        match self {
            Self::Qwen | Self::MistralJsonArray | Self::MistralNameArguments | Self::Llama => true,
            Self::GemmaPythonic
            | Self::FunctionGemma
            | Self::Gemma4
            | Self::Glm
            | Self::MiniMax
            | Self::Auto => false,
        }
    }
}

pub(super) struct ToolDefinition<'a> {
    pub(super) name: &'a str,
    parameters: Option<&'a serde_json::Value>,
}

pub(super) fn tool_definitions(tools: &[serde_json::Value]) -> Vec<ToolDefinition<'_>> {
    tools
        .iter()
        .filter_map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            Some(ToolDefinition {
                name: function.get("name")?.as_str()?,
                parameters: function.get("parameters").filter(|value| value.is_object()),
            })
        })
        .collect()
}

/// Escape `text` so it matches itself literally.
fn literal(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(
            character,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn arguments_pattern(tool: &ToolDefinition<'_>) -> Result<String, ForcedCallError> {
    let Some(parameters) = tool.parameters else {
        return Ok(literal("{}"));
    };
    let pattern =
        json_schema::regex_from_str(&parameters.to_string(), None, None).map_err(|error| {
            ForcedCallError::InvalidSchema {
                name: tool.name.to_owned(),
                reason: error.to_string(),
            }
        })?;
    Ok(format!("(?:{pattern})"))
}

fn json_string_literal(text: &str) -> String {
    literal(&serde_json::Value::String(text.to_owned()).to_string())
}

/// `{"name": "NAME", "<arguments_key>": ARGUMENTS}`
fn json_call_pattern(
    tool: &ToolDefinition<'_>,
    arguments_key: &str,
) -> Result<String, ForcedCallError> {
    Ok(format!(
        "{}{}{}{}{}",
        literal("{\"name\": "),
        json_string_literal(tool.name),
        literal(&format!(", \"{arguments_key}\": ")),
        arguments_pattern(tool)?,
        literal("}"),
    ))
}

fn alternatives(patterns: &[String]) -> String {
    format!("(?:{})", patterns.join("|"))
}

fn repeated(single: &str, separator: &str, allow_parallel: bool) -> String {
    if allow_parallel {
        format!("{single}(?:{separator}{single})*")
    } else {
        single.to_owned()
    }
}

/// Build a regular expression that only matches text containing the selected
/// tool call(s) in `format`.
///
/// With `allow_parallel` the model may emit several calls (where the format has
/// a way to write them); without it, exactly one. Llama's JSON convention has
/// no way to write more than one call, so it always yields exactly one.
pub fn forced_call_pattern(
    format: ToolCallFormat,
    tools: &[serde_json::Value],
    choice: ForcedChoice<'_>,
    allow_parallel: bool,
) -> Result<String, ForcedCallError> {
    if !format.supports_forced_choice() {
        return Err(ForcedCallError::UnsupportedFormat(format));
    }
    let definitions = tool_definitions(tools);
    let selected: Vec<&ToolDefinition<'_>> = match choice {
        ForcedChoice::AnyTool => definitions.iter().collect(),
        ForcedChoice::Function(name) => definitions
            .iter()
            .filter(|definition| definition.name == name)
            .collect(),
    };
    if selected.is_empty() {
        return Err(match choice {
            ForcedChoice::Function(name) if !definitions.is_empty() => {
                ForcedCallError::UnknownFunction(name.to_owned())
            }
            ForcedChoice::Function(_) | ForcedChoice::AnyTool => ForcedCallError::NoTools,
        });
    }

    let per_tool = |build: &dyn Fn(&ToolDefinition<'_>) -> Result<String, ForcedCallError>| {
        selected
            .iter()
            .map(|definition| build(definition))
            .collect::<Result<Vec<_>, _>>()
            .map(|patterns| alternatives(&patterns))
    };

    match format {
        ToolCallFormat::Qwen => {
            let call = per_tool(&|tool| {
                Ok(format!(
                    "{}{}{}",
                    literal("<tool_call>\n"),
                    json_call_pattern(tool, "arguments")?,
                    literal("\n</tool_call>"),
                ))
            })?;
            Ok(repeated(&call, "\n?", allow_parallel))
        }
        ToolCallFormat::MistralJsonArray => {
            let call = per_tool(&|tool| json_call_pattern(tool, "arguments"))?;
            Ok(format!(
                "{}{}{}",
                literal("[TOOL_CALLS] ["),
                repeated(&call, ", ", allow_parallel),
                literal("]"),
            ))
        }
        ToolCallFormat::MistralNameArguments => {
            let call = per_tool(&|tool| {
                Ok(format!(
                    "{}{}{}{}",
                    literal("[TOOL_CALLS]"),
                    literal(tool.name),
                    literal("[ARGS]"),
                    arguments_pattern(tool)?,
                ))
            })?;
            Ok(repeated(&call, "", allow_parallel))
        }
        ToolCallFormat::Llama => per_tool(&|tool| json_call_pattern(tool, "parameters")),
        ToolCallFormat::GemmaPythonic
        | ToolCallFormat::FunctionGemma
        | ToolCallFormat::Gemma4
        | ToolCallFormat::Glm
        | ToolCallFormat::MiniMax
        | ToolCallFormat::Auto => Err(ForcedCallError::UnsupportedFormat(format)),
    }
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrained::ConstrainedGenerator;
    use crate::tool_format::ToolCallParser;
    use outlines_core::vocabulary::Vocabulary;

    fn tools() -> Vec<serde_json::Value> {
        serde_json::json!([
            {"type": "function", "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["city"]
                }
            }},
            {"type": "function", "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {"zone": {"type": "string"}},
                    "required": ["zone"]
                }
            }},
            {"type": "function", "function": {"name": "ping"}}
        ])
        .as_array()
        .unwrap()
        .clone()
    }

    /// One token per printable ASCII byte, plus two-byte tokens that start
    /// with a structural character.
    ///
    /// `outlines-core` drops a single-byte token that continues a text which
    /// already matched the pattern, so a pattern whose later calls follow a
    /// complete call is only reachable through multi-byte tokens, as it is
    /// with a real tokenizer vocabulary.
    fn character_vocabulary() -> (Vocabulary, std::collections::HashMap<Vec<u8>, u32>) {
        let mut vocabulary = Vocabulary::new(0);
        let mut ids = std::collections::HashMap::new();
        let mut next_id = 1u32;
        let printable: Vec<u8> = std::iter::once(b'\n').chain(0x20..=0x7e).collect();
        let mut tokens: Vec<Vec<u8>> = printable.iter().map(|byte| vec![*byte]).collect();
        for first in [b'\n', b'[', b'<', b'{', b' '] {
            for second in &printable {
                tokens.push(vec![first, *second]);
            }
        }
        for token in tokens {
            vocabulary.try_insert(token.clone(), next_id).unwrap();
            ids.insert(token, next_id);
            next_id += 1;
        }
        (vocabulary, ids)
    }

    fn constraint_accepts(pattern: &str, text: &str) -> bool {
        let (vocabulary, ids) = character_vocabulary();
        let mut generator = ConstrainedGenerator::from_regex(pattern, &vocabulary).unwrap();
        let bytes = text.as_bytes();
        let mut position = 0;
        while position < bytes.len() {
            let allowed = generator.allowed_token_ids().unwrap_or_default();
            let candidates = [2usize, 1];
            let step = candidates.iter().find_map(|width| {
                let token = bytes.get(position..position + width)?;
                let id = ids.get(token)?;
                allowed.contains(id).then_some((*id, *width))
            });
            let Some((id, width)) = step else {
                return false;
            };
            assert!(generator.advance(id));
            position += width;
        }
        generator.is_finished()
    }

    fn regex_accepts(pattern: &str, text: &str) -> bool {
        regex::Regex::new(&format!("^(?:{pattern})$"))
            .unwrap()
            .is_match(text)
    }

    fn accepts(pattern: &str, text: &str) -> bool {
        let by_regex = regex_accepts(pattern, text);
        assert_eq!(
            by_regex,
            constraint_accepts(pattern, text),
            "regex and constraint disagree on {text:?}"
        );
        by_regex
    }

    struct Case {
        format: ToolCallFormat,
        choice: ForcedChoice<'static>,
        parallel: bool,
        valid: &'static [&'static str],
        invalid: &'static [&'static str],
    }

    fn cases() -> Vec<Case> {
        vec![
            Case {
                format: ToolCallFormat::Qwen,
                choice: ForcedChoice::AnyTool,
                parallel: false,
                valid: &[
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>",
                ],
                invalid: &[
                    "hello",
                    "<tool_call>\n{\"name\": \"other\", \"arguments\": {}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": 3}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>",
                ],
            },
            Case {
                format: ToolCallFormat::Qwen,
                choice: ForcedChoice::AnyTool,
                parallel: true,
                valid: &[
                    "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"UTC\"}}\n</tool_call>",
                ],
                invalid: &[
                    "",
                    "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>\nthanks",
                ],
            },
            Case {
                format: ToolCallFormat::Qwen,
                choice: ForcedChoice::Function("get_weather"),
                parallel: true,
                valid: &[
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": 3}}\n</tool_call>",
                ],
                invalid: &[
                    "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"UTC\"}}\n</tool_call>",
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\", \"days\": \"x\"}}\n</tool_call>",
                ],
            },
            Case {
                format: ToolCallFormat::MistralJsonArray,
                choice: ForcedChoice::AnyTool,
                parallel: true,
                valid: &[
                    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
                    "[TOOL_CALLS] [{\"name\": \"ping\", \"arguments\": {}}, {\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}]",
                ],
                invalid: &[
                    "[TOOL_CALLS] []",
                    "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": 1}}]",
                    "plain",
                ],
            },
            Case {
                format: ToolCallFormat::MistralJsonArray,
                choice: ForcedChoice::Function("get_time"),
                parallel: false,
                valid: &[
                    "[TOOL_CALLS] [{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}]",
                ],
                invalid: &[
                    "[TOOL_CALLS] [{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}, {\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}]",
                    "[TOOL_CALLS] [{\"name\": \"ping\", \"arguments\": {}}]",
                ],
            },
            Case {
                format: ToolCallFormat::MistralNameArguments,
                choice: ForcedChoice::AnyTool,
                parallel: true,
                valid: &[
                    "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}",
                    "[TOOL_CALLS]ping[ARGS]{}[TOOL_CALLS]get_time[ARGS]{\"zone\": \"CET\"}",
                ],
                invalid: &[
                    "[TOOL_CALLS]get_weather{\"city\": \"Paris\"}",
                    "[TOOL_CALLS]nope[ARGS]{}",
                ],
            },
            Case {
                format: ToolCallFormat::Llama,
                choice: ForcedChoice::AnyTool,
                parallel: true,
                valid: &[
                    "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
                    "{\"name\": \"ping\", \"parameters\": {}}",
                ],
                invalid: &[
                    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
                    "{\"name\": \"ping\", \"parameters\": {}}; {\"name\": \"ping\", \"parameters\": {}}",
                    "I will call it",
                ],
            },
        ]
    }

    #[test]
    fn valid_calls_are_accepted_and_invalid_ones_rejected_per_format() {
        let tools = tools();
        for case in cases() {
            let pattern =
                forced_call_pattern(case.format, &tools, case.choice, case.parallel).unwrap();
            for text in case.valid {
                assert!(
                    accepts(&pattern, text),
                    "{:?} {:?} should accept {text:?}",
                    case.format,
                    case.choice
                );
            }
            for text in case.invalid {
                assert!(
                    !accepts(&pattern, text),
                    "{:?} {:?} should reject {text:?}",
                    case.format,
                    case.choice
                );
            }
        }
    }

    #[test]
    fn accepted_text_is_extracted_by_the_parser_of_the_same_format() {
        let tools = tools();
        for case in cases() {
            let parser = ToolCallParser::new(case.format, Some(&tools));
            for text in case.valid {
                let parsed = parser.parse(text);
                assert!(
                    !parsed.tool_calls.is_empty(),
                    "{:?} produced no call from {text:?}",
                    case.format
                );
                assert!(
                    parsed.text.is_empty(),
                    "{:?} left text {:?}",
                    case.format,
                    parsed.text
                );
            }
        }
    }

    #[test]
    fn unsupported_formats_name_themselves() {
        let tools = tools();
        for format in [
            ToolCallFormat::GemmaPythonic,
            ToolCallFormat::FunctionGemma,
            ToolCallFormat::Gemma4,
            ToolCallFormat::Glm,
            ToolCallFormat::MiniMax,
            ToolCallFormat::Auto,
        ] {
            assert!(!format.supports_forced_choice());
            let error =
                forced_call_pattern(format, &tools, ForcedChoice::AnyTool, true).unwrap_err();
            assert_eq!(error, ForcedCallError::UnsupportedFormat(format));
            assert!(error.to_string().contains(format.name()), "{error}");
        }
    }

    #[test]
    fn missing_tools_and_unknown_functions_are_reported() {
        assert_eq!(
            forced_call_pattern(ToolCallFormat::Qwen, &[], ForcedChoice::AnyTool, true),
            Err(ForcedCallError::NoTools)
        );
        assert_eq!(
            forced_call_pattern(
                ToolCallFormat::Qwen,
                &tools(),
                ForcedChoice::Function("missing"),
                true
            ),
            Err(ForcedCallError::UnknownFunction("missing".to_owned()))
        );
    }

    #[test]
    fn unsatisfiable_parameter_schema_is_reported() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "bad", "parameters": {"type": "bogus"}}
        })];
        let error = forced_call_pattern(ToolCallFormat::Qwen, &tools, ForcedChoice::AnyTool, true)
            .unwrap_err();
        assert!(
            matches!(error, ForcedCallError::InvalidSchema { .. }),
            "{error}"
        );
    }

    #[test]
    fn tool_names_with_regex_metacharacters_are_matched_literally() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "a.b", "parameters": {"type": "object", "properties": {}}}
        })];
        let pattern = forced_call_pattern(
            ToolCallFormat::MistralNameArguments,
            &tools,
            ForcedChoice::AnyTool,
            false,
        )
        .unwrap();
        assert!(accepts(&pattern, "[TOOL_CALLS]a.b[ARGS]{}"));
        assert!(!accepts(&pattern, "[TOOL_CALLS]axb[ARGS]{}"));
    }
}
