//! Brace-delimited calls used by Gemma function-calling models:
//!
//! ```text
//! <|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>
//! <start_function_call>call:get_weather{city:<escape>Paris<escape>,days:3}<end_function_call>
//! ```
//!
//! Gemma 4 uses the first spelling, `FunctionGemma` the second; they differ only
//! in the call markers and the string delimiter. Keys are bare identifiers,
//! strings are wrapped in the delimiter (or JSON double quotes), and bare
//! values are numbers, booleans, `null`, or unquoted text up to the next
//! `,`/`}`/`]`.
//!
//! Some engines strip `<|"|>` as a control token before the text reaches the
//! parser, in which case string values arrive bare; they parse as long as they
//! contain no `,`, `}` or `]`.

use super::span::leading_whitespace;
use super::{ScanContext, Unit};
use crate::tool_parser::ParsedToolCall;

const MAX_NESTING: usize = 64;
const CALL_PREFIX: &str = "call:";

/// The markers and string delimiter of one brace-delimited dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Dialect {
    pub open: &'static str,
    pub close: &'static str,
    pub quote: &'static str,
}

pub(super) const GEMMA4: Dialect = Dialect {
    open: "<|tool_call>",
    close: "<tool_call|>",
    quote: "<|\"|>",
};

pub(super) const FUNCTION_GEMMA: Dialect = Dialect {
    open: "<start_function_call>",
    close: "<end_function_call>",
    quote: "<escape>",
};

/// Why a value could not be parsed.
enum Failure {
    /// The text ends before the value does.
    Incomplete,
    Invalid,
}

struct Reader<'a> {
    text: &'a str,
    position: usize,
    quote: &'static str,
}

impl<'a> Reader<'a> {
    fn rest(&self) -> &'a str {
        self.text.get(self.position..).unwrap_or_default()
    }

    const fn advance(&mut self, bytes: usize) {
        self.position += bytes;
    }

    fn skip_whitespace(&mut self) {
        self.advance(leading_whitespace(self.rest()));
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance(expected.len_utf8());
            true
        } else {
            false
        }
    }

    /// A string wrapped in the dialect's delimiter. `None` if not at one.
    fn delimited_string(&mut self) -> Option<Result<String, Failure>> {
        let after_open = self.rest().strip_prefix(self.quote)?;
        Some(
            after_open
                .find(self.quote)
                .map_or(Err(Failure::Incomplete), |length| {
                    let content = after_open.get(..length).unwrap_or_default().to_owned();
                    self.advance(self.quote.len() + length + self.quote.len());
                    Ok(content)
                }),
        )
    }

    /// A JSON double-quoted string. `None` if not at one.
    fn json_string(&mut self) -> Option<Result<String, Failure>> {
        let rest = self.rest();
        if !rest.starts_with('"') {
            return None;
        }
        let mut escaped = false;
        for (offset, character) in rest.char_indices().skip(1) {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                let literal = rest.get(..offset + 1).unwrap_or_default();
                let parsed = serde_json::from_str::<String>(literal).map_err(|_| Failure::Invalid);
                if parsed.is_ok() {
                    self.advance(offset + 1);
                }
                return Some(parsed);
            }
        }
        Some(Err(Failure::Incomplete))
    }

    fn key(&mut self) -> Result<String, Failure> {
        self.skip_whitespace();
        if let Some(delimited) = self.delimited_string() {
            return delimited;
        }
        if let Some(quoted) = self.json_string() {
            return quoted;
        }
        let rest = self.rest();
        let Some(length) = rest.find([':', ',', '{', '}', '[', ']']) else {
            return Err(Failure::Incomplete);
        };
        let key = rest.get(..length).unwrap_or_default().trim();
        if key.is_empty() || !rest.get(length..).is_some_and(|tail| tail.starts_with(':')) {
            return Err(Failure::Invalid);
        }
        self.advance(length);
        Ok(key.to_owned())
    }

    fn bare_value(&mut self) -> Result<serde_json::Value, Failure> {
        let rest = self.rest();
        let Some(length) = rest.find([',', '}', ']']) else {
            return Err(Failure::Incomplete);
        };
        let literal = rest.get(..length).unwrap_or_default().trim();
        if literal.is_empty() {
            return Err(Failure::Invalid);
        }
        self.advance(length);
        Ok(match literal {
            "true" | "True" => serde_json::Value::Bool(true),
            "false" | "False" => serde_json::Value::Bool(false),
            "null" | "None" => serde_json::Value::Null,
            other => serde_json::from_str::<serde_json::Value>(other)
                .ok()
                .filter(serde_json::Value::is_number)
                .unwrap_or_else(|| serde_json::Value::String(other.to_owned())),
        })
    }

    fn value(&mut self, depth: usize) -> Result<serde_json::Value, Failure> {
        if depth > MAX_NESTING {
            return Err(Failure::Invalid);
        }
        self.skip_whitespace();
        if let Some(delimited) = self.delimited_string() {
            return delimited.map(serde_json::Value::String);
        }
        if let Some(quoted) = self.json_string() {
            return quoted.map(serde_json::Value::String);
        }
        match self.peek() {
            None => Err(Failure::Incomplete),
            Some('{') => {
                self.advance(1);
                self.object_members(depth).map(serde_json::Value::Object)
            }
            Some('[') => {
                self.advance(1);
                self.array_items(depth).map(serde_json::Value::Array)
            }
            Some(_) => self.bare_value(),
        }
    }

    /// Members of an object whose `{` has been consumed.
    fn object_members(
        &mut self,
        depth: usize,
    ) -> Result<serde_json::Map<String, serde_json::Value>, Failure> {
        let mut members = serde_json::Map::new();
        loop {
            self.skip_whitespace();
            if self.eat('}') {
                return Ok(members);
            }
            if self.peek().is_none() {
                return Err(Failure::Incomplete);
            }
            let key = self.key()?;
            self.skip_whitespace();
            if !self.eat(':') {
                return Err(Failure::Invalid);
            }
            members.insert(key, self.value(depth + 1)?);
            self.skip_whitespace();
            if self.eat(',') {
                continue;
            }
            return match self.peek() {
                Some('}') => {
                    self.advance(1);
                    Ok(members)
                }
                Some(_) => Err(Failure::Invalid),
                None => Err(Failure::Incomplete),
            };
        }
    }

    /// Items of an array whose `[` has been consumed.
    fn array_items(&mut self, depth: usize) -> Result<Vec<serde_json::Value>, Failure> {
        let mut items = Vec::new();
        loop {
            self.skip_whitespace();
            if self.eat(']') {
                return Ok(items);
            }
            if self.peek().is_none() {
                return Err(Failure::Incomplete);
            }
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            if self.eat(',') {
                continue;
            }
            return match self.peek() {
                Some(']') => {
                    self.advance(1);
                    Ok(items)
                }
                Some(_) => Err(Failure::Invalid),
                None => Err(Failure::Incomplete),
            };
        }
    }
}

/// Scan one call. `text` starts at the dialect's opening marker.
pub(super) fn scan(dialect: Dialect, text: &str, context: &ScanContext<'_>) -> Unit {
    let rejected = Unit::Text {
        consumed: dialect.open.len(),
    };
    let after_open = text.get(dialect.open.len()..).unwrap_or_default();
    let skipped = leading_whitespace(after_open);
    let body = after_open.get(skipped..).unwrap_or_default();

    let Some(after_prefix) = body.strip_prefix(CALL_PREFIX) else {
        return if CALL_PREFIX.starts_with(body) {
            Unit::Incomplete
        } else {
            rejected
        };
    };
    let Some(brace_offset) = after_prefix.find(|character: char| {
        !(character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':'))
    }) else {
        return Unit::Incomplete;
    };
    let name = after_prefix.get(..brace_offset).unwrap_or_default();
    let arguments_text = after_prefix.get(brace_offset..).unwrap_or_default();
    if name.is_empty() || !arguments_text.starts_with('{') {
        return rejected;
    }

    let mut reader = Reader {
        text: arguments_text,
        position: 1,
        quote: dialect.quote,
    };
    let members = match reader.object_members(0) {
        Ok(members) => members,
        Err(Failure::Incomplete) => return Unit::Incomplete,
        Err(Failure::Invalid) => return rejected,
    };
    reader.skip_whitespace();
    let closer_text = reader.rest();
    let prefix_length = dialect.open.len() + skipped + CALL_PREFIX.len() + brace_offset;
    let arguments_length = reader.position;

    let closer_length = if closer_text.starts_with(dialect.close) {
        dialect.close.len()
    } else if dialect.close.starts_with(closer_text) {
        if !context.at_end {
            return Unit::Incomplete;
        }
        closer_text.len()
    } else {
        return rejected;
    };

    Unit::Calls {
        consumed: prefix_length + arguments_length + closer_length,
        calls: vec![ParsedToolCall {
            name: name.to_owned(),
            arguments: serde_json::Value::Object(members),
        }],
    }
}
