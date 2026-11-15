//! Pythonic tool calls, as Gemma 3 style prompts elicit them: keyword-argument
//! calls written as Python expressions.
//!
//! ````text
//! ```tool_code
//! get_weather(city="Paris", days=3)
//! ```
//! [get_weather(city="Paris"), get_time(zone="CET")]
//! ````
//!
//! A fenced `tool_code` block is a call wherever it appears. A bare
//! `[call, …]` list is only a call when it opens the response and every call
//! names a declared tool, so prose that merely starts with a bracket is left
//! alone.

use super::span::{Span, bracketed_span, leading_whitespace};
use super::{ScanContext, Unit};
use crate::tool_parser::ParsedToolCall;

pub(super) const FENCE_OPEN: &str = "```tool_code";
const FENCE_CLOSE: &str = "```";
const MAX_NESTING: usize = 64;

struct Cursor<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Cursor<'a> {
    const fn new(text: &'a str) -> Self {
        Self { text, position: 0 }
    }

    fn rest(&self) -> &'a str {
        self.text.get(self.position..).unwrap_or_default()
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    const fn advance(&mut self, bytes: usize) {
        self.position += bytes;
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.advance(character.len_utf8());
        Some(character)
    }

    fn skip_whitespace(&mut self) {
        self.advance(leading_whitespace(self.rest()));
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance(expected.len_utf8());
            true
        } else {
            false
        }
    }

    fn is_finished(&self) -> bool {
        self.rest().is_empty()
    }

    fn take_while(&mut self, accept: impl Fn(char) -> bool) -> &'a str {
        let rest = self.rest();
        let length = rest
            .find(|character: char| !accept(character))
            .unwrap_or(rest.len());
        self.advance(length);
        rest.get(..length).unwrap_or_default()
    }
}

fn is_name_character(character: char) -> bool {
    character.is_alphanumeric() || character == '_'
}

/// A dotted identifier such as `get_weather` or `tools.get_weather`.
fn parse_dotted_name(cursor: &mut Cursor<'_>) -> Option<String> {
    let mut name = String::new();
    loop {
        let segment = cursor.take_while(is_name_character);
        if segment.is_empty() || segment.starts_with(|character: char| character.is_ascii_digit()) {
            return None;
        }
        name.push_str(segment);
        if cursor.peek() == Some('.') {
            cursor.advance(1);
            name.push('.');
        } else {
            return Some(name);
        }
    }
}

fn parse_string(cursor: &mut Cursor<'_>) -> Option<String> {
    let quote = cursor.bump()?;
    let mut value = String::new();
    loop {
        match cursor.bump()? {
            character if character == quote => return Some(value),
            '\\' => match cursor.bump()? {
                'n' => value.push('\n'),
                't' => value.push('\t'),
                'r' => value.push('\r'),
                '0' => value.push('\0'),
                'u' => {
                    let digits = cursor.rest().get(..4)?;
                    let code = u32::from_str_radix(digits, 16).ok()?;
                    value.push(char::from_u32(code)?);
                    cursor.advance(4);
                }
                other => value.push(other),
            },
            character => value.push(character),
        }
    }
}

fn parse_number(cursor: &mut Cursor<'_>) -> Option<serde_json::Value> {
    let literal = cursor.take_while(|character| {
        character.is_ascii_digit() || matches!(character, '-' | '+' | '.' | 'e' | 'E')
    });
    if let Ok(integer) = literal.parse::<i64>() {
        return Some(serde_json::Value::from(integer));
    }
    if let Ok(unsigned) = literal.parse::<u64>() {
        return Some(serde_json::Value::from(unsigned));
    }
    let float = literal.parse::<f64>().ok()?;
    serde_json::Number::from_f64(float).map(serde_json::Value::Number)
}

fn parse_sequence(
    cursor: &mut Cursor<'_>,
    closer: char,
    depth: usize,
) -> Option<Vec<serde_json::Value>> {
    let mut items = Vec::new();
    loop {
        cursor.skip_whitespace();
        if cursor.eat(closer) {
            return Some(items);
        }
        items.push(parse_value(cursor, depth + 1)?);
        cursor.skip_whitespace();
        if !cursor.eat(',') {
            return cursor.eat(closer).then_some(items);
        }
    }
}

fn parse_dictionary(
    cursor: &mut Cursor<'_>,
    depth: usize,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut members = serde_json::Map::new();
    loop {
        cursor.skip_whitespace();
        if cursor.eat('}') {
            return Some(members);
        }
        if !matches!(cursor.peek(), Some('"' | '\'')) {
            return None;
        }
        let key = parse_string(cursor)?;
        cursor.skip_whitespace();
        if !cursor.eat(':') {
            return None;
        }
        members.insert(key, parse_value(cursor, depth + 1)?);
        cursor.skip_whitespace();
        if !cursor.eat(',') {
            return cursor.eat('}').then_some(members);
        }
    }
}

fn parse_value(cursor: &mut Cursor<'_>, depth: usize) -> Option<serde_json::Value> {
    if depth > MAX_NESTING {
        return None;
    }
    cursor.skip_whitespace();
    match cursor.peek()? {
        '"' | '\'' => parse_string(cursor).map(serde_json::Value::String),
        '[' => {
            cursor.advance(1);
            parse_sequence(cursor, ']', depth).map(serde_json::Value::Array)
        }
        '(' => {
            cursor.advance(1);
            parse_sequence(cursor, ')', depth).map(serde_json::Value::Array)
        }
        '{' => {
            cursor.advance(1);
            parse_dictionary(cursor, depth).map(serde_json::Value::Object)
        }
        character if character.is_ascii_digit() || matches!(character, '-' | '+' | '.') => {
            parse_number(cursor)
        }
        _ => match cursor.take_while(is_name_character) {
            "True" | "true" => Some(serde_json::Value::Bool(true)),
            "False" | "false" => Some(serde_json::Value::Bool(false)),
            "None" | "null" => Some(serde_json::Value::Null),
            _ => None,
        },
    }
}

fn parse_call(cursor: &mut Cursor<'_>) -> Option<ParsedToolCall> {
    cursor.skip_whitespace();
    let name = parse_dotted_name(cursor)?;
    cursor.skip_whitespace();
    if !cursor.eat('(') {
        return None;
    }
    let mut arguments = serde_json::Map::new();
    loop {
        cursor.skip_whitespace();
        if cursor.eat(')') {
            break;
        }
        let key = cursor.take_while(is_name_character);
        if key.is_empty() {
            return None;
        }
        cursor.skip_whitespace();
        if !cursor.eat('=') {
            return None;
        }
        arguments.insert(key.to_owned(), parse_value(cursor, 0)?);
        cursor.skip_whitespace();
        if !cursor.eat(',') {
            cursor.skip_whitespace();
            if !cursor.eat(')') {
                return None;
            }
            break;
        }
    }
    Some(ParsedToolCall {
        name,
        arguments: serde_json::Value::Object(arguments),
    })
}

/// Parse `call(…)` expressions separated by commas or line breaks, with an
/// optional surrounding `[…]`.
fn parse_calls(text: &str) -> Option<Vec<ParsedToolCall>> {
    let mut cursor = Cursor::new(text.trim());
    let bracketed = cursor.eat('[');
    let mut calls = Vec::new();
    loop {
        cursor.skip_whitespace();
        if bracketed && cursor.eat(']') {
            break;
        }
        if !bracketed && cursor.is_finished() {
            break;
        }
        calls.push(parse_call(&mut cursor)?);
        cursor.skip_whitespace();
        cursor.eat(',');
    }
    cursor.skip_whitespace();
    (cursor.is_finished() && !calls.is_empty()).then_some(calls)
}

/// Rename each call to its declared tool, or `None` if any call names an
/// undeclared one.
fn declared_calls(
    calls: Vec<ParsedToolCall>,
    context: &ScanContext<'_>,
) -> Option<Vec<ParsedToolCall>> {
    calls
        .into_iter()
        .map(|call| {
            let name = context.canonical_name(&call.name)?;
            Some(ParsedToolCall { name, ..call })
        })
        .collect()
}

/// Scan a ```` ```tool_code ```` block. `text` starts at the opening fence.
pub(super) fn scan_fence(text: &str, context: &ScanContext<'_>) -> Unit {
    let after_open = text.get(FENCE_OPEN.len()..).unwrap_or_default();
    let Some(close) = after_open.find(FENCE_CLOSE) else {
        return Unit::Incomplete;
    };
    let body = after_open.get(..close).unwrap_or_default();
    let consumed = FENCE_OPEN.len() + close + FENCE_CLOSE.len();
    parse_calls(body)
        .and_then(|calls| declared_calls(calls, context))
        .map_or(Unit::Text { consumed }, |calls| Unit::Calls {
            consumed,
            calls,
        })
}

/// Scan a bare `[call(…), …]` list. `text` starts at the opening `[`.
pub(super) fn scan_bare_list(text: &str, context: &ScanContext<'_>) -> Unit {
    let mut head = Cursor::new(text.get(1..).unwrap_or_default());
    head.skip_whitespace();
    let Some(name) = parse_dotted_name(&mut head) else {
        return if head.is_finished() {
            Unit::Incomplete
        } else {
            Unit::Text { consumed: 1 }
        };
    };
    head.skip_whitespace();
    if head.is_finished() {
        return Unit::Incomplete;
    }
    if head.peek() != Some('(') || context.canonical_name(&name).is_none() {
        return Unit::Text { consumed: 1 };
    }

    match bracketed_span(text, &['"', '\'']) {
        Span::Incomplete => Unit::Incomplete,
        Span::Invalid => Unit::Text { consumed: 1 },
        Span::Complete(end) => text
            .get(..end)
            .and_then(parse_calls)
            .and_then(|calls| declared_calls(calls, context))
            .map_or(Unit::Text { consumed: end }, |calls| Unit::Calls {
                consumed: end,
                calls,
            }),
    }
}
