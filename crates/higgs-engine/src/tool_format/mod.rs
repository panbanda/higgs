//! Per-model tool-call output formats.
//!
//! Each model family writes tool calls its own way. [`ToolCallFormat`] names
//! the dialect, is detected once when a model loads
//! ([`ToolCallFormat::detect`]), and drives both whole-text parsing
//! ([`ToolCallParser::parse`]) and incremental parsing of a token stream
//! ([`ToolCallTracker`]).
//!
//! Every format other than [`ToolCallFormat::Qwen`] is expressed as a list of
//! *openers*. Whenever one appears in the text, the owning scanner decides
//! whether the text from there on is a complete call, definitely not a call, or
//! not yet decidable. Parsing a whole response and streaming it chunk by chunk
//! run the same loop, so the two agree whatever the chunk boundaries are, and
//! no part of a half-received call is ever released as visible text.

mod braced;
pub mod forced;
mod llama;
mod mistral;
mod python;
mod span;
mod tagged;

use crate::tool_parser::{
    MAX_INSIDE_TOOL_CALL_BYTES, MINICPM_FUNCTION_OPEN, ParsedToolCall, StreamingToolCallTracker,
    StreamingToolOutput, TOOL_CALL_OPEN, ToolParseResult, ToolSchema, minicpm_function_end,
    parse_minicpm_function, parse_tool_calls,
};

const FUNCTION_CLOSE: &str = "</function>";

/// How a model family writes tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolCallFormat {
    /// `<tool_call>` JSON or XML bodies, and bare `MiniCPM` `<function>`
    /// blocks. The original behavior, kept exactly.
    Qwen,
    /// `[TOOL_CALLS] [{"name": …, "arguments": …}]`.
    MistralJsonArray,
    /// `[TOOL_CALLS]name[ARGS]{…}`.
    MistralNameArguments,
    /// Bare `{"name": …, "parameters": …}`, `<|python_tag|>` or
    /// `<function=name>{…}</function>`.
    Llama,
    /// Gemma 3 style Python calls in a `tool_code` fence or a bare
    /// `[call(arg=value)]` list.
    GemmaPythonic,
    /// `<start_function_call>call:name{…}<end_function_call>`.
    FunctionGemma,
    /// `<|tool_call>call:name{…}<tool_call|>`.
    Gemma4,
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
    Glm,
    /// `<minimax:tool_call><invoke name="…"><parameter name="…">…`.
    MiniMax,
    /// Family unknown: recognize any of the formats above that has a
    /// unique opening marker.
    Auto,
}

impl ToolCallFormat {
    /// Pick the format from the model's chat template and `model_type`.
    ///
    /// The template wins because it is what the model was taught to answer
    /// in. `model_type` decides when the template says nothing, but only for
    /// families whose output is unmistakable; Mistral, Llama and Gemma 3 write
    /// calls only when their template teaches them to, so a template that
    /// never mentions tools leaves them on [`Self::Auto`].
    pub fn detect(model_type: Option<&str>, chat_template: Option<&str>) -> Self {
        let from_model_type =
            |family_name: &str| Self::from_model_type(family_name, chat_template.is_none());
        chat_template
            .and_then(Self::from_template)
            .or_else(|| model_type.and_then(from_model_type))
            .unwrap_or(Self::Auto)
    }

    fn from_template(template: &str) -> Option<Self> {
        let mentions = |marker: &str| template.contains(marker);
        if mentions("<|tool_call>") {
            Some(Self::Gemma4)
        } else if mentions("<start_function_call>") {
            Some(Self::FunctionGemma)
        } else if mentions("[TOOL_CALLS]") {
            Some(if mentions("[ARGS]") {
                Self::MistralNameArguments
            } else {
                Self::MistralJsonArray
            })
        } else if mentions("minimax:tool_call") {
            Some(Self::MiniMax)
        } else if mentions("<arg_key>") {
            Some(Self::Glm)
        } else if mentions("<tool_call>") {
            Some(Self::Qwen)
        } else if mentions("<|python_tag|>")
            || (mentions("<|start_header_id|>") && mentions("\"parameters\""))
        {
            Some(Self::Llama)
        } else if mentions("```tool_code") {
            Some(Self::GemmaPythonic)
        } else {
            None
        }
    }

    fn from_model_type(model_type: &str, template_absent: bool) -> Option<Self> {
        let family = |prefix: &str| model_type.starts_with(prefix);
        if family("qwen") || family("minicpm") {
            Some(Self::Qwen)
        } else if family("glm") || family("chatglm") {
            Some(Self::Glm)
        } else if family("minimax") {
            Some(Self::MiniMax)
        } else if family("gemma4") {
            Some(Self::Gemma4)
        } else if !template_absent {
            None
        } else if family("gemma3") {
            Some(Self::GemmaPythonic)
        } else if family("mistral") || family("mixtral") || family("ministral") {
            Some(Self::MistralJsonArray)
        } else if family("llama") {
            Some(Self::Llama)
        } else {
            None
        }
    }

    /// Stable lowercase name for logs and error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen => "qwen",
            Self::MistralJsonArray => "mistral (json array)",
            Self::MistralNameArguments => "mistral (name and arguments)",
            Self::Llama => "llama",
            Self::GemmaPythonic => "gemma (pythonic)",
            Self::FunctionGemma => "function-gemma",
            Self::Gemma4 => "gemma 4",
            Self::Glm => "glm",
            Self::MiniMax => "minimax",
            Self::Auto => "auto-detected",
        }
    }

    const fn openers(self) -> &'static [Opener] {
        match self {
            Self::Qwen => &[],
            Self::MistralJsonArray | Self::MistralNameArguments => &[MISTRAL_OPENER],
            Self::Llama => &[PYTHON_TAG_OPENER, LLAMA_FUNCTION_OPENER, BARE_JSON_OPENER],
            Self::GemmaPythonic => &[TOOL_CODE_OPENER, BARE_LIST_OPENER],
            Self::FunctionGemma => &[FUNCTION_GEMMA_OPENER],
            Self::Gemma4 => &[GEMMA4_OPENER],
            Self::Glm => &[GLM_OPENER],
            Self::MiniMax => &[MINIMAX_OPENER],
            Self::Auto => &[
                QWEN_OPENER,
                MINICPM_OPENER,
                MISTRAL_OPENER,
                PYTHON_TAG_OPENER,
                LLAMA_FUNCTION_OPENER,
                TOOL_CODE_OPENER,
                FUNCTION_GEMMA_OPENER,
                GEMMA4_OPENER,
                MINIMAX_OPENER,
            ],
        }
    }
}

/// What kind of call unit an opener introduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitKind {
    QwenToolCall,
    GlmToolCall,
    MiniCpmFunction,
    Mistral,
    PythonTag,
    LlamaFunction,
    BareJson,
    ToolCodeFence,
    BareList,
    FunctionGemma,
    Gemma4,
    MiniMax,
}

impl UnitKind {
    /// Bare units have no marker of their own: they are a call only when they
    /// open the response (or follow another call).
    const fn is_bare(self) -> bool {
        matches!(self, Self::BareJson | Self::BareList)
    }
}

#[derive(Debug, Clone, Copy)]
struct Opener {
    token: &'static str,
    kind: UnitKind,
}

const QWEN_OPENER: Opener = Opener {
    token: TOOL_CALL_OPEN,
    kind: UnitKind::QwenToolCall,
};
const GLM_OPENER: Opener = Opener {
    token: TOOL_CALL_OPEN,
    kind: UnitKind::GlmToolCall,
};
const MINICPM_OPENER: Opener = Opener {
    token: MINICPM_FUNCTION_OPEN,
    kind: UnitKind::MiniCpmFunction,
};
const MISTRAL_OPENER: Opener = Opener {
    token: mistral::MARKER,
    kind: UnitKind::Mistral,
};
const PYTHON_TAG_OPENER: Opener = Opener {
    token: llama::PYTHON_TAG,
    kind: UnitKind::PythonTag,
};
const LLAMA_FUNCTION_OPENER: Opener = Opener {
    token: llama::FUNCTION_OPEN,
    kind: UnitKind::LlamaFunction,
};
const BARE_JSON_OPENER: Opener = Opener {
    token: "{",
    kind: UnitKind::BareJson,
};
const TOOL_CODE_OPENER: Opener = Opener {
    token: python::FENCE_OPEN,
    kind: UnitKind::ToolCodeFence,
};
const BARE_LIST_OPENER: Opener = Opener {
    token: "[",
    kind: UnitKind::BareList,
};
const FUNCTION_GEMMA_OPENER: Opener = Opener {
    token: braced::FUNCTION_GEMMA.open,
    kind: UnitKind::FunctionGemma,
};
const GEMMA4_OPENER: Opener = Opener {
    token: braced::GEMMA4.open,
    kind: UnitKind::Gemma4,
};
const MINIMAX_OPENER: Opener = Opener {
    token: tagged::MINIMAX_OPEN,
    kind: UnitKind::MiniMax,
};

/// What the scanner for one opener concluded about the text from the opener on.
#[derive(Debug)]
enum Unit {
    /// A complete call (or several); `consumed` bytes belong to it.
    Calls {
        consumed: usize,
        calls: Vec<ParsedToolCall>,
    },
    /// Not a call: release the first `consumed` bytes as visible text.
    Text { consumed: usize },
    /// Could still become a call once more text arrives.
    Incomplete,
}

/// Request-level facts a scanner needs.
struct ScanContext<'a> {
    schema: Option<&'a ToolSchema>,
    tool_names: &'a [String],
    /// The stream has ended, so a missing closing marker will not arrive.
    at_end: bool,
}

impl ScanContext<'_> {
    /// The declared tool that `name` refers to, if any.
    ///
    /// Models sometimes namespace names (`default_api.get_weather`); the last
    /// segment is accepted when it names a declared tool. When the request
    /// declares no tools the name is taken as written.
    fn canonical_name(&self, name: &str) -> Option<String> {
        if self.tool_names.is_empty() || self.tool_names.iter().any(|known| known == name) {
            return Some(name.to_owned());
        }
        let (_, last_segment) = name.rsplit_once('.')?;
        self.tool_names
            .iter()
            .any(|known| known == last_segment)
            .then(|| last_segment.to_owned())
    }
}

fn scan_minicpm(text: &str, context: &ScanContext<'_>) -> Unit {
    let Some(end) = minicpm_function_end(text) else {
        return Unit::Incomplete;
    };
    let consumed = end + FUNCTION_CLOSE.len();
    text.get(..end)
        .and_then(|block| parse_minicpm_function(block, context.schema))
        .map_or(Unit::Text { consumed }, |call| Unit::Calls {
            consumed,
            calls: vec![call],
        })
}

fn scan_unit(kind: UnitKind, text: &str, context: &ScanContext<'_>) -> Unit {
    match kind {
        UnitKind::QwenToolCall => tagged::scan_tool_call(text, context, true),
        UnitKind::GlmToolCall => tagged::scan_tool_call(text, context, false),
        UnitKind::MiniCpmFunction => scan_minicpm(text, context),
        UnitKind::Mistral => mistral::scan(text, context),
        UnitKind::PythonTag => llama::scan_python_tag(text, context),
        UnitKind::LlamaFunction => llama::scan_function(text, context),
        UnitKind::BareJson => llama::scan_bare_json(text, context),
        UnitKind::ToolCodeFence => python::scan_fence(text, context),
        UnitKind::BareList => python::scan_bare_list(text, context),
        UnitKind::FunctionGemma => braced::scan(braced::FUNCTION_GEMMA, text, context),
        UnitKind::Gemma4 => braced::scan(braced::GEMMA4, text, context),
        UnitKind::MiniMax => tagged::scan_minimax(text, context),
    }
}

/// The earliest opener in `text`, as `(offset, opener)`.
///
/// A bare opener only counts when nothing but whitespace precedes it in the
/// buffer and `bare_allowed` says nothing but whitespace and calls preceded the
/// buffer.
fn find_opener(
    openers: &'static [Opener],
    text: &str,
    bare_allowed: bool,
) -> Option<(usize, Opener)> {
    let first_content = text.len() - text.trim_start().len();
    openers
        .iter()
        .filter_map(|opener| {
            let offset = if opener.kind.is_bare() {
                (bare_allowed && text.get(first_content..)?.starts_with(opener.token))
                    .then_some(first_content)?
            } else {
                text.find(opener.token)?
            };
            Some((offset, *opener))
        })
        .min_by_key(|(offset, _)| *offset)
}

/// Length of the longest suffix of `text` that is a proper prefix of an
/// opener: those bytes might be the start of a marker split across chunks.
fn held_suffix_length(openers: &[Opener], text: &str) -> usize {
    openers
        .iter()
        .filter(|opener| !opener.kind.is_bare())
        .flat_map(|opener| {
            (1..opener.token.len()).filter_map(|length| {
                let prefix = opener.token.get(..length)?;
                text.ends_with(prefix).then_some(length)
            })
        })
        .max()
        .unwrap_or(0)
}

/// Incremental parser for every format except [`ToolCallFormat::Qwen`].
struct ScanningTracker {
    format: ToolCallFormat,
    openers: &'static [Opener],
    buffer: String,
    bare_allowed: bool,
    completed_count: usize,
    active: bool,
    schema: Option<ToolSchema>,
    tool_names: Vec<String>,
}

impl ScanningTracker {
    const fn new(
        format: ToolCallFormat,
        active: bool,
        schema: Option<ToolSchema>,
        tool_names: Vec<String>,
    ) -> Self {
        Self {
            format,
            openers: format.openers(),
            buffer: String::new(),
            bare_allowed: true,
            completed_count: 0,
            active,
            schema,
            tool_names,
        }
    }

    /// Release the first `length` buffered bytes as visible text.
    fn release(&mut self, length: usize, out: &mut StreamingToolOutput) {
        let mut boundary = length.min(self.buffer.len());
        while boundary > 0 && !self.buffer.is_char_boundary(boundary) {
            boundary -= 1;
        }
        let released: String = self.buffer.drain(..boundary).collect();
        if !released.trim().is_empty() {
            self.bare_allowed = false;
        }
        out.visible.push_str(&released);
    }

    /// Llama separates parallel JSON calls with `;`, which is not content.
    fn drop_call_separator(&mut self) {
        if self.format != ToolCallFormat::Llama || !self.bare_allowed || self.completed_count == 0 {
            return;
        }
        let whitespace = self.buffer.len() - self.buffer.trim_start().len();
        if self
            .buffer
            .get(whitespace..)
            .is_some_and(|rest| rest.starts_with(';'))
        {
            self.buffer.remove(whitespace);
        }
    }

    fn run(&mut self, out: &mut StreamingToolOutput, at_end: bool) {
        loop {
            self.drop_call_separator();
            let Some((offset, opener)) = find_opener(self.openers, &self.buffer, self.bare_allowed)
            else {
                let held = if at_end {
                    0
                } else {
                    held_suffix_length(self.openers, &self.buffer)
                };
                self.release(self.buffer.len() - held, out);
                return;
            };
            self.release(offset, out);

            let unit = {
                let context = ScanContext {
                    schema: self.schema.as_ref(),
                    tool_names: &self.tool_names,
                    at_end,
                };
                scan_unit(opener.kind, &self.buffer, &context)
            };
            match unit {
                Unit::Calls { consumed, calls } => {
                    self.completed_count += calls.len();
                    out.new_tool_calls.extend(calls);
                    self.buffer.drain(..consumed.min(self.buffer.len()));
                }
                Unit::Text { consumed } => self.release(consumed.max(1), out),
                Unit::Incomplete => {
                    if at_end || self.buffer.len() > MAX_INSIDE_TOOL_CALL_BYTES {
                        self.release(opener.token.len(), out);
                    } else {
                        return;
                    }
                }
            }
        }
    }

    fn process(&mut self, text: &str) -> StreamingToolOutput {
        if !self.active {
            return StreamingToolOutput {
                visible: text.to_owned(),
                new_tool_calls: Vec::new(),
            };
        }
        self.buffer.push_str(text);
        let mut out = StreamingToolOutput::default();
        self.run(&mut out, false);
        out
    }

    fn flush(&mut self) -> StreamingToolOutput {
        let mut out = StreamingToolOutput::default();
        if self.active {
            self.run(&mut out, true);
        }
        out
    }
}

/// Parses model output for tool calls in the format the model uses.
///
/// Built once per request from the model's [`ToolCallFormat`] and the
/// request's `tools`.
pub struct ToolCallParser {
    format: ToolCallFormat,
    schema: Option<ToolSchema>,
    tool_names: Vec<String>,
    max_calls: Option<usize>,
}

impl ToolCallParser {
    /// `tools` is the request's `OpenAI` `tools` array, if any.
    #[must_use]
    pub fn new(format: ToolCallFormat, tools: Option<&[serde_json::Value]>) -> Self {
        let tool_names = forced::tool_definitions(tools.unwrap_or_default())
            .into_iter()
            .map(|definition| definition.name.to_owned())
            .collect();
        Self {
            format,
            schema: ToolSchema::from_tools(tools),
            tool_names,
            max_calls: None,
        }
    }

    /// Keep at most `max_calls` calls (`parallel_tool_calls: false` is one);
    /// later calls are dropped from both the calls and the visible text.
    #[must_use]
    pub const fn with_max_calls(mut self, max_calls: Option<usize>) -> Self {
        self.max_calls = max_calls;
        self
    }

    pub const fn format(&self) -> ToolCallFormat {
        self.format
    }

    /// Parse a complete model response.
    pub fn parse(&self, text: &str) -> ToolParseResult {
        let mut result = if self.format == ToolCallFormat::Qwen {
            parse_tool_calls(text, self.schema.as_ref())
        } else {
            let mut tracker = ScanningTracker::new(
                self.format,
                true,
                self.schema.clone(),
                self.tool_names.clone(),
            );
            let mut output = tracker.process(text);
            let tail = tracker.flush();
            output.visible.push_str(&tail.visible);
            output.new_tool_calls.extend(tail.new_tool_calls);
            ToolParseResult {
                text: output.visible.trim().to_owned(),
                tool_calls: output.new_tool_calls,
            }
        };
        if let Some(limit) = self.max_calls {
            result.tool_calls.truncate(limit);
        }
        result
    }

    /// A streaming tracker for one response. With `active == false` it passes
    /// text through untouched.
    pub fn tracker(&self, active: bool) -> ToolCallTracker {
        let inner = if self.format == ToolCallFormat::Qwen {
            TrackerKind::Qwen(StreamingToolCallTracker::new(active, self.schema.clone()))
        } else {
            TrackerKind::Scanning(Box::new(ScanningTracker::new(
                self.format,
                active,
                self.schema.clone(),
                self.tool_names.clone(),
            )))
        };
        ToolCallTracker {
            inner,
            max_calls: self.max_calls,
            accepted_count: 0,
        }
    }
}

enum TrackerKind {
    Qwen(StreamingToolCallTracker),
    Scanning(Box<ScanningTracker>),
}

/// Streaming counterpart of [`ToolCallParser::parse`].
///
/// Text that could be the start of a tool call is held back until it is known
/// not to be one, so a half-received call never reaches the visible output.
pub struct ToolCallTracker {
    inner: TrackerKind,
    max_calls: Option<usize>,
    accepted_count: usize,
}

impl ToolCallTracker {
    fn limit(&mut self, mut output: StreamingToolOutput) -> StreamingToolOutput {
        if let Some(limit) = self.max_calls {
            let room = limit.saturating_sub(self.accepted_count);
            output.new_tool_calls.truncate(room);
        }
        self.accepted_count += output.new_tool_calls.len();
        output
    }

    /// Feed a chunk of streamed text.
    pub fn process(&mut self, text: &str) -> StreamingToolOutput {
        let output = match &mut self.inner {
            TrackerKind::Qwen(tracker) => tracker.process(text),
            TrackerKind::Scanning(tracker) => tracker.process(text),
        };
        self.limit(output)
    }

    /// Drain everything still buffered; call when the stream ends.
    pub fn flush(&mut self) -> StreamingToolOutput {
        let output = match &mut self.inner {
            TrackerKind::Qwen(tracker) => tracker.flush(),
            TrackerKind::Scanning(tracker) => tracker.flush(),
        };
        self.limit(output)
    }

    /// Calls handed out so far.
    pub const fn completed_count(&self) -> usize {
        self.accepted_count
    }

    pub const fn has_tool_calls(&self) -> bool {
        self.accepted_count > 0
    }
}

#[cfg(test)]
mod tests;
