//! Parse reasoning/thinking content from model-generated text.
//!
//! Qwen3 (and similar) models emit reasoning in `<think>...</think>` tags:
//! ```text
//! <think>
//! Let me analyze this step by step...
//! </think>
//! The answer is 42.
//! ```
//!
//! This module separates the reasoning content from the visible response.

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Result of parsing reasoning tags from model output.
#[derive(Debug, Clone)]
pub struct ReasoningParseResult {
    /// The visible response text (everything outside `<think>` tags).
    pub text: String,
    /// Concatenated reasoning content from all `<think>` blocks.
    /// `None` if no `<think>` tags were found.
    pub reasoning: Option<String>,
}

/// Parse model output for `<think>...</think>` reasoning blocks.
///
/// Extracts reasoning content and returns it separately from the visible text.
/// If no `<think>` tags are found, returns the full text with `reasoning = None`.
pub fn parse_reasoning(text: &str) -> ReasoningParseResult {
    let mut visible = String::new();
    let mut reasoning = String::new();
    let mut found_thinking = false;
    let mut remaining = text;

    loop {
        if let Some(start_pos) = remaining.find(THINK_OPEN) {
            visible.push_str(remaining.get(..start_pos).unwrap_or_default());

            let after_open = remaining
                .get(start_pos + THINK_OPEN.len()..)
                .unwrap_or_default();

            if let Some(end_pos) = after_open.find(THINK_CLOSE) {
                let think_content = after_open.get(..end_pos).unwrap_or_default().trim();
                if !think_content.is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(think_content);
                }
                found_thinking = true;

                remaining = after_open
                    .get(end_pos + THINK_CLOSE.len()..)
                    .unwrap_or_default();
            } else {
                // Unclosed <think> tag -- treat remaining as reasoning
                let unclosed = after_open.trim();
                if !unclosed.is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(unclosed);
                }
                found_thinking = true;
                break;
            }
        } else {
            visible.push_str(remaining);
            break;
        }
    }

    ReasoningParseResult {
        text: visible.trim().to_owned(),
        reasoning: found_thinking.then_some(reasoning),
    }
}

/// Streaming reasoning state tracker.
///
/// Tracks whether we are currently inside a `<think>` block during
/// token-by-token streaming, buffering partial tags.
pub struct StreamingReasoningTracker {
    buffer: String,
    inside_think: bool,
    started: bool,
}

impl Default for StreamingReasoningTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingReasoningTracker {
    pub const fn new() -> Self {
        Self {
            buffer: String::new(),
            inside_think: false,
            started: false,
        }
    }

    /// Create a tracker that starts inside a `<think>` block.
    /// Use when the chat template already opened `<think>` in the prompt.
    pub const fn new_inside_think() -> Self {
        Self {
            buffer: String::new(),
            inside_think: true,
            started: true,
        }
    }

    /// Process a new text chunk from the model. Returns `(visible_text, reasoning_text)`.
    ///
    /// Either or both may be empty on a given call (e.g. when buffering a partial tag).
    pub fn process(&mut self, text: &str) -> (String, String) {
        self.buffer.push_str(text);

        let mut visible = String::new();
        let mut reasoning = String::new();

        loop {
            if self.inside_think {
                if let Some(end_pos) = self.buffer.find(THINK_CLOSE) {
                    let think_text = self.buffer.get(..end_pos).unwrap_or_default();
                    reasoning.push_str(think_text);
                    self.buffer = self
                        .buffer
                        .get(end_pos + THINK_CLOSE.len()..)
                        .unwrap_or_default()
                        .to_owned();
                    self.inside_think = false;
                } else if self.buffer.len() > THINK_CLOSE.len() {
                    // Flush all but the last few chars (which could be a partial </think>)
                    let mut safe_len = self.buffer.len() - THINK_CLOSE.len();
                    while safe_len > 0 && !self.buffer.is_char_boundary(safe_len) {
                        safe_len -= 1;
                    }
                    reasoning.push_str(&self.buffer[..safe_len]);
                    self.buffer = self.buffer[safe_len..].to_owned();
                    break;
                } else {
                    break;
                }
            } else if let (Some(stray_pos), open_pos) =
                (self.buffer.find(THINK_CLOSE), self.buffer.find(THINK_OPEN))
            {
                // Strip stray </think> outside a thinking block, but only if
                // it appears before any <think> (otherwise process <think> first).
                if open_pos.is_none_or(|op| stray_pos < op) {
                    visible.push_str(self.buffer.get(..stray_pos).unwrap_or_default());
                    self.buffer = self
                        .buffer
                        .get(stray_pos + THINK_CLOSE.len()..)
                        .unwrap_or_default()
                        .to_owned();
                } else {
                    // open_pos is guaranteed to be Some here (else branch of none_or)
                    let op = open_pos.unwrap();
                    visible.push_str(self.buffer.get(..op).unwrap_or_default());
                    self.buffer = self
                        .buffer
                        .get(op + THINK_OPEN.len()..)
                        .unwrap_or_default()
                        .to_owned();
                    self.inside_think = true;
                    self.started = true;
                }
            } else if let Some(start_pos) = self.buffer.find(THINK_OPEN) {
                visible.push_str(self.buffer.get(..start_pos).unwrap_or_default());
                self.buffer = self
                    .buffer
                    .get(start_pos + THINK_OPEN.len()..)
                    .unwrap_or_default()
                    .to_owned();
                self.inside_think = true;
                self.started = true;
            } else if self.buffer.len() > THINK_CLOSE.len() {
                // Flush all but the last few chars (could be partial <think> or </think>)
                let mut safe_len = self.buffer.len() - THINK_CLOSE.len();
                while safe_len > 0 && !self.buffer.is_char_boundary(safe_len) {
                    safe_len -= 1;
                }
                visible.push_str(&self.buffer[..safe_len]);
                self.buffer = self.buffer[safe_len..].to_owned();
                break;
            } else {
                break;
            }
        }

        (visible, reasoning)
    }

    /// Flush any remaining buffered content. Call when generation is complete.
    pub fn flush(&mut self) -> (String, String) {
        let buf = std::mem::take(&mut self.buffer);
        let was_inside = self.inside_think;
        self.inside_think = false;
        self.started = false;
        if was_inside {
            (String::new(), buf)
        } else {
            (buf, String::new())
        }
    }

    /// Whether any `<think>` block has been encountered.
    pub const fn has_reasoning(&self) -> bool {
        self.started
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::shadow_unrelated)]
mod tests {
    use super::*;

    #[test]
    fn no_thinking_tags() {
        let result = parse_reasoning("Hello world");
        assert_eq!(result.text, "Hello world");
        assert!(result.reasoning.is_none());
    }

    #[test]
    fn simple_thinking_block() {
        let input = "<think>\nLet me think...\n</think>\nThe answer is 42.";
        let result = parse_reasoning(input);
        assert_eq!(result.text, "The answer is 42.");
        assert_eq!(result.reasoning.as_deref(), Some("Let me think..."));
    }

    #[test]
    fn thinking_with_text_before_and_after() {
        let input = "Preamble.\n<think>\nReasoning here.\n</think>\nConclusion.";
        let result = parse_reasoning(input);
        assert!(result.text.contains("Preamble."));
        assert!(result.text.contains("Conclusion."));
        assert_eq!(result.reasoning.as_deref(), Some("Reasoning here."));
    }

    #[test]
    fn multiple_thinking_blocks() {
        let input = "<think>First thought</think>Middle<think>Second thought</think>End";
        let result = parse_reasoning(input);
        assert!(result.text.contains("Middle"));
        assert!(result.text.contains("End"));
        let r = result.reasoning.unwrap();
        assert!(r.contains("First thought"));
        assert!(r.contains("Second thought"));
    }

    #[test]
    fn unclosed_think_tag() {
        let input = "Hello <think>still thinking...";
        let result = parse_reasoning(input);
        assert_eq!(result.text, "Hello");
        assert_eq!(result.reasoning.as_deref(), Some("still thinking..."));
    }

    #[test]
    fn empty_think_block() {
        let input = "<think></think>Answer here.";
        let result = parse_reasoning(input);
        assert_eq!(result.text, "Answer here.");
        assert_eq!(result.reasoning.as_deref(), Some(""));
    }

    #[test]
    fn only_think_block_no_visible() {
        let input = "<think>All reasoning, no response.</think>";
        let result = parse_reasoning(input);
        assert!(result.text.is_empty());
        assert_eq!(
            result.reasoning.as_deref(),
            Some("All reasoning, no response.")
        );
    }

    #[test]
    fn streaming_tracker_basic() {
        let mut tracker = StreamingReasoningTracker::new();

        let (vis, reas) = tracker.process("<think>thinking");
        assert!(vis.is_empty());
        assert!(reas.contains("thinking") || tracker.has_reasoning());

        let (vis, reas) = tracker.process("</think>answer");
        // After close tag, answer should appear in visible
        let (vis2, reas2) = tracker.flush();
        let total_visible = format!("{vis}{vis2}");
        let total_reasoning = format!("{reas}{reas2}");
        assert!(total_visible.contains("answer"));
        assert!(total_reasoning.contains("thinking"));
    }

    #[test]
    fn streaming_tracker_no_thinking() {
        let mut tracker = StreamingReasoningTracker::new();
        let (vis, reas) = tracker.process("Hello world, no thinking here at all.");
        let (vis2, reas2) = tracker.flush();
        let total_visible = format!("{vis}{vis2}");
        let total_reasoning = format!("{reas}{reas2}");
        assert!(total_visible.contains("Hello world"));
        assert!(total_reasoning.is_empty());
        assert!(!tracker.has_reasoning());
    }

    /// Tracker created with `new_inside_think()` starts in thinking mode.
    /// The first `</think>` transitions to visible output without needing `<think>`.
    #[test]
    fn streaming_tracker_new_inside_think() {
        let mut tracker = StreamingReasoningTracker::new_inside_think();
        assert!(tracker.has_reasoning());

        // Feed reasoning content (no opening tag needed)
        let (vis, reas) = tracker.process("step 1... step 2...");
        let mut total_reasoning = reas.clone();
        assert!(vis.is_empty(), "reasoning should not be visible");
        assert!(!reas.is_empty(), "should capture reasoning text");

        // Close thinking and emit visible answer
        let (vis, reas) = tracker.process("</think>The answer is 42.");
        total_reasoning.push_str(&reas);
        let (vis2, reas2) = tracker.flush();
        total_reasoning.push_str(&reas2);
        let total_visible = format!("{vis}{vis2}");
        assert!(
            total_visible.contains("The answer is 42."),
            "answer should be visible after </think>"
        );
        assert!(
            !total_visible.contains("</think>"),
            "</think> tag should not leak into visible"
        );
        assert!(
            total_reasoning.contains("step 1"),
            "reasoning should be captured"
        );
    }

    /// When `</think>` is emitted twice (e.g. from a thinking budget bug),
    /// the second `</think>` must NOT leak into visible output.
    #[test]
    fn streaming_tracker_double_close_does_not_leak() {
        let mut tracker = StreamingReasoningTracker::new_inside_think();

        // First close — legitimate end of thinking
        let (vis1, _reas1) = tracker.process("reasoning</think>");
        // Second close — erroneous duplicate (from budget enforcement bug)
        let (vis2, _reas2) = tracker.process("</think>answer");
        let (vis3, _reas3) = tracker.flush();
        let total_visible = format!("{vis1}{vis2}{vis3}");

        assert!(
            !total_visible.contains("</think>"),
            "double </think> should not leak into visible output, got: {total_visible}"
        );
        assert!(
            total_visible.contains("answer"),
            "content after duplicate close should remain visible, got: {total_visible}"
        );
    }

    /// Flush when still inside thinking returns buffered content as reasoning.
    #[test]
    fn streaming_tracker_flush_while_thinking() {
        let mut tracker = StreamingReasoningTracker::new_inside_think();
        let (_, reas) = tracker.process("partial reasoning");
        let (vis, reas2) = tracker.flush();
        let total_reasoning = format!("{reas}{reas2}");
        assert!(vis.is_empty(), "nothing visible while still thinking");
        assert!(
            total_reasoning.contains("partial reasoning"),
            "flush should return buffered reasoning"
        );
    }
}
