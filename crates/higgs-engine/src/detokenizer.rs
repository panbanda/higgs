//! Turning generated token ids into streamed text.
//!
//! [`IncrementalDetok`] decodes only a trailing window of tokens per step, and
//! [`find_stop_in_tail`] looks for stop sequences only where new text could
//! complete one, so both cost O(1) per token instead of O(completion length).

use tokenizers::Tokenizer;

use crate::error::EngineError;

/// Cap on the trailing token window re-decoded per streaming step.
/// Generous for any multi-token UTF-8 sequence; prevents a stream of
/// undecodable tokens from growing the window without bound.
const MAX_DETOK_WINDOW: usize = 64;

/// Special tokens shaped like control delimiters that carry tool-call content.
const CONTENT_BEARING_DELIMITERS: [&str; 1] = ["<|\"|>"];

/// Token IDs that must never surface in decoded output text.
///
/// Seeded with the model's EOS ids, plus every added *control* special token:
/// the `<|…|>` chat-control delimiters and the classic `<s>`/`</s>`/`<pad>`/
/// `<unk>`/`<mask>` sentinels. Content-bearing special tokens (`<think>`,
/// `<tool_call>`, `MiniCPM`'s `<function>`/`<param>`, and Gemma 4's `<|"|>`
/// string delimiter, which has the control-token shape) deliberately do NOT
/// match, so they survive decoding and reach the tool-call / reasoning parsers. Used by
/// the streaming detokenizer ([`IncrementalDetok`]).
pub(crate) fn content_preserving_skip_ids(
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> std::collections::HashSet<u32> {
    let mut ids: std::collections::HashSet<u32> = eos_token_ids.iter().copied().collect();
    for (id, added) in tokenizer.get_added_tokens_decoder() {
        let content = added.content.as_str();
        let is_control = (content.starts_with("<|")
            && content.ends_with("|>")
            && !CONTENT_BEARING_DELIMITERS.contains(&content))
            || matches!(content, "<s>" | "</s>" | "<pad>" | "<unk>" | "<mask>");
        if is_control {
            ids.insert(id);
        }
    }
    ids
}

/// Incremental streaming detokenizer.
///
/// Re-decoding the full completion on every generated token is O(n^2) in
/// completion length. Instead, decode only `tokens[prefix_offset..]` and emit
/// the difference against `tokens[prefix_offset..read_offset]`; both windows
/// start at the same token, so tokenizer boundary effects cancel out. Text
/// ending in an incomplete UTF-8 sequence (trailing replacement char) is held
/// back until a later token completes it.
pub(crate) struct IncrementalDetok {
    /// Start of the decode window; advanced whenever text is emitted.
    prefix_offset: usize,
    /// Number of leading tokens already represented in `text`.
    read_offset: usize,
    /// All text decoded so far (streamed to the client incrementally).
    pub(crate) text: String,
    /// Control-token IDs filtered out before decoding so content-bearing
    /// special tokens (tool-call markup) survive streaming. See
    /// [`content_preserving_skip_ids`].
    skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
}

impl IncrementalDetok {
    /// Start from text already decoded for the first `token_count` tokens.
    pub(crate) const fn new(
        text: String,
        token_count: usize,
        skip_ids: std::sync::Arc<std::collections::HashSet<u32>>,
    ) -> Self {
        Self {
            prefix_offset: 0,
            read_offset: token_count,
            text,
            skip_ids,
        }
    }

    /// Decode `tokens`, dropping only the control tokens in `skip_ids` while
    /// preserving content-bearing special tokens.
    fn decode(&self, tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String, EngineError> {
        let run = |ids: &[u32]| {
            tokenizer
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenization(e.to_string()))
        };
        if !tokens.iter().any(|id| self.skip_ids.contains(id)) {
            return run(tokens);
        }
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !self.skip_ids.contains(id))
            .collect();
        run(&filtered)
    }

    /// Decode the trailing window of `tokens`, appending newly stable text to
    /// `self.text` and returning it.
    pub(crate) fn append(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;

        let over_window = window_tokens.len() > MAX_DETOK_WINDOW;
        if window_text.len() > prefix_text.len()
            && (!window_text.ends_with('\u{FFFD}') || over_window)
        {
            let new_text = window_text
                .get(prefix_text.len()..)
                .unwrap_or_default()
                .to_owned();
            self.prefix_offset = self.read_offset;
            self.read_offset = tokens.len();
            self.text.push_str(&new_text);
            return Ok(new_text);
        }
        if over_window && window_text.len() == prefix_text.len() {
            // The pending tokens decode to nothing (e.g. skipped special
            // tokens); drop them so the window stays bounded.
            self.prefix_offset = tokens.len();
            self.read_offset = tokens.len();
        }
        Ok(String::new())
    }

    /// Emit any text still held back by `append` (a trailing incomplete UTF-8
    /// sequence). Called when generation finishes so the total streamed text
    /// matches a full decode of the token buffer.
    pub(crate) fn flush(
        &mut self,
        tokenizer: &Tokenizer,
        tokens: &[u32],
    ) -> Result<String, EngineError> {
        if self.read_offset >= tokens.len() {
            return Ok(String::new());
        }
        let prefix_tokens = tokens
            .get(self.prefix_offset..self.read_offset)
            .unwrap_or_default();
        let window_tokens = tokens.get(self.prefix_offset..).unwrap_or_default();
        let prefix_text = self.decode(tokenizer, prefix_tokens)?;
        let window_text = self.decode(tokenizer, window_tokens)?;
        let new_text = window_text
            .get(prefix_text.len()..)
            .unwrap_or_default()
            .to_owned();
        self.prefix_offset = self.read_offset;
        self.read_offset = tokens.len();
        self.text.push_str(&new_text);
        Ok(new_text)
    }
}

/// Find the earliest stop-sequence occurrence that could involve the newly appended text.
///
/// Scans only the tail of `text` rather than the whole buffer.
/// Returns the absolute byte position where the match starts.
pub(crate) fn find_stop_in_tail(
    text: &str,
    new_len: usize,
    stop_sequences: &[String],
) -> Option<usize> {
    let max_stop = stop_sequences.iter().map(String::len).max().unwrap_or(0);
    if max_stop == 0 {
        return None;
    }
    let mut start = text.len().saturating_sub(new_len + max_stop - 1);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let tail = text.get(start..)?;
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = tail.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| start + pos)
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::{IncrementalDetok, Tokenizer, content_preserving_skip_ids, find_stop_in_tail};

    // -----------------------------------------------------------------------
    // find_stop_in_tail
    // -----------------------------------------------------------------------

    #[test]
    fn find_stop_in_tail_empty_stops_returns_none() {
        assert_eq!(find_stop_in_tail("hello world", 5, &[]), None);
    }

    #[test]
    fn find_stop_in_tail_finds_stop_in_new_text() {
        let stops = vec!["</s>".to_owned()];
        // "hello</s>" with the last 4 bytes new
        assert_eq!(find_stop_in_tail("hello</s>", 4, &stops), Some(5));
    }

    #[test]
    fn find_stop_in_tail_finds_stop_spanning_boundary() {
        let stops = vec!["STOP".to_owned()];
        // "abcSTOP" where only "OP" is new; "ST" was emitted previously
        assert_eq!(find_stop_in_tail("abcSTOP", 2, &stops), Some(3));
    }

    #[test]
    fn find_stop_in_tail_ignores_stop_fully_in_old_text() {
        let stops = vec!["XY".to_owned()];
        // "XYabcdefgh" with 2 new bytes: the scan window covers the last
        // 2 + (2 - 1) = 3 bytes only, so the old "XY" is not rescanned.
        assert_eq!(find_stop_in_tail("XYabcdefgh", 2, &stops), None);
    }

    #[test]
    fn find_stop_in_tail_earliest_of_multiple() {
        let stops = vec!["BBB".to_owned(), "AAA".to_owned()];
        assert_eq!(find_stop_in_tail("xAAAyBBBz", 9, &stops), Some(1));
    }

    #[test]
    fn find_stop_in_tail_handles_multibyte_boundary() {
        let stops = vec!["端".to_owned()];
        // The tail start can land mid-codepoint; it must back up to a
        // char boundary instead of panicking. "日本語端" = 4 chars, 12 bytes.
        assert_eq!(find_stop_in_tail("日本語端", 3, &stops), Some(9));
    }

    // -----------------------------------------------------------------------
    // IncrementalDetok
    // -----------------------------------------------------------------------

    /// Minimal word-level tokenizer with a byte-level decoder for detok tests.
    fn word_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "WordLevel",
                "vocab": {"Hello": 0, "Ġworld": 1, "!": 2, "ðŁĺ": 3, "Ģ": 4},
                "unk_token": "Hello"
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    /// Empty skip-id set for detok tests that exercise plain (non-special)
    /// tokens, where `skip_special_tokens` makes no difference.
    fn no_skip() -> std::sync::Arc<std::collections::HashSet<u32>> {
        std::sync::Arc::new(std::collections::HashSet::new())
    }

    /// `content_preserving_skip_ids` strips control tokens (chat delimiters,
    /// sentinels, the explicit EOS list) but keeps content-bearing special
    /// tokens so tool-call markup reaches the parser.
    #[test]
    fn skip_ids_strip_control_but_keep_content_special_tokens() {
        let mut tok = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let _ = tok.add_special_tokens([
            tokenizers::AddedToken::from("<|im_end|>", true),
            tokenizers::AddedToken::from("</s>", true),
            tokenizers::AddedToken::from("<function>", true),
            tokenizers::AddedToken::from("<tool_call>", true),
            tokenizers::AddedToken::from("<|\"|>", true),
            tokenizers::AddedToken::from("<|endoftext|>", true),
        ]);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let skip = content_preserving_skip_ids(&tok, &[im_end, 7]);

        // Stripped: <|…|> delimiter, the </s> sentinel, and the explicit eos id.
        assert!(skip.contains(&im_end), "<|…|> delimiter is a control token");
        assert!(
            skip.contains(&tok.token_to_id("</s>").unwrap()),
            "</s> sentinel stripped"
        );
        assert!(skip.contains(&7), "explicit eos id retained");
        // Preserved: content-bearing tool-call markup the parser depends on.
        assert!(
            !skip.contains(&tok.token_to_id("<function>").unwrap()),
            "<function> preserved for the tool-call parser"
        );
        assert!(
            !skip.contains(&tok.token_to_id("<tool_call>").unwrap()),
            "<tool_call> preserved"
        );
        assert!(
            !skip.contains(&tok.token_to_id("<|\"|>").unwrap()),
            "Gemma 4 string delimiter preserved for the tool-call parser"
        );
        assert!(
            skip.contains(&tok.token_to_id("<|endoftext|>").unwrap()),
            "other <|…|> tokens stay control tokens"
        );
    }

    #[test]
    fn detok_keeps_the_gemma_string_delimiter_and_drops_end_of_text() {
        let mut tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let _ = tokenizer.add_special_tokens([
            tokenizers::AddedToken::from("<|\"|>", true),
            tokenizers::AddedToken::from("<|endoftext|>", true),
        ]);
        let quote = tokenizer.token_to_id("<|\"|>").unwrap();
        let end_of_text = tokenizer.token_to_id("<|endoftext|>").unwrap();
        let skip = std::sync::Arc::new(content_preserving_skip_ids(&tokenizer, &[]));

        let tokens = [quote, end_of_text, quote];
        let mut detok = IncrementalDetok::new(String::new(), 0, skip);
        let text = detok.append(&tokenizer, &tokens).unwrap();
        assert_eq!(
            text.split_whitespace().collect::<Vec<_>>(),
            ["<|\"|>", "<|\"|>"]
        );
    }

    #[test]
    fn incremental_detok_emits_per_token_diffs() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first.clone(), tokens.len(), no_skip());

        tokens.push(1);
        let second = detok.append(&tokenizer, &tokens).unwrap();
        tokens.push(2);
        let third = detok.append(&tokenizer, &tokens).unwrap();

        let full = tokenizer.decode(&tokens, true).unwrap();
        assert_eq!(format!("{first}{second}{third}"), full);
        assert_eq!(detok.text, full);
    }

    #[test]
    fn incremental_detok_flush_without_pending_is_empty() {
        let tokenizer = word_tokenizer();
        let mut tokens: Vec<u32> = vec![0];
        let first = tokenizer.decode(&tokens, true).unwrap();
        let mut detok = IncrementalDetok::new(first, tokens.len(), no_skip());

        tokens.push(1);
        detok.append(&tokenizer, &tokens).unwrap();
        assert_eq!(detok.flush(&tokenizer, &tokens).unwrap(), "");
    }

    // Tokens 3 and 4 are the byte-level pieces of 😀 (U+1F600).
    // Appending only token 3 must hold back the incomplete UTF-8 sequence
    // (return ""), and appending both tokens must emit the full emoji.
    #[test]
    fn incremental_detok_first_token_partial_utf8_held_back() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // First partial piece: held back, no replacement char emitted
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 token must be held back");

        // Completing the sequence emits the full emoji
        let emitted = detok.append(&tokenizer, &[3, 4]).unwrap();
        assert_eq!(emitted, "😀", "completing UTF-8 should emit the emoji");

        let full = tokenizer.decode(&[3u32, 4], true).unwrap();
        assert_eq!(detok.text, full, "detok.text must equal full decode");
    }

    // flush() must emit any bytes held back by append(), and a second flush
    // on the same (now-fully-drained) detok must return "".
    #[test]
    fn incremental_detok_flush_emits_pending() {
        let tokenizer = word_tokenizer();
        let mut detok = IncrementalDetok::new(String::new(), 0, no_skip());

        // Partial piece held back
        let held = detok.append(&tokenizer, &[3]).unwrap();
        assert_eq!(held, "", "partial UTF-8 must be held back before flush");

        // flush must emit something (a replacement char is acceptable here)
        let flushed = detok.flush(&tokenizer, &[3]).unwrap();
        assert!(!flushed.is_empty(), "flush must emit held-back text");

        // A second flush on an already-drained detok returns ""
        let flushed2 = detok.flush(&tokenizer, &[3]).unwrap();
        assert_eq!(flushed2, "", "second flush must return empty string");
    }

    // find_stop_in_tail must identify a stop whose prefix was already emitted,
    // returning the byte position that lets the caller emit the prefix ("hi")
    // before the stop ("STOP") in the same new-text window.
    #[test]
    fn find_stop_in_tail_first_token_prefix_and_stop() {
        let stops = vec!["STOP".to_owned()];
        // "hiSTOP": all 6 bytes are new, stop starts at byte 2
        assert_eq!(
            find_stop_in_tail("hiSTOP", 6, &stops),
            Some(2),
            "stop at pos 2 so 'hi' can be emitted as prefix"
        );
    }
}
