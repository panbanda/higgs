//! Incremental Server-Sent Events parser (the subset MCP uses).

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, Default)]
pub struct SseParser {
    pending: String,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Feed more text; returns the events completed by it.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.pending.push_str(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.pending.find(['\n', '\r']) {
            let terminator_is_cr = self.pending.as_bytes().get(end) == Some(&b'\r');
            let mut consumed = end + 1;
            if terminator_is_cr {
                match self.pending.as_bytes().get(end + 1) {
                    Some(b'\n') => consumed += 1,
                    // A lone trailing CR may be the first half of CRLF.
                    None => break,
                    Some(_) => {}
                }
            }
            let line: String = self.pending.drain(..consumed).collect();
            if let Some(event) = self.accept_line(line.trim_end_matches(['\r', '\n'])) {
                events.push(event);
            }
        }
        events
    }

    /// Flush a final event not terminated by a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            if let Some(event) = self.accept_line(line.trim_end_matches(['\r', '\n'])) {
                return Some(event);
            }
        }
        self.dispatch()
    }

    fn accept_line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let content = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(content.to_owned()),
            "data" => self.data.push(content.to_owned()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        if self.data.is_empty() {
            return None;
        }
        let data = self.data.join("\n");
        self.data.clear();
        Some(SseEvent { event, data })
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_split_across_chunks() {
        let mut parser = SseParser::default();
        assert!(parser.feed("event: message\nda").is_empty());
        let events = parser.feed("ta: {\"a\":1}\n\ndata: second\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: Some("message".to_owned()),
                    data: "{\"a\":1}".to_owned()
                },
                SseEvent {
                    event: None,
                    data: "second".to_owned()
                }
            ]
        );
    }

    #[test]
    fn joins_multiline_data_and_skips_comments() {
        let mut parser = SseParser::default();
        let events = parser.feed(": keepalive\ndata: a\ndata: b\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "a\nb");
    }

    #[test]
    fn handles_crlf_even_when_split() {
        let mut parser = SseParser::default();
        assert!(parser.feed("data: x\r").is_empty());
        let events = parser.feed("\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn blank_lines_without_data_emit_nothing() {
        let mut parser = SseParser::default();
        assert!(parser.feed("\n\nevent: ping\n\n").is_empty());
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut parser = SseParser::default();
        assert!(parser.feed("data: tail").is_empty());
        assert_eq!(parser.finish().unwrap().data, "tail");
        assert!(parser.finish().is_none());
    }
}
