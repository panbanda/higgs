//! Readable-text extraction from HTML, without a DOM.
//!
//! The extractor is a forgiving single-pass tokenizer: it drops scripts,
//! styles and page chrome, turns block elements into line breaks, keeps link
//! targets next to their text and decodes common entities. It is meant to
//! give a model the gist of a page, not to render it.

const SKIPPED_WITH_CONTENT: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "iframe", "object",
];
const SKIPPED_CHROME: &[&str] = &["nav", "footer", "aside", "form"];
const BLOCK_TAGS: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "header",
    "main",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "pre",
    "hr",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "details",
    "summary",
    "address",
];
const HEADINGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Extracted {
    pub title: Option<String>,
    pub text: String,
}

struct Tag<'a> {
    name: String,
    closing: bool,
    attributes: &'a str,
    /// Byte offset just past the closing `>`.
    end: usize,
}

pub fn extract(html: &str) -> Extracted {
    let mut writer = TextWriter::default();
    let mut title: Option<String> = None;
    let mut first_heading: Option<String> = None;
    let mut position = 0;
    let mut chrome_depth: Option<(String, usize)> = None;
    let mut preformatted = 0_usize;
    let mut link: Option<(String, usize)> = None;

    while position < html.len() {
        let rest = html.get(position..).unwrap_or_default();
        let Some(open) = rest.find('<') else {
            if chrome_depth.is_none() {
                writer.text(&decode_entities(rest), preformatted > 0);
            }
            break;
        };
        if chrome_depth.is_none() {
            writer.text(
                &decode_entities(rest.get(..open).unwrap_or_default()),
                preformatted > 0,
            );
        }
        position += open;
        let here = html.get(position..).unwrap_or_default();
        if let Some(after) = here.strip_prefix("<!--") {
            let skipped = 4 + after.find("-->").map_or(after.len(), |index| index + 3);
            position += skipped;
            continue;
        }
        let parsed = parse_tag(html, position);
        let Some(tag) = parsed else {
            // A lone '<' that does not start a tag is ordinary text.
            if chrome_depth.is_none() {
                writer.text("<", false);
            }
            position += 1;
            continue;
        };
        position = tag.end;

        if let Some((name, depth)) = &mut chrome_depth {
            if tag.name == *name {
                if tag.closing {
                    if *depth == 0 {
                        chrome_depth = None;
                    } else {
                        *depth -= 1;
                    }
                } else {
                    *depth += 1;
                }
            }
            continue;
        }
        if tag.closing {
            handle_closing(&tag, &mut writer, &mut preformatted, &mut link);
            continue;
        }
        if tag.name == "title" {
            let (content, next) = raw_text_until(html, position, "title");
            title = Some(collapse_whitespace(&decode_entities(content)));
            position = next;
            continue;
        }
        if SKIPPED_WITH_CONTENT.contains(&tag.name.as_str()) {
            position = raw_text_until(html, position, &tag.name).1;
            continue;
        }
        if SKIPPED_CHROME.contains(&tag.name.as_str()) {
            chrome_depth = Some((tag.name.clone(), 0));
            continue;
        }
        if HEADINGS.contains(&tag.name.as_str()) {
            if first_heading.is_none() {
                first_heading = Some(heading_text(html, position, &tag.name));
            }
            writer.block_break(2);
        } else if tag.name == "br" {
            writer.block_break(1);
        } else if tag.name == "li" {
            writer.block_break(1);
            writer.text("- ", false);
        } else if tag.name == "a" {
            if let Some(href) = attribute(tag.attributes, "href").filter(|href| is_web_link(href)) {
                link = Some((decode_entities(&href), writer.mark()));
            }
        } else if BLOCK_TAGS.contains(&tag.name.as_str()) {
            writer.block_break(if tag.name == "pre" { 2 } else { 1 });
            if tag.name == "pre" {
                preformatted += 1;
            }
        }
    }

    Extracted {
        title: title
            .filter(|value| !value.is_empty())
            .or_else(|| first_heading.filter(|value| !value.is_empty())),
        text: writer.finish(),
    }
}

fn handle_closing(
    tag: &Tag<'_>,
    writer: &mut TextWriter,
    preformatted: &mut usize,
    link: &mut Option<(String, usize)>,
) {
    if HEADINGS.contains(&tag.name.as_str()) {
        writer.block_break(2);
    } else if tag.name == "a" {
        if let Some((href, mark)) = link.take() {
            let shown = writer.since(mark).trim().to_owned();
            if !shown.is_empty() && shown != href {
                writer.text(&format!(" ({href})"), false);
            }
        }
    } else if tag.name == "li" || tag.name == "tr" {
        writer.block_break(1);
    } else if BLOCK_TAGS.contains(&tag.name.as_str()) {
        writer.block_break(if tag.name == "pre" { 2 } else { 1 });
        if tag.name == "pre" {
            *preformatted = preformatted.saturating_sub(1);
        }
    } else if tag.name == "td" || tag.name == "th" {
        writer.text(" ", false);
    }
}

fn is_web_link(href: &str) -> bool {
    let lowered = href.trim().to_ascii_lowercase();
    lowered.starts_with("http://") || lowered.starts_with("https://")
}

fn parse_tag(html: &str, start: usize) -> Option<Tag<'_>> {
    let body = html.get(start + 1..)?;
    let closing = body.starts_with('/');
    let name_start = usize::from(closing);
    let named = body.get(name_start..)?;
    let first = named.chars().next()?;
    if !first.is_ascii_alphabetic() {
        // `<!doctype ...>` and `<?xml ...?>` are skipped as markup, anything
        // else is text.
        return (first == '!' || first == '?').then(|| Tag {
            name: String::new(),
            closing: false,
            attributes: "",
            end: body
                .find('>')
                .map_or(html.len(), |index| start + 1 + index + 1),
        });
    }
    let name_length = named
        .find(|character: char| character.is_whitespace() || character == '/' || character == '>')
        .unwrap_or(named.len());
    let name = named.get(..name_length)?.to_ascii_lowercase();
    // Find the closing '>' while honoring quoted attribute values.
    let attributes_start = name_start + name_length;
    let mut quote: Option<char> = None;
    let mut end = None;
    for (offset, character) in body.get(attributes_start..)?.char_indices() {
        match (quote, character) {
            (None, '"' | '\'') => quote = Some(character),
            (Some(open), found) if found == open => quote = None,
            (None, '>') => {
                end = Some(attributes_start + offset);
                break;
            }
            _ => {}
        }
    }
    // A tag cut off by the end of the input is dropped, as browsers do.
    let Some(tag_end) = end else {
        return Some(Tag {
            name,
            closing,
            attributes: "",
            end: html.len(),
        });
    };
    Some(Tag {
        name,
        closing,
        attributes: body.get(attributes_start..tag_end).unwrap_or_default(),
        end: start + 1 + tag_end + 1,
    })
}

/// Text between `position` and the next `</name`, plus the offset after that
/// closing tag.
fn raw_text_until<'a>(html: &'a str, position: usize, name: &str) -> (&'a str, usize) {
    let rest = html.get(position..).unwrap_or_default();
    let needle = format!("</{name}");
    let lowered = rest.to_ascii_lowercase();
    lowered.find(&needle).map_or((rest, html.len()), |index| {
        let after = position + index;
        let end = html
            .get(after..)
            .and_then(|tail| tail.find('>'))
            .map_or(html.len(), |close| after + close + 1);
        (rest.get(..index).unwrap_or_default(), end)
    })
}

fn heading_text(html: &str, position: usize, name: &str) -> String {
    let (content, _) = raw_text_until(html, position, name);
    collapse_whitespace(&extract(content).text)
}

pub fn attribute(attributes: &str, wanted: &str) -> Option<String> {
    let mut rest = attributes;
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == '/');
        if rest.is_empty() {
            return None;
        }
        let name_end = rest
            .find(|c: char| c.is_whitespace() || c == '=' || c == '/')
            .unwrap_or(rest.len());
        let name = rest.get(..name_end)?;
        rest = rest.get(name_end..)?.trim_start();
        let value = if let Some(after_equals) = rest.strip_prefix('=') {
            let trimmed = after_equals.trim_start();
            let quote = trimmed.chars().next().filter(|c| matches!(c, '"' | '\''));
            let (value, remaining) = if let Some(quote_char) = quote {
                let inner = trimmed.get(1..)?;
                let close = inner.find(quote_char).unwrap_or(inner.len());
                (
                    inner.get(..close)?,
                    inner.get(close + 1..).unwrap_or_default(),
                )
            } else {
                let close = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
                (
                    trimmed.get(..close)?,
                    trimmed.get(close..).unwrap_or_default(),
                )
            };
            rest = remaining;
            value
        } else {
            ""
        };
        if name.eq_ignore_ascii_case(wanted) {
            return Some(value.to_owned());
        }
    }
}

#[derive(Default)]
struct TextWriter {
    output: String,
}

impl TextWriter {
    fn text(&mut self, text: &str, preformatted: bool) {
        if preformatted {
            self.output.push_str(text);
            return;
        }
        for character in text.chars() {
            if character.is_whitespace() {
                if !self.output.is_empty() && !self.output.ends_with([' ', '\n']) {
                    self.output.push(' ');
                }
            } else {
                self.output.push(character);
            }
        }
    }

    /// Ensure the output ends with at least `newlines` line breaks.
    fn block_break(&mut self, newlines: usize) {
        if self.output.is_empty() {
            return;
        }
        let trimmed_length = self.output.trim_end_matches(' ').len();
        self.output.truncate(trimmed_length);
        let existing = self.output.chars().rev().take_while(|c| *c == '\n').count();
        for _ in existing..newlines {
            self.output.push('\n');
        }
    }

    const fn mark(&self) -> usize {
        self.output.len()
    }

    fn since(&self, mark: usize) -> &str {
        self.output.get(mark..).unwrap_or_default()
    }

    fn finish(self) -> String {
        let mut lines: Vec<&str> = self.output.lines().map(str::trim_end).collect();
        while lines.first().is_some_and(|line| line.is_empty()) {
            lines.remove(0);
        }
        while lines.last().is_some_and(|line| line.is_empty()) {
            lines.pop();
        }
        let mut result = String::with_capacity(self.output.len());
        let mut blank_run = 0;
        for line in lines {
            if line.trim().is_empty() {
                blank_run += 1;
                if blank_run > 1 {
                    continue;
                }
            } else {
                blank_run = 0;
                result.push_str(line);
            }
            result.push('\n');
        }
        result.trim_end().to_owned()
    }
}

pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find('&') {
        output.push_str(rest.get(..index).unwrap_or_default());
        let candidate = rest.get(index + 1..).unwrap_or_default();
        let decoded = candidate
            .find(';')
            .filter(|end| *end <= 10)
            .and_then(|end| decode_entity(candidate.get(..end)?).map(|value| (value, end)));
        if let Some((value, end)) = decoded {
            output.push(value);
            rest = candidate.get(end + 1..).unwrap_or_default();
        } else {
            output.push('&');
            rest = candidate;
        }
    }
    output.push_str(rest);
    output
}

fn decode_entity(name: &str) -> Option<char> {
    if let Some(numeric) = name.strip_prefix('#') {
        let code = match numeric.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => numeric.parse().ok()?,
        };
        return char::from_u32(code).filter(|character| *character != '\0');
    }
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "ndash" => '\u{2013}',
        "mdash" => '\u{2014}',
        "hellip" => '\u{2026}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "laquo" => '\u{ab}',
        "raquo" => '\u{bb}',
        "bull" => '\u{2022}',
        "middot" => '\u{b7}',
        "copy" => '\u{a9}',
        "reg" => '\u{ae}',
        "trade" => '\u{2122}',
        "euro" => '\u{20ac}',
        "pound" => '\u{a3}',
        "deg" => '\u{b0}',
        _ => return None,
    })
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_title_and_paragraphs_dropping_scripts_styles_and_comments() {
        let page = extract(
            r#"<!DOCTYPE html><html><head><title> My  Page &amp; More </title>
            <style>p { color: red }</style><script>var x = "<p>not text</p>";</script></head>
            <body><!-- hidden --><h1>Heading</h1><p>First   paragraph
            with wrapping.</p><p>Second &lt;one&gt;.</p></body></html>"#,
        );
        assert_eq!(page.title.as_deref(), Some("My Page & More"));
        assert_eq!(
            page.text,
            "Heading\n\nFirst paragraph with wrapping.\nSecond <one>."
        );
    }

    #[test]
    fn page_chrome_is_dropped_including_nested_same_name_elements() {
        let page = extract(
            "<body><nav>menu <nav>inner</nav> more menu</nav><main><p>Content</p></main>\
             <aside>sidebar</aside><footer>copyright</footer><form>search box</form></body>",
        );
        assert_eq!(page.text, "Content");
    }

    #[test]
    fn lists_breaks_and_links_are_rendered_readably() {
        let page = extract(
            r#"<ul><li>one</li><li>two <a href="https://example.com/x?a=1&amp;b=2">link</a></li></ul>
               line<br>break <a href="/relative">skipped</a> <a href="https://same.example">https://same.example</a>"#,
        );
        assert_eq!(
            page.text,
            "- one\n- two link (https://example.com/x?a=1&b=2)\nline\nbreak skipped https://same.example"
        );
    }

    #[test]
    fn preformatted_text_keeps_its_whitespace() {
        let page = extract("<p>before</p><pre>fn main() {\n    run();\n}</pre><p>after</p>");
        assert_eq!(page.text, "before\n\nfn main() {\n    run();\n}\n\nafter");
    }

    #[test]
    fn title_falls_back_to_the_first_heading() {
        assert_eq!(
            extract("<h2>Only <b>heading</b></h2><p>x</p>")
                .title
                .as_deref(),
            Some("Only heading")
        );
        assert_eq!(extract("<p>no title</p>").title, None);
    }

    #[test]
    fn malformed_markup_degrades_to_text() {
        assert_eq!(extract("a < b and c > d").text, "a < b and c > d");
        assert_eq!(extract("<p>unterminated <b").text, "unterminated");
        assert_eq!(extract("<script>never closed").text, "");
        assert_eq!(extract("").text, "");
        assert_eq!(extract("plain text only").text, "plain text only");
    }

    #[test]
    fn quoted_attribute_values_may_contain_angle_brackets() {
        let page = extract(r#"<img alt="a > b" src="x"><p title="x>y">kept</p>"#);
        assert_eq!(page.text, "kept");
    }

    #[test]
    fn entities_decode_named_numeric_and_leave_unknown_alone() {
        assert_eq!(
            decode_entities("&lt;a&gt; &amp;amp; &#65;&#x42; &nbsp;| &mdash; &bogus; & &copy"),
            "<a> &amp; AB  | \u{2014} &bogus; & &copy"
        );
        assert_eq!(
            decode_entities("&#0; &#xD800; &#99999999;"),
            "&#0; &#xD800; &#99999999;"
        );
    }

    #[test]
    fn attribute_lookup_handles_quoting_styles() {
        assert_eq!(
            attribute(r#" href="a b" class=x"#, "href").as_deref(),
            Some("a b")
        );
        assert_eq!(attribute(" href='q' ", "HREF").as_deref(), Some("q"));
        assert_eq!(
            attribute(" href=bare class=c", "class").as_deref(),
            Some("c")
        );
        assert_eq!(attribute(" disabled ", "disabled").as_deref(), Some(""));
        assert_eq!(attribute(" a=1 ", "b"), None);
    }

    #[test]
    fn blank_lines_collapse_and_edges_trim() {
        let page = extract("<div>\n\n\n</div><p>a</p><div></div><div></div><p>b</p>");
        assert_eq!(page.text, "a\nb");
    }
}
