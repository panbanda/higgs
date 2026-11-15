//! A parsed TOML document with the bookkeeping needed to report problems against the file the
//! user wrote: which line each key was on, even after deprecated keys were rewritten.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

use toml_edit::{Document, DocumentMut, Item, Table};

/// One step into a document: a table key or an array-of-tables index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Segment {
    Key(String),
    Index(usize),
}

impl Segment {
    pub fn key(name: &str) -> Self {
        Self::Key(name.to_owned())
    }
}

/// Render a path the way users write keys: `models[1].kv_cache.bits`.
pub fn render_path(path: &[Segment]) -> String {
    let mut rendered = String::new();
    for segment in path {
        match segment {
            Segment::Key(name) => {
                if !rendered.is_empty() {
                    rendered.push('.');
                }
                rendered.push_str(name);
            }
            Segment::Index(index) => {
                let _ = write!(rendered, "[{index}]");
            }
        }
    }
    rendered
}

/// A problem found while loading a configuration, located in a file when that is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub file: Option<PathBuf>,
    pub line: Option<usize>,
    pub message: String,
}

impl Diagnostic {
    pub fn new<M: Into<String>>(file: Option<&Path>, line: Option<usize>, message: M) -> Self {
        Self {
            file: file.map(Path::to_path_buf),
            line,
            message: message.into(),
        }
    }

    pub fn bare<M: Into<String>>(message: M) -> Self {
        Self::new(None, None, message)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.file, self.line) {
            (Some(file), Some(line)) => {
                write!(formatter, "{}:{line}: {}", file.display(), self.message)
            }
            (Some(file), None) => write!(formatter, "{}: {}", file.display(), self.message),
            (None, Some(line)) => write!(formatter, "line {line}: {}", self.message),
            (None, None) => formatter.write_str(&self.message),
        }
    }
}

/// Every problem that stopped a configuration from loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub diagnostics: Vec<Diagnostic>,
}

impl ConfigError {
    pub fn single(diagnostic: Diagnostic) -> Self {
        Self {
            diagnostics: vec![diagnostic],
        }
    }

    pub fn message<M: Into<String>>(message: M) -> Self {
        Self::single(Diagnostic::bare(message))
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, diagnostic) in self.diagnostics.iter().enumerate() {
            if position > 0 {
                formatter.write_str("\n")?;
            }
            write!(formatter, "{diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for String {
    fn from(error: ConfigError) -> Self {
        error.to_string()
    }
}

/// Where each key of the original file was, by rendered path.
#[derive(Debug, Clone, Default)]
pub struct LineIndex {
    lines: HashMap<String, usize>,
}

impl LineIndex {
    pub fn build(source: &str, document: &Document<&str>) -> Self {
        let mut index = Self::default();
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(offset, _)| offset + 1))
            .collect();
        let line_of = |offset: usize| line_starts.partition_point(|start| *start <= offset);
        index.walk_table(document.as_table(), &mut Vec::new(), &line_of);
        index
    }

    fn walk_table(
        &mut self,
        table: &Table,
        path: &mut Vec<Segment>,
        line_of: &dyn Fn(usize) -> usize,
    ) {
        for (name, item) in table {
            path.push(Segment::key(name));
            let key_span = table.key(name).and_then(toml_edit::Key::span);
            if let Some(span) = key_span.or_else(|| item.span()) {
                self.lines.insert(render_path(path), line_of(span.start));
            }
            match item {
                Item::Table(child) => self.walk_table(child, path, line_of),
                Item::ArrayOfTables(rows) => {
                    for (position, row) in rows.iter().enumerate() {
                        path.push(Segment::Index(position));
                        if let Some(span) = row.span() {
                            self.lines.insert(render_path(path), line_of(span.start));
                        }
                        self.walk_table(row, path, line_of);
                        path.pop();
                    }
                }
                Item::Value(toml_edit::Value::InlineTable(inline)) => {
                    for (inner_name, inner) in inline {
                        path.push(Segment::key(inner_name));
                        if let Some(span) = inner.span() {
                            self.lines.insert(render_path(path), line_of(span.start));
                        }
                        path.pop();
                    }
                }
                Item::Value(_) | Item::None => {}
            }
            path.pop();
        }
    }

    /// The line a path was written on, falling back to the nearest enclosing path that has one.
    pub fn line_of(&self, path: &[Segment]) -> Option<usize> {
        let mut prefix = path;
        while !prefix.is_empty() {
            if let Some(line) = self.lines.get(&render_path(prefix)) {
                return Some(*line);
            }
            prefix = prefix.split_last().map_or(&[][..], |(_, rest)| rest);
        }
        None
    }

    /// Record that `new` now holds what was written at `old`.
    pub fn alias(&mut self, new: &[Segment], old: &[Segment]) {
        if let Some(line) = self.line_of(old) {
            self.lines.insert(render_path(new), line);
        }
    }
}

/// A document ready for rewriting, with the line index of the text it was parsed from.
pub struct ParsedDocument {
    pub document: DocumentMut,
    pub lines: LineIndex,
}

pub fn parse_document(source: &str, file: Option<&Path>) -> Result<ParsedDocument, ConfigError> {
    let spanned = Document::parse(source).map_err(|error| {
        let line = error.span().map(|span| {
            source
                .get(..span.start)
                .map_or(1, |before| before.matches('\n').count() + 1)
        });
        ConfigError::single(Diagnostic::new(
            file,
            line,
            format!("invalid TOML: {}", error.message()),
        ))
    })?;
    let lines = LineIndex::build(source, &spanned);
    Ok(ParsedDocument {
        document: spanned.into_mut(),
        lines,
    })
}

/// The table at `path` (every segment a key, or an index into an array of tables).
pub fn table_at<'a>(root: &'a Table, path: &[Segment]) -> Option<&'a Table> {
    let mut current = root;
    let mut remaining = path;
    while let Some((first, rest)) = remaining.split_first() {
        let Segment::Key(name) = first else {
            return None;
        };
        let item = current.get(name)?;
        match (item, rest.split_first()) {
            (Item::Table(table), _) => {
                current = table;
                remaining = rest;
            }
            (Item::ArrayOfTables(rows), Some((Segment::Index(index), after_index))) => {
                current = rows.get(*index)?;
                remaining = after_index;
            }
            _ => return None,
        }
    }
    Some(current)
}

pub fn table_at_mut<'a>(root: &'a mut Table, path: &[Segment]) -> Option<&'a mut Table> {
    let mut current = root;
    let mut remaining = path;
    while let Some((first, rest)) = remaining.split_first() {
        let Segment::Key(name) = first else {
            return None;
        };
        let item = current.get_mut(name)?;
        match (item, rest.split_first()) {
            (Item::Table(table), _) => {
                current = table;
                remaining = rest;
            }
            (Item::ArrayOfTables(rows), Some((Segment::Index(index), after_index))) => {
                current = rows.get_mut(*index)?;
                remaining = after_index;
            }
            _ => return None,
        }
    }
    Some(current)
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_render_as_users_write_them() {
        let path = [
            Segment::key("models"),
            Segment::Index(1),
            Segment::key("kv_cache"),
            Segment::key("bits"),
        ];
        assert_eq!(render_path(&path), "models[1].kv_cache.bits");
        assert_eq!(render_path(&[]), "");
    }

    #[test]
    fn line_index_finds_keys_tables_and_array_rows() {
        let source = "[server]\nport = 1\n\n[[models]]\npath = \"a\"\n\n[[models]]\npath = \"b\"\nname = \"n\"\n";
        let parsed = parse_document(source, None).unwrap();
        let line = |path: &[Segment]| parsed.lines.line_of(path);
        assert_eq!(
            line(&[Segment::key("server"), Segment::key("port")]),
            Some(2)
        );
        assert_eq!(line(&[Segment::key("models"), Segment::Index(1)]), Some(7));
        assert_eq!(
            line(&[
                Segment::key("models"),
                Segment::Index(1),
                Segment::key("name")
            ]),
            Some(9)
        );
    }

    #[test]
    fn a_missing_key_falls_back_to_its_enclosing_table() {
        let parsed = parse_document("\n[server]\nport = 1\n", None).unwrap();
        assert_eq!(
            parsed
                .lines
                .line_of(&[Segment::key("server"), Segment::key("nope")]),
            Some(2)
        );
    }

    #[test]
    fn invalid_toml_reports_the_line() {
        let error = parse_document("[server]\nport = \n", Some(Path::new("c.toml")))
            .err()
            .unwrap();
        let text = error.to_string();
        assert!(text.starts_with("c.toml:2:"), "{text}");
    }

    #[test]
    fn diagnostics_display_file_and_line_when_known() {
        assert_eq!(
            Diagnostic::new(Some(Path::new("a.toml")), Some(3), "bad").to_string(),
            "a.toml:3: bad"
        );
        assert_eq!(Diagnostic::bare("bad").to_string(), "bad");
    }
}
