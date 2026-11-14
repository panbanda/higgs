//! `higgs config migrate`: rewrite a file that uses deprecated keys in the current schema.
//!
//! The rewrite edits the parsed document, so comments and key order survive as far as the
//! structure allows. `${VAR}` references are never expanded and unknown keys are an error, never
//! silently dropped.

use std::path::Path;

use super::document::{ConfigError, Diagnostic, parse_document};
use super::legacy::{Deprecation, Rewrite};
use super::schema::find_unknown_keys;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// The file in the current schema.
    pub text: String,
    /// Every deprecated key that was rewritten.
    pub rewritten: Vec<Deprecation>,
}

impl Migration {
    pub const fn changed(&self) -> bool {
        !self.rewritten.is_empty()
    }
}

pub fn migrate(source: &str, file: Option<&Path>) -> Result<Migration, ConfigError> {
    let mut parsed = parse_document(source, file)?;
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let rewritten = {
        let mut rewrite = Rewrite::new(file, &mut parsed.lines);
        rewrite.apply(&mut parsed.document);
        diagnostics.extend(rewrite.errors);
        rewrite.deprecations
    };
    for unknown in find_unknown_keys(parsed.document.as_table()) {
        diagnostics.push(Diagnostic::new(
            file,
            parsed.lines.line_of(&unknown.path),
            unknown.message,
        ));
    }
    if !diagnostics.is_empty() {
        return Err(ConfigError { diagnostics });
    }
    let text = if rewritten.is_empty() {
        source.to_owned()
    } else {
        parsed.document.to_string()
    };
    Ok(Migration { text, rewritten })
}
