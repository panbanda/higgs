//! `${VAR}` and `${VAR:-default}` references in string values.
//!
//! One resolver serves every string that can hold a secret or a path, in the server config and in
//! the MCP server definitions. A reference to an unset variable without a default is an error, so a
//! typo can never silently turn a credential into an empty string. `$${` writes a literal `${`.

use std::fmt;

/// Look up an environment variable; injectable so tests do not touch the process environment.
pub type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<String>;

pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvRefError {
    /// A `${` with no closing `}`.
    Unterminated,
    /// The variable name is empty or has characters other than letters, digits and `_`.
    InvalidName(String),
    /// The variable is not set and the reference gave no default.
    Unset(String),
}

impl EnvRefError {
    /// The variable the failure concerns, when there is one.
    pub fn variable(&self) -> Option<&str> {
        match self {
            Self::Unset(name) | Self::InvalidName(name) => Some(name),
            Self::Unterminated => None,
        }
    }
}

impl fmt::Display for EnvRefError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unterminated => formatter.write_str("unterminated ${ reference"),
            Self::InvalidName(name) => write!(formatter, "invalid variable name '{name}'"),
            Self::Unset(name) => write!(
                formatter,
                "environment variable {name} is not set (use ${{{name}:-default}} to give it a default)"
            ),
        }
    }
}

impl std::error::Error for EnvRefError {}

/// Whether `value` contains a `${` that is not escaped as `$${`.
pub fn has_reference(value: &str) -> bool {
    let mut rest = value;
    while let Some(position) = rest.find('$') {
        let after = rest.get(position + 1..).unwrap_or_default();
        if after.starts_with("${") {
            rest = after.get(2..).unwrap_or_default();
        } else if after.starts_with('{') {
            return true;
        } else {
            rest = after;
        }
    }
    false
}

/// Expand every reference in `value`.
///
/// `${VAR}` is replaced by the variable's value. `${VAR:-default}` uses the default when the
/// variable is unset or empty.
pub fn expand(value: &str, lookup: EnvLookup<'_>) -> Result<String, EnvRefError> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(position) = rest.find('$') {
        output.push_str(rest.get(..position).unwrap_or_default());
        let after = rest.get(position + 1..).unwrap_or_default();
        if let Some(escaped) = after.strip_prefix("${") {
            output.push_str("${");
            rest = escaped;
        } else if let Some(reference) = after.strip_prefix('{') {
            let end = reference.find('}').ok_or(EnvRefError::Unterminated)?;
            let body = reference.get(..end).unwrap_or_default();
            let (name, default) = body
                .split_once(":-")
                .map_or((body, None), |(name, default)| (name, Some(default)));
            if name.is_empty()
                || !name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
            {
                return Err(EnvRefError::InvalidName(name.to_owned()));
            }
            match (lookup(name), default) {
                (Some(resolved), _) if default.is_none() || !resolved.is_empty() => {
                    output.push_str(&resolved);
                }
                (_, Some(fallback)) => output.push_str(fallback),
                (_, None) => return Err(EnvRefError::Unset(name.to_owned())),
            }
            rest = reference.get(end + 1..).unwrap_or_default();
        } else {
            output.push('$');
            rest = after;
        }
    }
    output.push_str(rest);
    Ok(output)
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::literal_string_with_formatting_args
)]
#[cfg(test)]
mod tests {
    use super::*;

    fn lookup(name: &str) -> Option<String> {
        match name {
            "TOKEN" => Some("abc".to_owned()),
            "EMPTY" => Some(String::new()),
            "HOME_DIR" => Some("/home/me".to_owned()),
            _ => None,
        }
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(
            expand("no refs, cost $5", &lookup).unwrap(),
            "no refs, cost $5"
        );
        assert_eq!(expand("", &lookup).unwrap(), "");
    }

    #[test]
    fn a_set_variable_is_substituted_anywhere_in_the_text() {
        assert_eq!(expand("${TOKEN}", &lookup).unwrap(), "abc");
        assert_eq!(expand("Bearer ${TOKEN}!", &lookup).unwrap(), "Bearer abc!");
        assert_eq!(
            expand("${HOME_DIR}/models/${TOKEN}", &lookup).unwrap(),
            "/home/me/models/abc"
        );
    }

    #[test]
    fn an_unset_variable_without_a_default_is_an_error_naming_it() {
        let error = expand("${MISSING}", &lookup).unwrap_err();
        assert_eq!(error, EnvRefError::Unset("MISSING".to_owned()));
        assert_eq!(error.variable(), Some("MISSING"));
        assert!(error.to_string().contains("MISSING"));
    }

    #[test]
    fn a_default_applies_when_the_variable_is_unset_or_empty() {
        assert_eq!(expand("${MISSING:-fallback}", &lookup).unwrap(), "fallback");
        assert_eq!(expand("${EMPTY:-fallback}", &lookup).unwrap(), "fallback");
        assert_eq!(expand("${TOKEN:-fallback}", &lookup).unwrap(), "abc");
        assert_eq!(expand("${MISSING:-}", &lookup).unwrap(), "");
        assert_eq!(expand("${MISSING:-a:-b}", &lookup).unwrap(), "a:-b");
    }

    #[test]
    fn an_empty_variable_without_a_default_expands_to_empty() {
        assert_eq!(expand("[${EMPTY}]", &lookup).unwrap(), "[]");
    }

    #[test]
    fn a_doubled_dollar_escapes_a_literal_reference() {
        assert_eq!(expand("$${TOKEN}", &lookup).unwrap(), "${TOKEN}");
        assert!(!has_reference("$${TOKEN}"));
    }

    #[test]
    fn malformed_references_are_errors() {
        assert_eq!(
            expand("${TOKEN", &lookup).unwrap_err(),
            EnvRefError::Unterminated
        );
        assert!(matches!(
            expand("${}", &lookup),
            Err(EnvRefError::InvalidName(_))
        ));
        assert!(matches!(
            expand("${A-B}", &lookup),
            Err(EnvRefError::InvalidName(_))
        ));
        assert!(matches!(
            expand("${:-x}", &lookup),
            Err(EnvRefError::InvalidName(_))
        ));
    }

    #[test]
    fn has_reference_detects_unescaped_references_only() {
        assert!(has_reference("a ${B} c"));
        assert!(!has_reference("plain $B"));
        assert!(has_reference("$$ ${X}"));
    }
}
