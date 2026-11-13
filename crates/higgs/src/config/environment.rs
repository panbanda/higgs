//! The process environment as an injectable snapshot, so tests never touch real variables.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct Environment {
    variables: BTreeMap<String, String>,
}

impl Environment {
    /// A snapshot of the running process's environment.
    pub fn process() -> Self {
        Self {
            variables: std::env::vars().collect(),
        }
    }

    pub fn from_pairs<'a, I: IntoIterator<Item = (&'a str, &'a str)>>(pairs: I) -> Self {
        Self {
            variables: pairs
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
                .collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(String::as_str)
    }

    pub fn lookup(&self, name: &str) -> Option<String> {
        self.get(name).map(str::to_owned)
    }

    /// Variables whose names start with `prefix`, in name order.
    pub fn with_prefix<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = (&'a str, &'a str)> + 'a {
        self.variables
            .iter()
            .filter(move |(name, _)| name.starts_with(prefix))
            .map(|(name, value)| (name.as_str(), value.as_str()))
    }
}
