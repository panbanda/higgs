use std::fs;
use std::path::Path;

use crate::config::{self, legacy, migrate, schema};

fn format_toml_value(value: &toml_edit::Value) -> String {
    value.as_str().map_or_else(
        || {
            value
                .as_bool()
                .map(|b| b.to_string())
                .or_else(|| value.as_integer().map(|n| n.to_string()))
                .unwrap_or_else(|| value.to_string())
        },
        str::to_owned,
    )
}

fn parse_toml_value(raw: &str) -> toml_edit::Item {
    raw.parse::<toml_edit::Value>()
        .map_or_else(|_| toml_edit::value(raw), toml_edit::Item::Value)
}

const TOP_LEVEL_SECTIONS: &str = "server, memory, defaults, providers, routing, observability, experimental, mcp, web; arrays such as models and routes must be edited as TOML";

/// Whether `segments` name a single value that `config set` may write: a leaf of the schema, with
/// user-chosen names for providers and MCP servers. Deprecated spellings are accepted too.
fn is_settable_key(segments: &[&str]) -> bool {
    let joined = segments.join(".");
    let current = schema::all_leaf_patterns().iter().any(|pattern| {
        let parts: Vec<&str> = pattern.split('.').collect();
        !pattern.contains("[]")
            && parts.len() == segments.len()
            && parts
                .iter()
                .zip(segments)
                .all(|(expected, given)| *expected == "*" || expected == given)
            && !matches!(parts.last(), Some(&"env" | &"headers"))
    });
    current || legacy_settable(&joined, segments)
}

fn legacy_settable(joined: &str, segments: &[&str]) -> bool {
    legacy::legacy_key_table().iter().any(|(old, _)| {
        if old.contains("[]") || old.contains('(') {
            return false;
        }
        let parts: Vec<&str> = old.split('.').collect();
        parts.len() == segments.len()
            && parts
                .iter()
                .zip(segments)
                .all(|(expected, given)| *expected == "*" || expected == given)
    }) || matches!(joined, "retention.enabled" | "retention.minutes")
}

#[allow(clippy::print_stderr)]
pub fn config_set(config_path: &Path, key: &str, value: &str) {
    let segments: Vec<&str> = key.split('.').collect();
    if segments.iter().any(|s| s.is_empty()) {
        eprintln!("invalid key: {key}");
        std::process::exit(1);
    }
    if !is_settable_key(&segments) {
        eprintln!(
            "unknown configuration key '{key}'; settable top-level sections: {TOP_LEVEL_SECTIONS}; arrays such as models and routes must be edited as TOML"
        );
        std::process::exit(1);
    }

    let original = fs::read_to_string(config_path).unwrap_or_default();
    let mut doc: toml_edit::DocumentMut = original.parse().unwrap_or_else(|e| {
        eprintln!("failed to parse {}: {e}", config_path.display());
        std::process::exit(1);
    });

    let (table_segments, leaf_slice) = segments.split_at(segments.len() - 1);
    let Some(&leaf) = leaf_slice.first() else {
        eprintln!("invalid key: {key}");
        std::process::exit(1);
    };

    let mut current = doc.as_table_mut();
    for &seg in table_segments {
        if !current.contains_key(seg) {
            current.insert(seg, toml_edit::Item::Table(toml_edit::Table::new()));
        }
        current = current[seg].as_table_mut().unwrap_or_else(|| {
            eprintln!("key segment '{seg}' is not a table");
            std::process::exit(1);
        });
    }
    current[leaf] = parse_toml_value(value);

    let rendered = doc.to_string();
    if let Err(err) = config::validate_config_contents(&rendered, config_path) {
        eprintln!("invalid value '{value}' for configuration key '{key}': {err}");
        std::process::exit(1);
    }
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("failed to create {}: {e}", parent.display());
            std::process::exit(1);
        });
    }
    config::write_private_file(config_path, &rendered).unwrap_or_else(|e| {
        eprintln!("failed to write {}: {e}", config_path.display());
        std::process::exit(1);
    });
}

fn lookup_in(doc: &toml_edit::DocumentMut, key: &str) -> Result<String, String> {
    let mut current = doc.as_item();
    for seg in key.split('.') {
        current = current
            .get(seg)
            .ok_or_else(|| format!("key not found: {key}"))?;
    }
    current
        .as_value()
        .map(format_toml_value)
        .ok_or_else(|| format!("key '{key}' is a table, not a value"))
}

/// Read `key` as written in the file, or, when the file uses deprecated spellings, under its
/// current name.
pub fn config_lookup(content: &str, key: &str) -> Result<String, String> {
    let doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|e| format!("failed to parse config: {e}"))?;
    lookup_in(&doc, key).or_else(|original_error| {
        let mut parsed =
            config::document::parse_document(content, None).map_err(|error| error.to_string())?;
        {
            let mut rewrite = legacy::Rewrite::new(None, &mut parsed.lines);
            rewrite.apply(&mut parsed.document);
        }
        lookup_in(&parsed.document, key).map_err(|_| original_error)
    })
}

fn read_config_or_exit(config_path: &Path) -> String {
    match fs::read_to_string(config_path) {
        Ok(contents) => contents,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            #[allow(clippy::print_stderr)]
            {
                eprintln!(
                    "config file not found: {}\nhint: run `higgs init` to create one",
                    config_path.display()
                );
            }
            std::process::exit(1);
        }
        Err(e) => {
            #[allow(clippy::print_stderr)]
            {
                eprintln!("failed to read {}: {e}", config_path.display());
            }
            std::process::exit(1);
        }
    }
}

/// `higgs config migrate`: print the file in the current schema, or rewrite it in place.
#[allow(clippy::print_stderr, clippy::print_stdout)]
pub fn config_migrate(config_path: &Path, write: bool) {
    let contents = read_config_or_exit(config_path);
    let migration = match migrate::migrate(&contents, Some(config_path)) {
        Ok(migration) => migration,
        Err(error) => {
            eprintln!("cannot migrate {}:\n{error}", config_path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = config::validate_config_contents(&migration.text, config_path) {
        eprintln!("the migrated file does not load, so nothing was changed:\n{error}");
        std::process::exit(1);
    }
    if !write {
        print!("{}", migration.text);
        for rewritten in &migration.rewritten {
            eprintln!(
                "migrated `{}` -> `{}`",
                rewritten.old_key, rewritten.new_key
            );
        }
        return;
    }
    if !migration.changed() {
        eprintln!("{} already uses the current schema", config_path.display());
        return;
    }
    let backup = backup_path(config_path);
    if let Err(error) = fs::copy(config_path, &backup) {
        eprintln!("cannot write the backup {}: {error}", backup.display());
        std::process::exit(1);
    }
    if let Err(error) = config::write_private_file(config_path, &migration.text) {
        eprintln!("failed to write {}: {error}", config_path.display());
        std::process::exit(1);
    }
    eprintln!(
        "migrated {} key(s) in {} (previous version saved as {})",
        migration.rewritten.len(),
        config_path.display(),
        backup.display()
    );
}

fn backup_path(config_path: &Path) -> std::path::PathBuf {
    let mut name = config_path.as_os_str().to_owned();
    name.push(".bak");
    let first = std::path::PathBuf::from(&name);
    if !first.exists() {
        return first;
    }
    (1..=u32::MAX)
        .map(|attempt| {
            let mut numbered = config_path.as_os_str().to_owned();
            numbered.push(format!(".bak.{attempt}"));
            std::path::PathBuf::from(numbered)
        })
        .find(|candidate| !candidate.exists())
        .unwrap_or(first)
}

/// `higgs config explain [model]`.
#[allow(clippy::print_stderr, clippy::print_stdout)]
pub fn config_explain(config_path: &Path, flags: &config::ServeArgs, target: Option<&str>) {
    let environment = config::Environment::process();
    let options = config::load::LoadOptions {
        flags: Some(flags),
        environment: environment.clone(),
    };
    let loaded = match config::load::load(config_path, &options) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    match config::explain::explain(&loaded, &environment, Some(flags), target) {
        Ok(text) => print!("{text}"),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    }
}

/// `higgs config check`: load and validate without touching models or the network.
#[allow(clippy::print_stderr, clippy::print_stdout)]
pub fn config_check(config_path: &Path) {
    let options = config::load::LoadOptions {
        flags: None,
        environment: config::Environment::process(),
    };
    match config::load::load(config_path, &options) {
        Ok(loaded) => {
            for deprecation in &loaded.config.notices.deprecations {
                eprintln!("warning: {deprecation}");
            }
            for entry in &loaded.config.notices.env_overrides {
                eprintln!(
                    "note: {} {} `{}`",
                    entry.variable,
                    if entry.applied {
                        "overrides"
                    } else {
                        "has an unusable value and does not override"
                    },
                    entry.key
                );
            }
            println!(
                "{}: ok ({} warning(s))",
                config_path.display(),
                loaded.config.notices.deprecations.len()
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[allow(clippy::print_stderr, clippy::print_stdout)]
pub fn config_get(config_path: &Path, key: &str) {
    let content = match fs::read_to_string(config_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "config file not found: {}\nhint: run `higgs init` to create one",
                config_path.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("failed to read {}: {e}", config_path.display());
            std::process::exit(1);
        }
    };

    match config_lookup(&content, key) {
        Ok(value) => println!("{value}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn set_and_parse(initial: &str, key: &str, value: &str) -> toml_edit::DocumentMut {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        if !initial.is_empty() {
            fs::write(&path, initial).unwrap();
        }
        config_set(&path, key, value);
        let content = fs::read_to_string(&path).unwrap();
        content.parse().unwrap()
    }

    #[test]
    fn set_creates_nested_tables() {
        let doc = set_and_parse("", "observability.metrics_log.enabled", "true");
        assert_eq!(
            doc["observability"]["metrics_log"]["enabled"].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn set_preserves_existing_values() {
        let doc = set_and_parse(
            "[server]\nhost = \"127.0.0.1\"\nport = 3100\n",
            "server.port",
            "3200",
        );
        assert_eq!(doc["server"]["host"].as_str(), Some("127.0.0.1"));
        assert_eq!(doc["server"]["port"].as_integer(), Some(3200));
    }

    #[test]
    fn set_integer_value() {
        let doc = set_and_parse("", "server.port", "3100");
        assert_eq!(doc["server"]["port"].as_integer(), Some(3100));
    }

    #[test]
    fn set_string_value() {
        let doc = set_and_parse("", "server.host", "localhost");
        assert_eq!(doc["server"]["host"].as_str(), Some("localhost"));
    }

    #[test]
    fn get_reads_nested_value() {
        let toml = "[server]\nhost = \"127.0.0.1\"\nport = 3100\n";
        assert_eq!(config_lookup(toml, "server.port").unwrap(), "3100");
        assert_eq!(config_lookup(toml, "server.host").unwrap(), "127.0.0.1");
    }

    #[test]
    fn get_missing_key_errors() {
        let toml = "[server]\nport = 3100\n";
        let err = config_lookup(toml, "server.host").unwrap_err();
        assert!(err.contains("key not found"));
    }

    #[test]
    fn get_table_key_errors() {
        let toml = "[server]\nport = 3100\n";
        let err = config_lookup(toml, "server").unwrap_err();
        assert!(err.contains("table, not a value"));
    }

    #[test]
    fn set_accepts_current_keys_of_every_section() {
        for (key, value) in [
            ("server.request_timeout", "90s"),
            ("server.max_body_size", "20MB"),
            ("memory.guard", "balanced"),
            ("defaults.idle_ttl", "15m"),
            ("defaults.kv_cache.bits", "4"),
            ("providers.openai.url", "https://api.openai.com"),
            ("routing.default", "openai"),
            ("routing.auto.timeout", "2s"),
            ("observability.retention", "30m"),
            ("observability.metrics_log.max_size", "10MB"),
            ("experimental.chunked_prefill.chunk", "512"),
            ("mcp.servers.fs.timeout", "10s"),
            ("web.timeout", "5s"),
        ] {
            assert!(
                is_settable_key(&key.split('.').collect::<Vec<_>>()),
                "{key}"
            );
            let doc = set_and_parse("", key, value);
            assert!(lookup_in(&doc, key).is_ok(), "{key}");
        }
    }

    #[test]
    fn set_still_accepts_deprecated_keys_and_rejects_unknown_ones() {
        for key in [
            "server.timeout",
            "local.idle_ttl_seconds",
            "provider.example.url",
            "default.provider",
            "auto_router.timeout_ms",
            "logging.metrics.max_size_mb",
            "retention.minutes",
        ] {
            assert!(
                is_settable_key(&key.split('.').collect::<Vec<_>>()),
                "{key}"
            );
        }
        for key in [
            "nonsense.key",
            "models.path",
            "server.prot",
            "mcp.servers.fs.env",
            "providers.p.bogus",
        ] {
            assert!(
                !is_settable_key(&key.split('.').collect::<Vec<_>>()),
                "{key}"
            );
        }
    }

    #[test]
    fn get_finds_a_current_key_in_a_file_that_uses_the_deprecated_spelling() {
        let toml = "[server]\ntimeout = 90\n[local]\nidle_ttl_seconds = 900\n";
        assert_eq!(
            config_lookup(toml, "server.request_timeout").unwrap(),
            "90s"
        );
        assert_eq!(config_lookup(toml, "defaults.idle_ttl").unwrap(), "15m");
        assert_eq!(config_lookup(toml, "server.timeout").unwrap(), "90");
        assert!(
            config_lookup(toml, "server.nope")
                .unwrap_err()
                .contains("key not found")
        );
    }

    #[test]
    fn backups_never_overwrite_an_earlier_backup() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        assert_eq!(backup_path(&path), dir.path().join("config.toml.bak"));
        fs::write(dir.path().join("config.toml.bak"), "").unwrap();
        assert_eq!(backup_path(&path), dir.path().join("config.toml.bak.1"));
        fs::write(dir.path().join("config.toml.bak.1"), "").unwrap();
        assert_eq!(backup_path(&path), dir.path().join("config.toml.bak.2"));
    }
}
