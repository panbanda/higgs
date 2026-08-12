use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use crate::config;

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
    if raw == "true" {
        toml_edit::value(true)
    } else if raw == "false" {
        toml_edit::value(false)
    } else if let Ok(n) = raw.parse::<i64>() {
        toml_edit::value(n)
    } else if let Ok(f) = raw.parse::<f64>() {
        toml_edit::value(f)
    } else {
        toml_edit::value(raw)
    }
}

#[allow(clippy::print_stderr)]
pub fn config_set(config_path: &Path, key: &str, value: &str) {
    let segments: Vec<&str> = key.split('.').collect();
    if segments.iter().any(|s| s.is_empty()) {
        eprintln!("invalid key: {key}");
        std::process::exit(1);
    }

    let original_exists = config_path.exists();
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
    if let Err(err) = config::load_config_file(config_path, None) {
        if !bootstrap_config_allowed(&doc) {
            let _ = restore_original_config(config_path, &original, original_exists);
            eprintln!("refusing to keep invalid config after setting {key}: {err}");
            std::process::exit(1);
        }
    }
}

fn bootstrap_config_allowed(doc: &toml_edit::DocumentMut) -> bool {
    let root = doc.as_table();
    !root.contains_key("models") && !root.contains_key("provider")
}

fn restore_original_config(
    config_path: &Path,
    original: &str,
    original_exists: bool,
) -> std::io::Result<()> {
    if original_exists {
        fs::write(config_path, original)
    } else {
        match fs::remove_file(config_path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }
}

pub fn config_lookup(content: &str, key: &str) -> Result<String, String> {
    let doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|e| format!("failed to parse config: {e}"))?;

    let segments: Vec<&str> = key.split('.').collect();
    let mut current = doc.as_item();
    for &seg in &segments {
        current = current
            .get(seg)
            .ok_or_else(|| format!("key not found: {key}"))?;
    }

    current
        .as_value()
        .map(format_toml_value)
        .ok_or_else(|| format!("key '{key}' is a table, not a value"))
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
        let doc = set_and_parse("", "logging.metrics.enabled", "true");
        assert_eq!(doc["logging"]["metrics"]["enabled"].as_bool(), Some(true));
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
    fn set_bool_value() {
        let doc = set_and_parse("", "flag", "true");
        assert_eq!(doc["flag"].as_bool(), Some(true));
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
    fn bootstrap_config_allowed_only_for_empty_models_and_providers() {
        let bootstrap: toml_edit::DocumentMut = "[server]\nport = 8000\n".parse().unwrap();
        assert!(bootstrap_config_allowed(&bootstrap));

        let with_provider: toml_edit::DocumentMut =
            "[provider.openai]\nurl = \"http://localhost\"\n"
                .parse()
                .unwrap();
        assert!(!bootstrap_config_allowed(&with_provider));

        let with_models: toml_edit::DocumentMut =
            "[[models]]\npath = \"mlx-community/Llama-3.2-1B-Instruct-4bit\"\n"
                .parse()
                .unwrap();
        assert!(!bootstrap_config_allowed(&with_models));
    }

    #[test]
    fn restore_original_config_rewrites_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "invalid = [\n").unwrap();

        restore_original_config(&path, "original = true\n", true).unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "original = true\n");
    }

    #[test]
    fn restore_original_config_removes_new_file_when_original_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "invalid = [\n").unwrap();

        restore_original_config(&path, "", false).unwrap();

        assert!(!path.exists());
    }
}
