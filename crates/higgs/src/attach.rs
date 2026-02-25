use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::config::MetricsLogConfig;
use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};
use crate::metrics_log::rotated_path;

#[derive(Debug, Deserialize)]
struct LogEntry {
    timestamp: DateTime<Utc>,
    model: String,
    provider: String,
    routing_method: Option<String>,
    status: u16,
    duration_ms: u64,
    input_tokens: u64,
    output_tokens: u64,
    error: Option<String>,
}

pub fn parse_log_entry(line: &str) -> Option<RequestRecord> {
    let entry: LogEntry = serde_json::from_str(line).ok()?;
    let age = (Utc::now() - entry.timestamp)
        .to_std()
        .unwrap_or(Duration::ZERO);
    let timestamp = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
    Some(RequestRecord {
        id: 0,
        timestamp,
        wallclock: entry.timestamp,
        model: entry.model,
        provider: entry.provider,
        routing_method: match entry.routing_method.as_deref() {
            Some("higgs") => RoutingMethod::Higgs,
            Some("pattern") => RoutingMethod::Pattern,
            Some("auto") => RoutingMethod::Auto,
            Some("default" | _) | None => RoutingMethod::Default,
        },
        status: entry.status,
        duration: Duration::from_millis(entry.duration_ms),
        input_tokens: entry.input_tokens,
        output_tokens: entry.output_tokens,
        error_body: entry.error,
    })
}

pub fn load_history(config: &MetricsLogConfig, store: &MetricsStore) {
    let base = Path::new(&config.path);
    let cutoff =
        Utc::now() - chrono::Duration::from_std(store.window()).unwrap_or(chrono::Duration::zero());

    // Read rotated files oldest-first: .max_files, .max_files-1, ..., .1, then current
    let mut paths = Vec::new();
    for i in (1..=config.max_files).rev() {
        paths.push(rotated_path(base, i));
    }
    paths.push(base.to_path_buf());

    for path in paths {
        let Ok(file) = std::fs::File::open(&path) else {
            continue;
        };
        let reader = BufReader::new(file);
        for result in reader.lines() {
            let Ok(line) = result else {
                continue;
            };
            if line.is_empty() {
                continue;
            }
            let Some(record) = parse_log_entry(&line) else {
                continue;
            };
            if record.wallclock < cutoff {
                continue;
            }
            store.record_silent(record);
        }
    }
}

pub fn tail_log(path: &Path, store: &Arc<MetricsStore>, stop: &Arc<AtomicBool>) {
    use std::os::unix::fs::MetadataExt;

    let mut position: u64 = std::fs::metadata(path).map_or(0, |m| m.len());
    let mut current_ino: u64 = std::fs::metadata(path).map_or(0, |m| m.ino());

    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(250));

        let Ok(file) = std::fs::File::open(path) else {
            continue;
        };

        let Ok(meta) = file.metadata() else {
            continue;
        };
        let file_len = meta.len();
        let file_ino = meta.ino();

        // Detect rotation: inode changed or file shrunk
        if file_ino != current_ino || file_len < position {
            position = 0;
            current_ino = file_ino;
        }

        if file_len == position {
            continue;
        }

        let mut reader = BufReader::new(file);
        if reader.seek(SeekFrom::Start(position)).is_err() {
            continue;
        }

        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let n_u64 = u64::try_from(n).unwrap_or(u64::MAX);
                    position = position.saturating_add(n_u64);
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if let Some(record) = parse_log_entry(trimmed) {
                        store.record_silent(record);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::option_if_let_else
)]
mod tests {
    use super::*;
    use std::fs;

    fn recent_timestamp() -> String {
        Utc::now().to_rfc3339()
    }

    fn old_timestamp() -> String {
        (Utc::now() - chrono::Duration::hours(2)).to_rfc3339()
    }

    fn make_entry(ts: &str, model: &str, error: Option<&str>) -> String {
        let error_json = match error {
            Some(e) => format!("\"{e}\""),
            None => "null".to_owned(),
        };
        format!(
            r#"{{"timestamp":"{ts}","model":"{model}","provider":"anthropic","status":200,"duration_ms":100,"input_tokens":50,"output_tokens":75,"error":{error_json}}}"#
        )
    }

    #[test]
    fn parse_valid_entry() {
        let ts = recent_timestamp();
        let line = make_entry(&ts, "claude-opus-4-6", None);
        let record = parse_log_entry(&line).expect("should parse");
        assert_eq!(record.model, "claude-opus-4-6");
        assert_eq!(record.provider, "anthropic");
        assert_eq!(record.status, 200);
        assert_eq!(record.duration.as_millis(), 100);
        assert_eq!(record.input_tokens, 50);
        assert_eq!(record.output_tokens, 75);
        assert!(record.error_body.is_none());
    }

    #[test]
    fn parse_entry_with_error() {
        let ts = recent_timestamp();
        let line = make_entry(&ts, "opus", Some("rate limited"));
        let record = parse_log_entry(&line).expect("should parse");
        assert_eq!(record.error_body.as_deref(), Some("rate limited"));
    }

    #[test]
    fn parse_missing_fields() {
        let line = r#"{"timestamp":"2025-01-01T00:00:00Z","model":"opus"}"#;
        assert!(parse_log_entry(line).is_none());
    }

    #[test]
    fn parse_malformed_json() {
        assert!(parse_log_entry("not json").is_none());
        assert!(parse_log_entry("").is_none());
        assert!(parse_log_entry("{").is_none());
        assert!(parse_log_entry("{}").is_none());
    }

    #[test]
    fn parse_routing_method_variants() {
        let ts = recent_timestamp();
        let make = |method: &str| -> String {
            format!(
                r#"{{"timestamp":"{ts}","model":"m","provider":"p","routing_method":"{method}","status":200,"duration_ms":10,"input_tokens":1,"output_tokens":1,"error":null}}"#
            )
        };
        assert_eq!(
            parse_log_entry(&make("higgs")).unwrap().routing_method,
            RoutingMethod::Higgs
        );
        assert_eq!(
            parse_log_entry(&make("pattern")).unwrap().routing_method,
            RoutingMethod::Pattern
        );
        assert_eq!(
            parse_log_entry(&make("auto")).unwrap().routing_method,
            RoutingMethod::Auto
        );
        assert_eq!(
            parse_log_entry(&make("default")).unwrap().routing_method,
            RoutingMethod::Default
        );
        assert_eq!(
            parse_log_entry(&make("unknown_future"))
                .unwrap()
                .routing_method,
            RoutingMethod::Default
        );
    }

    #[test]
    fn parse_routing_method_missing() {
        let ts = recent_timestamp();
        let line = format!(
            r#"{{"timestamp":"{ts}","model":"m","provider":"p","status":200,"duration_ms":10,"input_tokens":1,"output_tokens":1,"error":null}}"#
        );
        assert_eq!(
            parse_log_entry(&line).unwrap().routing_method,
            RoutingMethod::Default
        );
    }

    #[test]
    fn load_history_reads_rotated_files() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("metrics.jsonl");
        let ts = recent_timestamp();

        fs::write(
            rotated_path(&base, 2),
            format!("{}\n", make_entry(&ts, "oldest", None)),
        )
        .unwrap();
        fs::write(
            rotated_path(&base, 1),
            format!("{}\n", make_entry(&ts, "middle", None)),
        )
        .unwrap();
        fs::write(&base, format!("{}\n", make_entry(&ts, "newest", None))).unwrap();

        let config = MetricsLogConfig {
            enabled: true,
            path: base.to_string_lossy().to_string(),
            max_size_mb: 50,
            max_files: 5,
        };
        let store = MetricsStore::new(Duration::from_secs(3600));
        load_history(&config, &store);

        let snap = store.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].model, "oldest");
        assert_eq!(snap[1].model, "middle");
        assert_eq!(snap[2].model, "newest");
    }

    #[test]
    fn load_history_skips_old_entries() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("metrics.jsonl");
        let recent = recent_timestamp();
        let old = old_timestamp();

        let content = format!(
            "{}\n{}\n",
            make_entry(&old, "old-model", None),
            make_entry(&recent, "new-model", None)
        );
        fs::write(&base, content).unwrap();

        let config = MetricsLogConfig {
            enabled: true,
            path: base.to_string_lossy().to_string(),
            max_size_mb: 50,
            max_files: 5,
        };
        let store = MetricsStore::new(Duration::from_secs(3600));
        load_history(&config, &store);

        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].model, "new-model");
    }

    #[test]
    fn load_history_skips_malformed_lines() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("metrics.jsonl");
        let ts = recent_timestamp();

        let content = format!(
            "garbage\n{}\n{{}}\n{}\n",
            make_entry(&ts, "good1", None),
            make_entry(&ts, "good2", None)
        );
        fs::write(&base, content).unwrap();

        let config = MetricsLogConfig {
            enabled: true,
            path: base.to_string_lossy().to_string(),
            max_size_mb: 50,
            max_files: 5,
        };
        let store = MetricsStore::new(Duration::from_secs(3600));
        load_history(&config, &store);

        let snap = store.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn load_history_handles_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("metrics.jsonl");

        let config = MetricsLogConfig {
            enabled: true,
            path: base.to_string_lossy().to_string(),
            max_size_mb: 50,
            max_files: 5,
        };
        let store = MetricsStore::new(Duration::from_secs(3600));
        load_history(&config, &store);

        assert_eq!(store.snapshot().len(), 0);
    }
}
