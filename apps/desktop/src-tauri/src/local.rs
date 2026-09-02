//! Commands that touch the local machine rather than the HTTP API: the
//! Higgs config directory, the metrics JSONL log, pid files, and the CLI.

use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("HIGGS_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    directories::BaseDirs::new().map_or_else(
        || PathBuf::from("/tmp/higgs"),
        |dirs| dirs.home_dir().join(".config/higgs"),
    )
}

fn home_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

/// Resolves a user-supplied path and rejects anything outside the Higgs config
/// directory, so the renderer cannot read or write arbitrary files.
fn config_scoped(path: &str) -> Result<PathBuf, String> {
    let candidate = expand_home(path);
    let dir = config_dir();
    let absolute = if candidate.is_absolute() {
        candidate
    } else {
        dir.join(candidate)
    };
    if !normalized(&absolute).starts_with(normalized(&dir)) {
        return Err(format!(
            "path {} is outside the Higgs config directory",
            absolute.display()
        ));
    }
    Ok(absolute)
}

/// Like [`config_scoped`] but also accepts a `.jsonl` file anywhere under the
/// home directory, since `logging.metrics.path` may point outside the config
/// directory.
fn log_scoped(path: &str) -> Result<PathBuf, String> {
    if let Ok(scoped) = config_scoped(path) {
        return Ok(scoped);
    }
    let candidate = expand_home(path);
    let under_home =
        home_dir().is_some_and(|home| normalized(&candidate).starts_with(normalized(&home)));
    if candidate.extension().is_some_and(|ext| ext == "jsonl") && under_home {
        return Ok(candidate);
    }
    Err(format!(
        "path {} is not an allowed metrics log location",
        candidate.display()
    ))
}

/// Lexically removes `.` and `..` components without touching the filesystem,
/// so containment checks hold for files that do not exist yet.
fn normalized(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[derive(Debug, Serialize)]
pub struct Profile {
    /// `None` is the default `config.toml`.
    pub name: Option<String>,
    pub config_path: String,
}

#[derive(Debug, Serialize)]
pub struct ProfileList {
    pub config_dir: String,
    pub profiles: Vec<Profile>,
}

#[tauri::command]
pub fn list_profiles() -> ProfileList {
    let dir = config_dir();
    let mut profiles = vec![Profile {
        name: None,
        config_path: dir.join("config.toml").to_string_lossy().into_owned(),
    }];
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut named: Vec<Profile> = entries
            .flatten()
            .filter_map(|entry| {
                let file_name = entry.file_name();
                let file_name = file_name.to_string_lossy();
                let name = file_name
                    .strip_prefix("config.")?
                    .strip_suffix(".toml")?
                    .to_owned();
                if name.is_empty() {
                    return None;
                }
                Some(Profile {
                    name: Some(name),
                    config_path: entry.path().to_string_lossy().into_owned(),
                })
            })
            .collect();
        named.sort_by(|a, b| a.name.cmp(&b.name));
        profiles.extend(named);
    }
    ProfileList {
        config_dir: dir.to_string_lossy().into_owned(),
        profiles,
    }
}

#[derive(Debug, Serialize)]
pub struct ConfigFile {
    pub path: String,
    pub exists: bool,
    pub raw: String,
    pub parsed: serde_json::Value,
    pub parse_error: Option<String>,
}

#[tauri::command]
pub fn read_config(path: String) -> Result<ConfigFile, String> {
    let file = config_scoped(&path)?;
    let raw = std::fs::read_to_string(&file).ok();
    let exists = raw.is_some();
    let raw = raw.unwrap_or_default();
    let (parsed, parse_error) = match toml::from_str::<toml::Value>(&raw) {
        Ok(value) => (
            serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
            None,
        ),
        Err(error) => (serde_json::Value::Null, Some(error.to_string())),
    };
    Ok(ConfigFile {
        path: file.to_string_lossy().into_owned(),
        exists,
        raw,
        parsed,
        parse_error,
    })
}

fn write_private(path: &Path, contents: &str) -> Result<(), String> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt as _;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .map_err(|error| error.to_string())?;
    file.write_all(contents.as_bytes())
        .map_err(|error| error.to_string())
}

/// Writes raw TOML after checking it parses, so a typo cannot brick the daemon.
#[tauri::command]
pub fn write_config_raw(path: String, raw: String) -> Result<(), String> {
    toml::from_str::<toml::Value>(&raw).map_err(|error| format!("invalid TOML: {error}"))?;
    write_private(&config_scoped(&path)?, &raw)
}

/// Serializes a structured config (JSON from the form editor) to TOML.
/// Comments in the existing file are not preserved.
#[tauri::command]
pub fn write_config_structured(path: String, config: serde_json::Value) -> Result<String, String> {
    let value: toml::Value = serde_json::from_value(config).map_err(|error| error.to_string())?;
    let raw = toml::to_string_pretty(&value).map_err(|error| error.to_string())?;
    write_private(&config_scoped(&path)?, &raw)?;
    Ok(raw)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub timestamp: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub routing_method: Option<String>,
    pub status: u16,
    pub duration_ms: u64,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub ttft_ms: Option<u64>,
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct MetricsLog {
    pub path: String,
    pub exists: bool,
    pub records: Vec<RequestRecord>,
    /// Byte offset after the last record read; pass back as `since_offset`
    /// to fetch only new lines next time.
    pub offset: u64,
    /// True when the file was replaced or truncated since `since_offset`,
    /// meaning the caller should discard what it had.
    pub reset: bool,
}

/// Bytes read from the end of the log per call; enough for tens of
/// thousands of records without scanning multi-megabyte histories.
const LOG_TAIL_BYTES: u64 = 4 * 1024 * 1024;

#[tauri::command]
pub fn read_metrics_log(
    path: String,
    max_records: usize,
    since_offset: Option<u64>,
) -> Result<MetricsLog, String> {
    let file_path = log_scoped(&path)?;
    let display_path = file_path.to_string_lossy().into_owned();
    let Ok(file) = std::fs::File::open(&file_path) else {
        return Ok(MetricsLog {
            path: display_path,
            exists: false,
            records: Vec::new(),
            offset: 0,
            reset: since_offset.is_some(),
        });
    };
    let length = file.metadata().map_or(0, |meta| meta.len());
    let reset = since_offset.is_some_and(|offset| offset > length);
    let start = match since_offset {
        Some(offset) if !reset => offset,
        _ => length.saturating_sub(LOG_TAIL_BYTES),
    };
    let skip_partial_first_line = since_offset.is_none() && start > 0;
    let mut reader = BufReader::new(file);
    if reader.seek(SeekFrom::Start(start)).is_err() {
        return Ok(MetricsLog {
            path: display_path,
            exists: true,
            records: Vec::new(),
            offset: start,
            reset,
        });
    }
    let mut offset = start;
    let mut records = Vec::new();
    let mut line = String::new();
    let mut first = true;
    loop {
        line.clear();
        let Ok(read) = reader.read_line(&mut line) else {
            break;
        };
        if read == 0 {
            break;
        }
        // Only complete lines advance the offset so a half-written record is
        // re-read on the next call.
        if !line.ends_with('\n') {
            break;
        }
        offset += u64::try_from(read).unwrap_or(0);
        let skip = first && skip_partial_first_line;
        first = false;
        if skip {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<RequestRecord>(line.trim()) {
            records.push(record);
        }
    }
    if records.len() > max_records {
        records.drain(..records.len() - max_records);
    }
    Ok(MetricsLog {
        path: display_path,
        exists: true,
        records,
        offset,
        reset,
    })
}

#[derive(Debug, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub pid: Option<i32>,
    pub pid_path: String,
    pub log_path: String,
}

fn pid_alive(pid: i32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|status| status.success())
}

#[tauri::command]
pub fn daemon_status(profile: Option<String>) -> DaemonStatus {
    let dir = config_dir();
    let (pid_path, log_path) = match profile.as_deref() {
        Some(name) => (
            dir.join(format!("higgs.{name}.pid")),
            dir.join(format!("higgs.{name}.log")),
        ),
        None => (dir.join("higgs.pid"), dir.join("higgs.log")),
    };
    let pid = std::fs::read_to_string(&pid_path)
        .ok()
        .and_then(|text| text.trim().parse::<i32>().ok());
    DaemonStatus {
        running: pid.is_some_and(pid_alive),
        pid,
        pid_path: pid_path.to_string_lossy().into_owned(),
        log_path: log_path.to_string_lossy().into_owned(),
    }
}

#[tauri::command]
pub fn read_text_tail(path: String, max_bytes: u64) -> Result<String, String> {
    let file_path = config_scoped(&path)?;
    let file = std::fs::File::open(&file_path).map_err(|error| error.to_string())?;
    let length = file.metadata().map_or(0, |meta| meta.len());
    let mut reader = BufReader::new(file);
    let start = length.saturating_sub(max_bytes);
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|error| error.to_string())?;
    let mut text = String::new();
    std::io::Read::read_to_string(&mut reader, &mut text).map_err(|error| error.to_string())?;
    Ok(text)
}

#[derive(Debug, Serialize)]
pub struct CommandOutput {
    pub program: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// GUI apps do not inherit the shell PATH, so resolve the binary through a
/// login shell unless the user pinned a path in settings.
fn resolve_higgs(binary: Option<&str>) -> Result<String, String> {
    if let Some(path) = binary.filter(|value| !value.trim().is_empty()) {
        let is_higgs = Path::new(path)
            .file_name()
            .is_some_and(|name| name == "higgs");
        if !is_higgs {
            return Err(
                "the Higgs binary path must point at an executable named `higgs`".to_owned(),
            );
        }
        return Ok(path.to_owned());
    }
    let output = Command::new("/bin/zsh")
        .args(["-lc", "command -v higgs"])
        .output()
        .map_err(|error| error.to_string())?;
    let found = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && !found.is_empty() {
        Ok(found)
    } else {
        Err("could not find `higgs` on PATH; set the binary path in Settings".to_owned())
    }
}

/// Only whitelisted subcommands run from the UI so the bridge cannot be
/// turned into a general shell.
const ALLOWED_SUBCOMMANDS: &[&str] = &["doctor", "start", "stop", "config", "--version"];

#[tauri::command]
pub async fn run_higgs(binary: Option<String>, args: Vec<String>) -> Result<CommandOutput, String> {
    let subcommand = args.first().map(String::as_str).unwrap_or_default();
    if !ALLOWED_SUBCOMMANDS.contains(&subcommand) {
        return Err(format!("subcommand not allowed: {subcommand}"));
    }
    let program = resolve_higgs(binary.as_deref())?;
    let program_for_task = program.clone();
    let output = tauri::async_runtime::spawn_blocking(move || {
        Command::new(&program_for_task)
            .args(&args)
            .env("NO_COLOR", "1")
            .output()
    })
    .await
    .map_err(|error| error.to_string())?
    .map_err(|error| format!("failed to run {program}: {error}"))?;
    Ok(CommandOutput {
        program,
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: strip_ansi(&String::from_utf8_lossy(&output.stderr)),
    })
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

#[derive(Debug, Serialize)]
pub struct ModelCacheInfo {
    pub path: String,
    pub cached: bool,
    pub size_bytes: u64,
    pub location: Option<String>,
}

fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let Ok(meta) = entry.metadata() else { return 0 };
            if meta.is_dir() {
                dir_size(&entry.path())
            } else {
                meta.len()
            }
        })
        .sum()
}

#[tauri::command]
pub fn model_cache_info(path: String) -> ModelCacheInfo {
    let direct = expand_home(&path);
    if direct.is_dir() {
        return ModelCacheInfo {
            path,
            cached: true,
            size_bytes: dir_size(&direct),
            location: Some(direct.to_string_lossy().into_owned()),
        };
    }
    let hub = std::env::var("HF_HOME")
        .map(|home| PathBuf::from(home).join("hub"))
        .ok()
        .or_else(|| home_dir().map(|home| home.join(".cache/huggingface/hub")));
    let repo_dir = hub.map(|hub| hub.join(format!("models--{}", path.replace('/', "--"))));
    match repo_dir {
        Some(dir) if dir.is_dir() => ModelCacheInfo {
            path,
            cached: true,
            // Blobs hold the real bytes; snapshots are symlinks into them.
            size_bytes: dir_size(&dir.join("blobs")),
            location: Some(dir.to_string_lossy().into_owned()),
        },
        _ => ModelCacheInfo {
            path,
            cached: false,
            size_bytes: 0,
            location: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_sequences() {
        assert_eq!(strip_ansi("\x1b[32m[PASS]\x1b[0m ok"), "[PASS] ok");
    }

    #[test]
    fn parses_log_line() {
        let line = r#"{"timestamp":"2026-02-25T22:45:14Z","model":"m","provider":"higgs","routing_method":"pattern","status":200,"duration_ms":5,"input_tokens":1,"output_tokens":2,"error":null}"#;
        let record: RequestRecord = serde_json::from_str(line).expect("parses");
        assert_eq!(record.duration_ms, 5);
        assert_eq!(record.provider.as_deref(), Some("higgs"));
    }

    #[test]
    fn incremental_log_reads_only_new_lines() {
        // Under the home directory so `log_scoped` accepts it.
        let dir = home_dir()
            .expect("home dir")
            .join(format!(".cache/higgs-desktop-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("metrics.jsonl");
        let line = |model: &str| {
            format!(
                "{{\"timestamp\":\"2026-01-01T00:00:00Z\",\"model\":\"{model}\",\"status\":200,\"duration_ms\":1,\"input_tokens\":1,\"output_tokens\":1}}\n"
            )
        };
        std::fs::write(&path, line("a")).expect("write");
        let first = read_metrics_log(path.to_string_lossy().into_owned(), 100, None).expect("read");
        assert_eq!(first.records.len(), 1);
        assert!(!first.reset);

        let mut existing = std::fs::read_to_string(&path).expect("read");
        existing.push_str(&line("b"));
        existing.push_str("{\"partial\":");
        std::fs::write(&path, existing).expect("write");
        let second = read_metrics_log(path.to_string_lossy().into_owned(), 100, Some(first.offset))
            .expect("read");
        assert_eq!(second.records.len(), 1);
        assert_eq!(second.records[0].model.as_deref(), Some("b"));
        assert!(!second.reset);

        std::fs::write(&path, line("c")).expect("write");
        let third = read_metrics_log(
            path.to_string_lossy().into_owned(),
            100,
            Some(second.offset),
        )
        .expect("read");
        assert!(third.reset, "truncation must signal reset");
        assert_eq!(third.records.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn log_line_keeps_timing_fields() {
        let line = r#"{"timestamp":"2026-02-25T22:45:14Z","model":"m","provider":"higgs","status":200,"duration_ms":5,"input_tokens":1,"output_tokens":2,"error":null,"ttft_ms":112,"cached_tokens":54}"#;
        let record: RequestRecord = serde_json::from_str(line).expect("parses");
        assert_eq!(record.ttft_ms, Some(112));
        assert_eq!(record.cached_tokens, Some(54));
        let back = serde_json::to_value(&record).expect("serializes");
        assert_eq!(back["ttft_ms"], 112);
    }

    #[test]
    fn config_paths_stay_inside_the_config_dir() {
        assert!(config_scoped("config.toml").is_ok());
        assert!(config_scoped("../../etc/passwd").is_err());
        assert!(config_scoped("/etc/passwd").is_err());
        let inside = config_dir().join("logs/metrics.jsonl");
        assert!(log_scoped(&inside.to_string_lossy()).is_ok());
        assert!(log_scoped("/etc/passwd").is_err());
    }

    #[test]
    fn rejects_unknown_subcommand() {
        let allowed = ALLOWED_SUBCOMMANDS.contains(&"rm");
        assert!(!allowed);
    }
}
