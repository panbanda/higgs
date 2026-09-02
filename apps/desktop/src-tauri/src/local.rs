//! Commands that touch the local machine rather than the HTTP API: the
//! Higgs config directory, the metrics JSONL log, pid files, and the CLI.

use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager as _};

use crate::paths::is_contained_strict;

/// Pure computation behind [`config_dir`], parameterized on the
/// `HIGGS_CONFIG_DIR` value so tests can exercise it without touching the
/// process environment or the real home directory.
fn config_dir_from(env: Option<&str>) -> PathBuf {
    if let Some(dir) = env {
        return PathBuf::from(dir);
    }
    directories::BaseDirs::new().map_or_else(
        || PathBuf::from("/tmp/higgs"),
        |dirs| dirs.home_dir().join(".config/higgs"),
    )
}

fn config_dir() -> PathBuf {
    config_dir_from(std::env::var("HIGGS_CONFIG_DIR").ok().as_deref())
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

/// Pure logic behind [`config_scoped`], parameterized on the config
/// directory so tests can exercise it against a temp directory.
fn config_scoped_in(dir: &Path, path: &str) -> Result<PathBuf, String> {
    let candidate = expand_home(path);
    let absolute = if candidate.is_absolute() {
        candidate
    } else {
        dir.join(candidate)
    };
    if !is_contained_strict(dir, &absolute) {
        return Err(format!(
            "path {} is outside the Higgs config directory",
            absolute.display()
        ));
    }
    Ok(absolute)
}

/// Resolves a user-supplied path and rejects anything outside the Higgs config
/// directory, so the renderer cannot read or write arbitrary files.
fn config_scoped(path: &str) -> Result<PathBuf, String> {
    config_scoped_in(&config_dir(), path)
}

/// Pure logic behind [`log_scoped`], parameterized on the config directory
/// and home directory so tests can exercise it without touching either of
/// the real ones.
fn log_scoped_in(dir: &Path, path: &str) -> Result<PathBuf, String> {
    if let Ok(scoped) = config_scoped_in(dir, path) {
        return Ok(scoped);
    }
    let candidate = expand_home(path);
    if configured_metrics_logs(dir)
        .iter()
        .any(|allowed| is_same_or_rotation(allowed, &candidate))
    {
        return Ok(candidate);
    }
    Err(format!(
        "path {} is not an allowed metrics log location",
        candidate.display()
    ))
}

/// Metrics log paths declared as `logging.metrics.path` by the config files
/// in `dir`. Only these (and their rotations) may be read outside `dir`.
fn configured_metrics_logs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("config") && name.ends_with(".toml")
        })
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|raw| toml::from_str::<toml::Value>(&raw).ok())
        .filter_map(|value| {
            value
                .get("logging")?
                .get("metrics")?
                .get("path")?
                .as_str()
                .map(expand_home)
        })
        .collect()
}

/// `[[models]].path` from every config file in the config dir. Local model
/// directories may only be inspected when a profile actually references them.
fn configured_model_paths(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("config") && name.ends_with(".toml")
        })
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|raw| toml::from_str::<toml::Value>(&raw).ok())
        .flat_map(|value| {
            value
                .get("models")
                .and_then(|models| models.as_array())
                .map(|models| {
                    models
                        .iter()
                        .filter_map(|model| model.get("path")?.as_str().map(expand_home))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect()
}

/// True when `candidate` is `allowed` itself or one of its numbered
/// rotations (`metrics.jsonl.1`, `metrics.jsonl.2`, ...).
fn is_same_or_rotation(allowed: &Path, candidate: &Path) -> bool {
    if candidate == allowed {
        return true;
    }
    let (Some(parent), Some(name)) = (candidate.parent(), candidate.file_name()) else {
        return false;
    };
    let Some(allowed_name) = allowed.file_name() else {
        return false;
    };
    let name = name.to_string_lossy();
    let allowed_name = allowed_name.to_string_lossy();
    parent == allowed.parent().unwrap_or(Path::new(""))
        && name
            .strip_prefix(&*allowed_name)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
}

/// Like [`config_scoped`] but also accepts the metrics log declared by a
/// config file in the config directory (and its rotations), since
/// `logging.metrics.path` may point outside the config directory.
fn log_scoped(path: &str) -> Result<PathBuf, String> {
    log_scoped_in(&config_dir(), path)
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
    Ok(read_metrics_log_at(&file_path, max_records, since_offset))
}

/// Pure logic behind [`read_metrics_log`], operating on an already-resolved
/// path so tests can exercise it against a temp file without going through
/// [`log_scoped`].
fn read_metrics_log_at(
    file_path: &Path,
    max_records: usize,
    since_offset: Option<u64>,
) -> MetricsLog {
    let display_path = file_path.to_string_lossy().into_owned();
    let Ok(file) = std::fs::File::open(file_path) else {
        return MetricsLog {
            path: display_path,
            exists: false,
            records: Vec::new(),
            offset: 0,
            reset: since_offset.is_some(),
        };
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
        return MetricsLog {
            path: display_path,
            exists: true,
            records: Vec::new(),
            offset: start,
            reset,
        };
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
    MetricsLog {
        path: display_path,
        exists: true,
        records,
        offset,
        reset,
    }
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

/// Where a resolved `higgs` binary came from, in resolution-order
/// preference: an explicit path from Settings, `command -v higgs` on the
/// login shell's `PATH`, or the copy bundled inside the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinarySource {
    Settings,
    Path,
    Bundled,
}

impl BinarySource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Settings => "settings",
            Self::Path => "path",
            Self::Bundled => "bundled",
        }
    }
}

/// `command -v higgs` through a login shell, since GUI apps do not inherit
/// the shell `PATH`.
fn higgs_on_path() -> Option<String> {
    let output = Command::new("/bin/zsh")
        .args(["-lc", "command -v higgs"])
        .output()
        .ok()?;
    let found = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (output.status.success() && !found.is_empty()).then_some(found)
}

/// Resolves the `higgs` binary in order: an explicit path from Settings
/// (must be named `higgs`), `higgs` on `PATH`, then the copy bundled with
/// the app. Returns the resolved path together with where it came from.
fn resolve_higgs(app: &AppHandle, binary: Option<&str>) -> Result<(String, BinarySource), String> {
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?;
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?;
    let version = app.package_info().version.to_string();
    resolve_higgs_at(&resource_dir, &data_dir, &version, binary, higgs_on_path)
}

/// Pure logic behind [`resolve_higgs`], parameterized on the resource and
/// app-data directories and on how `PATH` lookup is performed, so tests can
/// exercise the resolution order against temporary directories without a
/// real Tauri app or a real `higgs` on `PATH`.
fn resolve_higgs_at(
    resource_dir: &Path,
    data_dir: &Path,
    version: &str,
    binary: Option<&str>,
    on_path: impl FnOnce() -> Option<String>,
) -> Result<(String, BinarySource), String> {
    if let Some(path) = binary.filter(|value| !value.trim().is_empty()) {
        let is_higgs = Path::new(path)
            .file_name()
            .is_some_and(|name| name == "higgs");
        if !is_higgs {
            return Err(
                "the Higgs binary path must point at an executable named `higgs`".to_owned(),
            );
        }
        return Ok((path.to_owned(), BinarySource::Settings));
    }
    if let Some(found) = on_path() {
        return Ok((found, BinarySource::Path));
    }
    let bundled = ensure_bundled_cli_at(resource_dir, data_dir, version)?;
    Ok((
        bundled.to_string_lossy().into_owned(),
        BinarySource::Bundled,
    ))
}

/// True for a regular file that exists and is not empty.
fn is_nonempty_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

/// True when `dest` does not already hold a copy of `src`: the destination
/// directory is keyed by app version (see [`ensure_bundled_cli_at`]), so an
/// existing file of the same size there is assumed to be that version's
/// copy already in place.
fn needs_copy(src: &Path, dest: &Path) -> bool {
    let Ok(src_len) = std::fs::metadata(src).map(|meta| meta.len()) else {
        return true;
    };
    std::fs::metadata(dest).map_or(true, |meta| meta.len() != src_len)
}

/// Copies the app-bundled `higgs` binary and `mlx.metallib` out of the
/// (read-only, code-signed on macOS) app resources directory into
/// `data_dir/bin/<version>`, where they can be executed and where MLX's
/// dladdr-based lookup finds `mlx.metallib` next to the binary. Keying the
/// destination by app version means an update copies a fresh binary rather
/// than reusing one from a prior version that might still be running.
/// Returns the copied binary's path, or an error if this build has no
/// bundled CLI.
fn ensure_bundled_cli_at(
    resource_dir: &Path,
    data_dir: &Path,
    version: &str,
) -> Result<PathBuf, String> {
    let bundled_binary = resource_dir.join("bin/higgs");
    let bundled_metallib = resource_dir.join("bin/mlx.metallib");
    // Every build (bundled or not) has these paths: `build.rs` writes empty
    // placeholders when the release workflow hasn't copied the real files
    // in, so an empty file means "not actually bundled" rather than present.
    if !is_nonempty_file(&bundled_binary) || !is_nonempty_file(&bundled_metallib) {
        return Err(
            "no `higgs` CLI is bundled with this build; install it or set the binary path in Settings"
                .to_owned(),
        );
    }

    let dest_dir = data_dir.join("bin").join(version);
    let dest_binary = dest_dir.join("higgs");
    let dest_metallib = dest_dir.join("mlx.metallib");

    if !needs_copy(&bundled_binary, &dest_binary) && !needs_copy(&bundled_metallib, &dest_metallib)
    {
        return Ok(dest_binary);
    }

    std::fs::create_dir_all(&dest_dir).map_err(|error| error.to_string())?;
    std::fs::copy(&bundled_binary, &dest_binary).map_err(|error| error.to_string())?;
    std::fs::copy(&bundled_metallib, &dest_metallib).map_err(|error| error.to_string())?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&dest_binary, std::fs::Permissions::from_mode(0o755))
            .map_err(|error| error.to_string())?;
    }

    prune_old_bin_versions(&data_dir.join("bin"), version);

    Ok(dest_binary)
}

/// Removes every version directory under `bin_root` other than
/// `current_version`, so a fresh copy of the bundled CLI does not leave
/// previous app versions' copies behind forever. Best-effort: an entry that
/// cannot be removed (in use, permissions) is left in place rather than
/// failing the caller.
fn prune_old_bin_versions(bin_root: &Path, current_version: &str) {
    let Ok(entries) = std::fs::read_dir(bin_root) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() == std::ffi::OsStr::new(current_version) {
            continue;
        }
        if entry.path().is_dir() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HiggsBinaryInfo {
    pub path: Option<String>,
    pub source: &'static str,
    pub version: Option<String>,
}

/// Resolves the `higgs` binary the same way [`run_higgs`] would and reports
/// where it came from and its `--version` output, for display in Settings.
#[tauri::command]
pub async fn higgs_binary_info(app: AppHandle, binary: Option<String>) -> HiggsBinaryInfo {
    let Ok((program, source)) = resolve_higgs(&app, binary.as_deref()) else {
        return HiggsBinaryInfo {
            path: None,
            source: "missing",
            version: None,
        };
    };
    let program_for_task = program.clone();
    let version = tauri::async_runtime::spawn_blocking(move || {
        Command::new(&program_for_task)
            .arg("--version")
            .env("NO_COLOR", "1")
            .output()
    })
    .await
    .ok()
    .and_then(Result::ok)
    .filter(|output| output.status.success())
    .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    HiggsBinaryInfo {
        path: Some(program),
        source: source.as_str(),
        version,
    }
}

/// Only whitelisted subcommands run from the UI so the bridge cannot be
/// turned into a general shell.
const ALLOWED_SUBCOMMANDS: &[&str] = &["doctor", "start", "stop", "config", "--version"];

fn is_profile_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// Validates a trailing run of `--profile <name>` / `--config <path>`
/// selector pairs, the only flags `doctor`, `start`, `stop`, and `config`
/// accept beyond their own subcommand arguments.
fn validate_selector_args(rest: &[String]) -> Result<(), String> {
    let mut iter = rest.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--profile" => {
                let name = iter
                    .next()
                    .ok_or_else(|| "--profile requires a value".to_owned())?;
                if !is_profile_name(name) {
                    return Err(format!("invalid profile name: {name}"));
                }
            }
            "--config" => {
                let path = iter
                    .next()
                    .ok_or_else(|| "--config requires a value".to_owned())?;
                config_scoped(path)?;
            }
            other => return Err(format!("unexpected argument: {other}")),
        }
    }
    Ok(())
}

fn validate_config_subcommand_args(rest: &[String]) -> Result<(), String> {
    match rest.first().map(String::as_str) {
        Some("get") => {
            let key = rest.get(1).ok_or("`config get` requires a key")?;
            if key.starts_with("--") {
                return Err(format!("unexpected argument in place of a key: {key}"));
            }
            validate_selector_args(&rest[2..])
        }
        Some("set") => {
            let key = rest.get(1).ok_or("`config set` requires a key")?;
            let value = rest.get(2).ok_or("`config set` requires a value")?;
            if key.starts_with("--") {
                return Err(format!("unexpected argument in place of a key: {key}"));
            }
            let _ = value;
            validate_selector_args(&rest[3..])
        }
        Some("path") => validate_selector_args(&rest[1..]),
        Some(other) => Err(format!("unexpected config subcommand: {other}")),
        None => Err("`config` requires a subcommand".to_owned()),
    }
}

/// Validates every argument after the subcommand, so the bridge cannot be
/// used to smuggle arbitrary flags to the `higgs` binary: `doctor`, `start`,
/// and `stop` accept only `--profile <name>` / `--config <path>` selectors;
/// `config` additionally accepts `get <key>`, `set <key> <value>`, or `path`
/// before those same selectors; `--version` accepts nothing else.
fn validate_subcommand_args(subcommand: &str, rest: &[String]) -> Result<(), String> {
    match subcommand {
        "doctor" | "start" | "stop" => validate_selector_args(rest),
        "config" => validate_config_subcommand_args(rest),
        "--version" => {
            if rest.is_empty() {
                Ok(())
            } else {
                Err("--version accepts no further arguments".to_owned())
            }
        }
        other => Err(format!("subcommand not allowed: {other}")),
    }
}

#[tauri::command]
pub async fn run_higgs(
    app: AppHandle,
    binary: Option<String>,
    args: Vec<String>,
) -> Result<CommandOutput, String> {
    let subcommand = args.first().map(String::as_str).unwrap_or_default();
    if !ALLOWED_SUBCOMMANDS.contains(&subcommand) {
        return Err(format!("subcommand not allowed: {subcommand}"));
    }
    validate_subcommand_args(subcommand, &args[1..])?;
    let (program, _source) = resolve_higgs(&app, binary.as_deref())?;
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
    model_cache_info_in(&config_dir(), path)
}

/// Sizes a model on disk. `path` is either a Hugging Face repo id resolved
/// inside the hub cache, or a local directory that some profile lists under
/// `[[models]].path`; arbitrary paths are never inspected.
fn model_cache_info_in(config_dir: &Path, path: String) -> ModelCacheInfo {
    let direct = expand_home(&path);
    let allowed_local = configured_model_paths(config_dir)
        .iter()
        .any(|allowed| allowed == &direct);
    if allowed_local && direct.is_dir() {
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
    let repo_dir = hub.and_then(|hub| {
        let dir = hub.join(format!("models--{}", path.replace('/', "--")));
        crate::paths::is_contained_strict(&hub, &dir).then_some(dir)
    });
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
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A directory under `std::env::temp_dir()` that is removed on drop even
    /// if the test panics, so tests never depend on or pollute the real home
    /// directory.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "higgs-desktop-test-{label}-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn model_cache_info_only_sizes_configured_local_paths() {
        let dir = TempDir::new("model-cache-info");
        let listed = dir.path.join("listed-model");
        let unlisted = dir.path.join("unlisted-model");
        std::fs::create_dir_all(&listed).expect("create listed");
        std::fs::create_dir_all(&unlisted).expect("create unlisted");
        std::fs::write(listed.join("weights.bin"), [0u8; 16]).expect("write weights");
        std::fs::write(
            dir.path.join("config.toml"),
            format!(
                "[[models]]\nname = \"m\"\npath = \"{}\"\n",
                listed.display()
            ),
        )
        .expect("write config");

        let info = model_cache_info_in(&dir.path, listed.to_string_lossy().into_owned());
        assert!(info.cached);
        assert_eq!(info.size_bytes, 16);

        let info = model_cache_info_in(&dir.path, unlisted.to_string_lossy().into_owned());
        assert!(!info.cached);
        assert_eq!(info.location, None);
    }

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
        let dir = TempDir::new("read-metrics-log");
        let path = dir.path.join("metrics.jsonl");
        let line = |model: &str| {
            format!(
                "{{\"timestamp\":\"2026-01-01T00:00:00Z\",\"model\":\"{model}\",\"status\":200,\"duration_ms\":1,\"input_tokens\":1,\"output_tokens\":1}}\n"
            )
        };
        std::fs::write(&path, line("a")).expect("write");
        let first = read_metrics_log_at(&path, 100, None);
        assert_eq!(first.records.len(), 1);
        assert!(!first.reset);

        let mut existing = std::fs::read_to_string(&path).expect("read");
        existing.push_str(&line("b"));
        existing.push_str("{\"partial\":");
        std::fs::write(&path, existing).expect("write");
        let second = read_metrics_log_at(&path, 100, Some(first.offset));
        assert_eq!(second.records.len(), 1);
        assert_eq!(second.records[0].model.as_deref(), Some("b"));
        assert!(!second.reset);

        std::fs::write(&path, line("c")).expect("write");
        let third = read_metrics_log_at(&path, 100, Some(second.offset));
        assert!(third.reset, "truncation must signal reset");
        assert_eq!(third.records.len(), 1);
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
        let dir = TempDir::new("config-scoped");
        assert!(config_scoped_in(&dir.path, "config.toml").is_ok());
        assert!(config_scoped_in(&dir.path, "../../etc/passwd").is_err());
        assert!(config_scoped_in(&dir.path, "/etc/passwd").is_err());
        let inside = dir.path.join("logs/metrics.jsonl");
        assert!(log_scoped_in(&dir.path, &inside.to_string_lossy()).is_ok());
        assert!(log_scoped_in(&dir.path, "/etc/passwd").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn config_scoped_rejects_a_symlink_escaping_the_config_dir() {
        let dir = TempDir::new("config-scoped-symlink");
        let outside = TempDir::new("config-scoped-symlink-outside");
        let linked = dir.path.join("linked");
        std::os::unix::fs::symlink(&outside.path, &linked).expect("symlink dir");
        let escaped = linked.join("secrets.toml").to_string_lossy().into_owned();
        assert!(config_scoped_in(&dir.path, &escaped).is_err());
    }

    #[test]
    fn config_dir_from_prefers_the_env_override() {
        let dir = TempDir::new("config-dir-from");
        let overridden = config_dir_from(Some(dir.path.to_str().expect("utf8 path")));
        assert_eq!(overridden, dir.path);
    }

    #[test]
    fn metrics_log_outside_config_dir_needs_a_declaring_config() {
        let guard = TempDir::new("log-scope");
        let dir = guard.path.as_path();
        let elsewhere = TempDir::new("log-elsewhere");
        let external = elsewhere.path.join("metrics.jsonl");
        std::fs::create_dir_all(external.parent().expect("parent")).expect("mkdir");
        std::fs::write(dir.join("config.toml"), "").expect("write");
        let external_str = external.to_string_lossy().into_owned();
        // The config directory itself is always allowed.
        assert!(log_scoped_in(dir, "logs/metrics.jsonl").is_ok());
        // An undeclared path outside it is not, even under a home-like tree.
        assert!(log_scoped_in(dir, &external_str).is_err());
        std::fs::write(
            dir.join("config.desk.toml"),
            format!("[logging.metrics]\npath = \"{external_str}\"\n"),
        )
        .expect("write");
        assert!(log_scoped_in(dir, &external_str).is_ok());
        assert!(log_scoped_in(dir, &format!("{external_str}.3")).is_ok());
        assert!(log_scoped_in(dir, &format!("{external_str}.bak")).is_err());
        assert!(
            log_scoped_in(
                dir,
                &format!("{}/other.jsonl", external.parent().expect("p").display())
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_unknown_subcommand() {
        let allowed = ALLOWED_SUBCOMMANDS.contains(&"rm");
        assert!(!allowed);
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn validates_selector_only_args() {
        assert!(validate_subcommand_args("doctor", &args(&[])).is_ok());
        assert!(validate_subcommand_args("start", &args(&["--profile", "work"])).is_ok());
        assert!(validate_subcommand_args("stop", &args(&["--profile", "../etc"])).is_err());
        assert!(validate_subcommand_args("doctor", &args(&["--config", "/etc/passwd"])).is_err());
        assert!(validate_subcommand_args("doctor", &args(&["rm", "-rf", "/"])).is_err());
    }

    #[test]
    fn validates_config_subcommand_args() {
        assert!(validate_subcommand_args("config", &args(&["path"])).is_ok());
        assert!(validate_subcommand_args("config", &args(&["get", "server.port"])).is_ok());
        assert!(validate_subcommand_args("config", &args(&["set", "server.port", "8080"])).is_ok());
        assert!(
            validate_subcommand_args(
                "config",
                &args(&["get", "server.port", "--profile", "work"])
            )
            .is_ok()
        );
        assert!(validate_subcommand_args("config", &args(&["get"])).is_err());
        assert!(validate_subcommand_args("config", &args(&["rm"])).is_err());
        assert!(validate_subcommand_args("config", &args(&[])).is_err());
    }

    #[test]
    fn validates_version_takes_no_args() {
        assert!(validate_subcommand_args("--version", &args(&[])).is_ok());
        assert!(validate_subcommand_args("--version", &args(&["--profile", "work"])).is_err());
    }

    fn write_bundled_cli(resource_dir: &Path, binary_contents: &str) {
        let bin = resource_dir.join("bin");
        std::fs::create_dir_all(&bin).expect("create resource bin dir");
        std::fs::write(bin.join("higgs"), binary_contents).expect("write bundled binary");
        std::fs::write(bin.join("mlx.metallib"), "metallib").expect("write bundled metallib");
    }

    #[test]
    fn settings_path_is_preferred_when_named_higgs() {
        let dir = TempDir::new("resolve-settings");
        let (path, source) = resolve_higgs_at(
            &dir.path,
            &dir.path,
            "1.0.0",
            Some("/opt/higgs/higgs"),
            || panic!("must not fall through to PATH lookup"),
        )
        .expect("resolves");
        assert_eq!(path, "/opt/higgs/higgs");
        assert_eq!(source, BinarySource::Settings);
    }

    #[test]
    fn settings_path_must_be_named_higgs() {
        let dir = TempDir::new("resolve-settings-invalid");
        let result = resolve_higgs_at(
            &dir.path,
            &dir.path,
            "1.0.0",
            Some("/opt/higgs/cli"),
            || None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn path_lookup_is_preferred_over_bundled() {
        let dir = TempDir::new("resolve-path");
        write_bundled_cli(&dir.path, "bundled");
        let (path, source) = resolve_higgs_at(&dir.path, &dir.path, "1.0.0", None, || {
            Some("/usr/local/bin/higgs".to_owned())
        })
        .expect("resolves");
        assert_eq!(path, "/usr/local/bin/higgs");
        assert_eq!(source, BinarySource::Path);
    }

    #[test]
    fn falls_back_to_bundled_cli_when_nothing_else_resolves() {
        let dir = TempDir::new("resolve-bundled");
        write_bundled_cli(&dir.path, "bundled");
        let (path, source) =
            resolve_higgs_at(&dir.path, &dir.path, "1.0.0", None, || None).expect("resolves");
        assert!(path.ends_with("bin/1.0.0/higgs"), "path was {path}");
        assert_eq!(source, BinarySource::Bundled);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read copied binary"),
            "bundled"
        );
    }

    #[test]
    fn errors_when_no_binary_is_available_anywhere() {
        let dir = TempDir::new("resolve-missing");
        let result = resolve_higgs_at(&dir.path, &dir.path, "1.0.0", None, || None);
        assert!(result.is_err());
    }

    #[test]
    fn bundled_cli_copy_is_keyed_by_app_version() {
        let dir = TempDir::new("resolve-bundled-version");
        write_bundled_cli(&dir.path, "bundled");
        let (path, _) =
            resolve_higgs_at(&dir.path, &dir.path, "2.5.0", None, || None).expect("resolves");
        assert!(path.contains("bin/2.5.0/"), "path was {path}");
    }

    #[test]
    fn bundled_cli_is_not_recopied_when_already_up_to_date() {
        let dir = TempDir::new("resolve-bundled-cached");
        write_bundled_cli(&dir.path, "bundled");
        let first = ensure_bundled_cli_at(&dir.path, &dir.path, "1.0.0").expect("first copy");
        // Overwrite the destination with same-length content to prove a
        // second call sees it as already up to date (same size) and leaves
        // it alone rather than copying again.
        assert_eq!("bundled".len(), "changed".len());
        std::fs::write(&first, "changed").expect("overwrite dest");
        let second = ensure_bundled_cli_at(&dir.path, &dir.path, "1.0.0").expect("second copy");
        assert_eq!(first, second);
        assert_eq!(
            std::fs::read_to_string(&second).expect("read"),
            "changed",
            "same-size destination should not be re-copied"
        );
    }

    #[test]
    fn bundled_cli_errors_when_resources_are_missing() {
        let dir = TempDir::new("resolve-no-resources");
        let result = ensure_bundled_cli_at(&dir.path, &dir.path, "1.0.0");
        assert!(result.is_err());
    }

    #[test]
    fn old_bin_versions_are_pruned_after_a_fresh_copy() {
        let dir = TempDir::new("prune-old-versions");
        write_bundled_cli(&dir.path, "bundled-v2");
        let old_version_dir = dir.path.join("bin").join("0.9.0");
        std::fs::create_dir_all(&old_version_dir).expect("create old version dir");
        std::fs::write(old_version_dir.join("higgs"), "old").expect("write old binary");

        let result = ensure_bundled_cli_at(&dir.path, &dir.path, "1.0.0").expect("copies");
        assert!(result.ends_with("bin/1.0.0/higgs"), "path was {result:?}");
        assert!(
            !old_version_dir.exists(),
            "old version directory should be pruned"
        );
        assert!(dir.path.join("bin").join("1.0.0").is_dir());
    }

    #[test]
    fn bundled_cli_errors_when_resources_are_empty_placeholders() {
        // build.rs writes empty files at these paths for every build that
        // did not have the release workflow copy the real CLI in; those
        // must not be mistaken for a bundled CLI.
        let dir = TempDir::new("resolve-placeholder-resources");
        let bin = dir.path.join("bin");
        std::fs::create_dir_all(&bin).expect("create resource bin dir");
        std::fs::write(bin.join("higgs"), b"").expect("write empty placeholder");
        std::fs::write(bin.join("mlx.metallib"), b"").expect("write empty placeholder");
        let result = ensure_bundled_cli_at(&dir.path, &dir.path, "1.0.0");
        assert!(result.is_err());
    }
}
