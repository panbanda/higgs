use std::fs;
use std::net::TcpStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;

use crate::attach;
use crate::config::{self, HiggsConfig};
use crate::metrics::MetricsStore;
use crate::metrics_log::MetricsLogger;

fn pid_path(profile: Option<&str>) -> std::path::PathBuf {
    config::pid_path(profile)
}

fn log_path(profile: Option<&str>) -> std::path::PathBuf {
    config::log_path(profile)
}

pub fn read_pid(profile: Option<&str>) -> Option<i32> {
    fs::read_to_string(pid_path(profile))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

pub fn pid_is_alive(pid: i32) -> bool {
    kill(Pid::from_raw(pid), None).is_ok()
}

pub fn remove_pid_file(profile: Option<&str>) {
    let _ = fs::remove_file(pid_path(profile));
}

pub fn write_pid_file(profile: Option<&str>) {
    let pid = std::process::id();
    if let Err(e) = fs::write(pid_path(profile), pid.to_string()) {
        tracing::warn!("failed to write pid file: {e}");
    }
}

#[allow(clippy::print_stderr)]
pub fn cmd_stop(profile: Option<&str>) {
    let label = profile.map_or_else(|| "higgs".to_owned(), |p| format!("higgs [{p}]"));
    match read_pid(profile) {
        Some(pid) if pid_is_alive(pid) => {
            if let Err(e) = kill(Pid::from_raw(pid), Signal::SIGTERM) {
                eprintln!("failed to send SIGTERM to {pid}: {e}");
                std::process::exit(1);
            }
            // Wait for process to exit before removing PID file
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while pid_is_alive(pid) {
                if std::time::Instant::now() >= deadline {
                    eprintln!("sent SIGTERM to {pid} but process still running after 5s");
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            remove_pid_file(profile);
            eprintln!("stopped {label} (pid {pid})");
        }
        Some(_) => {
            remove_pid_file(profile);
            eprintln!("{label} is not running (stale pid file removed)");
        }
        None => {
            eprintln!("{label} is not running (no pid file)");
        }
    }
}

#[allow(clippy::print_stderr)]
pub fn cmd_init(profile: Option<&str>) {
    let dir = config::config_dir();
    let filename = profile.map_or_else(
        || "config.toml".to_owned(),
        |name| format!("config.{name}.toml"),
    );
    let path = dir.join(filename);

    if path.exists() {
        eprintln!("config already exists: {}", path.display());
        return;
    }

    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("failed to create {}: {e}", dir.display());
        std::process::exit(1);
    }

    let metrics_path_example = profile.map_or_else(
        || "~/.config/higgs/logs/metrics.jsonl".to_owned(),
        |name| format!("~/.config/higgs/logs/metrics.{name}.jsonl"),
    );

    let default_config = format!(
        r#"[server]
host = "0.0.0.0"
port = 8000
# max_tokens = 32768
# timeout = 300.0
# api_key = "sk-..."
# rate_limit = 0

# --- Local models ---
# Each [[models]] entry loads an MLX model into GPU memory.
# Use a HuggingFace model ID or a local path.

# [[models]]
# path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# name = "llama"
# batch = false

# --- Remote providers ---
# Forward requests to external APIs via proxy routes.

# [provider.anthropic]
# url = "https://api.anthropic.com"
# format = "anthropic"

# [provider.openai]
# url = "https://api.openai.com"
# format = "openai"

# [provider.ollama]
# url = "http://localhost:11434"
# strip_auth = true
# api_key = "ollama"
# stub_count_tokens = true

# --- Routes ---
# Pattern routes match model names with regex. First match wins.

# [[routes]]
# pattern = "claude-.*"
# provider = "anthropic"

# [[routes]]
# pattern = "gpt-.*"
# provider = "openai"

# --- Default route ---
# When no pattern matches and the model isn't loaded locally.

[default]
provider = "higgs"

# --- Auto router ---
# Classify requests using a local model to pick the best route.

# [auto_router]
# enabled = true
# model = "mlx-community/Arch-Router-1.5B-4bit"
# timeout_ms = 2000

# --- Usage ---
# eval "$(higgs shellenv)"  -- set env vars in current shell
# higgs run -- claude        -- run a command with env vars set

# --- Retention ---
# How long to keep metrics in memory for the TUI dashboard.

# [retention]
# enabled = true
# minutes = 60

# --- Metrics logging ---
# Append request metrics to a JSONL file for attach/TUI history.

# [logging.metrics]
# enabled = true
# path = "{metrics_path_example}"
# max_size_mb = 50
# max_files = 5
"#
    );

    if let Err(e) = fs::write(&path, default_config) {
        eprintln!("failed to write {}: {e}", path.display());
        std::process::exit(1);
    }

    eprintln!("created {}", path.display());
}

#[allow(clippy::print_stdout)]
pub fn cmd_shellenv(config: &HiggsConfig) {
    let host = match config.server.host.as_str() {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    let addr = format!("{host}:{}", config.server.port);

    if TcpStream::connect(&addr).is_ok() {
        let base_url = format!("http://{addr}");
        println!("export ANTHROPIC_BASE_URL={base_url}");
        println!("export OPENAI_BASE_URL={base_url}");
    }
}

#[allow(clippy::print_stderr)]
pub fn cmd_run(config: &HiggsConfig, command: &[String]) -> ! {
    let Some((program, args)) = command.split_first() else {
        eprintln!("no command specified");
        std::process::exit(1);
    };

    let host = match config.server.host.as_str() {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    let addr = format!("{host}:{}", config.server.port);

    if TcpStream::connect(&addr).is_err() {
        eprintln!("higgs is not running on {addr}");
        eprintln!("hint: start with 'higgs start' or 'higgs serve'");
        std::process::exit(1);
    }

    let base_url = format!("http://{addr}");
    let err = std::process::Command::new(program)
        .args(args)
        .env("ANTHROPIC_BASE_URL", &base_url)
        .env("OPENAI_BASE_URL", &base_url)
        .exec();

    eprintln!("failed to exec '{program}': {err}");
    std::process::exit(1);
}

#[allow(clippy::print_stderr, clippy::too_many_lines, unsafe_code)]
pub fn detach(config_path: &Path, verbose: bool, profile: Option<&str>) {
    let label = profile.map_or_else(|| "higgs".to_owned(), |p| format!("higgs [{p}]"));

    if let Some(pid) = read_pid(profile) {
        if pid_is_alive(pid) {
            eprintln!("{label} is already running (pid {pid})");
            std::process::exit(1);
        }
        remove_pid_file(profile);
    }

    let dir = config::config_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("failed to create {}: {e}", dir.display());
        std::process::exit(1);
    }

    let Ok(log) = fs::File::create(log_path(profile)) else {
        eprintln!("failed to create log file");
        std::process::exit(1);
    };
    let Ok(log_err) = log.try_clone() else {
        eprintln!("failed to clone log file handle");
        std::process::exit(1);
    };

    let Ok(exe) = std::env::current_exe() else {
        eprintln!("failed to determine executable path");
        std::process::exit(1);
    };

    let Ok(devnull) = fs::File::open("/dev/null") else {
        eprintln!("failed to open /dev/null");
        std::process::exit(1);
    };

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("serve");
    if let Some(name) = profile {
        cmd.arg("--profile").arg(name);
    } else {
        cmd.arg("--config").arg(config_path);
    }
    if verbose {
        cmd.arg("--verbose");
    }
    cmd.stdin(devnull);

    // Create new session so child survives terminal close
    // SAFETY: setsid is async-signal-safe per POSIX
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut cmd, || {
            nix::unistd::setsid().map_err(std::io::Error::other)?;
            Ok(())
        });
    }

    let Ok(mut child) = cmd.stdout(log).stderr(log_err).spawn() else {
        eprintln!("failed to spawn detached process");
        std::process::exit(1);
    };

    let child_pid = child.id();

    // Detach: we don't want to wait on the child (it's the daemon).
    // Reap it so we don't leave a zombie during the brief startup check.
    std::thread::spawn(move || {
        let _ = child.wait();
    });

    if let Err(e) = fs::write(pid_path(profile), child_pid.to_string()) {
        eprintln!("failed to write pid file: {e}");
        std::process::exit(1);
    }

    let Ok(child_pid_i32) = i32::try_from(child_pid) else {
        eprintln!("child pid {child_pid} exceeds i32 range");
        std::process::exit(1);
    };

    // Probe the server address from config to check readiness
    let probe_path = profile.map_or_else(
        || config_path.to_path_buf(),
        crate::config::profile_config_path,
    );
    let Ok(probe_config) = crate::config::load_config_file(&probe_path, None) else {
        eprintln!(
            "{label} started (pid {child_pid}), log: {}",
            log_path(profile).display()
        );
        return;
    };
    let probe_host = match probe_config.server.host.as_str() {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    let probe_addr = format!("{probe_host}:{}", probe_config.server.port);

    // Poll until the daemon is accepting connections or the process dies
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if !pid_is_alive(child_pid_i32) {
            remove_pid_file(profile);
            eprintln!(
                "{label} failed to start, check {}",
                log_path(profile).display()
            );
            std::process::exit(1);
        }
        if TcpStream::connect(&probe_addr).is_ok() {
            eprintln!(
                "{label} started (pid {child_pid}), log: {}",
                log_path(profile).display()
            );
            return;
        }
        if std::time::Instant::now() >= deadline {
            eprintln!(
                "{label} started (pid {child_pid}) but not yet accepting connections, log: {}",
                log_path(profile).display()
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[allow(clippy::print_stderr)]
pub fn run_attached(config: &HiggsConfig, profile: Option<&str>) {
    if !config.logging.metrics.enabled {
        eprintln!("cannot attach: [logging.metrics] enabled = true required in config");
        std::process::exit(1);
    }

    let retention = retention_duration(config);
    let metrics = Arc::new(MetricsStore::new(retention));

    attach::load_history(&config.logging.metrics, &metrics);

    let log_path_buf = PathBuf::from(&config.logging.metrics.path);
    let stop = Arc::new(AtomicBool::new(false));

    let tail_store = Arc::clone(&metrics);
    let tail_stop = Arc::clone(&stop);
    let _tail_handle = std::thread::spawn(move || {
        attach::tail_log(&log_path_buf, &tail_store, &tail_stop);
    });

    let evict_metrics = Arc::clone(&metrics);
    let evict_stop = Arc::clone(&stop);
    let _evict_handle = std::thread::spawn(move || {
        while !evict_stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(60));
            evict_metrics.evict_expired();
        }
    });

    let tui_config = crate::tui::TuiConfig::from_higgs_config(config, profile);

    match crate::tui::run(Arc::clone(&metrics), true, Some(tui_config)) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("TUI error: {e}");
        }
    }

    stop.store(true, Ordering::Relaxed);
}

pub const fn retention_duration(config: &HiggsConfig) -> Duration {
    if config.retention.enabled {
        Duration::from_secs(config.retention.minutes.saturating_mul(60))
    } else {
        Duration::from_secs(365 * 24 * 60 * 60)
    }
}

pub fn create_metrics(config: &HiggsConfig) -> Arc<MetricsStore> {
    let retention = retention_duration(config);
    Arc::new(if config.logging.metrics.enabled {
        match MetricsLogger::new(&config.logging.metrics) {
            Ok(logger) => {
                tracing::info!(path = %config.logging.metrics.path, "metrics logging enabled");
                MetricsStore::with_logger(retention, logger)
            }
            Err(e) => {
                tracing::warn!("failed to initialize metrics logger: {e}");
                MetricsStore::new(retention)
            }
        }
    } else {
        MetricsStore::new(retention)
    })
}

pub fn spawn_eviction_task(metrics: &Arc<MetricsStore>) {
    let evict_metrics = Arc::clone(metrics);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            evict_metrics.evict_expired();
        }
    });
}

pub async fn await_shutdown_signal() {
    // Use explicit unix signals because crossterm's signal-hook
    // handler can interfere with tokio::signal::ctrl_c().
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).ok();
    let mut sigterm =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();

    tokio::select! {
        _ = async { if let Some(ref mut s) = sigint { s.recv().await } else { std::future::pending().await } } => {}
        _ = async { if let Some(ref mut s) = sigterm { s.recv().await } else { std::future::pending().await } } => {}
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
mod tests {
    use super::*;

    fn with_temp_config_dir<F: FnOnce(&std::path::Path)>(f: F) {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("HIGGS_CONFIG_DIR", dir.path());
        }
        f(dir.path());
        unsafe {
            std::env::remove_var("HIGGS_CONFIG_DIR");
        }
    }

    #[test]
    fn config_dir_respects_env_override() {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("HIGGS_CONFIG_DIR", dir.path());
        }
        assert_eq!(config::config_dir(), dir.path());
        unsafe {
            std::env::remove_var("HIGGS_CONFIG_DIR");
        }
    }

    #[test]
    fn config_dir_falls_back_to_home() {
        unsafe {
            std::env::remove_var("HIGGS_CONFIG_DIR");
        }
        let path = config::config_dir();
        assert!(path.to_string_lossy().ends_with(".config/higgs"));
    }

    #[test]
    fn pid_path_suffix() {
        with_temp_config_dir(|dir| {
            assert_eq!(pid_path(None), dir.join("higgs.pid"));
        });
    }

    #[test]
    fn pid_path_profile_suffix() {
        with_temp_config_dir(|dir| {
            assert_eq!(pid_path(Some("dev")), dir.join("higgs.dev.pid"));
        });
    }

    #[test]
    fn log_path_suffix() {
        with_temp_config_dir(|dir| {
            assert_eq!(log_path(None), dir.join("higgs.log"));
        });
    }

    #[test]
    fn log_path_profile_suffix() {
        with_temp_config_dir(|dir| {
            assert_eq!(log_path(Some("dev")), dir.join("higgs.dev.log"));
        });
    }

    #[test]
    fn read_pid_none_when_no_file() {
        with_temp_config_dir(|_| {
            assert!(read_pid(None).is_none());
        });
    }

    #[test]
    fn read_pid_returns_valid_pid() {
        with_temp_config_dir(|dir| {
            std::fs::write(dir.join("higgs.pid"), "12345").unwrap();
            assert_eq!(read_pid(None), Some(12345));
        });
    }

    #[test]
    fn read_pid_none_for_non_numeric() {
        with_temp_config_dir(|dir| {
            std::fs::write(dir.join("higgs.pid"), "not-a-number").unwrap();
            assert!(read_pid(None).is_none());
        });
    }

    #[test]
    fn write_and_read_pid_file() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            write_pid_file(None);
            let pid = read_pid(None).unwrap();
            assert_eq!(pid, i32::try_from(std::process::id()).unwrap());
        });
    }

    #[test]
    fn remove_pid_file_removes_file() {
        with_temp_config_dir(|dir| {
            let path = dir.join("higgs.pid");
            std::fs::write(&path, "12345").unwrap();
            assert!(path.exists());
            remove_pid_file(None);
            assert!(!path.exists());
        });
    }

    #[test]
    fn remove_pid_file_noop_when_missing() {
        with_temp_config_dir(|_| {
            remove_pid_file(None);
        });
    }

    #[test]
    fn pid_is_alive_for_own_process() {
        let pid = i32::try_from(std::process::id()).unwrap();
        assert!(pid_is_alive(pid));
    }

    #[test]
    fn pid_is_alive_false_for_nonexistent() {
        assert!(!pid_is_alive(99_999_999));
    }

    #[test]
    fn retention_duration_uses_configured_minutes() {
        let config = HiggsConfig {
            retention: crate::config::RetentionConfig {
                enabled: true,
                minutes: 30,
            },
            ..HiggsConfig::default()
        };
        assert_eq!(retention_duration(&config), Duration::from_secs(1800));
    }

    #[test]
    fn retention_duration_one_year_when_disabled() {
        let config = HiggsConfig {
            retention: crate::config::RetentionConfig {
                enabled: false,
                minutes: 30,
            },
            ..HiggsConfig::default()
        };
        assert_eq!(
            retention_duration(&config),
            Duration::from_secs(365 * 24 * 60 * 60)
        );
    }

    #[test]
    fn create_metrics_without_logger() {
        let config = HiggsConfig {
            logging: crate::config::LoggingConfig {
                metrics: crate::config::MetricsLogConfig {
                    enabled: false,
                    ..Default::default()
                },
            },
            ..HiggsConfig::default()
        };
        let store = create_metrics(&config);
        assert_eq!(store.window(), retention_duration(&config));
    }

    #[test]
    fn cmd_init_creates_config_file() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            assert!(dir.join("config.toml").exists());
        });
    }

    #[test]
    fn cmd_init_noop_when_config_exists() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            let path = dir.join("config.toml");
            std::fs::write(&path, "existing content").unwrap();
            cmd_init(None);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing content");
        });
    }

    #[test]
    fn cmd_init_profile_creates_named_config() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(Some("dev"));
            assert!(dir.join("config.dev.toml").exists());
            assert!(!dir.join("config.toml").exists());
        });
    }
}
