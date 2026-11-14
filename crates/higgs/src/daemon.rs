use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::Duration;

use nix::sys::signal::{SaFlags, SigAction, SigHandler, SigSet, Signal, kill, sigaction};
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

static EXEC_CHILD_PID: AtomicI32 = AtomicI32::new(0);
static EXEC_SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);
const EVICTION_INTERVAL_SECS: u64 = 60;
const DEFAULT_RETENTION_SECS: u64 = 365 * 24 * 60 * 60;

extern "C" fn exec_sigint_handler(_: i32) {
    EXEC_SIGINT_RECEIVED.store(true, Ordering::SeqCst);
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
    if let Err(e) = config::write_private_file(&pid_path(profile), &pid.to_string()) {
        tracing::warn!("failed to write pid file: {e}");
    }
}

/// Bundle identifier of the desktop app, matching `apps/desktop/src-tauri/tauri.conf.json`.
pub const DESKTOP_BUNDLE_ID: &str = "com.panbanda.higgs";

/// Launch the desktop app. Returns a process exit code.
#[allow(clippy::print_stderr)]
pub fn cmd_ui() -> i32 {
    if !cfg!(target_os = "macos") {
        eprintln!("the Higgs desktop app is only available on macOS");
        return 1;
    }
    // LaunchServices prints its own lookup failure to stderr; the hint below is enough.
    match std::process::Command::new("open")
        .args(["-b", DESKTOP_BUNDLE_ID])
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(status) if status.success() => 0,
        Ok(_) => {
            eprintln!(
                "Higgs.app is not installed. Install it with:\n  brew install --cask panbanda/brews/higgs-desktop\nor download it from https://github.com/panbanda/higgs/releases"
            );
            1
        }
        Err(e) => {
            eprintln!("failed to run open: {e}");
            1
        }
    }
}

#[allow(clippy::print_stderr)]
pub fn cmd_stop(profile: Option<&str>, force: bool) -> i32 {
    let label = profile.map_or_else(|| "higgs".to_owned(), |p| format!("higgs [{p}]"));
    match read_pid(profile) {
        Some(pid) if pid_is_alive(pid) => {
            if let Err(e) = kill(Pid::from_raw(pid), Signal::SIGTERM) {
                eprintln!("failed to send SIGTERM to {pid}: {e}");
                return 1;
            }
            // Wait for process to exit before removing PID file
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while pid_is_alive(pid) {
                if std::time::Instant::now() >= deadline {
                    if force {
                        eprintln!("graceful stop timed out; sending SIGKILL to {pid}");
                        if let Err(e) = kill(Pid::from_raw(pid), Signal::SIGKILL) {
                            eprintln!("failed to send SIGKILL to {pid}: {e}");
                            return 1;
                        }
                        let kill_deadline = std::time::Instant::now() + Duration::from_secs(2);
                        while pid_is_alive(pid) {
                            if std::time::Instant::now() >= kill_deadline {
                                eprintln!("sent SIGKILL to {pid} but process is still running");
                                return 1;
                            }
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        break;
                    }
                    eprintln!("sent SIGTERM to {pid} but process still running after 5s");
                    return 1;
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
    0
}

/// The commented config `higgs init` writes. `__METRICS_PATH__` is the profile-specific log path.
const INIT_TEMPLATE: &str = include_str!("config/init_template.toml");

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

    let default_config = INIT_TEMPLATE.replace("__METRICS_PATH__", &metrics_path_example);

    if let Err(e) = config::write_private_file(&path, &default_config) {
        eprintln!("failed to write {}: {e}", path.display());
        std::process::exit(1);
    }

    eprintln!("created {}", path.display());
}

/// Map wildcard bind addresses to localhost for client connections.
fn resolve_host(host: &str) -> &str {
    match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    }
}

fn resolved_addr(config: &HiggsConfig) -> String {
    format!(
        "{}:{}",
        resolve_host(&config.server.host),
        config.server.port
    )
}

/// Connect to the first address of `addr` that accepts. A name such as `localhost` can
/// resolve to `::1` while the server listens on `127.0.0.1` only.
fn connect_with_timeout(addr: &str) -> Result<TcpStream, String> {
    let socket_addrs: Vec<_> = addr
        .to_socket_addrs()
        .map_err(|e| format!("failed to resolve {addr}: {e}"))?
        .collect();
    let mut last_error = None;
    for socket_addr in &socket_addrs {
        match TcpStream::connect_timeout(socket_addr, Duration::from_secs(2)) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    last_error.map_or_else(
        || Err(format!("failed to resolve {addr}: no socket addresses found")),
        |error| Err(format!("higgs is not running on {addr}: {error}")),
    )
}

fn check_health(config: &HiggsConfig) -> Result<(), String> {
    check_health_at(&resolved_addr(config))
}

fn check_health_at(addr: &str) -> Result<(), String> {
    let mut stream = connect_with_timeout(addr)?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| format!("failed to set read timeout for {addr}: {e}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| format!("failed to set write timeout for {addr}: {e}"))?;
    let request = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("failed to probe /health on {addr}: {e}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|e| format!("failed to read /health response from {addr}: {e}"))?;

    if !(response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200")) {
        return Err(format!(
            "daemon health probe failed for {addr}: unexpected response status"
        ));
    }
    if !health_response_is_ok(&response) {
        return Err(format!(
            "daemon health probe failed for {addr}: unexpected response body"
        ));
    }
    Ok(())
}

fn health_response_is_ok(response: &str) -> bool {
    let Some((headers, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    let body_bytes = if response_is_chunked(headers) {
        let Some(decoded) = decode_chunked_body(body.as_bytes()) else {
            return false;
        };
        decoded
    } else {
        body.as_bytes().to_vec()
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body_bytes) else {
        return false;
    };
    value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| status == "ok")
}

fn response_is_chunked(headers: &str) -> bool {
    headers.lines().any(|line| {
        let Some((name, value)) = line.split_once(':') else {
            return false;
        };
        name.trim().eq_ignore_ascii_case("transfer-encoding")
            && value
                .split(',')
                .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"))
    })
}

fn decode_chunked_body(body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut offset = 0;

    loop {
        let line_end = find_crlf(body, offset)?;
        let size_line = std::str::from_utf8(body.get(offset..line_end)?).ok()?;
        let size = usize::from_str_radix(size_line.split(';').next()?.trim(), 16).ok()?;
        offset = line_end + 2;

        if size == 0 {
            return Some(out);
        }
        let chunk_end = offset.checked_add(size)?;
        let chunk = body.get(offset..chunk_end)?;
        out.extend_from_slice(chunk);
        offset = chunk_end;
        if body.get(offset..offset + 2)? != b"\r\n" {
            return None;
        }
        offset += 2;
    }
}

fn find_crlf(body: &[u8], offset: usize) -> Option<usize> {
    body.get(offset..)?
        .windows(2)
        .position(|window| window == b"\r\n")
        .map(|index| offset + index)
}

/// The server that `higgs shellenv` and `higgs exec` point client tools at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientTarget {
    addr: String,
    origin: TargetOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetOrigin {
    Config,
    Flag,
    DefaultAddress,
}

impl ClientTarget {
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub const fn description(&self) -> &'static str {
        match self.origin {
            TargetOrigin::Config => "from config",
            TargetOrigin::Flag => "from --url/--port",
            TargetOrigin::DefaultAddress => "default address",
        }
    }
}

/// Pick the server address for client tools.
///
/// `--url` and `--port` win, then the config file's `[server]` address. With neither, the
/// default `higgs serve` address is used so a server started with `--model` is found
/// without any config.
pub fn resolve_client_target(
    config: Option<&HiggsConfig>,
    url_flag: Option<&str>,
    port_flag: Option<u16>,
) -> Result<ClientTarget, String> {
    if let Some(url) = url_flag {
        return Ok(ClientTarget {
            addr: parse_client_url(url)?,
            origin: TargetOrigin::Flag,
        });
    }
    if let Some(port) = port_flag {
        return Ok(ClientTarget {
            addr: format!("{}:{port}", config::DEFAULT_HOST),
            origin: TargetOrigin::Flag,
        });
    }
    Ok(config.map_or_else(
        || ClientTarget {
            addr: format!("{}:{}", config::DEFAULT_HOST, config::DEFAULT_PORT),
            origin: TargetOrigin::DefaultAddress,
        },
        |loaded| ClientTarget {
            addr: resolved_addr(loaded),
            origin: TargetOrigin::Config,
        },
    ))
}

/// Reduce `http://host:port` (or bare `host:port`) to `host:port`.
fn parse_client_url(url: &str) -> Result<String, String> {
    let without_scheme = match url.split_once("://") {
        Some(("http", rest)) => rest,
        Some((scheme, _)) => {
            return Err(format!(
                "unsupported scheme '{scheme}' in --url {url}: only http:// is supported"
            ));
        }
        None => url,
    };
    let authority = without_scheme.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        return Err(format!("invalid --url {url}: missing host"));
    }
    let has_port = authority.rsplit_once(':').is_some_and(|(host, port)| {
        port.parse::<u16>().is_ok() && (!host.contains(':') || host.ends_with(']'))
    });
    Ok(if has_port {
        authority.to_owned()
    } else {
        format!("{authority}:80")
    })
}

/// Confirm a server answers at the target. A config-derived target keeps the plain
/// connection check; every other target must also pass `/health`, so a stray listener on
/// the default port is not mistaken for Higgs.
fn ensure_reachable(target: &ClientTarget) -> Result<(), String> {
    let result = match target.origin {
        TargetOrigin::Config => connect_with_timeout(&target.addr).map(|_| ()),
        TargetOrigin::Flag | TargetOrigin::DefaultAddress => check_health_at(&target.addr),
    };
    result.map_err(|err| {
        format!(
            "{err}\nhint: start a server with 'higgs serve --model <path or hf id>', \
             or pass --url or --port to use a server that runs elsewhere"
        )
    })
}

fn ensure_dashboard_ready(config: &HiggsConfig, profile: Option<&str>) -> Result<(), String> {
    if !config.observability.metrics_log.enabled {
        return Err(
            "cannot open dashboard: [observability.metrics_log] enabled = true required in config"
                .to_owned(),
        );
    }
    let Some(pid) = read_pid(profile) else {
        return Err("cannot open dashboard: no running daemon for this config/profile".to_owned());
    };
    if !pid_is_alive(pid) {
        remove_pid_file(profile);
        return Err(format!(
            "cannot open dashboard: daemon pid {pid} is not running"
        ));
    }
    check_health(config)?;

    let metrics_path = Path::new(&config.observability.metrics_log.path);
    if !metrics_path.exists() {
        return Err(format!(
            "cannot open dashboard: metrics log not found at {}",
            metrics_path.display()
        ));
    }
    Ok(())
}

#[allow(clippy::print_stdout)]
pub fn cmd_shellenv(target: &ClientTarget) -> Result<(), String> {
    ensure_reachable(target)?;
    let base_url = target.base_url();
    println!("export ANTHROPIC_BASE_URL={base_url}");
    println!("export OPENAI_BASE_URL={base_url}");
    Ok(())
}

/// Spawn a command with `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` pointing at the Higgs server.
///
/// Forwards signals to the child and exits with its status. Never returns.
#[allow(clippy::print_stderr, unsafe_code)]
pub fn cmd_exec(target: &ClientTarget, command: &[String]) -> ! {
    let Some((program, args)) = command.split_first() else {
        eprintln!("no command specified");
        std::process::exit(1);
    };

    if let Err(err) = ensure_reachable(target) {
        eprintln!("{err}");
        std::process::exit(1);
    }

    let base_url = target.base_url();
    eprintln!(
        "higgs: using server at {base_url} ({})",
        target.description()
    );
    let mut child = match std::process::Command::new(program)
        .args(args)
        .env("ANTHROPIC_BASE_URL", &base_url)
        .env("OPENAI_BASE_URL", &base_url)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn '{program}': {e}");
            std::process::exit(1);
        }
    };

    let Ok(child_pid_raw) = i32::try_from(child.id()) else {
        eprintln!("child pid {} exceeds i32 range", child.id());
        let _ = child.kill();
        std::process::exit(1);
    };
    let child_pid = nix::unistd::Pid::from_raw(child_pid_raw);
    EXEC_CHILD_PID.store(child_pid_raw, Ordering::SeqCst);
    EXEC_SIGINT_RECEIVED.store(false, Ordering::SeqCst);

    // Forward SIGINT (Ctrl+C) to the child as SIGTERM so wrappers like
    // `sleep` terminate cleanly and we preserve the child's exit status.
    let action = SigAction::new(
        SigHandler::Handler(exec_sigint_handler),
        SaFlags::empty(),
        SigSet::empty(),
    );
    if let Err(e) = unsafe { sigaction(Signal::SIGINT, &action) } {
        eprintln!("warning: failed to set SIGINT handler: {e}");
    }

    let status = loop {
        if EXEC_SIGINT_RECEIVED.swap(false, Ordering::SeqCst) {
            let _ = nix::sys::signal::kill(child_pid, nix::sys::signal::Signal::SIGTERM);
        }

        match child.try_wait() {
            Ok(Some(s)) => {
                use std::os::unix::process::ExitStatusExt;

                break s.code().unwrap_or_else(|| {
                    // Child killed by signal -- use the Unix convention of 128 + signal.
                    s.signal().map_or(1, |sig| 128 + sig)
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => {
                eprintln!("failed to wait for '{program}': {e}");
                break 1;
            }
        }
    };
    EXEC_CHILD_PID.store(0, Ordering::SeqCst);
    std::process::exit(status);
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

    if let Err(e) = config::write_private_file(&pid_path(profile), &child_pid.to_string()) {
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
    let probe_addr = format!(
        "{}:{}",
        resolve_host(&probe_config.server.host),
        probe_config.server.port
    );

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
    if let Err(err) = ensure_dashboard_ready(config, profile) {
        eprintln!("{err}");
        std::process::exit(1);
    }

    let retention = retention_duration(config);
    let metrics = Arc::new(MetricsStore::new(retention));

    attach::load_history(&config.observability.metrics_log, &metrics);

    let log_path_buf = PathBuf::from(&config.observability.metrics_log.path);
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
            std::thread::sleep(Duration::from_secs(EVICTION_INTERVAL_SECS));
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

pub fn retention_duration(config: &HiggsConfig) -> Duration {
    config
        .observability
        .retention
        .unwrap_or(Duration::from_secs(DEFAULT_RETENTION_SECS))
}

pub fn create_metrics(config: &HiggsConfig) -> Arc<MetricsStore> {
    let retention = retention_duration(config);
    Arc::new(if config.observability.metrics_log.enabled {
        match MetricsLogger::new(&config.observability.metrics_log) {
            Ok(logger) => {
                tracing::info!(path = %config.observability.metrics_log.path, "metrics logging enabled");
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
        let mut interval = tokio::time::interval(Duration::from_secs(EVICTION_INTERVAL_SECS));
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

#[allow(clippy::panic, clippy::unwrap_used, unsafe_code)]
#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_config_dir<F: FnOnce(&std::path::Path)>(f: F) {
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    fn client_url_accepts_scheme_and_bare_forms() {
        for (input, expected) in [
            ("http://127.0.0.1:8123", "127.0.0.1:8123"),
            ("http://localhost:8123/", "localhost:8123"),
            ("http://localhost:8123/v1", "localhost:8123"),
            ("localhost:9000", "localhost:9000"),
            ("http://example.test", "example.test:80"),
            ("http://[::1]:8123", "[::1]:8123"),
            ("http://[::1]", "[::1]:80"),
        ] {
            assert_eq!(parse_client_url(input).as_deref(), Ok(expected), "{input}");
        }
        assert!(parse_client_url("https://example.test").is_err());
        assert!(parse_client_url("http://").is_err());
    }

    #[test]
    fn client_target_prefers_flags_then_config_then_default_address() {
        let config = HiggsConfig {
            server: config::ServerConfig {
                host: "0.0.0.0".to_owned(),
                port: 3100,
                ..config::ServerConfig::default()
            },
            ..HiggsConfig::default()
        };
        let from_config = resolve_client_target(Some(&config), None, None).unwrap();
        assert_eq!(from_config.base_url(), "http://127.0.0.1:3100");
        assert_eq!(from_config.origin, TargetOrigin::Config);

        let by_port = resolve_client_target(Some(&config), None, Some(9000)).unwrap();
        assert_eq!(by_port.base_url(), "http://127.0.0.1:9000");

        let by_url = resolve_client_target(Some(&config), Some("http://10.0.0.5:7"), None).unwrap();
        assert_eq!(by_url.base_url(), "http://10.0.0.5:7");

        let default = resolve_client_target(None, None, None).unwrap();
        assert_eq!(default.base_url(), "http://127.0.0.1:8000");
        assert_eq!(default.origin, TargetOrigin::DefaultAddress);
    }

    #[test]
    fn connect_reaches_an_ipv4_listener_through_localhost() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        connect_with_timeout(&format!("localhost:{port}")).unwrap();
    }

    #[test]
    fn connect_failure_names_the_address() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let error = connect_with_timeout(&format!("localhost:{port}")).unwrap_err();
        assert!(error.contains("not running"), "{error}");
    }

    #[test]
    fn config_dir_respects_env_override() {
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let _guard = crate::test_env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            observability: crate::config::ObservabilityConfig {
                retention: Some(Duration::from_secs(30 * 60)),
                ..Default::default()
            },
            ..HiggsConfig::default()
        };
        assert_eq!(retention_duration(&config), Duration::from_secs(30 * 60));
    }

    #[test]
    fn retention_duration_one_year_when_disabled() {
        let config = HiggsConfig {
            observability: crate::config::ObservabilityConfig {
                retention: None,
                ..Default::default()
            },
            ..HiggsConfig::default()
        };
        assert_eq!(
            retention_duration(&config),
            Duration::from_secs(DEFAULT_RETENTION_SECS)
        );
    }

    #[test]
    fn create_metrics_without_logger() {
        let config = HiggsConfig {
            observability: crate::config::ObservabilityConfig {
                metrics_log: crate::config::MetricsLogConfig {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
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
    fn cmd_init_template_documents_thinking_budget() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            let template = std::fs::read_to_string(dir.join("config.toml")).unwrap();
            assert_eq!(template.matches("# thinking_budget = 256").count(), 2);
        });
    }

    /// The commented lines of the template section that starts at `heading`,
    /// uncommented, up to the next `# ---` heading.
    fn uncommented_section(template: &str, first_line: &str) -> String {
        template
            .lines()
            .skip_while(|line| *line != first_line)
            .take_while(|line| !line.starts_with("# ---"))
            .map(|line| line.strip_prefix("# ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn load_text_with(toml: &str, environment: &crate::config::Environment) -> HiggsConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        crate::config::load_config_file_in(&path, None, environment)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn load_text(toml: &str) -> HiggsConfig {
        load_text_with(toml, &crate::config::Environment::default())
    }

    const A_MODEL: &str = "\n[[models]]\npath = \"org/model\"\n";

    #[test]
    fn cmd_init_template_mcp_examples_are_valid_when_uncommented() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            let template = std::fs::read_to_string(dir.join("config.toml")).unwrap();
            assert!(template.contains("--mcp-config"));
            let section = uncommented_section(&template, "# [mcp]");
            let config = load_text(&format!("{section}{A_MODEL}"));
            assert_eq!(config.mcp.servers.len(), 2);
            let lookup = |name: &str| {
                (name == "EXAMPLE_TOKEN" || name == "MCP_TOKEN").then(|| "x".to_owned())
            };
            assert!(config.mcp.problems(&lookup).is_empty());
            assert!(config.notices.deprecations.is_empty());
        });
    }

    #[test]
    fn cmd_init_template_web_examples_are_valid_when_uncommented() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            let template = std::fs::read_to_string(dir.join("config.toml")).unwrap();
            let section = uncommented_section(&template, "# [web]");
            let config = load_text(&format!("{section}{A_MODEL}"));
            assert!(config.web.enabled);
            assert!(config.web.problems().is_empty());
            assert_eq!(config.web.search_provider, "duckduckgo");
        });
    }

    /// Every commented line of the template that is a table header or `key = value`, uncommented.
    fn uncomment_everything(template: &str) -> String {
        let config_line = regex::Regex::new(r"^# (\[\[?[A-Za-z_.*]+\]\]?|[a-z_]+ = .+)$").unwrap();
        template
            .lines()
            .filter(|line| !line.starts_with("# include ="))
            .map(|line| {
                if config_line.is_match(line) {
                    line.strip_prefix("# ").unwrap_or(line)
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn cmd_init_template_loads_without_warnings_when_every_example_is_uncommented() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            let template = std::fs::read_to_string(dir.join("config.toml")).unwrap();
            let everything = uncomment_everything(&template);
            let environment = crate::config::Environment::from_pairs([
                ("HIGGS_API_KEY", "sk-test"),
                ("ANTHROPIC_API_KEY", "sk-ant-test"),
            ]);
            let config = load_text_with(&everything, &environment);
            assert!(
                config.notices.deprecations.is_empty(),
                "{:?}",
                config.notices.deprecations
            );
            let model = config.models.first().unwrap();
            assert_eq!(model.path, "mlx-community/Llama-3.2-1B-Instruct-4bit");
            assert_eq!(model.aliases, ["default"]);
            assert_eq!(config.providers.len(), 3);
            assert!(config.web.enabled);
            assert_eq!(config.experimental.chunked_prefill_threshold, Some(4096));
        });
    }

    #[test]
    fn cmd_init_template_has_no_deprecated_keys_and_loads_as_written() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(None);
            let template = std::fs::read_to_string(dir.join("config.toml")).unwrap();
            let config = load_text(&format!("{template}{A_MODEL}"));
            assert!(config.notices.deprecations.is_empty());
            assert!(!template.contains("__METRICS_PATH__"));
        });
    }

    #[test]
    fn cmd_init_template_uses_the_profile_specific_metrics_path() {
        with_temp_config_dir(|dir| {
            std::fs::create_dir_all(dir).unwrap();
            cmd_init(Some("dev"));
            let template = std::fs::read_to_string(dir.join("config.dev.toml")).unwrap();
            assert!(template.contains("metrics.dev.jsonl"));
            assert!(!template.contains("__METRICS_PATH__"));
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

    #[test]
    fn resolve_host_maps_ipv4_wildcard() {
        assert_eq!(resolve_host("0.0.0.0"), "127.0.0.1");
    }

    #[test]
    fn resolve_host_maps_ipv6_wildcard() {
        assert_eq!(resolve_host("::"), "::1");
    }

    #[test]
    fn resolve_host_passes_through_other() {
        assert_eq!(resolve_host("10.0.0.1"), "10.0.0.1");
        assert_eq!(resolve_host("127.0.0.1"), "127.0.0.1");
    }

    #[test]
    fn health_response_is_ok_parses_json_body() {
        let response =
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"status\":\"ok\"}";
        assert!(health_response_is_ok(response));
    }

    #[test]
    fn health_response_is_ok_handles_pretty_printed_body() {
        let response = "HTTP/1.1 200 OK\r\nx-upstream-status: {\"status\":\"ok\"}\r\n\r\n{\n  \"status\": \"ok\"\n}";
        assert!(health_response_is_ok(response));
    }

    #[test]
    fn health_response_is_ok_rejects_non_ok_status() {
        let response = "HTTP/1.1 200 OK\r\n\r\n{\"status\":\"down\"}";
        assert!(!health_response_is_ok(response));
    }

    #[test]
    fn health_response_is_ok_handles_chunked_json_body() {
        let response = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nf\r\n{\"status\":\"ok\"}\r\n0\r\n\r\n";
        assert!(health_response_is_ok(response));
    }
}
