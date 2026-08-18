//! CLI integration tests covering the current command-contract changes.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]

use std::fs;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn higgs_bin() -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("higgs");
    path
}

fn write_config(dir: &Path, contents: &str) {
    fs::write(dir.join("config.toml"), contents).unwrap();
}

fn unused_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn gateway_config(port: u16, metrics_enabled: bool, metrics_path: &Path) -> String {
    format!(
        r#"[server]
host = "127.0.0.1"
port = {port}

[provider.dummy]
url = "http://127.0.0.1:1"

[logging.metrics]
enabled = {metrics_enabled}
path = "{}"
"#,
        metrics_path.display()
    )
}

fn bootstrap_config(port: u16) -> String {
    format!(
        r#"[server]
host = "127.0.0.1"
port = {port}

[default]
provider = "higgs"
"#
    )
}

#[test]
fn config_set_rejects_wrong_type_without_changing_file() {
    let dir = tempfile::tempdir().unwrap();
    let original = bootstrap_config(8000);
    write_config(dir.path(), &original);

    let output = Command::new(higgs_bin())
        .args(["config", "set", "server.port", "not_a_number"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(dir.path().join("config.toml")).unwrap(),
        original
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("server.port"), "stderr was: {stderr}");
    assert!(stderr.contains("not_a_number"), "stderr was: {stderr}");
    assert!(stderr.contains("u16"), "stderr was: {stderr}");
}

#[test]
fn config_set_rejects_unknown_key_without_changing_file() {
    let dir = tempfile::tempdir().unwrap();
    let original = bootstrap_config(8000);
    write_config(dir.path(), &original);

    let output = Command::new(higgs_bin())
        .args(["config", "set", "nonsense.key", "1"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(dir.path().join("config.toml")).unwrap(),
        original
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown configuration key 'nonsense.key'"),
        "stderr was: {stderr}"
    );
}

#[test]
fn config_set_then_get_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), &bootstrap_config(8000));

    let set = Command::new(higgs_bin())
        .args(["config", "set", "server.port", "18985"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();
    assert!(
        set.status.success(),
        "stderr was: {}",
        String::from_utf8_lossy(&set.stderr)
    );

    let get = Command::new(higgs_bin())
        .args(["config", "get", "server.port"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();
    assert!(get.status.success());
    assert_eq!(String::from_utf8_lossy(&get.stdout).trim(), "18985");
}

#[test]
fn config_set_dynamic_provider_key_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), &bootstrap_config(8000));

    let set = Command::new(higgs_bin())
        .args([
            "config",
            "set",
            "provider.example.url",
            "http://127.0.0.1:9000",
        ])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();
    assert!(
        set.status.success(),
        "stderr was: {}",
        String::from_utf8_lossy(&set.stderr)
    );

    let get = Command::new(higgs_bin())
        .args(["config", "get", "provider.example.url"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();
    assert!(get.status.success());
    assert_eq!(
        String::from_utf8_lossy(&get.stdout).trim(),
        "http://127.0.0.1:9000"
    );
}

#[test]
fn start_rejects_legacy_serve_flags() {
    let output = Command::new(higgs_bin())
        .args(["start", "--model", "mlx-community/Qwen3-1.7B-4bit"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("config/profile-only"),
        "stderr was: {stderr}"
    );
    assert!(stderr.contains("higgs serve"), "stderr was: {stderr}");
}

#[test]
fn shellenv_fails_on_invalid_config() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), "[server]\nport = 8000\n");

    let output = Command::new(higgs_bin())
        .arg("shellenv")
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("config must define at least one [[models]] entry or [provider.*]"),
        "stderr was: {stderr}"
    );
}

#[test]
fn shellenv_fails_when_server_not_running() {
    let dir = tempfile::tempdir().unwrap();
    let metrics_path = dir.path().join("metrics.jsonl");
    write_config(
        dir.path(),
        &gateway_config(unused_port(), true, &metrics_path),
    );

    let output = Command::new(higgs_bin())
        .arg("shellenv")
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not running"), "stderr was: {stderr}");
}

#[test]
fn attach_requires_live_daemon() {
    let dir = tempfile::tempdir().unwrap();
    let metrics_path = dir.path().join("metrics.jsonl");
    write_config(
        dir.path(),
        &gateway_config(unused_port(), true, &metrics_path),
    );

    let output = Command::new(higgs_bin())
        .arg("attach")
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no running daemon"), "stderr was: {stderr}");
}

#[test]
fn attach_requires_metrics_logging() {
    let dir = tempfile::tempdir().unwrap();
    let metrics_path = dir.path().join("metrics.jsonl");
    write_config(
        dir.path(),
        &gateway_config(unused_port(), false, &metrics_path),
    );

    let output = Command::new(higgs_bin())
        .arg("attach")
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("enabled = true required"),
        "stderr was: {stderr}"
    );
}

#[test]
fn stop_force_kills_term_ignoring_process() {
    let dir = tempfile::tempdir().unwrap();
    // Keep the shell alive as the parent so the TERM-ignoring child can be
    // reaped without reintroducing the PID-reuse race from the old fixture.
    let mut child = Command::new("sh")
        .args([
            "-c",
            "perl -e '$SIG{TERM} = q(IGNORE); sleep 60' >/dev/null 2>&1 & printf '%s\\n' \"$!\"; wait",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout_pipe = child.stdout.take().unwrap();
    let mut stdout = BufReader::new(stdout_pipe);
    let mut pid_line = String::new();
    stdout.read_line(&mut pid_line).unwrap();
    let pid: i32 = pid_line.trim().parse().unwrap();

    fs::write(dir.path().join("higgs.pid"), pid.to_string()).unwrap();

    let output = Command::new(higgs_bin())
        .args(["stop", "--force"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(output.status.success(), "stop failed: {:?}", output.status);

    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    panic!("term-ignoring process was still running after stop --force");
}

#[test]
fn exec_fails_on_invalid_config_instead_of_falling_back() {
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), &format!("[server]\nport = {}\n", unused_port()));

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "echo", "hello"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("config must define at least one [[models]] entry or [provider.*]"),
        "stderr was: {stderr}"
    );
}

#[test]
fn shellenv_succeeds_when_target_is_reachable() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = tempfile::tempdir().unwrap();
    let metrics_path = dir.path().join("metrics.jsonl");
    write_config(dir.path(), &gateway_config(port, true, &metrics_path));

    let output = Command::new(higgs_bin())
        .arg("shellenv")
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("OPENAI_BASE_URL=http://127.0.0.1:"),
        "stdout was: {stdout}"
    );
}
