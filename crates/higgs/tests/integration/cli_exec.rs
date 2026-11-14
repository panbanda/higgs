//! CLI integration tests for `higgs exec`.

#![allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]

use std::io::Write;
use std::net::TcpListener;
use std::process::Command;

fn higgs_bin() -> std::path::PathBuf {
    // cargo sets this during `cargo test`
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

#[test]
fn exec_exits_with_error_when_server_not_running() {
    // Bind a random port then drop the listener so nothing is listening on it.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();

    let dir = write_test_config(port);

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "echo", "hello"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not running"),
        "expected 'not running' in stderr, got: {stderr}"
    );
}

#[test]
fn exec_requires_command_argument() {
    let output = Command::new(higgs_bin()).args(["exec"]).output().unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    // clap should report missing required argument
    assert!(
        stderr.contains("required") || stderr.contains("Usage"),
        "expected clap error in stderr, got: {stderr}"
    );
}

/// Write a minimal valid config.toml pointing at the given port.
/// Includes a dummy provider so config validation passes (otherwise
/// `load_config_for_command` errors and `unwrap_or_default` silently
/// falls back to defaults, making the test flaky).
fn write_test_config(port: u16) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let mut f = std::fs::File::create(&config_path).unwrap();
    write!(
        f,
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\n\n\
         [provider.dummy]\nurl = \"http://127.0.0.1:1\"\n"
    )
    .unwrap();
    dir
}

/// Bind a random port and write a config pointing at it.
/// Returns (`TcpListener`, temp dir) -- keep the listener alive so `higgs exec`
/// sees a "running" server.
fn fake_server_env() -> (TcpListener, tempfile::TempDir) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = write_test_config(port);
    (listener, dir)
}

#[test]
fn exec_forwards_child_exit_code() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "sh", "-c", "exit 42"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(42),
        "expected exit code 42, got {:?}",
        output.status.code()
    );
}

#[test]
fn exec_forwards_zero_exit_code() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "true"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "expected exit code 0, got {:?}",
        output.status.code()
    );
}

#[test]
fn exec_reports_spawn_failure() {
    let (_listener, dir) = fake_server_env();

    let output = Command::new(higgs_bin())
        .args(["exec", "--", "/nonexistent/binary"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to spawn"),
        "expected 'failed to spawn' in stderr, got: {stderr}"
    );
}

#[test]
fn exec_sigint_terminates_child() {
    use std::os::unix::process::ExitStatusExt;
    use std::time::{Duration, Instant};

    let (_listener, dir) = fake_server_env();

    let mut child = Command::new(higgs_bin())
        .args(["exec", "--", "sleep", "60"])
        .env("HIGGS_CONFIG_DIR", dir.path())
        .spawn()
        .unwrap();

    // Give the child process time to start
    std::thread::sleep(Duration::from_millis(500));

    let pid = nix::unistd::Pid::from_raw(i32::try_from(child.id()).unwrap());
    nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGINT).unwrap();

    let start = Instant::now();
    let status = child.wait().unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "child should terminate within 5s after SIGINT, took {elapsed:?}"
    );
    // `higgs exec` should not hang on SIGINT. Depending on runtime/platform
    // timing we may observe either the wrapper's translated SIGTERM exit code
    // or the wrapper itself exiting from SIGINT.
    let translated_sigterm = status.code() == Some(143);
    let direct_sigint = status.signal() == Some(2);
    assert!(
        translated_sigterm || direct_sigint,
        "expected exit code 143 or signal 2, got: {status}"
    );
}

/// Serve `GET /health` with the Higgs health body on a random port until the process ends.
fn fake_health_server() -> u16 {
    use std::io::Read;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for connection in listener.incoming() {
            let Ok(mut stream) = connection else { continue };
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            let body = r#"{"status":"ok"}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    port
}

#[test]
fn exec_without_config_uses_url_flag_and_names_the_server() {
    let port = fake_health_server();
    let empty_config_dir = tempfile::tempdir().unwrap();

    let output = Command::new(higgs_bin())
        .args(["exec", "--url", &format!("http://127.0.0.1:{port}")])
        .args([
            "--",
            "sh",
            "-c",
            "echo $ANTHROPIC_BASE_URL $OPENAI_BASE_URL",
        ])
        .env("HIGGS_CONFIG_DIR", empty_config_dir.path())
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    let base_url = format!("http://127.0.0.1:{port}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        format!("{base_url} {base_url}")
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("using server at {base_url}")),
        "stderr was: {stderr}"
    );
}

#[test]
fn shellenv_without_config_uses_port_flag_and_keeps_stdout_clean() {
    let port = fake_health_server();
    let empty_config_dir = tempfile::tempdir().unwrap();

    let output = Command::new(higgs_bin())
        .args(["shellenv", "--port", &port.to_string()])
        .env("HIGGS_CONFIG_DIR", empty_config_dir.path())
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
    let base_url = format!("http://127.0.0.1:{port}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!("export ANTHROPIC_BASE_URL={base_url}\nexport OPENAI_BASE_URL={base_url}\n")
    );
    assert!(output.stderr.is_empty(), "{output:?}");
}

#[test]
fn exec_without_server_says_how_to_start_one() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let empty_config_dir = tempfile::tempdir().unwrap();

    let output = Command::new(higgs_bin())
        .args(["exec", "--port", &port.to_string(), "--", "true"])
        .env("HIGGS_CONFIG_DIR", empty_config_dir.path())
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not running"), "stderr was: {stderr}");
    assert!(
        stderr.contains("higgs serve --model"),
        "stderr was: {stderr}"
    );
    assert!(stderr.contains("--url or --port"), "stderr was: {stderr}");
}

#[test]
fn exec_rejects_url_and_port_together() {
    let output = Command::new(higgs_bin())
        .args([
            "exec",
            "--url",
            "http://127.0.0.1:1",
            "--port",
            "1",
            "--",
            "true",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
