#![allow(clippy::too_long_first_doc_paragraph)]
//! Helpers for launching and stopping a higgs server as a subprocess.
//!
//! Used by `bench_prefix_cache`, `bench_prefix_cache_turns`, `bench_tq_configs`,
//! and `bench_all`, which all spin up one higgs server per model and tear it
//! down between models.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};

/// Returns the path to the higgs binary. Honors `HIGGS_BIN` if set,
/// otherwise defaults to `./target/release/higgs` relative to the
/// workspace root.
#[must_use]
pub fn higgs_bin() -> PathBuf {
    if let Ok(p) = std::env::var("HIGGS_BIN") {
        return PathBuf::from(p);
    }
    PathBuf::from("./target/release/higgs")
}

/// Verifies the `higgs` binary is present before spawning. Auto-building
/// from inside a bench would hide setup work, so the caller is expected
/// to run `cargo build --release -p higgs` first or set `HIGGS_BIN`.
fn ensure_higgs_bin(bin: &std::path::Path) -> Result<()> {
    if bin.exists() {
        return Ok(());
    }
    if std::env::var_os("HIGGS_BIN").is_some() {
        anyhow::bail!(
            "HIGGS_BIN points to {} but that file does not exist. \
             Run `cargo build --release -p higgs` first or fix HIGGS_BIN.",
            bin.display()
        );
    }
    anyhow::bail!(
        "higgs binary not found at {}. \
         Run `cargo build --release -p higgs` first or set HIGGS_BIN=/path/to/higgs.",
        bin.display()
    );
}

/// Spawns `higgs serve --model <model> --port <port> [extra_args...]` as a
/// child process with stdio captured to /dev/null, then waits up to
/// `timeout` for the server to respond on `/v1/models`. Returns the live
/// child handle on success.
pub async fn start_higgs_server(
    model: &str,
    port: u16,
    extra_args: &[String],
    timeout: Duration,
) -> Result<Child> {
    let bin = higgs_bin();
    ensure_higgs_bin(&bin)?;
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .args(extra_args)
        .env("HIGGS_ENABLE_THINKING", "0")
        .env("HIGGS_NO_CONFIG", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {} serve --model {model}", bin.display()))?;

    // Race the readiness probe against the child process exiting. If
    // `higgs` exits during startup (bad model path, --port already in
    // use, panic during model load), surface that immediately instead
    // of waiting out the full readiness `timeout` and returning a
    // generic "not reachable" message.
    let base_url = format!("http://127.0.0.1:{port}");
    tokio::select! {
        ready = crate::server::wait_until_ready(&base_url, timeout) => {
            ready?;
        }
        wait_result = child.wait() => {
            let exit_status = wait_result.context("wait higgs child during startup")?;
            anyhow::bail!(
                "higgs exited before becoming ready (status: {exit_status}); \
                 check the model path and that port {port} is free"
            );
        }
    }
    Ok(child)
}

/// Forcefully terminates the higgs child and waits for it to exit.
///
/// `Child::start_kill` sends `SIGKILL` on Unix; there is no graceful
/// shutdown path. Errors signaling/reaping are returned but typically
/// swallowed by callers — teardown failures shouldn't mask the bench's
/// real result.
pub async fn stop_server(mut child: Child) -> Result<()> {
    child.start_kill().context("signal higgs child")?;
    child.wait().await.context("reap higgs child")?;
    Ok(())
}
