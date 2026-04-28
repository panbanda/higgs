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

    let child = cmd
        .spawn()
        .with_context(|| format!("spawn {} serve --model {model}", bin.display()))?;

    let base_url = format!("http://127.0.0.1:{port}");
    crate::server::wait_until_ready(&base_url, timeout).await?;
    Ok(child)
}

/// Sends `SIGTERM` (via `Child::kill`) and waits for the process to exit.
///
/// Errors signaling/reaping are returned but typically swallowed by
/// callers — teardown failures shouldn't mask the bench's real result.
pub async fn stop_server(mut child: Child) -> Result<()> {
    child.start_kill().context("signal higgs child")?;
    child.wait().await.context("reap higgs child")?;
    Ok(())
}
