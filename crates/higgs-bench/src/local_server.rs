//! A server process started for the length of one benchmark or self-test run.

use std::fs::File;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::server::wait_until_ready;

const LOG_TAIL_LINES: usize = 25;

/// How to start the server: an executable and its full argument list.
#[derive(Debug, Clone)]
pub struct ServerCommand {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
}

/// A running child server. It is killed when this value is dropped, and only that process.
pub struct LocalServer {
    child: Child,
    base_url: String,
    log_path: PathBuf,
}

/// A port nothing is listening on right now.
pub fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("find a free port")?;
    Ok(listener.local_addr().context("read the bound port")?.port())
}

impl LocalServer {
    /// Starts `command`, which must serve on `127.0.0.1:port`, with its output going to a log file
    /// shown if the server fails to start.
    pub fn spawn(command: &ServerCommand, port: u16) -> Result<Self> {
        let log_path = std::env::temp_dir().join(format!(
            "higgs-local-server-{}-{port}.log",
            std::process::id()
        ));
        let log = File::create(&log_path)
            .with_context(|| format!("create server log {}", log_path.display()))?;
        let child = Command::new(&command.executable)
            .args(&command.arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::from(
                log.try_clone().context("duplicate the server log handle")?,
            ))
            .stderr(Stdio::from(log))
            .spawn()
            .with_context(|| format!("start {}", command.executable.display()))?;
        Ok(Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
            log_path,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Waits until the server answers its health probe, failing early if the process exits.
    pub async fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().context("check the server process")? {
                bail!(
                    "the server exited during startup ({status})\n{}",
                    self.log_tail()
                );
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!(
                    "the server was not ready within {timeout:?}\n{}",
                    self.log_tail()
                );
            }
            let slice = remaining.min(Duration::from_secs(2));
            if wait_until_ready(&self.base_url, slice).await.is_ok() {
                return Ok(());
            }
        }
    }

    /// The last lines the server wrote.
    pub fn log_tail(&self) -> String {
        let text = std::fs::read_to_string(&self.log_path).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(LOG_TAIL_LINES);
        let tail = lines.get(start..).unwrap_or_default().join("\n");
        format!("--- server log ({}) ---\n{tail}", self.log_path.display())
    }
}

impl Drop for LocalServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_ports_are_nonzero_and_bindable() -> Result<()> {
        let port = free_port()?;
        assert_ne!(port, 0);
        TcpListener::bind(("127.0.0.1", port)).context("the port should be free")?;
        Ok(())
    }

    #[tokio::test]
    async fn a_server_that_exits_early_is_reported_with_its_log() -> Result<()> {
        let command = ServerCommand {
            executable: PathBuf::from("/bin/sh"),
            arguments: vec![
                "-c".to_owned(),
                "echo boom-from-server >&2; exit 3".to_owned(),
            ],
        };
        let mut server = LocalServer::spawn(&command, free_port()?)?;
        let error = server
            .wait_ready(Duration::from_secs(10))
            .await
            .err()
            .context("startup should fail")?;
        let message = format!("{error:#}");
        assert!(message.contains("exited during startup"), "{message}");
        assert!(message.contains("boom-from-server"), "{message}");
        Ok(())
    }

    #[tokio::test]
    async fn a_server_that_never_answers_times_out() -> Result<()> {
        let command = ServerCommand {
            executable: PathBuf::from("/bin/sleep"),
            arguments: vec!["30".to_owned()],
        };
        let mut server = LocalServer::spawn(&command, free_port()?)?;
        let pid = server.pid();
        let error = server
            .wait_ready(Duration::from_millis(600))
            .await
            .err()
            .context("startup should time out")?;
        assert!(format!("{error:#}").contains("not ready within"));
        drop(server);
        let still_running = Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        assert!(!still_running, "dropping the server must kill its process");
        Ok(())
    }
}
