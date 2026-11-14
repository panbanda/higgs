//! Stdio transport: a child process speaking newline-delimited JSON-RPC.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::Value;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use super::config::StdioSpec;
use super::jsonrpc::{LineError, encode_line, read_bounded_line};
use super::transport::{
    Connection, INCOMING_CAPACITY, Incoming, MAX_MESSAGE_BYTES, Transport, TransportError,
};

/// How long a server gets to exit after its stdin closes before it is killed.
const EXIT_GRACE: Duration = Duration::from_millis(1500);
const STDERR_LINE_LIMIT: usize = 8 * 1024;
const STDERR_TAIL_LINES: usize = 5;

type StderrTail = Arc<Mutex<VecDeque<String>>>;

pub struct StdioTransport {
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
    close_signal: Mutex<Option<oneshot::Sender<()>>>,
    supervisor: Mutex<Option<JoinHandle<()>>>,
}

pub fn spawn(name: &str, spec: &StdioSpec) -> Result<Connection, TransportError> {
    let mut command = tokio::process::Command::new(&spec.command);
    command
        .args(&spec.args)
        .envs(spec.env.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        // Own process group so closing the server also reaches the
        // grandchildren that launchers like `npx` start.
        .process_group(0);
    if let Some(directory) = &spec.cwd {
        command.current_dir(directory);
    }
    let mut child = command.spawn().map_err(|error| {
        TransportError::Start(format!("cannot run '{}': {error}", spec.command))
    })?;
    let (Some(stdin), Some(stdout), Some(stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        return Err(TransportError::Start(
            "child process pipes are unavailable".to_owned(),
        ));
    };
    let group = child.id().and_then(|pid| i32::try_from(pid).ok());
    let (sender, receiver) = mpsc::channel(INCOMING_CAPACITY);
    let tail: StderrTail = Arc::new(Mutex::new(VecDeque::new()));

    tokio::spawn(drain_stderr(name.to_owned(), stderr, Arc::clone(&tail)));
    tokio::spawn(read_stdout(stdout, sender.clone()));
    let (close_signal, close_receiver) = oneshot::channel();
    let supervisor = tokio::spawn(supervise(child, group, close_receiver, sender, tail));

    Ok(Connection {
        transport: Arc::new(StdioTransport {
            stdin: tokio::sync::Mutex::new(Some(stdin)),
            close_signal: Mutex::new(Some(close_signal)),
            supervisor: Mutex::new(Some(supervisor)),
        }),
        incoming: receiver,
    })
}

async fn read_stdout(stdout: tokio::process::ChildStdout, sender: mpsc::Sender<Incoming>) {
    let mut reader = BufReader::new(stdout);
    let mut line = Vec::new();
    loop {
        match read_bounded_line(&mut reader, &mut line, MAX_MESSAGE_BYTES).await {
            Ok(true) => {}
            // End of output: the supervisor reports the exit with its status.
            Ok(false) => return,
            Err(error) => {
                let _ = sender.send(Incoming::Closed(error.to_string())).await;
                return;
            }
        }
        let text = String::from_utf8_lossy(&line);
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Servers sometimes print banners or logs to stdout; skip anything
        // that is not JSON rather than treating it as a protocol failure.
        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                if sender.send(Incoming::Message(value)).await.is_err() {
                    return;
                }
            }
            Err(error) => {
                tracing::debug!(%error, "ignoring non-JSON line on MCP server stdout");
            }
        }
    }
}

async fn drain_stderr(name: String, stderr: tokio::process::ChildStderr, tail: StderrTail) {
    let mut reader = BufReader::new(stderr);
    let mut line = Vec::new();
    loop {
        match read_bounded_line(&mut reader, &mut line, STDERR_LINE_LIMIT).await {
            Ok(true) => {
                let text = String::from_utf8_lossy(&line).trim_end().to_owned();
                tracing::debug!(server = %name, "stderr: {text}");
                record_tail_line(&tail, text);
            }
            // The rest of an oversized line is read as the next line; the
            // pipe must keep draining or the child would block on it.
            Err(LineError::TooLong { .. }) => {}
            Ok(false) | Err(LineError::Io(_)) => return,
        }
    }
}

fn record_tail_line(tail: &StderrTail, text: String) {
    let mut lines = tail.lock().unwrap_or_else(PoisonError::into_inner);
    if lines.len() == STDERR_TAIL_LINES {
        lines.pop_front();
    }
    lines.push_back(text);
}

fn tail_summary(tail: &StderrTail) -> Option<String> {
    let lines = tail.lock().unwrap_or_else(PoisonError::into_inner);
    (!lines.is_empty()).then(|| {
        let joined = lines
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" | ");
        joined.chars().take(500).collect()
    })
}

fn kill_process_group(group: Option<i32>) {
    if let Some(id) = group {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(id),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
}

async fn supervise(
    mut child: Child,
    group: Option<i32>,
    close_receiver: oneshot::Receiver<()>,
    sender: mpsc::Sender<Incoming>,
    tail: StderrTail,
) {
    tokio::select! {
        waited = child.wait() => {
            let mut reason = match waited {
                Ok(status) => format!("server process exited ({status})"),
                Err(error) => format!("cannot wait for server process: {error}"),
            };
            if let Some(summary) = tail_summary(&tail) {
                reason.push_str("; stderr: ");
                reason.push_str(&summary);
            }
            let _ = sender.send(Incoming::Closed(reason)).await;
        }
        _ = close_receiver => {
            if tokio::time::timeout(EXIT_GRACE, child.wait()).await.is_err() {
                let _ = child.kill().await;
            }
            kill_process_group(group);
            let _ = sender.send(Incoming::Closed("closed by client".to_owned())).await;
        }
    }
}

async fn send_frame(
    stdin: &tokio::sync::Mutex<Option<ChildStdin>>,
    frame: &str,
) -> Result<(), TransportError> {
    let mut guard = stdin.lock().await;
    match guard.as_mut() {
        Some(pipe) => write_frame(pipe, frame)
            .await
            .map_err(|error| TransportError::Send(error.to_string())),
        None => Err(TransportError::Send("connection is closed".to_owned())),
    }
}

async fn write_frame(stdin: &mut ChildStdin, frame: &str) -> std::io::Result<()> {
    stdin.write_all(frame.as_bytes()).await?;
    stdin.flush().await
}

impl Transport for StdioTransport {
    fn send(&self, message: Value) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move { send_frame(&self.stdin, &encode_line(&message)).await })
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            // Dropping stdin is the polite shutdown request for stdio servers.
            self.stdin.lock().await.take();
            let pending_signal = self
                .close_signal
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(sender) = pending_signal {
                let _ = sender.send(());
            }
            let supervisor = self
                .supervisor
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(handle) = supervisor {
                let _ = tokio::time::timeout(EXIT_GRACE * 3, handle).await;
            }
        })
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::mcp::config::{ResolvedServer, ServerSpec};
    use crate::mcp::manager::{BackoffPolicy, McpManager};
    use crate::mcp::transport::{DefaultTransportFactory, TransportFactory};

    const INITIALIZE_REPLY: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"sh-server","version":"0.1"}}}"#;
    const LIST_REPLY: &str = r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"greet","description":"say hi","inputSchema":{"type":"object"}}]}}"#;

    /// A POSIX shell script playing an MCP server with fixed replies. The
    /// client numbers its requests 1, 2, 3, so canned ids line up.
    fn server_script(preamble: &str, call_reply: &str) -> String {
        format!(
            "{preamble}\nread line\nprintf '%s\\n' '{INITIALIZE_REPLY}'\nread line\nread line\nprintf '%s\\n' '{LIST_REPLY}'\nread line\nprintf '%s\\n' '{call_reply}'\nwhile read line; do :; done\n"
        )
    }

    fn shell_server(name: &str, script: &str, env: Vec<(String, String)>) -> ResolvedServer {
        ResolvedServer {
            name: name.to_owned(),
            enabled: true,
            timeout: Duration::from_secs(5),
            spec: ServerSpec::Stdio(StdioSpec {
                command: "sh".to_owned(),
                args: vec!["-c".to_owned(), script.to_owned()],
                env,
                cwd: None,
            }),
        }
    }

    fn manager_for(server: ResolvedServer) -> McpManager {
        let factory: Arc<dyn TransportFactory> = Arc::new(DefaultTransportFactory);
        McpManager::start_with(
            vec![server],
            4096,
            &factory,
            BackoffPolicy {
                initial: Duration::from_millis(20),
                maximum: Duration::from_millis(100),
            },
        )
    }

    async fn wait_for_state(
        manager: &McpManager,
        state: &str,
    ) -> crate::mcp::manager::ServerStatus {
        for _ in 0..300 {
            let status = manager.statuses().into_iter().next().unwrap();
            if status.state == state {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("never reached {state}: {:?}", manager.statuses());
    }

    fn process_is_alive(pid: i32) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
    }

    async fn read_pid(path: &std::path::Path) -> i32 {
        for _ in 0..100 {
            if let Ok(text) = std::fs::read_to_string(path) {
                if let Ok(pid) = text.trim().parse() {
                    return pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("pid file {} never appeared", path.display());
    }

    #[tokio::test]
    async fn real_child_process_speaks_the_protocol() {
        let script = server_script(
            "echo 'starting up, not JSON'",
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hello from sh"}]}}"#,
        );
        let manager = manager_for(shell_server("sh", &script, Vec::new()));
        let status = wait_for_state(&manager, "ready").await;
        assert_eq!(status.server_info.unwrap().name, "sh-server");
        assert_eq!(manager.tools()[0].name, "sh__greet");

        let output = manager
            .call("sh__greet", json!({"who": "x"}))
            .await
            .unwrap();
        assert_eq!(output.text, "hello from sh");
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn environment_and_working_directory_reach_the_child() {
        let directory = tempfile::tempdir().unwrap();
        let script = server_script(
            "",
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}]}}"#,
        )
        .replace(
            "read line\nprintf",
            "pwd > seen_cwd; echo \"$GREETING\" > seen_env; read line\nprintf",
        );
        let mut server = shell_server(
            "sh",
            &script,
            vec![("GREETING".to_owned(), "hello env".to_owned())],
        );
        if let ServerSpec::Stdio(stdio) = &mut server.spec {
            stdio.cwd = Some(directory.path().to_path_buf());
        }
        let manager = manager_for(server);
        wait_for_state(&manager, "ready").await;
        manager.call("sh__greet", json!({})).await.unwrap();
        manager.shutdown().await;
        assert_eq!(
            std::fs::read_to_string(directory.path().join("seen_env"))
                .unwrap()
                .trim(),
            "hello env"
        );
        let seen = std::fs::read_to_string(directory.path().join("seen_cwd")).unwrap();
        assert_eq!(
            std::fs::canonicalize(seen.trim()).unwrap(),
            std::fs::canonicalize(directory.path()).unwrap()
        );
    }

    #[tokio::test]
    async fn a_missing_executable_is_reported_as_a_failed_server() {
        let mut server = shell_server("nope", "", Vec::new());
        if let ServerSpec::Stdio(stdio) = &mut server.spec {
            "definitely-not-a-real-binary-xyz".clone_into(&mut stdio.command);
        }
        let manager = manager_for(server);
        let status = wait_for_state(&manager, "failed").await;
        assert!(status.error.unwrap().contains("cannot run"));
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn an_early_exit_reports_status_and_stderr() {
        let manager = manager_for(shell_server(
            "dies",
            "echo 'fatal: bad config' >&2; exit 3",
            Vec::new(),
        ));
        let status = wait_for_state(&manager, "failed").await;
        let error = status.error.unwrap();
        assert!(error.contains("exit status: 3"), "{error}");
        assert!(error.contains("fatal: bad config"), "{error}");
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn a_crashed_child_is_restarted_when_a_tool_is_next_used() {
        let directory = tempfile::tempdir().unwrap();
        let counter = directory.path().join("starts");
        // The first start answers the handshake and dies right after listing
        // tools; later starts answer the call as well.
        let script = format!(
            "echo x >> {counter}\nread line\nprintf '%s\\n' '{INITIALIZE_REPLY}'\nread line\nread line\nprintf '%s\\n' '{LIST_REPLY}'\nif [ \"$(wc -l < {counter})\" -lt 2 ]; then exit 1; fi\nread line\nprintf '%s\\n' '{{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"second life\"}}]}}}}'\nwhile read line; do :; done\n",
            counter = counter.display()
        );
        let manager = manager_for(shell_server("sh", &script, Vec::new()));
        // Give the client time to notice the exit: a call written to a dying
        // process has an unknown outcome and is not replayed.
        tokio::time::sleep(Duration::from_millis(500)).await;
        let output = manager.call("sh__greet", json!({})).await.unwrap();
        assert_eq!(output.text, "second life");
        let starts = std::fs::read_to_string(&counter).unwrap().lines().count();
        assert_eq!(starts, 2);
        manager.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_leaves_no_child_or_grandchild_behind() {
        let directory = tempfile::tempdir().unwrap();
        let server_pid_path = directory.path().join("server.pid");
        let grandchild_pid_path = directory.path().join("grandchild.pid");
        let script = server_script(
            &format!(
                "echo $$ > {}\nsleep 300 &\necho $! > {}",
                server_pid_path.display(),
                grandchild_pid_path.display()
            ),
            r#"{"jsonrpc":"2.0","id":3,"result":{"content":[]}}"#,
        );
        let manager = manager_for(shell_server("sh", &script, Vec::new()));
        wait_for_state(&manager, "ready").await;
        let server_pid = read_pid(&server_pid_path).await;
        let grandchild_pid = read_pid(&grandchild_pid_path).await;
        assert!(process_is_alive(server_pid));
        assert!(process_is_alive(grandchild_pid));

        manager.shutdown().await;
        for _ in 0..100 {
            if !process_is_alive(grandchild_pid) && !process_is_alive(server_pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("server or grandchild survived shutdown");
    }

    #[tokio::test]
    async fn a_server_that_ignores_stdin_close_is_killed() {
        let directory = tempfile::tempdir().unwrap();
        let pid_path = directory.path().join("stubborn.pid");
        let script = format!(
            "echo $$ > {}\nread line\nprintf '%s\\n' '{INITIALIZE_REPLY}'\nread line\nread line\nprintf '%s\\n' '{LIST_REPLY}'\nwhile true; do sleep 1; done\n",
            pid_path.display()
        );
        let manager = manager_for(shell_server("sh", &script, Vec::new()));
        wait_for_state(&manager, "ready").await;
        let pid = read_pid(&pid_path).await;
        manager.shutdown().await;
        for _ in 0..100 {
            if !process_is_alive(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("stubborn server survived shutdown");
    }
}
