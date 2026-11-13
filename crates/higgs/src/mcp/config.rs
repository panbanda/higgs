//! Resolved `[mcp]` configuration: server definitions, the desktop-client JSON file
//! format, `${VAR}` expansion and validation. The on-disk spelling lives in
//! `crate::config::raw`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::config::env_ref;
use crate::config::units::HumanDuration;

pub use crate::config::env_ref::{EnvLookup, process_env};

/// Namespace reserved for tools Higgs provides itself (see `[web]`).
pub const BUILTIN_SERVER_NAME: &str = "higgs";
/// Separator between server and tool name in namespaced tool names.
pub const NAMESPACE_SEPARATOR: &str = "__";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RESULT_BYTES: usize = 256 * 1024;
const MAX_TIMEOUT: Duration = Duration::from_secs(3600);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpConfig {
    /// `[mcp.servers.<name>]` tables.
    pub servers: BTreeMap<String, McpServerConfig>,
    /// Upper bound on the size of one tool result; larger results are
    /// truncated and flagged.
    pub max_result_bytes: usize,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            servers: BTreeMap::new(),
            max_result_bytes: DEFAULT_MAX_RESULT_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    pub enabled: bool,
    /// `"stdio"` or `"http"` (Streamable HTTP). Inferred from `command` or
    /// `url` when omitted.
    pub transport: Option<String>,
    /// Executable for stdio servers. Run directly, never through a shell.
    pub command: Option<String>,
    pub args: Vec<String>,
    /// Extra environment for the child process; values may reference the
    /// Higgs process environment as `${VAR}`.
    pub env: BTreeMap<String, String>,
    pub cwd: Option<String>,
    /// Endpoint for Streamable HTTP servers.
    pub url: Option<String>,
    /// Extra request headers; values may reference `${VAR}`.
    pub headers: BTreeMap<String, String>,
    /// Bound on connecting, listing and each tool call.
    pub timeout: Duration,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            transport: None,
            command: None,
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            url: None,
            headers: BTreeMap::new(),
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    Stdio,
    Http,
}

impl TransportKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Http => "http",
        }
    }
}

impl McpServerConfig {
    pub fn transport_kind(&self) -> Result<TransportKind, String> {
        match self.transport.as_deref().map(str::trim) {
            Some("stdio") => Ok(TransportKind::Stdio),
            Some("http" | "streamable-http" | "streamable_http" | "streamableHttp") => {
                Ok(TransportKind::Http)
            }
            Some("sse") => Err(
                "the legacy HTTP+SSE transport is not supported; use a Streamable HTTP endpoint"
                    .to_owned(),
            ),
            Some(other) => Err(format!(
                "unknown transport '{other}' (expected \"stdio\" or \"http\")"
            )),
            None => match (self.command.is_some(), self.url.is_some()) {
                (true, false) => Ok(TransportKind::Stdio),
                (false, true) => Ok(TransportKind::Http),
                (true, true) => {
                    Err("both command and url are set; set transport to choose one".to_owned())
                }
                (false, false) => Err("needs either command (stdio) or url (http)".to_owned()),
            },
        }
    }

    /// The timeout, if it is within the range the client can honor.
    pub fn valid_timeout(&self) -> Option<Duration> {
        (!self.timeout.is_zero() && self.timeout <= MAX_TIMEOUT).then_some(self.timeout)
    }
}

/// A server definition with `${VAR}` references expanded, ready to connect.
#[derive(Debug, Clone)]
pub struct ResolvedServer {
    pub name: String,
    pub enabled: bool,
    pub timeout: Duration,
    pub spec: ServerSpec,
}

#[derive(Debug, Clone)]
pub enum ServerSpec {
    Stdio(StdioSpec),
    Http(HttpSpec),
}

impl ServerSpec {
    pub const fn kind(&self) -> TransportKind {
        match self {
            Self::Stdio(_) => TransportKind::Stdio,
            Self::Http(_) => TransportKind::Http,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StdioSpec {
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct HttpSpec {
    pub url: reqwest::Url,
    pub headers: Vec<(String, String)>,
    pub request_timeout: Duration,
}

fn valid_server_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("server name must not be empty".to_owned());
    }
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_' || character == '-')
    {
        return Err(format!(
            "server name '{name}' may only contain letters, digits, '-' and '_'"
        ));
    }
    if name.contains(NAMESPACE_SEPARATOR) || name.starts_with('_') || name.ends_with('_') {
        return Err(format!(
            "server name '{name}' must not contain '__' or start/end with '_' (it namespaces tool names)"
        ));
    }
    if name == BUILTIN_SERVER_NAME {
        return Err(format!(
            "server name '{BUILTIN_SERVER_NAME}' is reserved for built-in tools"
        ));
    }
    Ok(())
}

impl McpConfig {
    /// Every problem with this configuration, one message each. Empty when the
    /// configuration can be started. Checks that need the environment use
    /// `lookup`.
    pub fn problems(&self, lookup: EnvLookup<'_>) -> Vec<String> {
        let mut problems = Vec::new();
        if self.max_result_bytes == 0 {
            problems.push("mcp.max_result_size must be greater than zero".to_owned());
        }
        for (name, server) in &self.servers {
            if let Err(message) = Self::resolve_server(name, server, lookup) {
                problems.push(format!("mcp server '{name}': {message}"));
            }
        }
        problems
    }

    /// Expand and validate every enabled and disabled server, failing on the
    /// first problem.
    pub fn resolve(&self, lookup: EnvLookup<'_>) -> Result<Vec<ResolvedServer>, String> {
        self.servers
            .iter()
            .map(|(name, server)| {
                Self::resolve_server(name, server, lookup)
                    .map_err(|message| format!("mcp server '{name}': {message}"))
            })
            .collect()
    }

    /// Expand and validate one server definition.
    pub fn resolve_server(
        name: &str,
        server: &McpServerConfig,
        lookup: EnvLookup<'_>,
    ) -> Result<ResolvedServer, String> {
        valid_server_name(name)?;
        let kind = server.transport_kind()?;
        let timeout = server.valid_timeout().ok_or_else(|| {
            format!(
                "timeout must be longer than zero and at most {}, got {}",
                HumanDuration::new(MAX_TIMEOUT),
                HumanDuration::new(server.timeout)
            )
        })?;
        // A disabled server is never started, so its environment is not
        // required to resolve.
        let expand = |value: &str| -> Result<String, String> {
            if server.enabled {
                env_ref::expand(value, lookup).map_err(|error| error.to_string())
            } else {
                Ok(value.to_owned())
            }
        };
        let spec = match kind {
            TransportKind::Stdio => {
                let command = server.command.as_deref().map(str::trim).unwrap_or_default();
                if command.is_empty() {
                    return Err("stdio transport requires a non-empty command".to_owned());
                }
                if server.url.is_some() {
                    return Err("url is only valid for the http transport".to_owned());
                }
                let mut env = Vec::new();
                for (key, value) in &server.env {
                    if key.is_empty() || key.contains('=') || key.contains('\0') {
                        return Err(format!("invalid environment variable name '{key}'"));
                    }
                    env.push((
                        key.clone(),
                        expand(value).map_err(|e| format!("env {key}: {e}"))?,
                    ));
                }
                ServerSpec::Stdio(StdioSpec {
                    command: command.to_owned(),
                    args: server.args.clone(),
                    env,
                    cwd: server.cwd.as_ref().map(PathBuf::from),
                })
            }
            TransportKind::Http => {
                let raw = server.url.as_deref().map(str::trim).unwrap_or_default();
                if raw.is_empty() {
                    return Err("http transport requires a url".to_owned());
                }
                if server.command.is_some() {
                    return Err("command is only valid for the stdio transport".to_owned());
                }
                let url = reqwest::Url::parse(raw)
                    .map_err(|error| format!("invalid url '{raw}': {error}"))?;
                if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                    return Err(format!("url '{raw}' must be an http(s) URL with a host"));
                }
                let mut headers = Vec::new();
                for (key, value) in &server.headers {
                    http::HeaderName::from_bytes(key.as_bytes())
                        .map_err(|_| format!("invalid header name '{key}'"))?;
                    let expanded = expand(value).map_err(|e| format!("header {key}: {e}"))?;
                    http::HeaderValue::from_str(&expanded)
                        .map_err(|_| format!("invalid value for header '{key}'"))?;
                    headers.push((key.clone(), expanded));
                }
                ServerSpec::Http(HttpSpec {
                    url,
                    headers,
                    request_timeout: timeout,
                })
            }
        };
        Ok(ResolvedServer {
            name: name.to_owned(),
            enabled: server.enabled,
            timeout,
            spec,
        })
    }

    /// Merge servers from a desktop-client style JSON file. A name already
    /// defined in the config is an error rather than a silent override.
    pub fn merge_file(&mut self, path: &Path) -> Result<(), String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read MCP config {}: {error}", path.display()))?;
        let servers = parse_servers_file(&contents)
            .map_err(|error| format!("invalid MCP config {}: {error}", path.display()))?;
        for (name, server) in servers {
            if self.servers.contains_key(&name) {
                return Err(format!(
                    "MCP server '{name}' from {} is already defined in the config file",
                    path.display()
                ));
            }
            self.servers.insert(name, server);
        }
        Ok(())
    }
}

/// One entry of the `mcpServers` file. Unknown keys written by other clients
/// (`autoApprove`, `alwaysAllow`, ...) are ignored.
#[derive(Debug, Deserialize)]
struct FileServer {
    #[serde(default, alias = "type")]
    transport: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default, alias = "serverUrl")]
    url: Option<String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    disabled: Option<bool>,
    #[serde(default, alias = "timeout")]
    timeout_seconds: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ServersFile {
    #[serde(default, rename = "mcpServers", alias = "servers")]
    servers: BTreeMap<String, FileServer>,
}

pub fn parse_servers_file(contents: &str) -> Result<BTreeMap<String, McpServerConfig>, String> {
    let file: ServersFile = serde_json::from_str(contents).map_err(|error| error.to_string())?;
    file.servers
        .into_iter()
        .map(|(name, entry)| {
            let enabled = entry
                .enabled
                .unwrap_or_else(|| !entry.disabled.unwrap_or(false));
            let server = McpServerConfig {
                enabled,
                transport: entry.transport,
                command: entry.command,
                args: entry.args,
                env: entry.env,
                cwd: entry.cwd,
                url: entry.url,
                headers: entry.headers,
                timeout: match entry.timeout_seconds {
                    None => DEFAULT_TIMEOUT,
                    Some(seconds) => Duration::try_from_secs_f64(seconds).map_err(|_| {
                        format!("server '{name}': timeout must be a non-negative number of seconds")
                    })?,
                },
            };
            Ok((name, server))
        })
        .collect()
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env_with(name: &str, value: &str) -> impl Fn(&str) -> Option<String> {
        let (wanted, answer) = (name.to_owned(), value.to_owned());
        move |requested| (requested == wanted).then(|| answer.clone())
    }

    fn stdio_server(command: &str) -> McpServerConfig {
        McpServerConfig {
            command: Some(command.to_owned()),
            ..McpServerConfig::default()
        }
    }

    fn config_with(name: &str, server: McpServerConfig) -> McpConfig {
        McpConfig {
            servers: BTreeMap::from([(name.to_owned(), server)]),
            ..McpConfig::default()
        }
    }

    #[test]
    fn server_defaults_are_a_thirty_second_timeout_and_enabled() {
        let server = McpServerConfig::default();
        assert!(server.enabled);
        assert_eq!(server.timeout, Duration::from_secs(30));
        assert_eq!(
            server.transport_kind().unwrap_err(),
            "needs either command (stdio) or url (http)"
        );
        assert_eq!(McpConfig::default().max_result_bytes, 256 * 1024);
    }

    #[test]
    fn default_config_is_empty() {
        let config = McpConfig::default();
        assert!(config.servers.is_empty());
        assert!(config.problems(&no_env).is_empty());
    }

    #[test]
    fn transport_is_inferred_or_explicit() {
        assert_eq!(
            stdio_server("x").transport_kind().unwrap(),
            TransportKind::Stdio
        );
        let explicit = McpServerConfig {
            transport: Some("streamable-http".to_owned()),
            url: Some("http://x".to_owned()),
            ..McpServerConfig::default()
        };
        assert_eq!(explicit.transport_kind().unwrap(), TransportKind::Http);
        let unknown = McpServerConfig {
            transport: Some("carrier-pigeon".to_owned()),
            ..stdio_server("x")
        };
        assert!(
            unknown
                .transport_kind()
                .unwrap_err()
                .contains("unknown transport")
        );
        let sse = McpServerConfig {
            transport: Some("sse".to_owned()),
            ..stdio_server("x")
        };
        assert!(sse.transport_kind().unwrap_err().contains("legacy"));
        assert!(McpServerConfig::default().transport_kind().is_err());
        let both = McpServerConfig {
            url: Some("http://x".to_owned()),
            ..stdio_server("x")
        };
        assert!(both.transport_kind().unwrap_err().contains("both"));
    }

    #[test]
    fn problems_cover_each_misconfiguration() {
        let cases: Vec<(McpServerConfig, &str)> = vec![
            (McpServerConfig::default(), "needs either command"),
            (stdio_server("   "), "non-empty command"),
            (
                McpServerConfig {
                    transport: Some("http".to_owned()),
                    ..McpServerConfig::default()
                },
                "requires a url",
            ),
            (
                McpServerConfig {
                    url: Some("not a url".to_owned()),
                    ..McpServerConfig::default()
                },
                "invalid url",
            ),
            (
                McpServerConfig {
                    url: Some("ftp://example.com/x".to_owned()),
                    ..McpServerConfig::default()
                },
                "http(s)",
            ),
            (
                McpServerConfig {
                    timeout: Duration::ZERO,
                    ..stdio_server("x")
                },
                "timeout must be longer than zero",
            ),
            (
                McpServerConfig {
                    timeout: Duration::from_secs(4000),
                    ..stdio_server("x")
                },
                "timeout must be longer than zero",
            ),
            (
                McpServerConfig {
                    timeout: Duration::from_secs(3601),
                    ..stdio_server("x")
                },
                "timeout must be longer than zero",
            ),
            (
                McpServerConfig {
                    env: BTreeMap::from([("A".to_owned(), "${NOPE}".to_owned())]),
                    ..stdio_server("x")
                },
                "NOPE is not set",
            ),
            (
                McpServerConfig {
                    url: Some("http://x".to_owned()),
                    headers: BTreeMap::from([("Authorization".to_owned(), "${NOPE}".to_owned())]),
                    ..McpServerConfig::default()
                },
                "NOPE is not set",
            ),
            (
                McpServerConfig {
                    url: Some("http://x".to_owned()),
                    headers: BTreeMap::from([("bad header".to_owned(), "v".to_owned())]),
                    ..McpServerConfig::default()
                },
                "invalid header name",
            ),
        ];
        for (server, expected) in cases {
            let problems = config_with("s", server).problems(&no_env);
            assert_eq!(problems.len(), 1, "{expected}: {problems:?}");
            assert!(problems[0].contains(expected), "{expected}: {problems:?}");
            assert!(problems[0].starts_with("mcp server 's':"));
        }
    }

    #[test]
    fn disabled_servers_skip_environment_resolution() {
        let server = McpServerConfig {
            enabled: false,
            env: BTreeMap::from([("A".to_owned(), "${NOPE}".to_owned())]),
            ..stdio_server("x")
        };
        assert!(config_with("s", server).problems(&no_env).is_empty());
    }

    #[test]
    fn server_names_are_restricted() {
        for bad in ["", "a b", "a__b", "_a", "higgs", "a/b"] {
            let problems = config_with(bad, stdio_server("x")).problems(&no_env);
            assert_eq!(problems.len(), 1, "{bad}");
        }
        assert!(
            config_with("files-2_x", stdio_server("x"))
                .problems(&no_env)
                .is_empty()
        );
    }

    #[test]
    fn zero_result_cap_is_a_problem() {
        let config = McpConfig {
            max_result_bytes: 0,
            ..McpConfig::default()
        };
        assert_eq!(config.problems(&no_env).len(), 1);
    }

    #[test]
    fn resolve_expands_environment_for_stdio_and_headers() {
        let lookup = env_with("TOKEN", "s3cret");
        let mut config = config_with(
            "files",
            McpServerConfig {
                env: BTreeMap::from([("API_KEY".to_owned(), "k-${TOKEN}".to_owned())]),
                cwd: Some("/work".to_owned()),
                args: vec!["--flag".to_owned()],
                ..stdio_server("npx")
            },
        );
        config.servers.insert(
            "remote".to_owned(),
            McpServerConfig {
                url: Some("https://example.com/mcp".to_owned()),
                headers: BTreeMap::from([(
                    "Authorization".to_owned(),
                    "Bearer ${TOKEN}".to_owned(),
                )]),
                timeout: Duration::from_millis(2500),
                ..McpServerConfig::default()
            },
        );
        let resolved = config.resolve(&lookup).unwrap();
        let ServerSpec::Stdio(stdio) = &resolved[0].spec else {
            panic!("expected stdio")
        };
        assert_eq!(resolved[0].name, "files");
        assert_eq!(
            stdio.env,
            vec![("API_KEY".to_owned(), "k-s3cret".to_owned())]
        );
        assert_eq!(stdio.cwd, Some(PathBuf::from("/work")));
        let ServerSpec::Http(http) = &resolved[1].spec else {
            panic!("expected http")
        };
        assert_eq!(http.headers[0].1, "Bearer s3cret");
        assert_eq!(resolved[1].timeout, Duration::from_millis(2500));
    }

    #[test]
    fn variable_expansion_rules() {
        let lookup = env_with("A", "1");
        assert_eq!(
            crate::config::env_ref::expand("x${A}y${A}", &lookup).unwrap(),
            "x1y1"
        );
        assert_eq!(
            crate::config::env_ref::expand("$A $ 5$", &lookup).unwrap(),
            "$A $ 5$"
        );
        assert_eq!(
            crate::config::env_ref::expand("$${A}", &lookup).unwrap(),
            "${A}"
        );
        assert!(
            crate::config::env_ref::expand("${B}", &lookup)
                .unwrap_err()
                .to_string()
                .contains("B is not set")
        );
        assert!(
            crate::config::env_ref::expand("${A", &lookup)
                .unwrap_err()
                .to_string()
                .contains("unterminated")
        );
        assert!(
            crate::config::env_ref::expand("${}", &lookup)
                .unwrap_err()
                .to_string()
                .contains("invalid variable")
        );
        assert!(
            crate::config::env_ref::expand("${a-b}", &lookup)
                .unwrap_err()
                .to_string()
                .contains("invalid variable")
        );
        assert_eq!(crate::config::env_ref::expand("", &lookup).unwrap(), "");
    }

    #[test]
    fn parses_desktop_client_file_format() {
        let servers = parse_servers_file(
            r#"{
              "mcpServers": {
                "filesystem": {
                  "command": "npx",
                  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                  "env": {"A": "b"},
                  "autoApprove": ["read_file"]
                },
                "remote": {"type": "http", "url": "http://localhost:3000/mcp",
                           "headers": {"X-Key": "v"}},
                "off": {"command": "x", "disabled": true},
                "also_off": {"command": "x", "enabled": false},
                "slow": {"command": "x", "timeout": 90}
              }
            }"#,
        )
        .unwrap();
        assert_eq!(servers.len(), 5);
        assert_eq!(servers["filesystem"].args.len(), 3);
        assert_eq!(servers["filesystem"].env["A"], "b");
        assert_eq!(
            servers["remote"].transport_kind().unwrap(),
            TransportKind::Http
        );
        assert!(!servers["off"].enabled);
        assert!(!servers["also_off"].enabled);
        assert_eq!(servers["slow"].timeout, Duration::from_secs(90));
        assert!(servers["filesystem"].enabled);
    }

    #[test]
    fn servers_key_is_accepted_and_garbage_rejected() {
        let servers = parse_servers_file(r#"{"servers": {"a": {"command": "x"}}}"#).unwrap();
        assert!(servers.contains_key("a"));
        assert!(parse_servers_file("not json").is_err());
        assert!(parse_servers_file(r#"{"mcpServers": {"a": "str"}}"#).is_err());
        assert!(parse_servers_file("{}").unwrap().is_empty());
    }

    #[test]
    fn merge_file_adds_servers_and_rejects_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp.json");
        std::fs::write(&path, r#"{"mcpServers": {"extra": {"command": "x"}}}"#).unwrap();
        let mut config = config_with("existing", stdio_server("y"));
        config.merge_file(&path).unwrap();
        assert_eq!(config.servers.len(), 2);

        let error = config.merge_file(&path).unwrap_err();
        assert!(error.contains("already defined"), "{error}");

        let missing = dir.path().join("missing.json");
        assert!(
            config
                .merge_file(&missing)
                .unwrap_err()
                .contains("cannot read")
        );

        std::fs::write(&path, "[").unwrap();
        assert!(
            config
                .merge_file(&path)
                .unwrap_err()
                .contains("invalid MCP config")
        );
    }
}
