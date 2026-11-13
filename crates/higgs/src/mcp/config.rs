//! `[mcp]` configuration: server definitions, the desktop-client JSON file
//! format, `${VAR}` expansion and validation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Namespace reserved for tools Higgs provides itself (see `[web]`).
pub const BUILTIN_SERVER_NAME: &str = "higgs";
/// Separator between server and tool name in namespaced tool names.
pub const NAMESPACE_SEPARATOR: &str = "__";

const DEFAULT_TIMEOUT_SECONDS: f64 = 30.0;
const DEFAULT_MAX_RESULT_BYTES: usize = 256 * 1024;
const MAX_TIMEOUT_SECONDS: f64 = 3600.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// `[mcp.servers.<name>]` tables.
    #[serde(default)]
    pub servers: BTreeMap<String, McpServerConfig>,
    /// Upper bound on the size of one tool result; larger results are
    /// truncated and flagged.
    #[serde(default = "default_max_result_bytes")]
    pub max_result_bytes: usize,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            servers: BTreeMap::new(),
            max_result_bytes: default_max_result_bytes(),
        }
    }
}

const fn default_max_result_bytes() -> usize {
    DEFAULT_MAX_RESULT_BYTES
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// `"stdio"` or `"http"` (Streamable HTTP). Inferred from `command` or
    /// `url` when omitted.
    #[serde(default, alias = "type")]
    pub transport: Option<String>,
    /// Executable for stdio servers. Run directly, never through a shell.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment for the child process; values may reference the
    /// Higgs process environment as `${VAR}`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Endpoint for Streamable HTTP servers.
    #[serde(default)]
    pub url: Option<String>,
    /// Extra request headers; values may reference `${VAR}`.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Bound on connecting, listing and each tool call.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
}

const fn default_enabled() -> bool {
    true
}

const fn default_timeout_seconds() -> f64 {
    DEFAULT_TIMEOUT_SECONDS
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
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
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

    pub fn timeout(&self) -> Option<Duration> {
        (self.timeout_seconds.is_finite()
            && self.timeout_seconds > 0.0
            && self.timeout_seconds <= MAX_TIMEOUT_SECONDS)
            .then(|| Duration::from_secs_f64(self.timeout_seconds))
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

/// Look up an environment variable; injectable so tests do not touch the
/// process environment.
pub type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<String>;

pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Expand `${VAR}` references. A reference to an unset variable is an error
/// rather than an empty string, so a typo cannot silently drop a credential.
/// `$${` produces a literal `${`.
pub fn expand_variables(value: &str, lookup: EnvLookup<'_>) -> Result<String, String> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(position) = rest.find('$') {
        output.push_str(rest.get(..position).unwrap_or_default());
        let after = rest.get(position + 1..).unwrap_or_default();
        if let Some(escaped) = after.strip_prefix("${") {
            output.push_str("${");
            rest = escaped;
        } else if let Some(reference) = after.strip_prefix('{') {
            let end = reference
                .find('}')
                .ok_or_else(|| format!("unterminated variable reference in '{value}'"))?;
            let name = reference.get(..end).unwrap_or_default();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
            {
                return Err(format!("invalid variable name '{name}' in '{value}'"));
            }
            let resolved =
                lookup(name).ok_or_else(|| format!("environment variable {name} is not set"))?;
            output.push_str(&resolved);
            rest = reference.get(end + 1..).unwrap_or_default();
        } else {
            output.push('$');
            rest = after;
        }
    }
    output.push_str(rest);
    Ok(output)
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
            problems.push("mcp.max_result_bytes must be greater than zero".to_owned());
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
        let timeout = server.timeout().ok_or_else(|| {
            format!(
                "timeout_seconds must be a finite number in (0, {MAX_TIMEOUT_SECONDS}], got {}",
                server.timeout_seconds
            )
        })?;
        // A disabled server is never started, so its environment is not
        // required to resolve.
        let expand = |value: &str| -> Result<String, String> {
            if server.enabled {
                expand_variables(value, lookup)
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
    Ok(file
        .servers
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
                timeout_seconds: entry.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS),
            };
            (name, server)
        })
        .collect())
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use figment::providers::Format;

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
    fn toml_tables_parse_with_defaults() {
        let config: McpConfig = figment::Figment::new()
            .merge(figment::providers::Toml::string(
                r#"
            max_result_bytes = 1000
            [servers.files]
            command = "npx"
            args = ["-y", "server"]
            env = { TOKEN = "${TOKEN}" }
            cwd = "/tmp"
            timeout_seconds = 5
            [servers.remote]
            url = "https://example.com/mcp"
            headers = { Authorization = "Bearer ${TOKEN}" }
            "#,
            ))
            .extract()
            .unwrap();
        assert_eq!(config.max_result_bytes, 1000);
        let files = &config.servers["files"];
        assert!(files.enabled);
        assert_eq!(files.transport_kind().unwrap(), TransportKind::Stdio);
        assert!((files.timeout_seconds - 5.0).abs() < f64::EPSILON);
        let remote = &config.servers["remote"];
        assert_eq!(remote.transport_kind().unwrap(), TransportKind::Http);
        assert!((remote.timeout_seconds - 30.0).abs() < f64::EPSILON);
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
                    timeout_seconds: 0.0,
                    ..stdio_server("x")
                },
                "timeout_seconds",
            ),
            (
                McpServerConfig {
                    timeout_seconds: -1.0,
                    ..stdio_server("x")
                },
                "timeout_seconds",
            ),
            (
                McpServerConfig {
                    timeout_seconds: f64::NAN,
                    ..stdio_server("x")
                },
                "timeout_seconds",
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
                timeout_seconds: 2.5,
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
        assert_eq!(expand_variables("x${A}y${A}", &lookup).unwrap(), "x1y1");
        assert_eq!(expand_variables("$A $ 5$", &lookup).unwrap(), "$A $ 5$");
        assert_eq!(expand_variables("$${A}", &lookup).unwrap(), "${A}");
        assert!(
            expand_variables("${B}", &lookup)
                .unwrap_err()
                .contains("B is not set")
        );
        assert!(
            expand_variables("${A", &lookup)
                .unwrap_err()
                .contains("unterminated")
        );
        assert!(
            expand_variables("${}", &lookup)
                .unwrap_err()
                .contains("invalid variable")
        );
        assert!(
            expand_variables("${a-b}", &lookup)
                .unwrap_err()
                .contains("invalid variable")
        );
        assert_eq!(expand_variables("", &lookup).unwrap(), "");
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
        assert!((servers["slow"].timeout_seconds - 90.0).abs() < f64::EPSILON);
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
