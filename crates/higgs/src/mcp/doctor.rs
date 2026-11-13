//! `higgs doctor` checks for the `[mcp]` and `[web]` sections.
//!
//! Nothing here starts a server or makes a request; it validates what the
//! configuration says and what is visible on this machine.

use std::path::Path;

use crate::config::HiggsConfig;

use super::config::{EnvLookup, McpConfig, McpServerConfig, ServerSpec, process_env};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub level: Level,
    pub message: String,
}

impl Finding {
    const fn new(level: Level, message: String) -> Self {
        Self { level, message }
    }
}

/// Findings for MCP servers and web settings against the real environment.
pub fn tool_integration_findings(config: &HiggsConfig) -> Vec<Finding> {
    let mut findings = mcp_findings(&config.mcp, &process_env, &command_available);
    findings.extend(web_findings(config));
    findings
}

pub fn mcp_findings(
    config: &McpConfig,
    lookup: EnvLookup<'_>,
    command_exists: &dyn Fn(&str) -> bool,
) -> Vec<Finding> {
    let mut findings = Vec::new();
    if config.max_result_bytes == 0 {
        findings.push(Finding::new(
            Level::Fail,
            "mcp.max_result_size must be greater than zero".to_owned(),
        ));
    }
    for (name, server) in &config.servers {
        match McpConfig::resolve_server(name, server, lookup) {
            Err(message) => findings.push(Finding::new(
                Level::Fail,
                format!("mcp server '{name}': {message}"),
            )),
            Ok(resolved) => {
                let state = if resolved.enabled {
                    "configured"
                } else {
                    "disabled"
                };
                findings.push(Finding::new(
                    Level::Pass,
                    format!(
                        "mcp server '{name}': {} transport, {state} ({}; not started)",
                        resolved.spec.kind().label(),
                        describe(&resolved.spec),
                    ),
                ));
                if resolved.enabled {
                    findings.extend(stdio_warnings(name, server, &resolved.spec, command_exists));
                }
            }
        }
    }
    findings
}

fn describe(spec: &ServerSpec) -> String {
    match spec {
        ServerSpec::Stdio(stdio) => format!("command {}", stdio.command),
        ServerSpec::Http(http) => {
            // Query strings and credentials are left out: they may hold secrets.
            let url = &http.url;
            format!(
                "url {}://{}{}{}",
                url.scheme(),
                url.host_str().unwrap_or_default(),
                url.port()
                    .map_or_else(String::new, |port| format!(":{port}")),
                url.path()
            )
        }
    }
}

fn stdio_warnings(
    name: &str,
    server: &McpServerConfig,
    spec: &ServerSpec,
    command_exists: &dyn Fn(&str) -> bool,
) -> Vec<Finding> {
    let ServerSpec::Stdio(stdio) = spec else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    if !command_exists(&stdio.command) {
        findings.push(Finding::new(
            Level::Warn,
            format!(
                "mcp server '{name}': command '{}' was not found on PATH or does not exist",
                stdio.command
            ),
        ));
    }
    if let Some(directory) = &server.cwd
        && !Path::new(directory).is_dir()
    {
        findings.push(Finding::new(
            Level::Warn,
            format!("mcp server '{name}': cwd '{directory}' is not a directory"),
        ));
    }
    findings
}

pub fn web_findings(config: &HiggsConfig) -> Vec<Finding> {
    let web = &config.web;
    let problems = web.problems();
    let mut findings: Vec<Finding> = problems
        .iter()
        .map(|problem| Finding::new(Level::Fail, problem.clone()))
        .collect();
    findings.extend(
        web.warnings()
            .into_iter()
            .map(|warning| Finding::new(Level::Warn, warning)),
    );
    if web.enabled && problems.is_empty() {
        findings.push(Finding::new(
            Level::Pass,
            format!(
                "web: search ({}) and fetch enabled; private addresses {}",
                web.search_provider,
                if web.allow_private_addresses {
                    "allowed"
                } else {
                    "blocked"
                }
            ),
        ));
    }
    findings
}

/// Whether `command` names an existing executable, either as a path or
/// somewhere on `PATH`.
pub fn command_available(command: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;

    let is_executable = |candidate: &Path| {
        candidate
            .metadata()
            .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
    };
    if command.contains('/') {
        return is_executable(Path::new(command));
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|directory| is_executable(&directory.join(command)))
    })
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::web::config::WebConfig;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn always(_: &str) -> bool {
        true
    }

    fn never(_: &str) -> bool {
        false
    }

    fn stdio(command: &str) -> McpServerConfig {
        McpServerConfig {
            command: Some(command.to_owned()),
            ..McpServerConfig::default()
        }
    }

    fn config_with(servers: Vec<(&str, McpServerConfig)>) -> McpConfig {
        McpConfig {
            servers: servers
                .into_iter()
                .map(|(name, server)| (name.to_owned(), server))
                .collect::<BTreeMap<_, _>>(),
            ..McpConfig::default()
        }
    }

    fn levels(findings: &[Finding]) -> Vec<Level> {
        findings.iter().map(|finding| finding.level).collect()
    }

    #[test]
    fn no_servers_means_no_findings() {
        assert!(mcp_findings(&McpConfig::default(), &no_env, &always).is_empty());
        assert!(web_findings(&HiggsConfig::default()).is_empty());
    }

    #[test]
    fn valid_servers_report_their_transport_without_starting() {
        let config = config_with(vec![
            ("files", stdio("npx")),
            (
                "remote",
                McpServerConfig {
                    url: Some("https://user:pw@example.com:8443/mcp?token=secret".to_owned()),
                    ..McpServerConfig::default()
                },
            ),
        ]);
        let findings = mcp_findings(&config, &no_env, &always);
        assert_eq!(levels(&findings), vec![Level::Pass, Level::Pass]);
        assert_eq!(
            findings[0].message,
            "mcp server 'files': stdio transport, configured (command npx; not started)"
        );
        assert_eq!(
            findings[1].message,
            "mcp server 'remote': http transport, configured (url https://example.com:8443/mcp; not started)"
        );
        assert!(!findings[1].message.contains("secret"));
        assert!(!findings[1].message.contains("pw"));
    }

    #[test]
    fn disabled_servers_are_reported_as_such() {
        let config = config_with(vec![(
            "off",
            McpServerConfig {
                enabled: false,
                ..stdio("npx")
            },
        )]);
        let findings = mcp_findings(&config, &no_env, &never);
        assert_eq!(levels(&findings), vec![Level::Pass]);
        assert!(findings[0].message.contains("disabled"));
    }

    #[test]
    fn each_misconfiguration_is_a_failure() {
        let config = config_with(vec![
            ("no_transport", McpServerConfig::default()),
            (
                "bad_transport",
                McpServerConfig {
                    transport: Some("pigeon".to_owned()),
                    ..stdio("x")
                },
            ),
            ("empty_command", stdio("  ")),
            (
                "bad_url",
                McpServerConfig {
                    url: Some("::nope::".to_owned()),
                    ..McpServerConfig::default()
                },
            ),
            (
                "bad_timeout",
                McpServerConfig {
                    timeout: std::time::Duration::ZERO,
                    ..stdio("x")
                },
            ),
            (
                "missing_env",
                McpServerConfig {
                    env: BTreeMap::from([("K".to_owned(), "${UNSET_VARIABLE}".to_owned())]),
                    ..stdio("x")
                },
            ),
        ]);
        let findings = mcp_findings(&config, &no_env, &always);
        assert_eq!(findings.len(), 6);
        assert!(findings.iter().all(|finding| finding.level == Level::Fail));
        let joined = findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for expected in [
            "needs either command",
            "unknown transport 'pigeon'",
            "non-empty command",
            "invalid url",
            "timeout",
            "UNSET_VARIABLE is not set",
        ] {
            assert!(
                joined.contains(expected),
                "{expected} missing from {joined}"
            );
        }
    }

    #[test]
    fn missing_command_and_cwd_are_warnings() {
        let config = config_with(vec![(
            "files",
            McpServerConfig {
                cwd: Some("/definitely/not/a/directory".to_owned()),
                ..stdio("npx")
            },
        )]);
        let findings = mcp_findings(&config, &no_env, &never);
        assert_eq!(
            levels(&findings),
            vec![Level::Pass, Level::Warn, Level::Warn]
        );
        assert!(findings[1].message.contains("not found on PATH"));
        assert!(findings[2].message.contains("not a directory"));
    }

    #[test]
    fn zero_result_cap_fails() {
        let config = McpConfig {
            max_result_bytes: 0,
            ..McpConfig::default()
        };
        assert_eq!(
            levels(&mcp_findings(&config, &no_env, &always)),
            vec![Level::Fail]
        );
    }

    #[test]
    fn web_findings_cover_provider_url_timeout_and_warnings() {
        let invalid = HiggsConfig {
            web: WebConfig {
                enabled: true,
                search_provider: "searxng".to_owned(),
                timeout: std::time::Duration::ZERO,
                ..WebConfig::default()
            },
            ..HiggsConfig::default()
        };
        let findings = web_findings(&invalid);
        assert_eq!(levels(&findings), vec![Level::Fail, Level::Fail]);

        let valid = HiggsConfig {
            web: WebConfig {
                enabled: true,
                search_provider: "searxng".to_owned(),
                searxng_url: Some("http://localhost:8888".to_owned()),
                allow_private_addresses: true,
                ..WebConfig::default()
            },
            ..HiggsConfig::default()
        };
        let valid_findings = web_findings(&valid);
        assert_eq!(levels(&valid_findings), vec![Level::Warn, Level::Pass]);
        assert!(
            valid_findings[1]
                .message
                .contains("private addresses allowed")
        );

        let inert = HiggsConfig {
            web: WebConfig {
                expose_as_tools: true,
                ..WebConfig::default()
            },
            ..HiggsConfig::default()
        };
        assert_eq!(levels(&web_findings(&inert)), vec![Level::Warn]);
    }

    #[test]
    fn command_lookup_finds_executables_by_path_and_by_name() {
        assert!(command_available("/bin/sh"));
        assert!(command_available("sh"));
        assert!(!command_available("definitely-not-a-real-binary-xyz"));
        assert!(!command_available("/definitely/not/here"));
        let directory = tempfile::tempdir().unwrap();
        let plain_file = directory.path().join("plain");
        std::fs::write(&plain_file, "data").unwrap();
        assert!(!command_available(plain_file.to_str().unwrap()));
        assert!(!command_available(directory.path().to_str().unwrap()));
    }
}
