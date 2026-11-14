//! `serve` flags, simple mode, path helpers and the MCP and web sections.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::literal_string_with_formatting_args
)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use higgs_models::turboquant::KvCacheMode;

use super::{
    BatchMode, Cli, Commands, Environment, LoadMode, MemoryGuard, ServeArgs, build_simple_config,
    load_config_file, load_config_file_in, validate_profile_name,
};

fn simple(args: &[&str]) -> Result<super::HiggsConfig, String> {
    let mut argv = vec!["serve"];
    argv.extend_from_slice(args);
    build_simple_config(&ServeArgs::try_parse_from(argv).unwrap())
}

fn write_config(contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, contents).unwrap();
    (dir, path)
}

// -- Simple mode ----------------------------------------------------------------------------

#[test]
fn simple_mode_builds_a_model_per_flag() {
    let config = simple(&[
        "--model",
        "org/model-a",
        "--model",
        "org/model-b",
        "--batch",
    ])
    .unwrap();
    assert_eq!(config.models.len(), 2);
    assert!(config.models.iter().all(|m| m.batch == BatchMode::On));
    assert_eq!(config.models[0].path, "org/model-a");
}

#[test]
fn simple_mode_server_flags_override_the_built_in_values() {
    let config = simple(&[
        "--model",
        "some/model",
        "--host",
        "10.1.2.3",
        "--port",
        "9000",
        "--max-tokens",
        "1024",
        "--api-key",
        "sk-test",
        "--rate-limit",
        "60",
        "--timeout",
        "60",
    ])
    .unwrap();
    assert_eq!(config.server.host, "10.1.2.3");
    assert_eq!(config.server.port, 9000);
    assert_eq!(config.server.max_tokens, 1024);
    assert_eq!(config.server.api_key.as_ref().unwrap().expose(), "sk-test");
    assert_eq!(config.server.rate_limit, 60);
    assert_eq!(config.server.request_timeout, Duration::from_secs(60));
}

#[test]
fn simple_mode_turboquant_flags() {
    let config = simple(&[
        "--model",
        "some/model",
        "--kv-cache",
        "turboquant",
        "--kv-bits",
        "4",
        "--kv-seed",
        "99",
        "--kv-no-norm-correction",
    ])
    .unwrap();
    let kv = config.models[0].kv_cache;
    assert_eq!(kv.mode, KvCacheMode::Turboquant);
    assert_eq!(kv.bits, 4);
    assert_eq!(kv.seed, 99);
    assert!(!kv.norm_correction);
}

#[test]
fn simple_mode_rejects_missing_empty_and_duplicate_models() {
    assert!(simple(&[]).unwrap_err().contains("--model"));
    assert!(simple(&["--model", "  "]).is_err());
    assert!(simple(&["--model", "org/m", "--model", "org/m"]).is_err());
}

#[test]
fn turboquant_with_auto_batch_is_allowed() {
    let config = simple(&["--model", "org/model", "--kv-cache", "turboquant"]).unwrap();
    assert_eq!(config.models[0].batch, BatchMode::Auto);
}

#[test]
fn lifecycle_flags_set_the_defaults_in_simple_mode() {
    let config = simple(&[
        "--model",
        "some/model",
        "--load",
        "on-demand",
        "--idle-ttl",
        "90",
        "--memory-guard",
        "safe",
        "--max-loaded-models",
        "3",
    ])
    .unwrap();
    assert_eq!(config.models[0].load, LoadMode::OnDemand);
    assert_eq!(config.models[0].idle_ttl, Some(Duration::from_secs(90)));
    assert_eq!(config.memory.guard, MemoryGuard::Safe);
    assert_eq!(config.memory.max_loaded_models, Some(3));
}

#[test]
fn simple_mode_reads_the_documented_server_variables() {
    let args = ServeArgs::try_parse_from(["serve", "--model", "org/m", "--port", "1234"]).unwrap();
    let environment = Environment::from_pairs([
        ("HIGGS_HOST", "0.0.0.0"),
        ("HIGGS_PORT", "5555"),
        ("HIGGS_API_KEY", "sk-env"),
        ("HIGGS_TIMEOUT", "12.5"),
        ("HIGGS_CORS_ORIGINS", "[\"*\"]"),
        ("HIGGS_MAX_BODY_SIZE", "1MB"),
    ]);
    let config = super::load::load_simple(&args, &environment)
        .unwrap()
        .config;
    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 1234, "the flag beats the variable");
    assert_eq!(config.server.api_key.as_ref().unwrap().expose(), "sk-env");
    assert_eq!(config.server.request_timeout, Duration::from_millis(12_500));
    assert_eq!(
        config.server.cors_origins.as_deref(),
        Some(&["*".to_owned()][..])
    );
    assert_eq!(config.server.max_body_size.bytes(), 1 << 20);

    let bad = Environment::from_pairs([("HIGGS_PORT", "not-a-port")]);
    assert!(
        super::load::load_simple(&args, &bad)
            .unwrap_err()
            .to_string()
            .contains("HIGGS_PORT")
    );
}

#[test]
fn invalid_timeouts_are_rejected() {
    assert!(simple(&["--model", "org/m", "--timeout", "0"]).is_err());
    assert!(simple(&["--model", "org/m", "--timeout=-5"]).is_err());
}

// -- Flags with a config file ---------------------------------------------------------------

#[test]
fn flags_overlay_a_config_file() {
    let (_dir, path) = write_config(
        "[server]\nhost = \"127.0.0.1\"\nport = 3100\n[[models]]\npath = \"some/model\"\n",
    );
    let args = ServeArgs::try_parse_from(["serve", "--port", "9000"]).unwrap();
    let config = load_config_file(&path, Some(&args)).unwrap();
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 9000);
}

#[test]
fn model_flags_append_to_the_files_models_and_keep_their_batch_modes() {
    let (_dir, path) = write_config(
        "[[models]]\npath = \"org/forced\"\nbatch = true\n\n[[models]]\npath = \"org/off\"\nbatch = false\n\n[[models]]\npath = \"org/auto\"\nbatch = \"auto\"\n",
    );
    let args = ServeArgs::try_parse_from(["serve", "--model", "org/extra"]).unwrap();
    let config = load_config_file(&path, Some(&args)).unwrap();
    let modes: Vec<BatchMode> = config.models.iter().map(|m| m.batch).collect();
    assert_eq!(
        modes,
        [
            BatchMode::On,
            BatchMode::Off,
            BatchMode::Auto,
            BatchMode::Auto
        ]
    );
}

#[test]
fn defaults_apply_to_models_added_with_flags() {
    let (_dir, path) =
        write_config("[defaults]\nthinking_budget = 77\n[[models]]\npath = \"org/a\"\n");
    let args =
        ServeArgs::try_parse_from(["serve", "--model", "org/extra", "--kv-bits", "5"]).unwrap();
    let config = load_config_file(&path, Some(&args)).unwrap();
    assert_eq!(config.models[1].thinking_budget, 77);
    assert_eq!(config.models[1].kv_cache.bits, 5);
    assert_eq!(config.models[0].kv_cache.bits, 3);
}

// -- Flag parsing -----------------------------------------------------------------------------

#[test]
fn lifecycle_flags_parse_in_kebab_case() {
    let cli = Cli::try_parse_from([
        "higgs",
        "serve",
        "--model",
        "m/m",
        "--load",
        "on-demand",
        "--idle-ttl",
        "30",
        "--memory-guard",
        "aggressive",
        "--max-loaded-models",
        "2",
    ])
    .unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve");
    };
    assert_eq!(args.load, Some(LoadMode::OnDemand));
    assert_eq!(args.idle_ttl, Some(30));
    assert_eq!(args.memory_guard, Some(MemoryGuard::Aggressive));
    assert_eq!(args.max_loaded_models, Some(2));
}

#[test]
fn serve_flags_pick_the_batch_mode() {
    let mode = |flags: &[&str]| {
        let mut argv = vec!["--model", "org/model"];
        argv.extend_from_slice(flags);
        simple(&argv).unwrap().models[0].batch
    };
    assert_eq!(mode(&[]), BatchMode::Auto);
    assert_eq!(mode(&["--batch"]), BatchMode::On);
    assert_eq!(mode(&["--no-batch"]), BatchMode::Off);
}

#[test]
fn serve_rejects_batch_with_no_batch() {
    assert!(ServeArgs::try_parse_from(["serve", "--batch", "--no-batch"]).is_err());
}

#[test]
fn config_subcommands_parse() {
    for (argv, expect_target, expect_write) in [
        (vec!["higgs", "config", "check"], None, false),
        (vec!["higgs", "config", "migrate"], None, false),
        (vec!["higgs", "config", "migrate", "--write"], None, true),
        (vec!["higgs", "config", "explain"], None, false),
        (
            vec!["higgs", "config", "explain", "qwen"],
            Some("qwen"),
            false,
        ),
    ] {
        let cli =
            Cli::try_parse_from(argv.clone()).unwrap_or_else(|error| panic!("{argv:?}: {error}"));
        let Commands::Config { action } = cli.command else {
            panic!("expected config");
        };
        match action {
            super::ConfigAction::Explain { target, .. } => {
                assert_eq!(target.as_deref(), expect_target);
            }
            super::ConfigAction::Migrate { write } => assert_eq!(write, expect_write),
            super::ConfigAction::Check => {}
            super::ConfigAction::Get { .. }
            | super::ConfigAction::Set { .. }
            | super::ConfigAction::Path => panic!("unexpected action for {argv:?}"),
        }
    }
    let cli = Cli::try_parse_from(["higgs", "config", "explain", "--port", "9", "qwen"]).unwrap();
    let Commands::Config {
        action: super::ConfigAction::Explain { target, flags },
    } = cli.command
    else {
        panic!("expected explain");
    };
    assert_eq!(target.as_deref(), Some("qwen"));
    assert_eq!(flags.port, Some(9));
}

#[test]
fn get_set_and_path_subcommands_still_parse() {
    assert!(Cli::try_parse_from(["higgs", "config", "get", "server.port"]).is_ok());
    assert!(Cli::try_parse_from(["higgs", "config", "set", "server.port", "1"]).is_ok());
    assert!(Cli::try_parse_from(["higgs", "config", "path"]).is_ok());
}

#[test]
fn mcp_config_flag_parses_on_serve() {
    let cli = Cli::try_parse_from(["higgs", "serve", "--mcp-config", "/tmp/mcp.json"]).unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve");
    };
    assert_eq!(args.mcp_config, Some(PathBuf::from("/tmp/mcp.json")));
}

// -- Profile names and paths ------------------------------------------------------------------

#[test]
fn profile_names_are_validated() {
    for good in ["dev", "my-profile", "a_b", "x1"] {
        assert!(validate_profile_name(good).is_ok(), "{good}");
    }
    for bad in [
        "",
        "..",
        "a/b",
        "a\\b",
        "a b",
        "a;b",
        "caf\u{e9}",
        "a\nb",
        "$(x)",
    ] {
        assert!(validate_profile_name(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn profile_paths_share_the_config_directory() {
    assert_eq!(
        super::profile_config_path("dev").file_name().unwrap(),
        "config.dev.toml"
    );
    assert_eq!(
        super::profile_config_path("dev").parent(),
        super::default_config_path().parent()
    );
    assert_ne!(super::pid_path(None), super::pid_path(Some("dev")));
    assert_ne!(super::log_path(None), super::log_path(Some("dev")));
    assert!(super::default_metrics_log_path_for_profile("dev").contains("metrics.dev.jsonl"));
    assert_ne!(
        super::default_metrics_log_path_for_profile("a"),
        super::default_metrics_log_path_for_profile("b")
    );
}

#[test]
#[cfg(unix)]
fn private_files_are_owner_only() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    super::write_private_file(&path, "[server]\n").unwrap();
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

// -- MCP and web -------------------------------------------------------------------------------

fn load_text(toml: &str) -> super::HiggsConfig {
    let (_dir, path) = write_config(toml);
    load_config_file_in(&path, None, &Environment::default()).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn mcp_defaults_to_no_servers() {
    let config = load_text("[[models]]\npath = \"org/model\"\n");
    assert!(config.mcp.servers.is_empty());
    assert_eq!(config.mcp.max_result_bytes, 262_144);
}

#[test]
fn mcp_servers_parse_with_durations_and_sizes() {
    let config = load_text(
        r#"
        [[models]]
        path = "org/model"

        [mcp]
        max_result_size = "4KB"

        [mcp.servers.files]
        command = "npx"
        args = ["-y", "server"]
        env = { TOKEN = "${TOKEN}" }
        cwd = "/tmp"
        timeout = "12500ms"
        enabled = false

        [mcp.servers.remote]
        type = "http"
        url = "https://example.com/mcp"
        headers = { Authorization = "Bearer ${TOKEN}" }
        "#,
    );
    assert_eq!(config.mcp.max_result_bytes, 4096);
    let files = &config.mcp.servers["files"];
    assert_eq!(files.command.as_deref(), Some("npx"));
    assert_eq!(files.args, ["-y", "server"]);
    assert_eq!(files.env["TOKEN"], "${TOKEN}");
    assert!(!files.enabled);
    assert_eq!(files.timeout, Duration::from_millis(12_500));
    let remote = &config.mcp.servers["remote"];
    assert_eq!(remote.transport.as_deref(), Some("http"));
    assert_eq!(remote.timeout, Duration::from_secs(30));
}

fn write_mcp_json(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("mcp.json");
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn mcp_config_file_is_merged_in_config_mode_and_collisions_are_errors() {
    let (dir, config_path) =
        write_config("[[models]]\npath = \"org/model\"\n[mcp.servers.local]\ncommand = \"a\"\n");
    let json = write_mcp_json(
        dir.path(),
        r#"{"mcpServers": {"fs": {"command": "npx", "args": ["x"], "env": {"A": "b"}, "timeout": 5}}}"#,
    );
    let args = ServeArgs {
        mcp_config: Some(json),
        ..ServeArgs::default()
    };
    let config = load_config_file(&config_path, Some(&args)).unwrap();
    let names: Vec<&str> = config.mcp.servers.keys().map(String::as_str).collect();
    assert_eq!(names, ["fs", "local"]);
    assert_eq!(config.mcp.servers["fs"].env["A"], "b");
    assert_eq!(config.mcp.servers["fs"].timeout, Duration::from_secs(5));

    std::fs::write(
        &config_path,
        "[[models]]\npath = \"org/model\"\n[mcp.servers.fs]\ncommand = \"a\"\n",
    )
    .unwrap();
    let error = load_config_file(&config_path, Some(&args)).unwrap_err();
    assert!(error.contains("already defined"), "{error}");
}

#[test]
fn mcp_config_file_is_merged_in_simple_mode() {
    let dir = tempfile::tempdir().unwrap();
    let json = write_mcp_json(
        dir.path(),
        r#"{"mcpServers": {"remote": {"type": "http", "url": "http://localhost:9/mcp"}}}"#,
    );
    let args = ServeArgs {
        models: vec!["org/model".to_owned()],
        mcp_config: Some(json),
        ..ServeArgs::default()
    };
    let config = build_simple_config(&args).unwrap();
    assert_eq!(config.mcp.servers.len(), 1);
    assert_eq!(
        config.mcp.servers["remote"].transport_kind().unwrap(),
        crate::mcp::config::TransportKind::Http
    );
}

#[test]
fn unreadable_or_invalid_mcp_config_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let missing = ServeArgs {
        models: vec!["org/model".to_owned()],
        mcp_config: Some(dir.path().join("missing.json")),
        ..ServeArgs::default()
    };
    assert!(
        build_simple_config(&missing)
            .unwrap_err()
            .contains("cannot read MCP config")
    );
    let bad = ServeArgs {
        models: vec!["org/model".to_owned()],
        mcp_config: Some(write_mcp_json(dir.path(), "{not json")),
        ..ServeArgs::default()
    };
    assert!(
        build_simple_config(&bad)
            .unwrap_err()
            .contains("invalid MCP config")
    );
}

#[test]
fn web_defaults_off_and_every_field_parses() {
    let defaults = load_text("[[models]]\npath = \"org/model\"\n");
    assert!(!defaults.web.enabled);
    assert_eq!(defaults.web.timeout, Duration::from_secs(15));

    let config = load_text(
        r#"
        [[models]]
        path = "org/model"
        [web]
        enabled = true
        search_provider = "searxng"
        searxng_url = "http://localhost:8888"
        allow_private_addresses = true
        expose_as_tools = true
        timeout = "7s"
        "#,
    );
    assert!(config.web.enabled && config.web.allow_private_addresses && config.web.expose_as_tools);
    assert_eq!(config.web.search_provider, "searxng");
    assert_eq!(
        config.web.searxng_url.as_deref(),
        Some("http://localhost:8888")
    );
    assert_eq!(config.web.timeout, Duration::from_secs(7));
    assert!(config.web.problems().is_empty());
}

// -- Other defaults -----------------------------------------------------------------------------

#[test]
fn the_default_configuration_matches_the_documented_defaults() {
    let config = super::HiggsConfig::default();
    assert!(config.models.is_empty() && config.providers.is_empty() && config.routes.is_empty());
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 8000);
    assert_eq!(config.server.max_tokens, 32768);
    assert_eq!(config.server.request_timeout, Duration::from_secs(300));
    assert!(config.server.api_key.is_none());
    assert_eq!(config.server.rate_limit, 0);
    assert_eq!(config.routing.default_provider, "higgs");
    assert_eq!(
        config.observability.retention,
        Some(Duration::from_secs(3600))
    );
    assert!(config.observability.metrics_log.enabled);
    assert_eq!(config.observability.metrics_log.max_files, 5);
    assert!(!config.routing.auto.enabled);
    assert_eq!(config.routing.auto.timeout, Duration::from_secs(2));
}

#[test]
fn the_memory_ceiling_comes_from_the_limit_or_the_guard() {
    let physical = Some(100 * 1024 * 1024 * 1024_u64);
    let guard = |guard| super::MemoryConfig {
        guard,
        ..super::MemoryConfig::default()
    };
    assert_eq!(guard(MemoryGuard::Off).ceiling_bytes(physical), None);
    assert_eq!(
        guard(MemoryGuard::Safe).ceiling_bytes(physical),
        Some(80 * 1024 * 1024 * 1024)
    );
    assert_eq!(
        guard(MemoryGuard::Balanced).ceiling_bytes(physical),
        Some(92 * 1024 * 1024 * 1024)
    );
    assert_eq!(
        guard(MemoryGuard::Aggressive).ceiling_bytes(physical),
        Some(98 * 1024 * 1024 * 1024)
    );
    assert_eq!(guard(MemoryGuard::Safe).ceiling_bytes(None), None);
    let explicit = super::MemoryConfig {
        guard: MemoryGuard::Safe,
        limit: Some(super::units::ByteSize::from_bytes(5 << 30)),
        ..super::MemoryConfig::default()
    };
    assert_eq!(explicit.ceiling_bytes(physical), Some(5 << 30));
}
