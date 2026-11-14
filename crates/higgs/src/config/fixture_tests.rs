//! Whole-file behavior against fixtures: the pre-redesign files must resolve exactly as they did,
//! `config migrate` must produce the committed golden output, and `config explain` must name
//! the source of each value.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::literal_string_with_formatting_args
)]

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::explain::{self, Entry, Source};
use super::load::{self, LoadOptions, Loaded};
use super::migrate::migrate;
use super::{Environment, HiggsConfig};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/config")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(fixtures().join(name)).unwrap()
}

const TEMPLATE_MODEL: &str = "\n[[models]]\npath = \"mlx-community/Llama-3.2-1B-Instruct-4bit\"\n";

/// The text of a fixture as it is loaded: the init template has no model, so it gets one.
fn loadable(name: &str) -> String {
    let text = read(&format!("{name}.toml"));
    if name == "init_template_legacy" {
        format!("{text}{TEMPLATE_MODEL}")
    } else {
        text
    }
}

fn load_text(text: &str, environment: &Environment) -> Loaded {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, text).unwrap();
    let options = LoadOptions {
        flags: None,
        environment: environment.clone(),
    };
    load::load(&path, &options).unwrap_or_else(|error| panic!("{error}"))
}

fn home() -> String {
    directories::BaseDirs::new()
        .unwrap()
        .home_dir()
        .to_string_lossy()
        .into_owned()
}

/// The resolved configuration in the shape the pre-redesign code dumped it, for comparing a
/// file's meaning before and after the redesign.
fn snapshot(config: &HiggsConfig) -> Value {
    let models: Vec<Value> = config
        .models
        .iter()
        .map(|m| {
            let kv = m.kv_cache;
            json!({
                "path": m.path, "name": m.name,
                "profile": format!("{:?}", m.profile),
                "batch": m.batch.to_string(),
                "prefill_yield_tokens": m.prefill_yield_tokens,
                "thinking_budget": m.thinking_budget,
                "kv_mode": format!("{:?}", kv.mode), "kv_bits": kv.bits,
                "kv_key_bits": kv.key_bits_override, "kv_value_bits": kv.value_bits_override,
                "kv_norm_correction": kv.norm_correction, "kv_adaptive_dense_layers": kv.adaptive_dense_layers,
                "kv_seed": kv.seed, "mla_latent": kv.mla_latent,
                "kv_disk_dir": m.prefix_cache.disk_dir, "kv_disk_bytes": m.prefix_cache.disk_budget.bytes(),
                "load": m.load.as_str(),
                "idle_ttl_secs": m.idle_ttl.map(|d| d.as_secs_f64()),
                "pinned": m.pinned,
            })
        })
        .collect();
    let providers: Vec<Value> = config
        .providers
        .iter()
        .map(|(name, p)| {
            json!({
                "name": name, "url": p.url, "format": format!("{:?}", p.format),
                "api_key": p.api_key.as_ref().map(|key| key.expose().to_owned()),
                "strip_auth": p.strip_auth, "stub_count_tokens": p.stub_count_tokens})
        })
        .collect();
    let routes: Vec<Value> = config
        .routes
        .iter()
        .map(|r| {
            json!({
                "name": r.name, "description": r.description, "pattern": r.pattern,
                "provider": r.provider, "model": r.model})
        })
        .collect();
    let mcp: Vec<Value> = config
        .mcp
        .servers
        .iter()
        .map(|(name, s)| {
            json!({
                "name": name, "enabled": s.enabled, "transport": s.transport, "command": s.command,
                "args": s.args, "env": s.env, "cwd": s.cwd, "url": s.url, "headers": s.headers,
                "timeout_secs": s.timeout.as_secs_f64()})
        })
        .collect();
    json!({
        "server": {
            "host": config.server.host, "port": config.server.port, "max_tokens": config.server.max_tokens,
            "api_key": config.server.api_key.as_ref().map(|key| key.expose().to_owned()),
            "rate_limit": config.server.rate_limit,
            "request_timeout_secs": config.server.request_timeout.as_secs_f64(),
            "max_body_size": config.server.max_body_size.bytes(),
            "cors_origins": config.server.cors_origins},
        "memory": {
            "raise_wired_limit": config.memory.raise_wired_limit,
            "max_loaded_models": config.memory.max_loaded_models,
            "ceiling_bytes_at_128gib": config.memory.ceiling_bytes(Some(128 * 1024 * 1024 * 1024))},
        "models": models, "providers": providers, "routes": routes,
        "routing": {"default": config.routing.default_provider,
            "auto": {"enabled": config.routing.auto.enabled, "force": config.routing.auto.force,
                "model": config.routing.auto.model,
                "timeout_ms": u64::try_from(config.routing.auto.timeout.as_millis()).unwrap()}},
        "observability": {
            "retention_secs": config.observability.retention.map(|d| d.as_secs()),
            "metrics_log": {"enabled": config.observability.metrics_log.enabled,
                "path": config.observability.metrics_log.path,
                "max_size_bytes": config.observability.metrics_log.max_size.bytes(),
                "max_files": config.observability.metrics_log.max_files}},
        "mcp": {"max_result_bytes": config.mcp.max_result_bytes, "servers": mcp},
        "web": {"enabled": config.web.enabled, "search_provider": config.web.search_provider,
            "searxng_url": config.web.searxng_url, "allow_private_addresses": config.web.allow_private_addresses,
            "expose_as_tools": config.web.expose_as_tools, "timeout_secs": config.web.timeout.as_secs_f64()},
    })
}

/// The snapshot the pre-redesign code produced, with its home directory made portable.
fn recorded_snapshot(name: &str) -> Value {
    let mut recorded: Value =
        serde_json::from_str(&read(&format!("{name}.resolved.json")).replace("$HOME", &home()))
            .unwrap();
    if let Some(memory) = recorded.get_mut("memory").and_then(Value::as_object_mut) {
        memory.remove("default_profile");
    }
    recorded
}

const FIXTURE_NAMES: [&str; 3] = ["maintainer", "all_legacy_keys", "init_template_legacy"];

#[test]
fn legacy_fixtures_resolve_exactly_as_before_the_redesign() {
    for name in FIXTURE_NAMES {
        let loaded = load_text(&loadable(name), &Environment::default());
        assert_eq!(snapshot(&loaded.config), recorded_snapshot(name), "{name}");
    }
}

#[test]
fn the_maintainers_file_warns_only_about_the_keys_it_uses() {
    let loaded = load_text(&read("maintainer.toml"), &Environment::default());
    let keys: Vec<&str> = loaded
        .config
        .notices
        .deprecations
        .iter()
        .map(|deprecation| deprecation.old_key.as_str())
        .collect();
    assert_eq!(
        keys,
        [
            "models[0].mlx_profile",
            "default.provider",
            "retention.enabled",
            "retention.minutes"
        ]
    );
}

// -- migrate ------------------------------------------------------------------------------

fn assert_golden(name: &str, actual: &str) {
    let path = fixtures().join(name);
    if std::env::var_os("HIGGS_BLESS_GOLDEN").is_some() {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing golden file {name}; run with HIGGS_BLESS_GOLDEN=1"));
    assert_eq!(actual, expected, "{name} differs from the golden output");
}

#[test]
fn migrate_matches_the_golden_output_for_each_fixture() {
    for name in FIXTURE_NAMES {
        let migration = migrate(&read(&format!("{name}.toml")), None).unwrap();
        assert_golden(&format!("{name}.migrated.toml"), &migration.text);
    }
}

#[test]
fn migrated_files_load_without_deprecations_and_mean_the_same_thing() {
    for name in FIXTURE_NAMES {
        let original = read(&format!("{name}.toml"));
        let migration = migrate(&original, None).unwrap();
        let (before_text, after_text) = if name == "init_template_legacy" {
            (
                format!("{original}{TEMPLATE_MODEL}"),
                format!("{}{TEMPLATE_MODEL}", migration.text),
            )
        } else {
            (original, migration.text.clone())
        };
        let before = load_text(&before_text, &Environment::default());
        let after = load_text(&after_text, &Environment::default());
        assert!(
            after.config.notices.deprecations.is_empty(),
            "{name}: {:?}",
            after.config.notices.deprecations
        );
        assert_eq!(
            before.config.without_notices(),
            after.config.without_notices(),
            "{name}"
        );
        assert_eq!(snapshot(&after.config), recorded_snapshot(name), "{name}");
    }
}

#[test]
fn migrating_twice_changes_nothing() {
    for name in FIXTURE_NAMES {
        let once = migrate(&read(&format!("{name}.toml")), None).unwrap();
        let twice = migrate(&once.text, None).unwrap();
        assert!(!twice.changed(), "{name}");
        assert_eq!(once.text, twice.text, "{name}");
    }
}

#[test]
fn migrate_keeps_every_comment() {
    for name in FIXTURE_NAMES {
        let original = read(&format!("{name}.toml"));
        let migrated = migrate(&original, None).unwrap().text;
        for line in original.lines() {
            let comment = line.trim();
            if comment.starts_with('#') {
                assert!(
                    migrated.contains(comment),
                    "{name}: comment {comment:?} was lost"
                );
            }
            if let Some((_, trailing)) = line.split_once(" # ") {
                assert!(
                    migrated.contains(&format!("# {trailing}")),
                    "{name}: trailing comment {trailing:?} was lost"
                );
            }
        }
    }
}

#[test]
fn migrate_never_expands_references_or_drops_unknown_content() {
    let source = "# keep me\n[provider.p]\nurl = \"http://${HOST_NOT_SET}\"\napi_key = \"${KEY_NOT_SET}\"\n[mcp.servers.a]\ncommand = \"x\"\nenv = { T = \"${TOKEN_NOT_SET}\" }\n";
    let migration = migrate(source, None).unwrap();
    assert!(migration.text.contains("${HOST_NOT_SET}"));
    assert!(migration.text.contains("${KEY_NOT_SET}"));
    assert!(migration.text.contains("${TOKEN_NOT_SET}"));
    assert!(migration.text.contains("[providers.p]"));

    let error = migrate("[server]\nprot = 8000\n", None)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("`prot`") && error.contains("`port`"),
        "{error}"
    );
    let legacy_error = migrate("[local]\nloaad = 1\n", None)
        .unwrap_err()
        .to_string();
    assert!(legacy_error.contains("`loaad`"), "{legacy_error}");
}

#[test]
fn migrate_converts_units() {
    let text = migrate(
        "[server]\ntimeout = 300.0\nmax_body_size = 10485760\n[local]\nidle_ttl_seconds = 900\nmemory_limit_gb = 48\n[[models]]\npath = \"a/b\"\nkv_disk_space_mb = 4096\n[auto_router]\ntimeout_ms = 2000\n[logging.metrics]\nmax_size_mb = 50\n[retention]\nenabled = true\nminutes = 90\n[web]\ntimeout_seconds = 15\n[mcp]\nmax_result_bytes = 262144\n",
        None,
    )
    .unwrap()
    .text;
    for expected in [
        "request_timeout = \"5m\"",
        "max_body_size = 10485760",
        "idle_ttl = \"15m\"",
        "limit = \"48GB\"",
        "disk_budget = \"4GB\"",
        "timeout = \"2s\"",
        "max_size = \"50MB\"",
        "retention = \"90m\"",
        "timeout = \"15s\"",
        "max_result_size = \"256KB\"",
    ] {
        assert!(text.contains(expected), "missing {expected} in:\n{text}");
    }
}

// -- explain ------------------------------------------------------------------------------

fn value_and_source<'a>(entries: &'a [Entry], key: &str) -> (&'a str, &'a Source) {
    let entry = entries
        .iter()
        .find(|entry| entry.key == key)
        .unwrap_or_else(|| panic!("{key}"));
    (entry.value.as_str(), &entry.source)
}

#[test]
fn explain_names_the_source_of_every_server_value() {
    let environment =
        Environment::from_pairs([("HIGGS_SERVER__MAX_TOKENS", "100"), ("HIGGS_MTP", "off")]);
    let loaded = load_text(
        "[server]\nport = 9000\napi_key = \"${EXPLAIN_KEY:-fallback}\"\n[[models]]\npath = \"org/m\"\n",
        &environment,
    );
    let flags = super::ServeArgs {
        host: Some("0.0.0.0".to_owned()),
        ..super::ServeArgs::default()
    };
    let provenance = explain::Provenance::new(&loaded, &environment, Some(&flags));
    let entries = explain::global_entries(&provenance);
    assert_eq!(
        value_and_source(&entries, "server.port"),
        ("9000", &Source::ConfigFile)
    );
    assert_eq!(
        value_and_source(&entries, "server.max_tokens"),
        (
            "100",
            &Source::Environment("HIGGS_SERVER__MAX_TOKENS".to_owned())
        )
    );
    assert_eq!(
        value_and_source(&entries, "server.host").1,
        &Source::Flag("--host".to_owned())
    );
    assert_eq!(
        value_and_source(&entries, "server.request_timeout"),
        ("\"5m\"", &Source::BuiltIn)
    );
    assert_eq!(
        value_and_source(&entries, "experimental.mtp"),
        ("\"off\"", &Source::Environment("HIGGS_MTP".to_owned()))
    );
}

#[test]
fn explain_prints_references_unresolved_and_masks_literal_secrets() {
    let environment = Environment::from_pairs([("EXPLAIN_KEY", "sk-resolved-secret")]);
    let referenced = load_text(
        "[server]\napi_key = \"${EXPLAIN_KEY}\"\n[[models]]\npath = \"org/m\"\n",
        &environment,
    );
    let text = explain::explain(&referenced, &environment, None, None).unwrap();
    assert!(text.contains("${EXPLAIN_KEY}"), "{text}");
    assert!(!text.contains("sk-resolved-secret"), "{text}");

    let literal = load_text(
        "[server]\napi_key = \"sk-literal-secret\"\n[providers.p]\nurl = \"http://x\"\napi_key = \"sk-provider-secret\"\n[[models]]\npath = \"org/m\"\n",
        &Environment::default(),
    );
    let masked = explain::explain(&literal, &Environment::default(), None, None).unwrap();
    assert!(!masked.contains("sk-literal-secret"), "{masked}");
    assert!(!masked.contains("sk-provider-secret"), "{masked}");
    assert!(masked.contains("********"), "{masked}");
}

#[test]
fn explain_for_a_model_shows_defaults_model_and_built_in_sources() {
    let environment = Environment::from_pairs([("HIGGS_MLX_PROFILE", "latency")]);
    let loaded = load_text(
        "[defaults]\nload = \"on_demand\"\nidle_ttl = \"15m\"\nprofile = \"throughput\"\n[defaults.kv_cache]\nbits = 4\n[[models]]\npath = \"org/m\"\nname = \"m\"\nidle_ttl = \"0\"\n[models.kv_cache]\nmode = \"turboquant\"\n",
        &environment,
    );
    let flags = super::ServeArgs {
        idle_ttl: Some(60),
        ..super::ServeArgs::default()
    };
    let provenance = explain::Provenance::new(&loaded, &environment, Some(&flags));
    let (index, model) = explain::find_model(&loaded.config, "m").unwrap();
    let entries = explain::model_entries(&provenance, index, model);
    assert_eq!(
        value_and_source(&entries, "load"),
        ("\"on_demand\"", &Source::Defaults)
    );
    assert_eq!(
        value_and_source(&entries, "idle_ttl").1,
        &Source::Model,
        "the model's own idle_ttl beats the --idle-ttl flag"
    );
    assert_eq!(
        value_and_source(&entries, "profile"),
        (
            "\"latency\"",
            &Source::Environment("HIGGS_MLX_PROFILE".to_owned())
        )
    );
    assert_eq!(
        value_and_source(&entries, "kv_cache.mode"),
        ("\"turboquant\"", &Source::Model)
    );
    assert_eq!(
        value_and_source(&entries, "kv_cache.bits"),
        ("4", &Source::Defaults)
    );
    assert_eq!(
        value_and_source(&entries, "kv_cache.seed"),
        ("0", &Source::BuiltIn)
    );
    assert_eq!(value_and_source(&entries, "batch").1, &Source::BuiltIn);
    assert_eq!(
        value_and_source(&entries, "prefix_cache.checkpoint_interval_tokens"),
        ("512", &Source::BuiltIn)
    );
    assert_eq!(
        value_and_source(&entries, "prefix_cache.checkpoint_budget"),
        ("\"2GB\"", &Source::BuiltIn)
    );
}

#[test]
fn explain_credits_the_checkpoint_budget_variable() {
    let environment = Environment::from_pairs([("HIGGS_PREFIX_CHECKPOINT_BUDGET_MB", "256")]);
    let loaded = load_text(
        "[[models]]\npath = \"org/m\"\nname = \"m\"\n[models.prefix_cache]\ncheckpoint_budget = \"1GB\"\nmax_entries = 3\n",
        &environment,
    );
    let provenance = explain::Provenance::new(&loaded, &environment, None);
    let (index, model) = explain::find_model(&loaded.config, "m").unwrap();
    let entries = explain::model_entries(&provenance, index, model);
    assert_eq!(
        value_and_source(&entries, "prefix_cache.checkpoint_budget"),
        (
            "\"256MB\"",
            &Source::Environment("HIGGS_PREFIX_CHECKPOINT_BUDGET_MB".to_owned())
        )
    );
    assert_eq!(
        value_and_source(&entries, "prefix_cache.max_entries"),
        ("3", &Source::Model)
    );
}

#[test]
fn explain_text_lists_files_models_and_deprecations() {
    let loaded = load_text(&read("maintainer.toml"), &Environment::default());
    let text = explain::explain(&loaded, &Environment::default(), None, None).unwrap();
    assert!(text.contains("server.port"), "{text}");
    assert!(
        text.contains("(config file)") && text.contains("(built-in default)"),
        "{text}"
    );
    assert!(text.contains("thinklite"), "{text}");
    assert!(text.contains("4 deprecated key(s)"), "{text}");

    let model_text =
        explain::explain(&loaded, &Environment::default(), None, Some("thinklite")).unwrap();
    assert!(
        model_text.contains("profile") && model_text.contains("\"balanced\""),
        "{model_text}"
    );
    let error = explain::explain(&loaded, &Environment::default(), None, Some("nope")).unwrap_err();
    assert!(error.contains("thinklite"), "{error}");
}
