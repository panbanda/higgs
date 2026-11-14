//! Loading behavior: deprecated keys, strictness, inheritance, precedence, environment
//! references and includes.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::literal_string_with_formatting_args
)]

use std::path::Path;
use std::time::Duration;

use higgs_engine::mlx_tuning::RequestedMlxProfile;
use higgs_models::turboquant::KvCacheMode;

use super::cli::ServeArgs;
use super::units::ByteSize;
use super::{
    ApiFormat, BatchMode, Environment, HiggsConfig, LoadMode, MemoryGuard, MtpMode,
    load_config_file_in,
};

fn write(dir: &Path, name: &str, contents: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    path
}

fn load_with(
    toml: &str,
    environment: &Environment,
    flags: Option<&ServeArgs>,
) -> Result<HiggsConfig, String> {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "config.toml", toml);
    load_config_file_in(&path, flags, environment)
}

fn load(toml: &str) -> Result<HiggsConfig, String> {
    load_with(toml, &Environment::default(), None)
}

fn load_ok(toml: &str) -> HiggsConfig {
    load(toml).unwrap_or_else(|error| panic!("{error}"))
}

const MODEL: &str = "[[models]]\npath = \"org/model\"\nname = \"m\"\n";

fn with_model(prefix: &str) -> String {
    format!("{prefix}\n{MODEL}")
}

// -- Legacy keys --------------------------------------------------------------------------

struct LegacyRow {
    toml: &'static str,
    old: &'static str,
    new: &'static str,
    check: fn(&HiggsConfig),
}

fn model_of(config: &HiggsConfig) -> &super::ModelConfig {
    config.models.first().unwrap()
}

const LEGACY_ROWS: &[LegacyRow] = &[
    LegacyRow {
        toml: "[server]\ntimeout = 90.5\n",
        old: "server.timeout",
        new: "server.request_timeout",
        check: |c| assert_eq!(c.server.request_timeout, Duration::from_millis(90_500)),
    },
    LegacyRow {
        toml: "[local]\nmlx_profile = \"latency\"\n",
        old: "local.mlx_profile",
        new: "defaults.profile",
        check: |c| assert_eq!(model_of(c).profile, RequestedMlxProfile::Latency),
    },
    LegacyRow {
        toml: "[local]\nraise_wired_limit = true\n",
        old: "local.raise_wired_limit",
        new: "memory.raise_wired_limit",
        check: |c| assert!(c.memory.raise_wired_limit),
    },
    LegacyRow {
        toml: "[local]\nthinking_budget = 99\n",
        old: "local.thinking_budget",
        new: "defaults.thinking_budget",
        check: |c| assert_eq!(model_of(c).thinking_budget, 99),
    },
    LegacyRow {
        toml: "[local]\nload = \"on_demand\"\n",
        old: "local.load",
        new: "defaults.load",
        check: |c| assert_eq!(model_of(c).load, LoadMode::OnDemand),
    },
    LegacyRow {
        toml: "[local]\nidle_ttl_seconds = 900\n",
        old: "local.idle_ttl_seconds",
        new: "defaults.idle_ttl",
        check: |c| assert_eq!(model_of(c).idle_ttl, Some(Duration::from_secs(900))),
    },
    LegacyRow {
        toml: "[local]\nmax_loaded_models = 3\n",
        old: "local.max_loaded_models",
        new: "memory.max_loaded_models",
        check: |c| assert_eq!(c.memory.max_loaded_models, Some(3)),
    },
    LegacyRow {
        toml: "[local]\nmemory_limit_gb = 48.5\n",
        old: "local.memory_limit_gb",
        new: "memory.limit",
        check: |c| {
            assert_eq!(
                c.memory.limit,
                Some(ByteSize::from_bytes(
                    48 * 1024 * 1024 * 1024 + 512 * 1024 * 1024
                ))
            );
        },
    },
    LegacyRow {
        toml: "[local]\nmemory_guard = \"safe\"\n",
        old: "local.memory_guard",
        new: "memory.guard",
        check: |c| assert_eq!(c.memory.guard, MemoryGuard::Safe),
    },
    LegacyRow {
        toml: "mlx_profile = \"throughput\"\n",
        old: "models[0].mlx_profile",
        new: "models[0].profile",
        check: |c| assert_eq!(model_of(c).profile, RequestedMlxProfile::Throughput),
    },
    LegacyRow {
        toml: "idle_ttl_seconds = 30\n",
        old: "models[0].idle_ttl_seconds",
        new: "models[0].idle_ttl",
        check: |c| assert_eq!(model_of(c).idle_ttl, Some(Duration::from_secs(30))),
    },
    LegacyRow {
        toml: "kv_cache = \"turboquant\"\n",
        old: "models[0].kv_cache",
        new: "models[0].kv_cache.mode",
        check: |c| assert_eq!(model_of(c).kv_cache.mode, KvCacheMode::Turboquant),
    },
    LegacyRow {
        toml: "kv_bits = 4\n",
        old: "models[0].kv_bits",
        new: "models[0].kv_cache.bits",
        check: |c| assert_eq!(model_of(c).kv_cache.bits, 4),
    },
    LegacyRow {
        toml: "kv_key_bits = 2\n",
        old: "models[0].kv_key_bits",
        new: "models[0].kv_cache.key_bits",
        check: |c| assert_eq!(model_of(c).kv_cache.key_bits_override, Some(2)),
    },
    LegacyRow {
        toml: "kv_value_bits = 5\n",
        old: "models[0].kv_value_bits",
        new: "models[0].kv_cache.value_bits",
        check: |c| assert_eq!(model_of(c).kv_cache.value_bits_override, Some(5)),
    },
    LegacyRow {
        toml: "kv_norm_correction = false\n",
        old: "models[0].kv_norm_correction",
        new: "models[0].kv_cache.norm_correction",
        check: |c| assert!(!model_of(c).kv_cache.norm_correction),
    },
    LegacyRow {
        toml: "kv_adaptive_dense_layers = 2\n",
        old: "models[0].kv_adaptive_dense_layers",
        new: "models[0].kv_cache.adaptive_dense_layers",
        check: |c| assert_eq!(model_of(c).kv_cache.adaptive_dense_layers, 2),
    },
    LegacyRow {
        toml: "kv_seed = 11\n",
        old: "models[0].kv_seed",
        new: "models[0].kv_cache.seed",
        check: |c| assert_eq!(model_of(c).kv_cache.seed, 11),
    },
    LegacyRow {
        toml: "mla_latent_cache = true\n",
        old: "models[0].mla_latent_cache",
        new: "models[0].kv_cache.mla_latent",
        check: |c| assert!(model_of(c).kv_cache.mla_latent),
    },
    LegacyRow {
        toml: "kv_disk_dir = \"/tmp/higgs-kv\"\n",
        old: "models[0].kv_disk_dir",
        new: "models[0].prefix_cache.disk_dir",
        check: |c| {
            assert_eq!(
                model_of(c).prefix_cache.disk_dir.as_deref(),
                Some("/tmp/higgs-kv")
            );
        },
    },
    LegacyRow {
        toml: "kv_disk_space_mb = 8192\n",
        old: "models[0].kv_disk_space_mb",
        new: "models[0].prefix_cache.disk_budget",
        check: |c| {
            assert_eq!(
                model_of(c).prefix_cache.disk_budget,
                ByteSize::from_bytes(8 << 30)
            );
        },
    },
    LegacyRow {
        toml: "prefill_yield_tokens = 512\n",
        old: "models[0].prefill_yield_tokens",
        new: "models[0].prefill.yield_tokens",
        check: |c| assert_eq!(model_of(c).prefill_yield_tokens, Some(512)),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\n",
        old: "provider.p.url",
        new: "providers.p.url",
        check: |c| assert_eq!(c.providers["p"].url, "http://a"),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\nformat = \"anthropic\"\n",
        old: "provider.p.format",
        new: "providers.p.format",
        check: |c| assert_eq!(c.providers["p"].format, ApiFormat::Anthropic),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\napi_key = \"k\"\n",
        old: "provider.p.api_key",
        new: "providers.p.api_key",
        check: |c| assert_eq!(c.providers["p"].api_key.as_ref().unwrap().expose(), "k"),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\nstrip_auth = true\n",
        old: "provider.p.strip_auth",
        new: "providers.p.strip_auth",
        check: |c| assert!(c.providers["p"].strip_auth),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\nstub_count_tokens = true\n",
        old: "provider.p.stub_count_tokens",
        new: "providers.p.stub_count_tokens",
        check: |c| assert!(c.providers["p"].stub_count_tokens),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\n[[routes]]\npattern = \"x-.*\"\nprovider = \"p\"\n",
        old: "routes[0].pattern",
        new: "routes[0].match",
        check: |c| assert_eq!(c.routes[0].pattern.as_deref(), Some("x-.*")),
    },
    LegacyRow {
        toml: "[provider.p]\nurl = \"http://a\"\n[default]\nprovider = \"p\"\n",
        old: "default.provider",
        new: "routing.default",
        check: |c| assert_eq!(c.routing.default_provider, "p"),
    },
    LegacyRow {
        toml: "[auto_router]\nenabled = true\nmodel = \"m\"\n",
        old: "auto_router.enabled",
        new: "routing.auto.enabled",
        check: |c| assert!(c.routing.auto.enabled),
    },
    LegacyRow {
        toml: "[auto_router]\nforce = true\n",
        old: "auto_router.force",
        new: "routing.auto.force",
        check: |c| assert!(c.routing.auto.force),
    },
    LegacyRow {
        toml: "[auto_router]\nmodel = \"other\"\n",
        old: "auto_router.model",
        new: "routing.auto.model",
        check: |c| assert_eq!(c.routing.auto.model, "other"),
    },
    LegacyRow {
        toml: "[auto_router]\ntimeout_ms = 1500\n",
        old: "auto_router.timeout_ms",
        new: "routing.auto.timeout",
        check: |c| assert_eq!(c.routing.auto.timeout, Duration::from_millis(1500)),
    },
    LegacyRow {
        toml: "[logging.metrics]\nenabled = false\n",
        old: "logging.metrics.enabled",
        new: "observability.metrics_log.enabled",
        check: |c| assert!(!c.observability.metrics_log.enabled),
    },
    LegacyRow {
        toml: "[logging.metrics]\npath = \"/tmp/m.jsonl\"\n",
        old: "logging.metrics.path",
        new: "observability.metrics_log.path",
        check: |c| assert_eq!(c.observability.metrics_log.path, "/tmp/m.jsonl"),
    },
    LegacyRow {
        toml: "[logging.metrics]\nmax_size_mb = 25\n",
        old: "logging.metrics.max_size_mb",
        new: "observability.metrics_log.max_size",
        check: |c| {
            assert_eq!(
                c.observability.metrics_log.max_size,
                ByteSize::from_bytes(25 << 20)
            );
        },
    },
    LegacyRow {
        toml: "[logging.metrics]\nmax_files = 2\n",
        old: "logging.metrics.max_files",
        new: "observability.metrics_log.max_files",
        check: |c| assert_eq!(c.observability.metrics_log.max_files, 2),
    },
    LegacyRow {
        toml: "[retention]\nenabled = false\n",
        old: "retention.enabled",
        new: "observability.retention",
        check: |c| assert_eq!(c.observability.retention, None),
    },
    LegacyRow {
        toml: "[retention]\nminutes = 30\n",
        old: "retention.minutes",
        new: "observability.retention",
        check: |c| {
            assert_eq!(
                c.observability.retention,
                Some(Duration::from_secs(30 * 60))
            );
        },
    },
    LegacyRow {
        toml: "[mcp]\nmax_result_bytes = 1000\n",
        old: "mcp.max_result_bytes",
        new: "mcp.max_result_size",
        check: |c| assert_eq!(c.mcp.max_result_bytes, 1000),
    },
    LegacyRow {
        toml: "[mcp.servers.fs]\ncommand = \"npx\"\ntimeout_seconds = 45\n",
        old: "mcp.servers.fs.timeout_seconds",
        new: "mcp.servers.fs.timeout",
        check: |c| assert_eq!(c.mcp.servers["fs"].timeout, Duration::from_secs(45)),
    },
    LegacyRow {
        toml: "[web]\ntimeout_seconds = 5.5\n",
        old: "web.timeout_seconds",
        new: "web.timeout",
        check: |c| assert_eq!(c.web.timeout, Duration::from_millis(5500)),
    },
];

/// Legacy keys that some snippets need in order to load, besides the key under test.
const SNIPPET_SUPPORT_KEYS: &[&str] = &["provider.p.url", "auto_router.model"];

/// Rows whose key lives in a `[[models]]` entry are written as the entry's extra lines.
fn row_document(row: &LegacyRow) -> String {
    if row.old.starts_with("models[0].") {
        format!(
            "[[models]]\npath = \"org/model\"\nname = \"m\"\n{}",
            row.toml
        )
    } else {
        with_model(row.toml)
    }
}

#[test]
fn every_legacy_key_loads_resolves_and_warns_once() {
    for row in LEGACY_ROWS {
        let config =
            load(&row_document(row)).unwrap_or_else(|error| panic!("{}: {error}", row.old));
        (row.check)(&config);
        let matching: Vec<_> = config
            .notices
            .deprecations
            .iter()
            .filter(|deprecation| deprecation.old_key == row.old)
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "{}: {:?}",
            row.old,
            config.notices.deprecations
        );
        assert_eq!(matching[0].new_key, row.new, "{}", row.old);
        assert!(matching[0].line.is_some(), "{} has no line", row.old);
        // The only other deprecated keys are the ones a snippet needs to be valid on its own.
        for other in &config.notices.deprecations {
            assert!(
                other.old_key == row.old || SNIPPET_SUPPORT_KEYS.contains(&other.old_key.as_str()),
                "{} also warned about {}",
                row.old,
                other.old_key
            );
        }
    }
}

/// `models[0].x` becomes `models[].x`, and provider and server names become `*`.
fn generic_key(old: &str) -> String {
    let mut parts: Vec<String> = old
        .split('.')
        .map(|part| {
            part.split_once('[')
                .map_or_else(|| part.to_owned(), |(name, _)| format!("{name}[]"))
        })
        .collect();
    if parts.first().map(String::as_str) == Some("provider") {
        if let Some(name) = parts.get_mut(1) {
            "*".clone_into(name);
        }
    }
    if parts.starts_with(&["mcp".to_owned(), "servers".to_owned()]) {
        if let Some(name) = parts.get_mut(2) {
            "*".clone_into(name);
        }
    }
    parts.join(".")
}

#[test]
fn the_mapping_table_has_a_row_for_every_legacy_key() {
    let exercised: std::collections::HashSet<String> =
        LEGACY_ROWS.iter().map(|row| generic_key(row.old)).collect();
    for (old, _) in super::legacy::legacy_key_table() {
        let key = old.replace(" (string)", "");
        assert!(exercised.contains(&key), "no table row exercises {key}");
    }
}

#[test]
fn a_file_using_only_current_keys_has_no_deprecations() {
    let config = load_ok(
        "[server]\nrequest_timeout = \"2m\"\n[defaults]\nidle_ttl = \"5m\"\n[[models]]\npath = \"org/m\"\n[models.kv_cache]\nbits = 4\n",
    );
    assert!(config.notices.deprecations.is_empty());
}

#[test]
fn old_and_new_spelling_of_one_setting_conflict_naming_both() {
    for (toml, old, new) in [
        (
            "[local]\nload = \"startup\"\n[defaults]\nload = \"on_demand\"\n",
            "local.load",
            "defaults.load",
        ),
        (
            "[server]\ntimeout = 30\nrequest_timeout = \"1m\"\n",
            "server.timeout",
            "server.request_timeout",
        ),
        (
            "[provider.p]\nurl = \"http://a\"\n[providers.p]\nurl = \"http://b\"\n",
            "provider.p.url",
            "providers.p.url",
        ),
        (
            "[local]\nmemory_guard = \"safe\"\n[memory]\nguard = \"off\"\n",
            "local.memory_guard",
            "memory.guard",
        ),
        (
            "[retention]\nminutes = 5\n[observability]\nretention = \"1h\"\n",
            "retention.minutes",
            "observability.retention",
        ),
        (
            "[default]\nprovider = \"higgs\"\n[routing]\ndefault = \"higgs\"\n",
            "default.provider",
            "routing.default",
        ),
        (
            "[auto_router]\ntimeout_ms = 5\n[routing.auto]\ntimeout = \"1s\"\n",
            "auto_router.timeout_ms",
            "routing.auto.timeout",
        ),
    ] {
        let error = load(&with_model(toml)).unwrap_err();
        assert!(
            error.contains(old) && error.contains(new),
            "{toml}: {error}"
        );
        assert!(error.contains("both"), "{error}");
    }
}

#[test]
fn per_model_old_and_new_spelling_conflict() {
    let error = load("[[models]]\npath = \"org/m\"\nkv_bits = 4\n[models.kv_cache]\nbits = 5\n")
        .unwrap_err();
    assert!(error.contains("models[0].kv_bits"), "{error}");
    assert!(error.contains("models[0].kv_cache.bits"), "{error}");
}

#[test]
fn a_legacy_table_can_split_across_new_tables() {
    let config = load_ok(&with_model(
        "[local]\nload = \"on_demand\"\nmemory_guard = \"safe\"\nmax_loaded_models = 2\n",
    ));
    assert_eq!(model_of(&config).load, LoadMode::OnDemand);
    assert_eq!(config.memory.guard, MemoryGuard::Safe);
    assert_eq!(config.memory.max_loaded_models, Some(2));
    assert_eq!(config.notices.deprecations.len(), 3);
}

#[test]
fn inline_tables_in_the_old_schema_still_load() {
    let config = load_ok(
        "provider = { p = { url = \"http://a\" } }\nmodels = [{ path = \"org/m\", kv_bits = 4 }]\n",
    );
    assert_eq!(config.providers["p"].url, "http://a");
    assert_eq!(model_of(&config).kv_cache.bits, 4);
}

// -- Strict keys --------------------------------------------------------------------------

#[test]
fn an_unknown_key_is_an_error_with_file_line_section_and_suggestion() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(
        dir.path(),
        "config.toml",
        "[server]\nport = 1\n\n[defaults]\nidel_ttl = \"5m\"\n",
    );
    let error = load_config_file_in(&path, None, &Environment::default()).unwrap_err();
    assert!(
        error.starts_with(&format!("{}:5:", path.display())),
        "{error}"
    );
    assert!(
        error.contains("unknown key `idel_ttl` in [defaults]"),
        "{error}"
    );
    assert!(error.contains("did you mean `idle_ttl`"), "{error}");
}

#[test]
fn every_unknown_key_is_reported_at_once() {
    let error = load("[server]\nprot = 1\n[memory]\nguarde = \"off\"\n").unwrap_err();
    assert!(
        error.contains("`prot`") && error.contains("`port`"),
        "{error}"
    );
    assert!(
        error.contains("`guarde`") && error.contains("`guard`"),
        "{error}"
    );
}

#[test]
fn unknown_keys_inside_legacy_tables_are_errors_too() {
    let error = load(&with_model("[local]\nidel_ttl_seconds = 5\n")).unwrap_err();
    assert!(
        error.contains("unknown key `idel_ttl_seconds` in [local]"),
        "{error}"
    );
    assert!(error.contains("did you mean `idle_ttl_seconds`"), "{error}");
    let provider_error =
        load(&with_model("[provider.p]\nurl = \"u\"\nstrp_auth = true\n")).unwrap_err();
    assert!(
        provider_error.contains("`strp_auth`") && provider_error.contains("`strip_auth`"),
        "{provider_error}"
    );
}

#[test]
fn a_misspelled_top_level_section_suggests_the_closest() {
    let error = load("[servre]\nport = 1\n").unwrap_err();
    assert!(error.contains("did you mean `server`"), "{error}");
}

#[test]
fn a_wrong_type_names_the_key_and_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(
        dir.path(),
        "config.toml",
        &with_model("[server]\nport = \"abc\"\n"),
    );
    let error = load_config_file_in(&path, None, &Environment::default()).unwrap_err();
    assert!(error.contains("server.port"), "{error}");
    assert!(error.contains(":2:"), "{error}");
}

#[test]
fn an_invalid_duration_names_the_key() {
    let error = load(&with_model("[server]\nrequest_timeout = 30\n")).unwrap_err();
    assert!(
        error.contains("server.request_timeout") && error.contains("no unit"),
        "{error}"
    );
    let unit_error = load(&with_model("[defaults]\nidle_ttl = \"5 fortnights\"\n")).unwrap_err();
    assert!(unit_error.contains("defaults.idle_ttl"), "{unit_error}");
}

// -- Inheritance --------------------------------------------------------------------------

#[test]
fn built_in_values_apply_when_nothing_is_set() {
    let config = load_ok(MODEL);
    let model = model_of(&config);
    assert_eq!(model.load, LoadMode::Startup);
    assert_eq!(model.idle_ttl, None);
    assert_eq!(model.batch, BatchMode::Auto);
    assert_eq!(model.profile, RequestedMlxProfile::Auto);
    assert_eq!(
        model.thinking_budget,
        higgs_engine::engine::DEFAULT_THINKING_BUDGET
    );
    assert!(!model.pinned);
    assert_eq!(model.kv_cache.mode, KvCacheMode::Off);
    assert_eq!(model.kv_cache.bits, 3);
    assert!(model.kv_cache.norm_correction);
    assert_eq!(
        model.prefix_cache.disk_budget,
        ByteSize::from_bytes(4 << 30)
    );
    assert_eq!(model.prefill_yield_tokens, None);
    assert_eq!(config.server.request_timeout, Duration::from_secs(300));
    assert_eq!(config.server.max_body_size, ByteSize::from_bytes(10 << 20));
}

/// (default text, model text, expected from defaults, expected from model)
struct InheritanceRow {
    key: &'static str,
    defaults: &'static str,
    model: &'static str,
    check_defaults: fn(&super::ModelConfig),
    check_model: fn(&super::ModelConfig),
}

#[test]
#[allow(clippy::too_many_lines)]
fn every_per_model_key_inherits_from_defaults_and_is_overridden_by_the_entry() {
    let rows = [
        InheritanceRow {
            key: "load",
            defaults: "load = \"on_demand\"",
            model: "load = \"startup\"",
            check_defaults: |m| assert_eq!(m.load, LoadMode::OnDemand),
            check_model: |m| assert_eq!(m.load, LoadMode::Startup),
        },
        InheritanceRow {
            key: "idle_ttl",
            defaults: "idle_ttl = \"15m\"",
            model: "idle_ttl = \"0\"",
            check_defaults: |m| assert_eq!(m.idle_ttl, Some(Duration::from_secs(900))),
            check_model: |m| assert_eq!(m.idle_ttl, None),
        },
        InheritanceRow {
            key: "batch",
            defaults: "batch = false",
            model: "batch = \"auto\"",
            check_defaults: |m| assert_eq!(m.batch, BatchMode::Off),
            check_model: |m| assert_eq!(m.batch, BatchMode::Auto),
        },
        InheritanceRow {
            key: "profile",
            defaults: "profile = \"throughput\"",
            model: "profile = \"latency\"",
            check_defaults: |m| assert_eq!(m.profile, RequestedMlxProfile::Throughput),
            check_model: |m| assert_eq!(m.profile, RequestedMlxProfile::Latency),
        },
        InheritanceRow {
            key: "thinking_budget",
            defaults: "thinking_budget = 100",
            model: "thinking_budget = 7",
            check_defaults: |m| assert_eq!(m.thinking_budget, 100),
            check_model: |m| assert_eq!(m.thinking_budget, 7),
        },
        InheritanceRow {
            key: "pinned",
            defaults: "pinned = true",
            model: "pinned = false",
            check_defaults: |m| assert!(m.pinned),
            check_model: |m| assert!(!m.pinned),
        },
        InheritanceRow {
            key: "kv_cache.mode",
            defaults: "[defaults.kv_cache]\nmode = \"turboquant\"",
            model: "[models.kv_cache]\nmode = \"off\"",
            check_defaults: |m| assert_eq!(m.kv_cache.mode, KvCacheMode::Turboquant),
            check_model: |m| assert_eq!(m.kv_cache.mode, KvCacheMode::Off),
        },
        InheritanceRow {
            key: "kv_cache.bits",
            defaults: "[defaults.kv_cache]\nbits = 4",
            model: "[models.kv_cache]\nbits = 5",
            check_defaults: |m| assert_eq!(m.kv_cache.bits, 4),
            check_model: |m| assert_eq!(m.kv_cache.bits, 5),
        },
        InheritanceRow {
            key: "kv_cache.key_bits",
            defaults: "[defaults.kv_cache]\nkey_bits = 2",
            model: "[models.kv_cache]\nkey_bits = 6",
            check_defaults: |m| assert_eq!(m.kv_cache.key_bits_override, Some(2)),
            check_model: |m| assert_eq!(m.kv_cache.key_bits_override, Some(6)),
        },
        InheritanceRow {
            key: "kv_cache.value_bits",
            defaults: "[defaults.kv_cache]\nvalue_bits = 3",
            model: "[models.kv_cache]\nvalue_bits = 7",
            check_defaults: |m| assert_eq!(m.kv_cache.value_bits_override, Some(3)),
            check_model: |m| assert_eq!(m.kv_cache.value_bits_override, Some(7)),
        },
        InheritanceRow {
            key: "kv_cache.norm_correction",
            defaults: "[defaults.kv_cache]\nnorm_correction = false",
            model: "[models.kv_cache]\nnorm_correction = true",
            check_defaults: |m| assert!(!m.kv_cache.norm_correction),
            check_model: |m| assert!(m.kv_cache.norm_correction),
        },
        InheritanceRow {
            key: "kv_cache.adaptive_dense_layers",
            defaults: "[defaults.kv_cache]\nadaptive_dense_layers = 1",
            model: "[models.kv_cache]\nadaptive_dense_layers = 4",
            check_defaults: |m| assert_eq!(m.kv_cache.adaptive_dense_layers, 1),
            check_model: |m| assert_eq!(m.kv_cache.adaptive_dense_layers, 4),
        },
        InheritanceRow {
            key: "kv_cache.seed",
            defaults: "[defaults.kv_cache]\nseed = 9",
            model: "[models.kv_cache]\nseed = 10",
            check_defaults: |m| assert_eq!(m.kv_cache.seed, 9),
            check_model: |m| assert_eq!(m.kv_cache.seed, 10),
        },
        InheritanceRow {
            key: "kv_cache.mla_latent",
            defaults: "[defaults.kv_cache]\nmla_latent = true",
            model: "[models.kv_cache]\nmla_latent = false",
            check_defaults: |m| assert!(m.kv_cache.mla_latent),
            check_model: |m| assert!(!m.kv_cache.mla_latent),
        },
        InheritanceRow {
            key: "prefix_cache.disk_dir",
            defaults: "[defaults.prefix_cache]\ndisk_dir = \"/a\"",
            model: "[models.prefix_cache]\ndisk_dir = \"/b\"",
            check_defaults: |m| assert_eq!(m.prefix_cache.disk_dir.as_deref(), Some("/a")),
            check_model: |m| assert_eq!(m.prefix_cache.disk_dir.as_deref(), Some("/b")),
        },
        InheritanceRow {
            key: "prefix_cache.disk_budget",
            defaults: "[defaults.prefix_cache]\ndisk_budget = \"1GB\"",
            model: "[models.prefix_cache]\ndisk_budget = \"2GB\"",
            check_defaults: |m| {
                assert_eq!(m.prefix_cache.disk_budget, ByteSize::from_bytes(1 << 30));
            },
            check_model: |m| assert_eq!(m.prefix_cache.disk_budget, ByteSize::from_bytes(2 << 30)),
        },
        InheritanceRow {
            key: "prefix_cache.max_entries",
            defaults: "[defaults.prefix_cache]\nmax_entries = 4",
            model: "[models.prefix_cache]\nmax_entries = 16",
            check_defaults: |m| assert_eq!(m.prefix_cache.max_entries, 4),
            check_model: |m| assert_eq!(m.prefix_cache.max_entries, 16),
        },
        InheritanceRow {
            key: "prefix_cache.checkpoint_interval_tokens",
            defaults: "[defaults.prefix_cache]\ncheckpoint_interval_tokens = 1024",
            model: "[models.prefix_cache]\ncheckpoint_interval_tokens = 2048",
            check_defaults: |m| assert_eq!(m.prefix_cache.checkpoint_interval_tokens, 1024),
            check_model: |m| assert_eq!(m.prefix_cache.checkpoint_interval_tokens, 2048),
        },
        InheritanceRow {
            key: "prefix_cache.max_checkpoints_per_entry",
            defaults: "[defaults.prefix_cache]\nmax_checkpoints_per_entry = 2",
            model: "[models.prefix_cache]\nmax_checkpoints_per_entry = 0",
            check_defaults: |m| assert_eq!(m.prefix_cache.max_checkpoints_per_entry, 2),
            check_model: |m| assert_eq!(m.prefix_cache.max_checkpoints_per_entry, 0),
        },
        InheritanceRow {
            key: "prefix_cache.checkpoint_budget",
            defaults: "[defaults.prefix_cache]\ncheckpoint_budget = \"1GB\"",
            model: "[models.prefix_cache]\ncheckpoint_budget = \"512MB\"",
            check_defaults: |m| {
                assert_eq!(
                    m.prefix_cache.checkpoint_budget,
                    ByteSize::from_bytes(1 << 30)
                );
            },
            check_model: |m| {
                assert_eq!(
                    m.prefix_cache.checkpoint_budget,
                    ByteSize::from_bytes(512 << 20)
                );
            },
        },
        InheritanceRow {
            key: "prefill.yield_tokens",
            defaults: "[defaults.prefill]\nyield_tokens = 256",
            model: "[models.prefill]\nyield_tokens = 1024",
            check_defaults: |m| assert_eq!(m.prefill_yield_tokens, Some(256)),
            check_model: |m| assert_eq!(m.prefill_yield_tokens, Some(1024)),
        },
    ];
    for row in rows {
        let defaults = if row.defaults.starts_with('[') {
            format!("{}\n", row.defaults)
        } else {
            format!("[defaults]\n{}\n", row.defaults)
        };
        let own = format!("{}\n", row.model);
        let document =
            format!("{defaults}[[models]]\npath = \"org/a\"\n[[models]]\npath = \"org/b\"\n{own}");
        let config =
            load(&document).unwrap_or_else(|error| panic!("{}: {error}\n{document}", row.key));
        (row.check_defaults)(config.models.first().unwrap());
        (row.check_model)(config.models.get(1).unwrap());
    }
}

#[test]
fn nested_tables_merge_key_by_key() {
    let config = load_ok(
        "[defaults.kv_cache]\nmode = \"turboquant\"\nbits = 2\n[[models]]\npath = \"org/m\"\n[models.kv_cache]\nbits = 4\n",
    );
    let kv = model_of(&config).kv_cache;
    assert_eq!(kv.mode, KvCacheMode::Turboquant);
    assert_eq!(kv.bits, 4);
}

// -- Precedence ---------------------------------------------------------------------------

#[test]
fn server_keys_follow_flag_then_environment_then_file() {
    let file = with_model("[server]\nport = 1111\nhost = \"10.0.0.1\"\n");
    let environment = Environment::from_pairs([("HIGGS_SERVER__PORT", "2222")]);
    let from_file = load_with(&file, &Environment::default(), None).unwrap();
    assert_eq!(from_file.server.port, 1111);
    let from_env = load_with(&file, &environment, None).unwrap();
    assert_eq!(from_env.server.port, 2222);
    assert_eq!(from_env.server.host, "10.0.0.1");
    let flags = ServeArgs {
        port: Some(3333),
        ..ServeArgs::default()
    };
    let from_flag = load_with(&file, &environment, Some(&flags)).unwrap();
    assert_eq!(from_flag.server.port, 3333);
}

#[test]
fn the_environment_overlay_accepts_deprecated_names_with_a_warning() {
    let environment = Environment::from_pairs([("HIGGS_SERVER__TIMEOUT", "45")]);
    let config = load_with(MODEL, &environment, None).unwrap();
    assert_eq!(config.server.request_timeout, Duration::from_secs(45));
    assert_eq!(config.notices.deprecations.len(), 1);
}

#[test]
fn profile_follows_model_then_flag_then_env_then_defaults() {
    let file = "[defaults]\nprofile = \"balanced\"\n[[models]]\npath = \"org/a\"\n[[models]]\npath = \"org/b\"\nprofile = \"latency\"\n";
    let none = Environment::default();
    let env = Environment::from_pairs([("HIGGS_MLX_PROFILE", "throughput")]);
    let flag = ServeArgs {
        mlx_profile: Some(super::MlxProfile::Auto),
        ..ServeArgs::default()
    };
    let profiles = |config: &HiggsConfig| -> Vec<RequestedMlxProfile> {
        config.models.iter().map(|m| m.profile).collect()
    };
    assert_eq!(
        profiles(&load_with(file, &none, None).unwrap()),
        [RequestedMlxProfile::Balanced, RequestedMlxProfile::Latency]
    );
    assert_eq!(
        profiles(&load_with(file, &env, None).unwrap()),
        [
            RequestedMlxProfile::Throughput,
            RequestedMlxProfile::Latency
        ]
    );
    assert_eq!(
        profiles(&load_with(file, &env, Some(&flag)).unwrap()),
        [RequestedMlxProfile::Auto, RequestedMlxProfile::Latency]
    );
}

#[test]
fn an_invalid_profile_variable_is_an_error() {
    let env = Environment::from_pairs([("HIGGS_MLX_PROFILE", "warp9")]);
    let error = load_with(MODEL, &env, None).unwrap_err();
    assert!(error.contains("HIGGS_MLX_PROFILE"), "{error}");
}

#[test]
fn lifecycle_flags_override_defaults_but_not_the_model_entry() {
    let file = "[defaults]\nload = \"on_demand\"\nidle_ttl = \"1h\"\n[[models]]\npath = \"org/a\"\n[[models]]\npath = \"org/b\"\nload = \"on_demand\"\n";
    let flags = ServeArgs {
        load: Some(LoadMode::Startup),
        idle_ttl: Some(30),
        memory_guard: Some(MemoryGuard::Aggressive),
        max_loaded_models: Some(2),
        ..ServeArgs::default()
    };
    let config = load_with(file, &Environment::default(), Some(&flags)).unwrap();
    assert_eq!(config.models[0].load, LoadMode::Startup);
    assert_eq!(config.models[1].load, LoadMode::OnDemand);
    assert_eq!(config.models[0].idle_ttl, Some(Duration::from_secs(30)));
    assert_eq!(config.memory.guard, MemoryGuard::Aggressive);
    assert_eq!(config.memory.max_loaded_models, Some(2));
}

#[test]
fn experimental_keys_follow_environment_then_file_then_built_in() {
    let file = with_model(
        "[experimental]\nmtp = \"off\"\nmtp_draft_tokens = 3\nprompt_lookup = true\n[experimental.chunked_prefill]\nthreshold = 4096\nchunk = 1024\n",
    );
    let from_file = load_with(&file, &Environment::default(), None).unwrap();
    assert_eq!(from_file.experimental.mtp, MtpMode::Off);
    assert_eq!(from_file.experimental.mtp_draft_tokens, Some(3));
    assert!(from_file.experimental.prompt_lookup);
    assert_eq!(from_file.experimental.chunked_prefill_threshold, Some(4096));
    assert!(from_file.notices.env_overrides.is_empty());

    let environment = Environment::from_pairs([
        ("HIGGS_MTP", "1"),
        ("HIGGS_MTP_DRAFT_N_MAX", "5"),
        ("HIGGS_PROMPT_LOOKUP", "0"),
        ("HIGGS_CHUNKED_PREFILL_THRESHOLD", "100"),
        ("HIGGS_CHUNKED_PREFILL_CHUNK_SIZE", "nope"),
    ]);
    let from_env = load_with(&file, &environment, None).unwrap();
    assert_eq!(from_env.experimental.mtp, MtpMode::On);
    assert_eq!(from_env.experimental.mtp_draft_tokens, Some(5));
    assert!(!from_env.experimental.prompt_lookup);
    assert_eq!(from_env.experimental.chunked_prefill_threshold, Some(100));
    assert_eq!(from_env.experimental.chunked_prefill_chunk, Some(1024));
    let unusable: Vec<_> = from_env
        .notices
        .env_overrides
        .iter()
        .filter(|entry| !entry.applied)
        .map(|entry| entry.variable)
        .collect();
    assert_eq!(unusable, ["HIGGS_CHUNKED_PREFILL_CHUNK_SIZE"]);
}

#[test]
fn environment_overrides_name_the_key_they_replace() {
    let env = Environment::from_pairs([("HIGGS_MTP", "off"), ("HIGGS_MLA_LATENT_CACHE", "1")]);
    let config = load_with(MODEL, &env, None).unwrap();
    let pairs: Vec<_> = config
        .notices
        .env_overrides
        .iter()
        .map(|entry| (entry.variable, entry.key))
        .collect();
    assert!(pairs.contains(&("HIGGS_MTP", "experimental.mtp")));
    assert!(pairs.contains(&("HIGGS_MLA_LATENT_CACHE", "models[].kv_cache.mla_latent")));
}

#[test]
fn prefix_cache_defaults_match_the_engine_constants() {
    let config = load_ok(MODEL);
    let cache = &config.models[0].prefix_cache;
    assert_eq!(cache.max_entries, 8);
    assert_eq!(cache.checkpoint_interval_tokens, 512);
    assert_eq!(cache.max_checkpoints_per_entry, 8);
    assert_eq!(cache.checkpoint_budget, ByteSize::from_bytes(2 << 30));
}

#[test]
fn the_checkpoint_budget_variable_replaces_the_key_for_every_model() {
    let file = "[defaults.prefix_cache]\ncheckpoint_budget = \"1GB\"\n[[models]]\npath = \"org/a\"\n[[models]]\npath = \"org/b\"\n[models.prefix_cache]\ncheckpoint_budget = \"3GB\"\n";
    for (value, expected) in [("512", 512_u64 << 20), ("0", 0)] {
        let environment = Environment::from_pairs([("HIGGS_PREFIX_CHECKPOINT_BUDGET_MB", value)]);
        let config = load_with(file, &environment, None).unwrap();
        for model in &config.models {
            assert_eq!(model.prefix_cache.checkpoint_budget.bytes(), expected);
        }
        assert!(config.notices.env_overrides.iter().any(|entry| {
            entry.variable == "HIGGS_PREFIX_CHECKPOINT_BUDGET_MB"
                && entry.key == "models[].prefix_cache.checkpoint_budget"
                && entry.applied
        }));
    }
}

#[test]
fn an_unusable_checkpoint_budget_variable_is_reported_and_ignored() {
    let environment = Environment::from_pairs([("HIGGS_PREFIX_CHECKPOINT_BUDGET_MB", "lots")]);
    let config = load_with(
        "[[models]]\npath = \"org/a\"\n[models.prefix_cache]\ncheckpoint_budget = \"1GB\"\n",
        &environment,
        None,
    )
    .unwrap();
    assert_eq!(
        config.models[0].prefix_cache.checkpoint_budget,
        ByteSize::from_bytes(1 << 30)
    );
    assert!(
        config.notices.env_overrides.iter().any(|entry| {
            entry.variable == "HIGGS_PREFIX_CHECKPOINT_BUDGET_MB" && !entry.applied
        })
    );
}

#[test]
fn out_of_range_prefix_cache_sizes_are_rejected_at_load() {
    for (setting, needle) in [
        (
            "checkpoint_interval_tokens = 0",
            "checkpoint_interval_tokens",
        ),
        (
            "max_checkpoints_per_entry = 65",
            "max_checkpoints_per_entry",
        ),
        ("max_entries = 100000", "max_entries"),
    ] {
        let error = load(&format!(
            "[[models]]\npath = \"org/a\"\n[models.prefix_cache]\n{setting}\n"
        ))
        .unwrap_err();
        assert!(error.contains(needle), "{setting}: {error}");
    }
    assert!(
        load("[[models]]\npath = \"org/a\"\n[models.prefix_cache]\nmax_checkpoints_per_entry = 0\nmax_entries = 0\ncheckpoint_budget = \"0\"\n")
            .is_ok()
    );
}

// -- Environment references ---------------------------------------------------------------

#[test]
fn references_resolve_in_secret_and_path_keys() {
    let env = Environment::from_pairs([
        ("API", "sk-live"),
        ("PROV", "sk-prov"),
        ("MODELS", "/models"),
        ("HOME", "/home/me"),
    ]);
    let config = load_with(
        r#"
        [server]
        api_key = "${API}"
        cors_origins = ["https://${MISSING_HOST:-app.example.com}", "${API}"]
        [providers.p]
        url = "http://${MISSING_HOST:-localhost}:1"
        api_key = "Bearer ${PROV}"
        [[models]]
        path = "${MODELS}/qwen"
        [models.prefix_cache]
        disk_dir = "~/kv"
        [observability.metrics_log]
        path = "~/logs/m.jsonl"
        "#,
        &env,
        None,
    )
    .unwrap();
    assert_eq!(config.server.api_key.as_ref().unwrap().expose(), "sk-live");
    assert_eq!(
        config.server.cors_origins.as_deref(),
        Some(&["https://app.example.com".to_owned(), "sk-live".to_owned()][..])
    );
    assert_eq!(config.providers["p"].url, "http://localhost:1");
    assert_eq!(
        config.providers["p"].api_key.as_ref().unwrap().expose(),
        "Bearer sk-prov"
    );
    assert_eq!(model_of(&config).path, "/models/qwen");
    assert_eq!(
        model_of(&config).prefix_cache.disk_dir.as_deref(),
        Some("/home/me/kv")
    );
    assert_eq!(
        config.observability.metrics_log.path,
        "/home/me/logs/m.jsonl"
    );
}

#[test]
fn a_missing_variable_is_an_error_naming_the_key_and_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(
        dir.path(),
        "config.toml",
        &with_model("[server]\nport = 1\napi_key = \"${NOT_SET_ANYWHERE}\"\n"),
    );
    let error = load_config_file_in(&path, None, &Environment::default()).unwrap_err();
    assert!(error.contains("server.api_key"), "{error}");
    assert!(error.contains("NOT_SET_ANYWHERE"), "{error}");
    assert!(error.contains(":3:"), "{error}");
}

#[test]
fn resolved_secrets_never_appear_in_debug_output_or_errors() {
    let env = Environment::from_pairs([("API", "sk-super-secret-value")]);
    let config = load_with(&with_model("[server]\napi_key = \"${API}\"\n"), &env, None).unwrap();
    let rendered = format!("{config:?}");
    assert!(!rendered.contains("sk-super-secret-value"), "{rendered}");
    let error = load_with(
        &with_model("[server]\napi_key = \"${API}\"\nport = \"x\"\n"),
        &env,
        None,
    )
    .unwrap_err();
    assert!(!error.contains("sk-super-secret-value"), "{error}");
}

#[test]
fn a_literal_dollar_brace_can_be_escaped() {
    let config = load_ok(&with_model("[server]\napi_key = \"a$${B}c\"\n"));
    assert_eq!(config.server.api_key.as_ref().unwrap().expose(), "a${B}c");
}

#[test]
fn mcp_values_keep_their_references_for_the_server_that_uses_them() {
    let config = load_ok(&with_model(
        "[mcp.servers.fs]\ncommand = \"npx\"\nenv = { TOKEN = \"${TOKEN_NOT_SET_HERE}\" }\n",
    ));
    assert_eq!(
        config.mcp.servers["fs"].env["TOKEN"],
        "${TOKEN_NOT_SET_HERE}"
    );
}

// -- Includes -----------------------------------------------------------------------------

#[test]
fn includes_merge_relative_to_the_including_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    write(
        dir.path().join("sub").as_path(),
        "models.toml",
        "[[models]]\npath = \"org/included\"\n[server]\nport = 1\nhost = \"1.1.1.1\"\n",
    );
    write(
        dir.path(),
        "providers.toml",
        "[providers.p]\nurl = \"http://p\"\n[[routes]]\nmatch = \"p-.*\"\nprovider = \"p\"\n",
    );
    let main = write(
        dir.path(),
        "config.toml",
        "include = [\"sub/models.toml\", \"providers.toml\"]\n[server]\nport = 2\n[[models]]\npath = \"org/own\"\n[[routes]]\nmatch = \"own-.*\"\nprovider = \"p\"\n",
    );
    let config = load_config_file_in(&main, None, &Environment::default()).unwrap();
    assert_eq!(config.server.port, 2, "the including file wins");
    assert_eq!(
        config.server.host, "1.1.1.1",
        "untouched keys come from includes"
    );
    let paths: Vec<&str> = config.models.iter().map(|m| m.path.as_str()).collect();
    assert_eq!(
        paths,
        ["org/included", "org/own"],
        "arrays of tables append"
    );
    let matches: Vec<_> = config
        .routes
        .iter()
        .map(|r| r.pattern.clone().unwrap())
        .collect();
    assert_eq!(matches, ["p-.*", "own-.*"]);
    assert!(config.providers.contains_key("p"));
}

#[test]
fn a_provider_may_take_its_url_from_an_include_and_its_key_from_the_includer() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "providers.toml",
        "[providers.test]\nurl = \"http://upstream\"\n",
    );
    let main = write(
        dir.path(),
        "config.toml",
        "include = [\"providers.toml\"]\n[providers.test]\napi_key = \"secret\"\n",
    );
    let config = load_config_file_in(&main, None, &Environment::default()).unwrap();
    let provider = &config.providers["test"];
    assert_eq!(provider.url, "http://upstream");
    assert_eq!(provider.api_key.as_ref().unwrap().expose(), "secret");
}

#[test]
fn a_provider_still_missing_its_url_after_merging_is_reported_by_name() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "providers.toml",
        "[providers.test]\nformat = \"openai\"\n",
    );
    let main = write(
        dir.path(),
        "config.toml",
        "include = [\"providers.toml\"]\n[providers.test]\napi_key = \"secret\"\n",
    );
    let error = load_config_file_in(&main, None, &Environment::default()).unwrap_err();
    assert!(error.contains("providers.test"), "{error}");
    assert!(error.contains("missing field `url`"), "{error}");
}

#[test]
fn unknown_provider_keys_are_still_rejected_with_a_suggestion_when_layered() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "providers.toml",
        "[providers.test]\nurl = \"http://upstream\"\n",
    );
    let main = write(
        dir.path(),
        "config.toml",
        "include = [\"providers.toml\"]\n[providers.test]\napi_kye = \"secret\"\n",
    );
    let error = load_config_file_in(&main, None, &Environment::default()).unwrap_err();
    assert!(error.contains("unknown key `api_kye`"), "{error}");
    assert!(error.contains("did you mean `api_key`"), "{error}");
}

#[test]
fn includes_may_nest_and_later_files_override_earlier_ones() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "leaf.toml",
        "[server]\nport = 10\nmax_tokens = 5\n",
    );
    write(
        dir.path(),
        "a.toml",
        "include = [\"leaf.toml\"]\n[server]\nport = 20\n",
    );
    write(dir.path(), "b.toml", "[server]\nport = 30\n");
    let main = write(
        dir.path(),
        "config.toml",
        &with_model("include = [\"a.toml\", \"b.toml\"]\n"),
    );
    let config = load_config_file_in(&main, None, &Environment::default()).unwrap();
    assert_eq!(config.server.port, 30);
    assert_eq!(config.server.max_tokens, 5);
}

#[test]
fn a_missing_include_is_an_error_naming_the_file_and_line() {
    let dir = tempfile::tempdir().unwrap();
    let main = write(
        dir.path(),
        "config.toml",
        &with_model("include = [\"nope.toml\"]\n"),
    );
    let error = load_config_file_in(&main, None, &Environment::default()).unwrap_err();
    assert!(
        error.contains("nope.toml") && error.contains("does not exist"),
        "{error}"
    );
    assert!(error.contains(&format!("{}:1:", main.display())), "{error}");
}

#[test]
fn an_include_cycle_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.toml", "include = [\"b.toml\"]\n");
    write(dir.path(), "b.toml", "include = [\"a.toml\"]\n");
    let main = write(dir.path(), "config.toml", "include = [\"a.toml\"]\n");
    let error = load_config_file_in(&main, None, &Environment::default()).unwrap_err();
    assert!(error.contains("include cycle"), "{error}");
    assert!(
        error.contains("a.toml") && error.contains("b.toml"),
        "{error}"
    );
}

#[test]
fn a_problem_in_an_included_file_names_that_file() {
    let dir = tempfile::tempdir().unwrap();
    let included = write(dir.path(), "inc.toml", "[server]\nprot = 1\n");
    let main = write(
        dir.path(),
        "config.toml",
        &with_model("include = [\"inc.toml\"]\n"),
    );
    let error = load_config_file_in(&main, None, &Environment::default()).unwrap_err();
    assert!(
        error.contains(&format!("{}:2:", included.display())),
        "{error}"
    );
}

#[test]
fn deprecated_keys_in_included_files_warn_with_their_file() {
    let dir = tempfile::tempdir().unwrap();
    let included = write(dir.path(), "inc.toml", "[local]\nload = \"on_demand\"\n");
    let main = write(
        dir.path(),
        "config.toml",
        &with_model("include = [\"inc.toml\"]\n"),
    );
    let config = load_config_file_in(&main, None, &Environment::default()).unwrap();
    let deprecation = &config.notices.deprecations[0];
    assert_eq!(deprecation.file.as_deref(), Some(included.as_path()));
    assert_eq!(model_of(&config).load, LoadMode::OnDemand);
}

#[test]
fn include_must_be_an_array_of_strings() {
    assert!(
        load("include = \"x.toml\"\n")
            .unwrap_err()
            .contains("include")
    );
    assert!(load("include = [1]\n").unwrap_err().contains("include"));
}

// -- Names and aliases --------------------------------------------------------------------

#[test]
fn aliases_load_and_must_not_collide() {
    let config = load_ok(
        "[[models]]\npath = \"org/a\"\nname = \"a\"\naliases = [\"default\", \"x\"]\n[[models]]\npath = \"org/b\"\nname = \"b\"\naliases = [\"y\"]\n",
    );
    assert_eq!(config.models[0].aliases, ["default", "x"]);

    for (toml, needle) in [
        (
            "[[models]]\npath = \"org/a\"\nname = \"a\"\naliases = [\"b\"]\n[[models]]\npath = \"org/b\"\nname = \"b\"\n",
            "collides",
        ),
        (
            "[[models]]\npath = \"org/a\"\nname = \"a\"\naliases = [\"z\"]\n[[models]]\npath = \"org/b\"\nname = \"b\"\naliases = [\"z\"]\n",
            "collides",
        ),
        (
            "[[models]]\npath = \"org/a\"\nname = \"a\"\naliases = [\"a\"]\n",
            "own name",
        ),
        (
            "[[models]]\npath = \"org/a\"\nname = \"a\"\naliases = [\" \"]\n",
            "aliases must not be empty",
        ),
    ] {
        let error = load(toml).unwrap_err();
        assert!(error.contains(needle), "{toml}: {error}");
    }
}

// -- Validation carried over --------------------------------------------------------------

#[test]
fn semantic_problems_are_all_reported() {
    let error = load(
        "[memory]\nmax_loaded_models = 0\nlimit = \"0\"\n[server]\nrequest_timeout = \"0\"\n[[models]]\npath = \" \"\n",
    )
    .unwrap_err();
    assert!(error.contains("max_loaded_models"), "{error}");
    assert!(error.contains("memory.limit"), "{error}");
    assert!(error.contains("request_timeout"), "{error}");
    assert!(error.contains("path must not be empty"), "{error}");
}

#[test]
fn nothing_to_serve_is_an_error() {
    let error = load("[server]\nport = 1\n").unwrap_err();
    assert!(error.contains("at least one"), "{error}");
}

#[test]
fn the_auto_router_classifier_is_added_when_missing() {
    let config = load_ok(
        "[routing.auto]\nenabled = true\nmodel = \"org/router-model\"\n[[models]]\npath = \"org/chat\"\n",
    );
    assert_eq!(config.routing.auto.model, "router-model");
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.models[1].name.as_deref(), Some("router-model"));
}
