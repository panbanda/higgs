//! `higgs config explain`: every resolved setting and where its value came from.

use std::fmt::{self, Write as _};

use toml_edit::{Item, Table};

use super::cli::ServeArgs;
use super::env_ref;
use super::environment::Environment;
use super::load::{Layers, Loaded};
use super::settings::{HiggsConfig, ModelConfig};
use super::types::Secret;
use super::units::{ByteSize, HumanDuration};

/// Where a setting's value comes from, highest precedence first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Flag(String),
    Environment(String),
    Model,
    Defaults,
    ConfigFile,
    BuiltIn,
}

impl fmt::Display for Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flag(flag) => write!(formatter, "CLI flag {flag}"),
            Self::Environment(variable) => write!(formatter, "environment {variable}"),
            Self::Model => formatter.write_str("model entry"),
            Self::Defaults => formatter.write_str("[defaults]"),
            Self::ConfigFile => formatter.write_str("config file"),
            Self::BuiltIn => formatter.write_str("built-in default"),
        }
    }
}

/// One line of the explanation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    pub value: String,
    pub source: Source,
}

fn table_item<'a>(root: &'a Table, dotted: &str) -> Option<&'a Item> {
    let mut parts = dotted.split('.');
    let first = parts.next()?;
    let mut current = root.get(first)?;
    for part in parts {
        current = current.as_table()?.get(part)?;
    }
    Some(current)
}

fn environment_variable_for(dotted: &str) -> String {
    format!("HIGGS_{}", dotted.to_ascii_uppercase().replace('.', "__"))
}

/// The flag that sets a server-wide key, if there is one and it was given.
fn server_flag(dotted: &str, flags: &ServeArgs) -> Option<&'static str> {
    match dotted {
        "server.host" => flags.host.is_some().then_some("--host"),
        "server.port" => flags.port.is_some().then_some("--port"),
        "server.max_tokens" => flags.max_tokens.is_some().then_some("--max-tokens"),
        "server.api_key" => flags.api_key.is_some().then_some("--api-key"),
        "server.rate_limit" => flags.rate_limit.is_some().then_some("--rate-limit"),
        "server.request_timeout" => flags.timeout.is_some().then_some("--timeout"),
        "memory.guard" => flags.memory_guard.is_some().then_some("--memory-guard"),
        "memory.max_loaded_models" => flags
            .max_loaded_models
            .is_some()
            .then_some("--max-loaded-models"),
        _ => None,
    }
}

/// The flag that sets a per-model key at the `[defaults]` level.
fn defaults_flag(key: &str, flags: &ServeArgs) -> Option<&'static str> {
    match key {
        "load" => flags.load.is_some().then_some("--load"),
        "idle_ttl" => flags.idle_ttl.is_some().then_some("--idle-ttl"),
        "profile" => flags.mlx_profile.is_some().then_some("--mlx-profile"),
        _ => None,
    }
}

/// The flag that sets a key on a model given with `--model`.
fn cli_model_flag(key: &str, flags: &ServeArgs) -> Option<&'static str> {
    match key {
        "batch" => {
            if flags.batch {
                Some("--batch")
            } else if flags.no_batch {
                Some("--no-batch")
            } else {
                None
            }
        }
        "kv_cache.mode" => flags.kv_cache.is_some().then_some("--kv-cache"),
        "kv_cache.bits" => flags.kv_bits.is_some().then_some("--kv-bits"),
        "kv_cache.key_bits" => flags.kv_key_bits.is_some().then_some("--kv-key-bits"),
        "kv_cache.value_bits" => flags.kv_value_bits.is_some().then_some("--kv-value-bits"),
        "kv_cache.norm_correction" => flags
            .kv_no_norm_correction
            .then_some("--kv-no-norm-correction"),
        "kv_cache.adaptive_dense_layers" => flags
            .kv_adaptive_dense_layers
            .is_some()
            .then_some("--kv-adaptive-dense-layers"),
        "kv_cache.seed" => flags.kv_seed.is_some().then_some("--kv-seed"),
        _ => None,
    }
}

const EXPERIMENTAL_VARIABLES: &[(&str, &str)] = &[
    ("experimental.mtp", "HIGGS_MTP"),
    ("experimental.mtp_draft_tokens", "HIGGS_MTP_DRAFT_N_MAX"),
    (
        "experimental.mtp_adaptive_draft",
        "HIGGS_MTP_ADAPTIVE_DRAFT",
    ),
    ("experimental.mtp_prompt_lookup", "HIGGS_MTP_PROMPT_LOOKUP"),
    ("experimental.prompt_lookup", "HIGGS_PROMPT_LOOKUP"),
    (
        "experimental.clear_cache_after_prefill",
        "HIGGS_CLEAR_CACHE_AFTER_PREFILL",
    ),
    (
        "experimental.chunked_prefill.threshold",
        "HIGGS_CHUNKED_PREFILL_THRESHOLD",
    ),
    (
        "experimental.chunked_prefill.chunk",
        "HIGGS_CHUNKED_PREFILL_CHUNK_SIZE",
    ),
];

/// Answers "where did this value come from" against the documents a configuration loaded from.
pub struct Provenance<'a> {
    layers: &'a Layers,
    environment: &'a Environment,
    flags: Option<&'a ServeArgs>,
    config: &'a HiggsConfig,
}

impl<'a> Provenance<'a> {
    pub const fn new(
        loaded: &'a Loaded,
        environment: &'a Environment,
        flags: Option<&'a ServeArgs>,
    ) -> Self {
        Self {
            layers: &loaded.layers,
            environment,
            flags,
            config: &loaded.config,
        }
    }

    /// The files the configuration was read from: the main file, then its includes.
    pub fn files(&self) -> &[std::path::PathBuf] {
        &self.layers.files
    }

    fn file_root(&self) -> &Table {
        self.layers.file_document.as_table()
    }

    fn environment_root(&self) -> &Table {
        self.layers.environment_document.as_table()
    }

    /// The source of a server-wide key such as `server.port`.
    pub fn global(&self, dotted: &str) -> Source {
        if let Some(flag) = self.flags.and_then(|flags| server_flag(dotted, flags)) {
            return Source::Flag(flag.to_owned());
        }
        if let Some((_, variable)) = EXPERIMENTAL_VARIABLES
            .iter()
            .find(|(key, _)| *key == dotted)
        {
            if self
                .config
                .notices
                .env_overrides
                .iter()
                .any(|entry| entry.variable == *variable && entry.applied)
            {
                return Source::Environment((*variable).to_owned());
            }
        }
        if table_item(self.environment_root(), dotted).is_some() {
            return Source::Environment(environment_variable_for(dotted));
        }
        if table_item(self.file_root(), dotted).is_some() {
            return Source::ConfigFile;
        }
        Source::BuiltIn
    }

    fn model_row(&self, index: usize) -> Option<&Table> {
        match self.file_root().get("models") {
            Some(Item::ArrayOfTables(rows)) => rows.get(index),
            _ => None,
        }
    }

    fn defaults_table(&self) -> Option<&Table> {
        self.file_root().get("defaults").and_then(Item::as_table)
    }

    /// The source of a per-model key such as `kv_cache.bits` for the model at `index`.
    pub fn model(&self, index: usize, key: &str) -> Source {
        let is_file_model = index < self.layers.file_model_count;
        if !is_file_model {
            if let Some(flag) = self.flags.and_then(|flags| cli_model_flag(key, flags)) {
                return Source::Flag(flag.to_owned());
            }
        } else if self.model_row(index).is_some_and(|row| row_has(row, key)) {
            return Source::Model;
        }
        if let Some(flag) = self.flags.and_then(|flags| defaults_flag(key, flags)) {
            return Source::Flag(flag.to_owned());
        }
        if key == "profile" && self.environment.get("HIGGS_MLX_PROFILE").is_some() {
            return Source::Environment("HIGGS_MLX_PROFILE".to_owned());
        }
        if self
            .environment_root()
            .get("defaults")
            .and_then(Item::as_table)
            .is_some_and(|table| row_has(table, key))
        {
            return Source::Environment(environment_variable_for(&format!("defaults.{key}")));
        }
        if self
            .defaults_table()
            .is_some_and(|table| row_has(table, key))
        {
            return Source::Defaults;
        }
        Source::BuiltIn
    }

    /// Whether the model's own entry (not `[defaults]`) sets `key`.
    pub fn model_entry_sets(&self, index: usize, key: &str) -> bool {
        index < self.layers.file_model_count
            && self.model_row(index).is_some_and(|row| row_has(row, key))
    }

    /// The text written in the file for a string setting, `${VAR}` references intact.
    fn raw_string(&self, dotted: &str) -> Option<String> {
        [self.environment_root(), self.file_root()]
            .into_iter()
            .find_map(|root| {
                table_item(root, dotted)
                    .and_then(Item::as_value)
                    .and_then(toml_edit::Value::as_str)
                    .map(str::to_owned)
            })
    }

    fn raw_model_string(&self, index: usize, key: &str) -> Option<String> {
        if index >= self.layers.file_model_count {
            return None;
        }
        self.model_row(index)
            .into_iter()
            .chain(self.defaults_table())
            .find_map(|table| {
                table_item(table, key)
                    .and_then(Item::as_value)
                    .and_then(toml_edit::Value::as_str)
                    .map(str::to_owned)
            })
    }
}

fn row_has(table: &Table, dotted: &str) -> bool {
    table_item(table, dotted).is_some()
}

// -- Rendering --------------------------------------------------------------

fn quote(text: &str) -> String {
    format!("\"{text}\"")
}

fn show_secret(raw: Option<&str>, secret: &Secret) -> String {
    match raw {
        Some(text) if env_ref::has_reference(text) => quote(text),
        _ => quote(&secret.to_string()),
    }
}

fn show_raw_or(raw: Option<String>, resolved: &str) -> String {
    quote(&raw.unwrap_or_else(|| resolved.to_owned()))
}

fn show_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|item| quote(item)).collect();
    format!("[{}]", quoted.join(", "))
}

fn show_optional<T: fmt::Display>(value: Option<T>, none: &str) -> String {
    value.map_or_else(|| none.to_owned(), |inner| inner.to_string())
}

fn entry(key: &str, value: String, source: Source) -> Entry {
    Entry {
        key: key.to_owned(),
        value,
        source,
    }
}

/// The server-wide settings with their sources.
#[allow(clippy::too_many_lines)]
pub fn global_entries(provenance: &Provenance<'_>) -> Vec<Entry> {
    let config = provenance.config;
    let global = |key: &str, value: String| entry(key, value, provenance.global(key));
    let server = &config.server;
    let mut entries = vec![
        global(
            "server.host",
            show_raw_or(provenance.raw_string("server.host"), &server.host),
        ),
        global("server.port", server.port.to_string()),
        global(
            "server.api_key",
            server.api_key.as_ref().map_or_else(
                || "(none)".to_owned(),
                |key| show_secret(provenance.raw_string("server.api_key").as_deref(), key),
            ),
        ),
        global(
            "server.request_timeout",
            quote(&HumanDuration::new(server.request_timeout).to_string()),
        ),
        global(
            "server.max_body_size",
            quote(&server.max_body_size.to_string()),
        ),
        global("server.max_tokens", server.max_tokens.to_string()),
        global("server.rate_limit", server.rate_limit.to_string()),
        global(
            "server.cors_origins",
            server
                .cors_origins
                .as_ref()
                .map_or_else(|| "(none)".to_owned(), |origins| show_list(origins)),
        ),
    ];
    let memory = &config.memory;
    entries.extend([
        global("memory.guard", quote(memory.guard.as_str())),
        global(
            "memory.limit",
            memory
                .limit
                .map_or_else(|| "(none)".to_owned(), |limit| quote(&limit.to_string())),
        ),
        global(
            "memory.max_loaded_models",
            show_optional(memory.max_loaded_models, "(none)"),
        ),
        global(
            "memory.raise_wired_limit",
            memory.raise_wired_limit.to_string(),
        ),
    ]);
    let routing = &config.routing;
    entries.extend([
        global("routing.default", quote(&routing.default_provider)),
        global("routing.auto.enabled", routing.auto.enabled.to_string()),
        global("routing.auto.model", quote(&routing.auto.model)),
        global(
            "routing.auto.timeout",
            quote(&HumanDuration::new(routing.auto.timeout).to_string()),
        ),
        global("routing.auto.force", routing.auto.force.to_string()),
    ]);
    let observability = &config.observability;
    entries.extend([
        global(
            "observability.retention",
            observability.retention.map_or_else(
                || "false".to_owned(),
                |window| quote(&HumanDuration::new(window).to_string()),
            ),
        ),
        global(
            "observability.metrics_log.enabled",
            observability.metrics_log.enabled.to_string(),
        ),
        global(
            "observability.metrics_log.path",
            show_raw_or(
                provenance.raw_string("observability.metrics_log.path"),
                &observability.metrics_log.path,
            ),
        ),
        global(
            "observability.metrics_log.max_size",
            quote(&observability.metrics_log.max_size.to_string()),
        ),
        global(
            "observability.metrics_log.max_files",
            observability.metrics_log.max_files.to_string(),
        ),
    ]);
    let experimental = &config.experimental;
    entries.extend([
        global("experimental.mtp", quote(experimental.mtp.as_str())),
        global(
            "experimental.mtp_draft_tokens",
            show_optional(experimental.mtp_draft_tokens, "(profile)"),
        ),
        global(
            "experimental.mtp_adaptive_draft",
            experimental.mtp_adaptive_draft.to_string(),
        ),
        global(
            "experimental.mtp_prompt_lookup",
            experimental.mtp_prompt_lookup.to_string(),
        ),
        global(
            "experimental.prompt_lookup",
            experimental.prompt_lookup.to_string(),
        ),
        global(
            "experimental.clear_cache_after_prefill",
            show_optional(experimental.clear_cache_after_prefill, "(profile)"),
        ),
        global(
            "experimental.chunked_prefill.threshold",
            show_optional(experimental.chunked_prefill_threshold, "(profile)"),
        ),
        global(
            "experimental.chunked_prefill.chunk",
            show_optional(experimental.chunked_prefill_chunk, "(profile)"),
        ),
    ]);
    let web = &config.web;
    entries.extend([
        global("web.enabled", web.enabled.to_string()),
        global("web.search_provider", quote(&web.search_provider)),
        global(
            "web.timeout",
            quote(&HumanDuration::new(web.timeout).to_string()),
        ),
        global(
            "mcp.max_result_size",
            quote(
                &ByteSize::from_bytes(
                    u64::try_from(config.mcp.max_result_bytes).unwrap_or(u64::MAX),
                )
                .to_string(),
            ),
        ),
    ]);
    entries
}

/// The settings of one model with their sources.
pub fn model_entries(provenance: &Provenance<'_>, index: usize, model: &ModelConfig) -> Vec<Entry> {
    let source = |key: &str| provenance.model(index, key);
    let at = |key: &str, value: String| entry(key, value, source(key));
    let kv = &model.kv_cache;
    vec![
        entry(
            "path",
            show_raw_or(provenance.raw_model_string(index, "path"), &model.path),
            if index < provenance.layers.file_model_count {
                Source::Model
            } else {
                Source::Flag("--model".to_owned())
            },
        ),
        entry(
            "name",
            model
                .name
                .as_ref()
                .map_or_else(|| "(derived from the model)".to_owned(), |name| quote(name)),
            if model.name.is_some() {
                Source::Model
            } else {
                Source::BuiltIn
            },
        ),
        entry(
            "aliases",
            show_list(&model.aliases),
            if model.aliases.is_empty() {
                Source::BuiltIn
            } else {
                Source::Model
            },
        ),
        at("profile", quote(model.profile.as_str())),
        at("batch", quote(&model.batch.to_string())),
        at("load", quote(model.load.as_str())),
        at(
            "idle_ttl",
            model.idle_ttl.map_or_else(
                || "\"0s\" (never)".to_owned(),
                |ttl| quote(&HumanDuration::new(ttl).to_string()),
            ),
        ),
        at("pinned", model.pinned.to_string()),
        at("thinking_budget", model.thinking_budget.to_string()),
        at(
            "kv_cache.mode",
            quote(&format!("{:?}", kv.mode).to_lowercase()),
        ),
        at("kv_cache.bits", kv.bits.to_string()),
        at(
            "kv_cache.key_bits",
            show_optional(kv.key_bits_override, "(bits - 1)"),
        ),
        at(
            "kv_cache.value_bits",
            show_optional(kv.value_bits_override, "(bits)"),
        ),
        at("kv_cache.norm_correction", kv.norm_correction.to_string()),
        at(
            "kv_cache.adaptive_dense_layers",
            kv.adaptive_dense_layers.to_string(),
        ),
        at("kv_cache.seed", kv.seed.to_string()),
        at("kv_cache.mla_latent", kv.mla_latent.to_string()),
        at(
            "prefix_cache.disk_dir",
            model.prefix_cache.disk_dir.as_ref().map_or_else(
                || "(none)".to_owned(),
                |dir| {
                    show_raw_or(
                        provenance.raw_model_string(index, "prefix_cache.disk_dir"),
                        dir,
                    )
                },
            ),
        ),
        at(
            "prefix_cache.disk_budget",
            quote(&model.prefix_cache.disk_budget.to_string()),
        ),
        at(
            "prefill.yield_tokens",
            show_optional(model.prefill_yield_tokens, "(synchronous prefill)"),
        ),
    ]
}

fn render_entries(entries: &[Entry], indent: &str, out: &mut String) {
    let key_width = entries.iter().map(|e| e.key.len()).max().unwrap_or(0);
    let value_width = entries.iter().map(|e| e.value.len()).max().unwrap_or(0);
    for line in entries {
        let _ = writeln!(
            out,
            "{indent}{:<key_width$} = {:<value_width$}  ({})",
            line.key, line.value, line.source
        );
    }
}

/// Find the model a user means by name, alias, path or served name.
pub fn find_model<'a>(config: &'a HiggsConfig, wanted: &str) -> Option<(usize, &'a ModelConfig)> {
    config.models.iter().enumerate().find(|(_, model)| {
        model.name.as_deref() == Some(wanted)
            || model.path == wanted
            || model.aliases.iter().any(|alias| alias == wanted)
            || crate::engine_loader::served_name(model) == wanted
    })
}

/// The explanation as text: the whole server, or one model.
pub fn explain(
    loaded: &Loaded,
    environment: &Environment,
    flags: Option<&ServeArgs>,
    model: Option<&str>,
) -> Result<String, String> {
    let provenance = Provenance::new(loaded, environment, flags);
    let config = &loaded.config;
    let mut out = String::new();
    if let Some(wanted) = model {
        let (index, found) = find_model(config, wanted).ok_or_else(|| {
            let names: Vec<String> = config
                .models
                .iter()
                .map(crate::engine_loader::served_name)
                .collect();
            format!(
                "no model named '{wanted}'; configured models: {}",
                if names.is_empty() {
                    "(none)".to_owned()
                } else {
                    names.join(", ")
                }
            )
        })?;
        let _ = writeln!(
            out,
            "model \"{}\" ({})",
            crate::engine_loader::served_name(found),
            found.path
        );
        render_entries(&model_entries(&provenance, index, found), "  ", &mut out);
    } else {
        for file in &loaded.layers.files {
            let _ = writeln!(out, "config file: {}", file.display());
        }
        render_entries(&global_entries(&provenance), "", &mut out);
        if !config.models.is_empty() {
            out.push_str("\nmodels (explain one with `higgs config explain <name>`):\n");
            for configured in &config.models {
                let _ = writeln!(
                    out,
                    "  {}  {}",
                    crate::engine_loader::served_name(configured),
                    configured.path
                );
            }
        }
        if !config.providers.is_empty() {
            out.push_str("\nproviders:\n");
            for (name, provider) in &config.providers {
                let key = provider.api_key.as_ref().map_or_else(
                    || "no api_key".to_owned(),
                    |key| {
                        let raw = provenance.raw_string(&format!("providers.{name}.api_key"));
                        format!("api_key = {}", show_secret(raw.as_deref(), key))
                    },
                );
                let _ = writeln!(
                    out,
                    "  {name}: {} ({}, {key})",
                    provider.url,
                    provider.format.as_str()
                );
            }
        }
        if !config.notices.deprecations.is_empty() {
            let _ = writeln!(
                out,
                "\n{} deprecated key(s); run `higgs config migrate` to update the file",
                config.notices.deprecations.len()
            );
        }
    }
    Ok(out)
}
