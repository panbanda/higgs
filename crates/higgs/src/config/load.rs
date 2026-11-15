//! Loading a configuration: files and their includes, the environment overlay, expansion of
//! `${VAR}` references, typing, flags and resolution.

use std::path::{Path, PathBuf};

use figment::Figment;
use figment::providers::{Format, Toml};
use higgs_models::turboquant::KvCacheMode;
use toml_edit::{DocumentMut, Item, Table, Value};

use super::cli::ServeArgs;
use super::document::{
    ConfigError, Diagnostic, LineIndex, Segment, parse_document, render_path, table_at_mut,
};
use super::env_ref::{self, EnvRefError};
use super::environment::Environment;
use super::legacy::{Deprecation, Rewrite, find_matches};
use super::raw::{RawConfig, RawModel};
use super::schema::find_unknown_keys;
use super::settings::{HiggsConfig, ResolveContext, resolve};
use super::types::BatchMode;
use super::units::HumanDuration;
use super::validate;

/// Keys whose string values may contain `${VAR}` references.
const EXPANDED_KEYS: &[&str] = &[
    "server.host",
    "server.api_key",
    "server.cors_origins",
    "models[].path",
    "models[].prefix_cache.disk_dir",
    "defaults.prefix_cache.disk_dir",
    "providers.*.url",
    "providers.*.api_key",
    "observability.metrics_log.path",
    "web.searxng_url",
];

/// Keys that hold filesystem paths and understand a leading `~/`.
const PATH_KEYS: &[&str] = &[
    "models[].path",
    "models[].prefix_cache.disk_dir",
    "defaults.prefix_cache.disk_dir",
    "observability.metrics_log.path",
];

const ENVIRONMENT_PREFIX: &str = "HIGGS_";
const ENVIRONMENT_SEPARATOR: &str = "__";

/// The documents a configuration was resolved from, kept so `config explain` can say where each
/// value came from.
#[derive(Debug, Clone, Default)]
pub struct Layers {
    pub files: Vec<PathBuf>,
    /// Every file merged, deprecated keys rewritten, `${VAR}` references unexpanded.
    pub file_document: DocumentMut,
    /// `HIGGS_<SECTION>__<KEY>` variables as a document.
    pub environment_document: DocumentMut,
    /// How many `[[models]]` entries came from files; the rest were added by `--model`.
    pub file_model_count: usize,
}

#[derive(Debug, Clone)]
pub struct Loaded {
    pub config: HiggsConfig,
    pub layers: Layers,
}

/// What a load needs besides the file.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions<'a> {
    pub flags: Option<&'a ServeArgs>,
    pub environment: Environment,
}

// -- Merging ----------------------------------------------------------------

/// Merge `over` onto `base`: tables merge key by key, arrays of tables append, everything else
/// is replaced.
pub fn merge_tables(base: &mut Table, over: Table) {
    for (key, item) in over {
        match (base.get_mut(&key), item) {
            (Some(Item::Table(existing)), Item::Table(incoming)) => {
                merge_tables(existing, incoming);
            }
            (Some(Item::ArrayOfTables(existing)), Item::ArrayOfTables(incoming)) => {
                for row in incoming {
                    existing.push(row);
                }
            }
            (_, replacement) => {
                base.insert(&key, replacement);
            }
        }
    }
}

// -- Expansion --------------------------------------------------------------

fn expand_string(
    text: &str,
    environment: &Environment,
    tilde: bool,
) -> Result<String, EnvRefError> {
    let lookup = |name: &str| environment.lookup(name);
    let expanded = env_ref::expand(text, &lookup)?;
    if tilde {
        if let (Some(rest), Some(home)) = (
            expanded
                .strip_prefix("~/")
                .or_else(|| (expanded == "~").then_some("")),
            environment.get("HOME"),
        ) {
            return Ok(format!("{}/{rest}", home.trim_end_matches('/')));
        }
    }
    Ok(expanded)
}

fn replace_string(value: &mut Value, new_text: String) {
    let decor = value.decor().clone();
    *value = Value::from(new_text);
    *value.decor_mut() = decor;
}

/// Expand `${VAR}` references and `~/` in the keys that allow them. Returns each failure with the
/// path of the value that caused it.
pub fn expand_document(
    root: &mut Table,
    environment: &Environment,
) -> Vec<(Vec<Segment>, EnvRefError)> {
    let mut failures = Vec::new();
    for pattern in EXPANDED_KEYS {
        let tilde = PATH_KEYS.contains(pattern);
        for found in find_matches(root, pattern) {
            let Some((Segment::Key(leaf), parent_path)) = found.path.split_last() else {
                continue;
            };
            let Some(parent) = table_at_mut(root, parent_path) else {
                continue;
            };
            let Some(Item::Value(value)) = parent.get_mut(leaf) else {
                continue;
            };
            match value {
                Value::String(text) => match expand_string(text.value(), environment, tilde) {
                    Ok(expanded) => replace_string(value, expanded),
                    Err(error) => failures.push((found.path.clone(), error)),
                },
                Value::Array(array) => {
                    for (position, element) in array.iter_mut().enumerate() {
                        if let Value::String(text) = element {
                            match expand_string(text.value(), environment, tilde) {
                                Ok(expanded) => replace_string(element, expanded),
                                Err(error) => {
                                    let mut at = found.path.clone();
                                    at.push(Segment::Index(position));
                                    failures.push((at, error));
                                }
                            }
                        }
                    }
                }
                Value::Integer(_)
                | Value::Float(_)
                | Value::Boolean(_)
                | Value::Datetime(_)
                | Value::InlineTable(_) => {}
            }
        }
    }
    failures
}

fn expansion_diagnostic(
    file: Option<&Path>,
    lines: &LineIndex,
    path: &[Segment],
    error: &EnvRefError,
) -> Diagnostic {
    Diagnostic::new(
        file,
        lines.line_of(path),
        format!("`{}`: {error}", render_path(path)),
    )
}

// -- Typing -----------------------------------------------------------------

fn segments_of(figment_path: &[String]) -> Vec<Segment> {
    figment_path
        .iter()
        .map(|part| {
            part.parse::<usize>()
                .map_or_else(|_| Segment::key(part), Segment::Index)
        })
        .collect()
}

/// Deserialize a normalized document into the on-disk structs, locating each type error.
pub fn extract_raw(
    text: &str,
    file: Option<&Path>,
    lines: &LineIndex,
) -> Result<RawConfig, Vec<Diagnostic>> {
    Figment::new()
        .merge(Toml::string(text))
        .extract::<RawConfig>()
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|error| {
                    let path = segments_of(&error.path);
                    let location = render_path(&path);
                    let message = if location.is_empty() {
                        error.kind.to_string()
                    } else {
                        format!("`{location}`: {}", error.kind)
                    };
                    Diagnostic::new(file, lines.line_of(&path), message)
                })
                .collect()
        })
}

// -- Files and includes -----------------------------------------------------

struct FileState<'a> {
    environment: &'a Environment,
    errors: Vec<Diagnostic>,
    deprecations: Vec<Deprecation>,
    files: Vec<PathBuf>,
    stack: Vec<PathBuf>,
}

fn include_entries(
    table: &Table,
    file: &Path,
    lines: &LineIndex,
    state: &mut FileState<'_>,
) -> Vec<(String, Option<usize>)> {
    let Some(item) = table.get("include") else {
        return Vec::new();
    };
    let line = lines.line_of(&[Segment::key("include")]);
    let Some(array) = item.as_value().and_then(Value::as_array) else {
        state.errors.push(Diagnostic::new(
            Some(file),
            line,
            "`include` must be an array of file paths",
        ));
        return Vec::new();
    };
    let mut entries = Vec::new();
    for (position, element) in array.iter().enumerate() {
        let path = [Segment::key("include"), Segment::Index(position)];
        match element.as_str() {
            Some(text) => match expand_string(text, state.environment, true) {
                Ok(expanded) => entries.push((expanded, lines.line_of(&path).or(line))),
                Err(error) => {
                    state
                        .errors
                        .push(expansion_diagnostic(Some(file), lines, &path, &error));
                }
            },
            None => state.errors.push(Diagnostic::new(
                Some(file),
                line,
                "`include` entries must be strings",
            )),
        }
    }
    entries
}

#[allow(clippy::too_many_lines)]
fn load_file_tree(path: &Path, state: &mut FileState<'_>) -> Option<Table> {
    let canonical = match path.canonicalize() {
        Ok(canonical) => canonical,
        Err(error) => {
            state.errors.push(Diagnostic::new(
                Some(path),
                None,
                format!("cannot read config file: {error}"),
            ));
            return None;
        }
    };
    if state.stack.contains(&canonical) {
        let chain: Vec<String> = state
            .stack
            .iter()
            .chain(std::iter::once(&canonical))
            .map(|entry| entry.display().to_string())
            .collect();
        state.errors.push(Diagnostic::new(
            Some(path),
            None,
            format!("include cycle: {}", chain.join(" -> ")),
        ));
        return None;
    }
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            state.errors.push(Diagnostic::new(
                Some(path),
                None,
                format!("cannot read config file: {error}"),
            ));
            return None;
        }
    };
    let mut parsed = match parse_document(&source, Some(path)) {
        Ok(parsed) => parsed,
        Err(error) => {
            state.errors.extend(error.diagnostics);
            return None;
        }
    };
    state.files.push(path.to_path_buf());

    let (rewrite_errors, rewrite_deprecations) = {
        let mut rewrite = Rewrite::new(Some(path), &mut parsed.lines);
        rewrite.apply(&mut parsed.document);
        (rewrite.errors, rewrite.deprecations)
    };
    let mut failed = !rewrite_errors.is_empty();
    state.errors.extend(rewrite_errors);
    state.deprecations.extend(rewrite_deprecations);

    for unknown in find_unknown_keys(parsed.document.as_table()) {
        failed = true;
        state.errors.push(Diagnostic::new(
            Some(path),
            parsed.lines.line_of(&unknown.path),
            unknown.message,
        ));
    }
    if failed {
        return None;
    }

    let mut expanded = parsed.document.clone();
    for (failed_path, error) in expand_document(expanded.as_table_mut(), state.environment) {
        failed = true;
        state.errors.push(expansion_diagnostic(
            Some(path),
            &parsed.lines,
            &failed_path,
            &error,
        ));
    }
    if let Err(diagnostics) = extract_raw(&expanded.to_string(), Some(path), &parsed.lines) {
        state.errors.extend(diagnostics);
        failed = true;
    }
    if failed {
        return None;
    }

    let base_directory = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let entries = include_entries(parsed.document.as_table(), path, &parsed.lines, state);
    let mut merged = Table::new();
    state.stack.push(canonical);
    for (entry, line) in entries {
        let included = {
            let candidate = PathBuf::from(&entry);
            if candidate.is_absolute() {
                candidate
            } else {
                base_directory.join(candidate)
            }
        };
        if !included.exists() {
            state.errors.push(Diagnostic::new(
                Some(path),
                line,
                format!("included file {} does not exist", included.display()),
            ));
            continue;
        }
        if let Some(tree) = load_file_tree(&included, state) {
            merge_tables(&mut merged, tree);
        }
    }
    state.stack.pop();

    let mut own = parsed.document.into_table();
    own.remove("include");
    merge_tables(&mut merged, own);
    Some(merged)
}

// -- Environment overlay ----------------------------------------------------

const OVERLAY_ROOTS: &[&str] = &[
    "server",
    "memory",
    "defaults",
    "providers",
    "routing",
    "observability",
    "experimental",
    "mcp",
    "web",
    "local",
    "provider",
    "default",
    "auto_router",
    "logging",
    "retention",
];

/// Set `value` at the nested key `segments`, creating tables on the way. False when a segment
/// already holds a plain value.
fn insert_at(table: &mut Table, segments: &[String], value: Value) -> bool {
    let Some((first, rest)) = segments.split_first() else {
        return false;
    };
    if rest.is_empty() {
        table.insert(first, Item::Value(value));
        return true;
    }
    if !table.contains_key(first) {
        let mut child = Table::new();
        child.set_implicit(true);
        table.insert(first, Item::Table(child));
    }
    match table.get_mut(first) {
        Some(Item::Table(child)) => insert_at(child, rest, value),
        _ => false,
    }
}

/// `HIGGS_<SECTION>__<KEY>` variables as a document, so any key can be set from the environment.
fn environment_overlay(environment: &Environment) -> (DocumentMut, Vec<(String, String)>) {
    let mut document = DocumentMut::new();
    let mut applied = Vec::new();
    for (name, value) in environment.with_prefix(ENVIRONMENT_PREFIX) {
        let Some(rest) = name.strip_prefix(ENVIRONMENT_PREFIX) else {
            continue;
        };
        if !rest.contains(ENVIRONMENT_SEPARATOR) {
            continue;
        }
        let segments: Vec<String> = rest
            .split(ENVIRONMENT_SEPARATOR)
            .map(str::to_ascii_lowercase)
            .collect();
        if !segments
            .first()
            .is_some_and(|root| OVERLAY_ROOTS.contains(&root.as_str()))
        {
            continue;
        }
        let parsed = value
            .parse::<Value>()
            .unwrap_or_else(|_| Value::from(value));
        if insert_at(document.as_table_mut(), &segments, parsed) {
            applied.push((name.to_owned(), segments.join(".")));
        }
    }
    (document, applied)
}

// -- CLI models and flags ---------------------------------------------------

fn cli_models(args: &ServeArgs) -> Vec<RawModel> {
    args.models
        .iter()
        .map(|path| {
            let mut entry = RawModel {
                path: path.clone(),
                name: None,
                aliases: Vec::new(),
                settings: super::raw::RawModelSettings::default(),
            };
            let settings = &mut entry.settings;
            if args.batch || args.no_batch {
                settings.batch = Some(BatchMode::from_flags(args.batch, args.no_batch));
            }
            settings.kv_cache.mode = args.kv_cache.as_deref().and_then(|mode| match mode {
                "turboquant" => Some(KvCacheMode::Turboquant),
                "off" => Some(KvCacheMode::Off),
                _ => None,
            });
            settings.kv_cache.bits = args.kv_bits;
            settings.kv_cache.key_bits = args.kv_key_bits;
            settings.kv_cache.value_bits = args.kv_value_bits;
            if args.kv_no_norm_correction {
                settings.kv_cache.norm_correction = Some(false);
            }
            settings.kv_cache.adaptive_dense_layers = args.kv_adaptive_dense_layers;
            settings.kv_cache.seed = args.kv_seed;
            entry
        })
        .collect()
}

fn check_cli_kv_cache(mode: Option<&str>) -> Result<(), ConfigError> {
    match mode {
        None | Some("off" | "turboquant") => Ok(()),
        Some(other) => Err(ConfigError::message(format!(
            "unknown kv_cache mode '{other}'"
        ))),
    }
}

/// Apply `serve` flags to the typed file: server keys, lifecycle defaults and `--model` entries.
fn apply_flags(raw: &mut RawConfig, args: &ServeArgs) -> Result<(), ConfigError> {
    check_cli_kv_cache(args.kv_cache.as_deref())?;
    if let Some(host) = &args.host {
        raw.server.host = Some(host.clone());
    }
    if let Some(port) = args.port {
        raw.server.port = Some(port);
    }
    if let Some(max_tokens) = args.max_tokens {
        raw.server.max_tokens = Some(max_tokens);
    }
    if let Some(api_key) = &args.api_key {
        raw.server.api_key = Some(api_key.clone());
    }
    if let Some(rate_limit) = args.rate_limit {
        raw.server.rate_limit = Some(rate_limit);
    }
    if let Some(seconds) = args.timeout {
        let duration = std::time::Duration::try_from_secs_f64(seconds)
            .ok()
            .filter(|duration| !duration.is_zero())
            .ok_or_else(|| {
                ConfigError::message("server.request_timeout must be a finite, positive duration")
            })?;
        raw.server.request_timeout = Some(HumanDuration::new(duration));
    }
    if let Some(load) = args.load {
        raw.defaults.load = Some(load);
    }
    if let Some(seconds) = args.idle_ttl {
        raw.defaults.idle_ttl = Some(HumanDuration::new(std::time::Duration::from_secs(seconds)));
    }
    if let Some(guard) = args.memory_guard {
        raw.memory.guard = Some(guard);
    }
    if let Some(count) = args.max_loaded_models {
        raw.memory.max_loaded_models = Some(count);
    }
    raw.models.extend(cli_models(args));
    Ok(())
}

/// `HIGGS_HOST` and `HIGGS_PORT`, the bind address variables of simple mode.
fn apply_simple_mode_address(
    raw: &mut RawConfig,
    environment: &Environment,
) -> Result<(), ConfigError> {
    if let Some(value) = environment.get("HIGGS_HOST") {
        raw.server.host = Some(value.to_owned());
    }
    if let Some(value) = environment.get("HIGGS_PORT") {
        raw.server.port = Some(value.trim().parse().map_err(|_| {
            ConfigError::message("env overlay failed: HIGGS_PORT has an invalid value")
        })?);
    }
    Ok(())
}

/// The host and port `higgs serve --model` binds to when it has no config file.
pub fn simple_mode_address(environment: &Environment) -> Result<(String, u16), ConfigError> {
    let mut raw = RawConfig::default();
    apply_simple_mode_address(&mut raw, environment)?;
    Ok((
        raw.server
            .host
            .unwrap_or_else(|| super::settings::DEFAULT_HOST.to_owned()),
        raw.server.port.unwrap_or(super::settings::DEFAULT_PORT),
    ))
}

/// The server variables simple mode has always read (`HIGGS_PORT`, `HIGGS_API_KEY`, ...).
fn apply_simple_mode_environment(
    raw: &mut RawConfig,
    environment: &Environment,
) -> Result<(), ConfigError> {
    fn parse<T: std::str::FromStr>(name: &str, value: &str) -> Result<T, ConfigError> {
        value.trim().parse().map_err(|_| {
            ConfigError::message(format!("env overlay failed: {name} has an invalid value"))
        })
    }
    apply_simple_mode_address(raw, environment)?;
    if let Some(value) = environment.get("HIGGS_MAX_TOKENS") {
        raw.server.max_tokens = Some(parse("HIGGS_MAX_TOKENS", value)?);
    }
    if let Some(value) = environment.get("HIGGS_API_KEY") {
        raw.server.api_key = Some(value.to_owned());
    }
    if let Some(value) = environment.get("HIGGS_RATE_LIMIT") {
        raw.server.rate_limit = Some(parse("HIGGS_RATE_LIMIT", value)?);
    }
    if let Some(value) = environment.get("HIGGS_TIMEOUT") {
        let seconds: f64 = parse("HIGGS_TIMEOUT", value)?;
        let duration = std::time::Duration::try_from_secs_f64(seconds).map_err(|_| {
            ConfigError::message("env overlay failed: HIGGS_TIMEOUT must be a positive number")
        })?;
        raw.server.request_timeout = Some(HumanDuration::new(duration));
    }
    if let Some(value) = environment.get("HIGGS_MAX_BODY_SIZE") {
        raw.server.max_body_size = Some(value.trim().parse().map_err(|error| {
            ConfigError::message(format!("env overlay failed: HIGGS_MAX_BODY_SIZE: {error}"))
        })?);
    }
    if let Some(value) = environment.get("HIGGS_CORS_ORIGINS") {
        let origins: Vec<String> = serde_json::from_str(value).map_err(|_| {
            ConfigError::message(
                "env overlay failed: HIGGS_CORS_ORIGINS must be a JSON array of strings",
            )
        })?;
        raw.server.cors_origins = Some(origins);
    }
    Ok(())
}

// -- Assembly ---------------------------------------------------------------

fn profile_flag(args: Option<&ServeArgs>) -> Option<higgs_engine::mlx_tuning::RequestedMlxProfile> {
    args.and_then(|serve| serve.mlx_profile)
        .map(super::types::MlxProfile::to_requested)
}

/// Make sure the auto-router classifier is a configured model and referenced by name.
///
/// Mirrors `routes[].model`: always a name reference into the loaded models, never a raw path.
fn inject_auto_router_model(raw: &mut RawConfig) -> bool {
    let enabled = raw.routing.auto.enabled.unwrap_or(false);
    let Some(reference) = raw
        .routing
        .auto
        .model
        .clone()
        .filter(|model| !model.is_empty())
    else {
        return false;
    };
    if !enabled {
        return false;
    }
    if raw
        .models
        .iter()
        .any(|model| model.name.as_deref() == Some(reference.as_str()))
    {
        return false;
    }
    if let Some(model) = raw.models.iter_mut().find(|model| model.path == reference) {
        let name = model
            .name
            .get_or_insert_with(|| path_basename(&reference))
            .clone();
        raw.routing.auto.model = Some(name);
        return true;
    }
    let name = path_basename(&reference);
    raw.models.push(RawModel {
        path: reference,
        name: Some(name.clone()),
        aliases: Vec::new(),
        settings: super::raw::RawModelSettings::default(),
    });
    raw.routing.auto.model = Some(name);
    true
}

fn path_basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_owned()
}

/// Everything after the documents are merged and typed.
fn finish(
    mut raw: RawConfig,
    options: &LoadOptions<'_>,
    simple_mode: bool,
) -> Result<HiggsConfig, ConfigError> {
    if let Some(args) = options.flags {
        apply_flags(&mut raw, args)?;
    }
    let context = ResolveContext {
        environment: options.environment.clone(),
        profile_flag: profile_flag(options.flags),
    };
    let mut config = resolve(&raw, &context).map_err(ConfigError::message)?;
    let problems = validate::problems(&config, simple_mode);
    if !problems.is_empty() {
        return Err(ConfigError {
            diagnostics: problems.into_iter().map(Diagnostic::bare).collect(),
        });
    }
    if inject_auto_router_model(&mut raw) {
        let notices = std::mem::take(&mut config.notices);
        config = resolve(&raw, &context).map_err(ConfigError::message)?;
        config.notices = notices;
    }
    if let Some(path) = options.flags.and_then(|args| args.mcp_config.as_ref()) {
        config.mcp.merge_file(path).map_err(ConfigError::message)?;
    }
    Ok(config)
}

/// Load a configuration file with its includes, the environment overlay and `serve` flags.
#[allow(clippy::too_many_lines)]
pub fn load(path: &Path, options: &LoadOptions<'_>) -> Result<Loaded, ConfigError> {
    let mut state = FileState {
        environment: &options.environment,
        errors: Vec::new(),
        deprecations: Vec::new(),
        files: Vec::new(),
        stack: Vec::new(),
    };
    let tree = load_file_tree(path, &mut state);

    let (mut overlay, applied) = environment_overlay(&options.environment);
    let mut overlay_lines = LineIndex::default();
    let overlay_deprecations = {
        let mut rewrite = Rewrite::new(None, &mut overlay_lines);
        rewrite.apply(&mut overlay);
        state
            .errors
            .extend(rewrite.errors.into_iter().map(|diagnostic| {
                Diagnostic::bare(format!(
                    "{} (from the environment: {})",
                    diagnostic.message,
                    applied
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }));
        rewrite.deprecations
    };
    for unknown in find_unknown_keys(overlay.as_table()) {
        state.errors.push(Diagnostic::bare(format!(
            "{} (from an environment variable: {})",
            unknown.message,
            applied
                .iter()
                .find(|(_, key)| *key == render_path(&unknown.path))
                .map_or("HIGGS_*", |(name, _)| name.as_str())
        )));
    }
    state.deprecations.extend(overlay_deprecations);

    let Some(file_tree) = tree else {
        return Err(ConfigError {
            diagnostics: state.errors,
        });
    };
    if !state.errors.is_empty() {
        return Err(ConfigError {
            diagnostics: state.errors,
        });
    }

    let mut file_document = DocumentMut::new();
    *file_document.as_table_mut() = file_tree;
    let file_model_count = match file_document.get("models") {
        Some(Item::ArrayOfTables(rows)) => rows.len(),
        _ => 0,
    };
    let mut combined = file_document.clone();
    merge_tables(combined.as_table_mut(), overlay.clone().into_table());
    let failures = expand_document(combined.as_table_mut(), &options.environment);
    if !failures.is_empty() {
        return Err(ConfigError {
            diagnostics: failures
                .iter()
                .map(|(failed_path, error)| {
                    expansion_diagnostic(None, &LineIndex::default(), failed_path, error)
                })
                .collect(),
        });
    }
    let raw = extract_raw(&combined.to_string(), None, &LineIndex::default())
        .map_err(|diagnostics| ConfigError { diagnostics })?;
    let mut config = finish(raw, options, false)?;
    config.notices.deprecations = state.deprecations;
    Ok(Loaded {
        config,
        layers: Layers {
            files: state.files,
            file_document,
            environment_document: overlay,
            file_model_count,
        },
    })
}

/// Build a configuration from `serve` flags alone (simple mode, no config file).
pub fn load_simple(args: &ServeArgs, environment: &Environment) -> Result<Loaded, ConfigError> {
    let mut raw = RawConfig::default();
    apply_simple_mode_environment(&mut raw, environment)?;
    let options = LoadOptions {
        flags: Some(args),
        environment: environment.clone(),
    };
    let config = finish(raw, &options, true)?;
    Ok(Loaded {
        config,
        layers: Layers::default(),
    })
}

/// Parse and type-check a config file's text on its own, without includes, environment overlay
/// or semantic validation. Used by `config set` so one field can be edited at a time.
pub fn check_contents(contents: &str, source: &Path) -> Result<(), ConfigError> {
    let mut parsed = parse_document(contents, Some(source))?;
    let mut diagnostics = Vec::new();
    {
        let mut rewrite = Rewrite::new(Some(source), &mut parsed.lines);
        rewrite.apply(&mut parsed.document);
        diagnostics.extend(rewrite.errors);
    }
    for unknown in find_unknown_keys(parsed.document.as_table()) {
        diagnostics.push(Diagnostic::new(
            Some(source),
            parsed.lines.line_of(&unknown.path),
            unknown.message,
        ));
    }
    if diagnostics.is_empty() {
        if let Err(errors) = extract_raw(&parsed.document.to_string(), Some(source), &parsed.lines)
        {
            diagnostics.extend(errors);
        }
    }
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ConfigError { diagnostics })
    }
}
