//! The pre-redesign schema, rewritten to the current one.
//!
//! Every old key is described once in [`RULES`]. The same rewrite serves the loader (which warns
//! and carries on with the new spelling) and `higgs config migrate` (which writes the new spelling
//! back to disk, keeping the user's comments because the rewrite edits the parsed document).

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use toml_edit::{Decor, DocumentMut, Item, Key, Table, Value};

use super::document::{Diagnostic, LineIndex, Segment, render_path, table_at, table_at_mut};
use super::suggest::{Candidate, unknown_key_message};
use super::units::{ByteSize, HumanDuration};

/// A deprecated key, found in a file, with the key that replaces it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deprecation {
    pub file: Option<PathBuf>,
    pub line: Option<usize>,
    pub old_key: String,
    pub new_key: String,
}

impl fmt::Display for Deprecation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let location = match (&self.file, self.line) {
            (Some(file), Some(line)) => format!("{}:{line}: ", file.display()),
            (Some(file), None) => format!("{}: ", file.display()),
            (None, Some(line)) => format!("line {line}: "),
            (None, None) => String::new(),
        };
        write!(
            formatter,
            "{location}deprecated key `{}`; use `{}`",
            self.old_key, self.new_key
        )
    }
}

/// How a legacy value becomes a value of the new key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Convert {
    Keep,
    Seconds,
    Milliseconds,
    Bytes,
    Mebibytes,
    Gibibytes,
}

struct Rule {
    old: &'static str,
    new: &'static str,
    convert: Convert,
}

const fn rule(old: &'static str, new: &'static str, convert: Convert) -> Rule {
    Rule { old, new, convert }
}

/// One row per legacy key. `*` stands for a map key (provider or MCP server name) and `[]` for a
/// row of an array of tables; each appears at the same position in `old` and `new`.
const RULES: &[Rule] = &[
    rule("server.timeout", "server.request_timeout", Convert::Seconds),
    rule("local.mlx_profile", "defaults.profile", Convert::Keep),
    rule(
        "local.raise_wired_limit",
        "memory.raise_wired_limit",
        Convert::Keep,
    ),
    rule(
        "local.thinking_budget",
        "defaults.thinking_budget",
        Convert::Keep,
    ),
    rule("local.load", "defaults.load", Convert::Keep),
    rule(
        "local.idle_ttl_seconds",
        "defaults.idle_ttl",
        Convert::Seconds,
    ),
    rule(
        "local.max_loaded_models",
        "memory.max_loaded_models",
        Convert::Keep,
    ),
    rule("local.memory_limit_gb", "memory.limit", Convert::Gibibytes),
    rule("local.memory_guard", "memory.guard", Convert::Keep),
    rule("models[].mlx_profile", "models[].profile", Convert::Keep),
    rule(
        "models[].idle_ttl_seconds",
        "models[].idle_ttl",
        Convert::Seconds,
    ),
    rule("models[].kv_bits", "models[].kv_cache.bits", Convert::Keep),
    rule(
        "models[].kv_key_bits",
        "models[].kv_cache.key_bits",
        Convert::Keep,
    ),
    rule(
        "models[].kv_value_bits",
        "models[].kv_cache.value_bits",
        Convert::Keep,
    ),
    rule(
        "models[].kv_norm_correction",
        "models[].kv_cache.norm_correction",
        Convert::Keep,
    ),
    rule(
        "models[].kv_adaptive_dense_layers",
        "models[].kv_cache.adaptive_dense_layers",
        Convert::Keep,
    ),
    rule("models[].kv_seed", "models[].kv_cache.seed", Convert::Keep),
    rule(
        "models[].mla_latent_cache",
        "models[].kv_cache.mla_latent",
        Convert::Keep,
    ),
    rule(
        "models[].kv_disk_dir",
        "models[].prefix_cache.disk_dir",
        Convert::Keep,
    ),
    rule(
        "models[].kv_disk_space_mb",
        "models[].prefix_cache.disk_budget",
        Convert::Mebibytes,
    ),
    rule(
        "models[].prefill_yield_tokens",
        "models[].prefill.yield_tokens",
        Convert::Keep,
    ),
    rule("provider.*.url", "providers.*.url", Convert::Keep),
    rule("provider.*.format", "providers.*.format", Convert::Keep),
    rule("provider.*.api_key", "providers.*.api_key", Convert::Keep),
    rule(
        "provider.*.strip_auth",
        "providers.*.strip_auth",
        Convert::Keep,
    ),
    rule(
        "provider.*.stub_count_tokens",
        "providers.*.stub_count_tokens",
        Convert::Keep,
    ),
    rule("routes[].pattern", "routes[].match", Convert::Keep),
    rule("default.provider", "routing.default", Convert::Keep),
    rule("auto_router.enabled", "routing.auto.enabled", Convert::Keep),
    rule("auto_router.force", "routing.auto.force", Convert::Keep),
    rule("auto_router.model", "routing.auto.model", Convert::Keep),
    rule(
        "auto_router.timeout_ms",
        "routing.auto.timeout",
        Convert::Milliseconds,
    ),
    rule(
        "logging.metrics.enabled",
        "observability.metrics_log.enabled",
        Convert::Keep,
    ),
    rule(
        "logging.metrics.path",
        "observability.metrics_log.path",
        Convert::Keep,
    ),
    rule(
        "logging.metrics.max_size_mb",
        "observability.metrics_log.max_size",
        Convert::Mebibytes,
    ),
    rule(
        "logging.metrics.max_files",
        "observability.metrics_log.max_files",
        Convert::Keep,
    ),
    rule(
        "mcp.max_result_bytes",
        "mcp.max_result_size",
        Convert::Bytes,
    ),
    rule(
        "mcp.servers.*.timeout_seconds",
        "mcp.servers.*.timeout",
        Convert::Seconds,
    ),
    rule("web.timeout_seconds", "web.timeout", Convert::Seconds),
];

const RETENTION_ENABLED: &str = "retention.enabled";
const RETENTION_MINUTES: &str = "retention.minutes";
const RETENTION_NEW: &str = "observability.retention";
const DEFAULT_RETENTION: &str = "60m";

/// Tables that exist only in the old schema: whatever the rules leave in one is an unknown key.
const LEGACY_ROOTS: &[&str] = &[
    "local",
    "provider",
    "default",
    "auto_router",
    "logging",
    "retention",
];

/// Keys whose tables hold arbitrary user-chosen entries and are never reshaped.
const FREE_FORM_KEYS: &[&str] = &["env", "headers"];

/// The names a legacy table accepts, for "did you mean" suggestions.
pub fn legacy_keys_under(table_pattern: &str) -> Vec<&'static str> {
    let prefix: Vec<&str> = table_pattern.split('.').collect();
    let mut names: Vec<&'static str> = Vec::new();
    for entry in RULES {
        let segments: Vec<&'static str> = entry.old.split('.').collect();
        if segments.len() > prefix.len()
            && segments
                .iter()
                .zip(&prefix)
                .all(|(left, right)| left == right)
        {
            if let Some(next) = segments.get(prefix.len()) {
                if !names.contains(next) {
                    names.push(next);
                }
            }
        }
    }
    if table_pattern == "retention" {
        names.extend(["enabled", "minutes"]);
    }
    names
}

/// The legacy names valid inside a `[[models]]` entry.
pub fn legacy_model_keys() -> Vec<&'static str> {
    let mut names = vec!["kv_cache"];
    names.extend(
        RULES
            .iter()
            .filter_map(|entry| entry.old.strip_prefix("models[]."))
            .filter(|name| !name.contains('.')),
    );
    names
}

/// The current name for a legacy per-model key, for messages about keys that are not valid where
/// they were written.
pub fn model_key_replacement(old: &str) -> Option<&'static str> {
    if old == "kv_cache" {
        return Some("kv_cache.mode");
    }
    RULES
        .iter()
        .find(|entry| entry.old.strip_prefix("models[].") == Some(old))
        .and_then(|entry| entry.new.strip_prefix("models[]."))
}

/// Every legacy key paired with the key that replaces it, for documentation and tests.
pub fn legacy_key_table() -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = RULES
        .iter()
        .map(|entry| (entry.old.to_owned(), entry.new.to_owned()))
        .collect();
    rows.push((
        "models[].kv_cache (string)".to_owned(),
        "models[].kv_cache.mode".to_owned(),
    ));
    rows.push((RETENTION_ENABLED.to_owned(), RETENTION_NEW.to_owned()));
    rows.push((RETENTION_MINUTES.to_owned(), RETENTION_NEW.to_owned()));
    rows
}

// -- Patterns ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step<'a> {
    Name(&'a str),
    Key,
    Row,
}

fn steps(pattern: &str) -> Vec<Step<'_>> {
    let mut result = Vec::new();
    for part in pattern.split('.') {
        if let Some(name) = part.strip_suffix("[]") {
            result.push(Step::Name(name));
            result.push(Step::Row);
        } else if part == "*" {
            result.push(Step::Key);
        } else {
            result.push(Step::Name(part));
        }
    }
    result
}

pub struct Match {
    pub path: Vec<Segment>,
    pub captures: Vec<Segment>,
}

/// Every place in `root` where the old pattern's leaf key exists.
pub fn find_matches(root: &Table, pattern: &str) -> Vec<Match> {
    fn recurse(
        table: &Table,
        remaining: &[Step<'_>],
        path: &mut Vec<Segment>,
        captures: &mut Vec<Segment>,
        found: &mut Vec<Match>,
    ) {
        let Some((step, rest)) = remaining.split_first() else {
            return;
        };
        match *step {
            Step::Name(name) if rest.is_empty() => {
                if table.contains_key(name) {
                    let mut leaf = path.clone();
                    leaf.push(Segment::key(name));
                    found.push(Match {
                        path: leaf,
                        captures: captures.clone(),
                    });
                }
            }
            Step::Name(name) => match (table.get(name), rest.split_first()) {
                (Some(Item::Table(child)), _) => {
                    path.push(Segment::key(name));
                    recurse(child, rest, path, captures, found);
                    path.pop();
                }
                (Some(Item::ArrayOfTables(rows)), Some((Step::Row, after_row))) => {
                    path.push(Segment::key(name));
                    for (index, row) in rows.iter().enumerate() {
                        path.push(Segment::Index(index));
                        captures.push(Segment::Index(index));
                        recurse(row, after_row, path, captures, found);
                        captures.pop();
                        path.pop();
                    }
                    path.pop();
                }
                _ => {}
            },
            Step::Key => {
                let names: Vec<String> = table.iter().map(|(name, _)| name.to_owned()).collect();
                for name in names {
                    if let Some(Item::Table(child)) = table.get(&name) {
                        path.push(Segment::Key(name.clone()));
                        captures.push(Segment::Key(name));
                        recurse(child, rest, path, captures, found);
                        captures.pop();
                        path.pop();
                    }
                }
            }
            Step::Row => {}
        }
    }

    let mut found = Vec::new();
    recurse(
        root,
        &steps(pattern),
        &mut Vec::new(),
        &mut Vec::new(),
        &mut found,
    );
    found
}

fn instantiate(pattern: &str, captures: &[Segment]) -> Vec<Segment> {
    let mut values = captures.iter();
    let mut path = Vec::new();
    for step in steps(pattern) {
        match step {
            Step::Name(name) => path.push(Segment::key(name)),
            Step::Key | Step::Row => {
                if let Some(value) = values.next() {
                    path.push(value.clone());
                }
            }
        }
    }
    path
}

// -- Document edits ---------------------------------------------------------

/// What a freshly created table inherits from the legacy table it replaces, so the header's
/// comments and its place in the file survive the rename.
#[derive(Clone)]
struct HeaderTemplate {
    decor: Decor,
    position: Option<isize>,
}

fn header_template(root: &Table, parent: &[Segment]) -> Option<HeaderTemplate> {
    let table = table_at(root, parent)?;
    Some(HeaderTemplate {
        decor: table.decor().clone(),
        position: table.position(),
    })
}

/// Make sure the table at `path` exists, creating the missing ones. Only the innermost created
/// table is shown with a header of its own; the ones above it are implicit.
fn ensure_table<'a>(
    table: &'a mut Table,
    path: &[Segment],
    template: Option<&HeaderTemplate>,
) -> Option<&'a mut Table> {
    let Some((first, rest)) = path.split_first() else {
        return Some(table);
    };
    let Segment::Key(name) = first else {
        return None;
    };
    if !table.contains_key(name) {
        let mut created = Table::new();
        if rest.is_empty() {
            if let Some(header) = template {
                *created.decor_mut() = header.decor.clone();
                created.set_position(header.position);
            }
        } else {
            created.set_implicit(true);
        }
        table.insert(name, Item::Table(created));
    }
    match table.get_mut(name)? {
        Item::Table(child) => {
            if rest.is_empty() && child.is_implicit() {
                if let Some(header) = template {
                    *child.decor_mut() = header.decor.clone();
                    child.set_implicit(false);
                }
            }
            ensure_table(child, rest, template)
        }
        Item::ArrayOfTables(rows) => {
            let (Segment::Index(index), after_index) = rest.split_first()? else {
                return None;
            };
            ensure_table(rows.get_mut(*index)?, after_index, template)
        }
        Item::Value(_) | Item::None => None,
    }
}

/// Remove and return every entry after `name`, so a renamed key can be re-inserted in its old
/// place and the entries that followed it put back after it.
fn take_entries_after(table: &mut Table, name: &str) -> Vec<(Key, Item)> {
    let following: Vec<String> = table
        .iter()
        .map(|(key, _)| key.to_owned())
        .skip_while(|key| key != name)
        .skip(1)
        .collect();
    following
        .iter()
        .filter_map(|key| table.remove_entry(key))
        .collect()
}

fn copy_key_decor(from: &Key, to: &mut Key) {
    *to.leaf_decor_mut() = from.leaf_decor().clone();
}

// -- Conversion -------------------------------------------------------------

fn conversion_error(old: &str, expectation: &str) -> String {
    format!("`{old}` must be {expectation}")
}

fn number_of(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(integer) => integer.value().to_string().parse::<f64>().ok(),
        Value::Float(float) => Some(*float.value()),
        Value::String(_)
        | Value::Boolean(_)
        | Value::Datetime(_)
        | Value::Array(_)
        | Value::InlineTable(_) => None,
    }
}

fn unsigned_of(value: &Value) -> Option<u64> {
    match value {
        Value::Integer(integer) => u64::try_from(*integer.value()).ok(),
        Value::String(_)
        | Value::Float(_)
        | Value::Boolean(_)
        | Value::Datetime(_)
        | Value::Array(_)
        | Value::InlineTable(_) => None,
    }
}

fn convert(value: &Value, how: Convert, old: &str) -> Result<Value, String> {
    let rendered: String = match how {
        Convert::Keep => return Ok(value.clone()),
        Convert::Seconds => {
            let seconds =
                number_of(value).ok_or_else(|| conversion_error(old, "a number of seconds"))?;
            let duration = std::time::Duration::try_from_secs_f64(seconds)
                .map_err(|_| conversion_error(old, "a non-negative, finite number of seconds"))?;
            HumanDuration::new(duration).to_string()
        }
        Convert::Milliseconds => {
            let millis = unsigned_of(value).ok_or_else(|| {
                conversion_error(old, "a non-negative whole number of milliseconds")
            })?;
            HumanDuration::new(std::time::Duration::from_millis(millis)).to_string()
        }
        Convert::Bytes => {
            let bytes = unsigned_of(value)
                .ok_or_else(|| conversion_error(old, "a non-negative whole number of bytes"))?;
            ByteSize::from_bytes(bytes).to_string()
        }
        Convert::Mebibytes => {
            let mebibytes = unsigned_of(value)
                .ok_or_else(|| conversion_error(old, "a non-negative whole number of MiB"))?;
            ByteSize::from_mebibytes(mebibytes).to_string()
        }
        Convert::Gibibytes => {
            let gibibytes = number_of(value)
                .filter(|gib| gib.is_finite() && *gib > 0.0)
                .ok_or_else(|| conversion_error(old, "a positive, finite number of GiB"))?;
            ByteSize::from_bytes(gibibytes_to_bytes(gibibytes)).to_string()
        }
    };
    let mut converted = Value::from(rendered);
    *converted.decor_mut() = value.decor().clone();
    Ok(converted)
}

/// GiB to whole bytes, saturating for absurdly large limits (which then never bind).
fn gibibytes_to_bytes(gibibytes: f64) -> u64 {
    let bytes = gibibytes * 1024.0 * 1024.0 * 1024.0;
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    {
        bytes as u64
    }
}

// -- The rewrite ------------------------------------------------------------

/// Collects what a rewrite found.
pub struct Rewrite<'a> {
    pub file: Option<&'a Path>,
    pub lines: &'a mut LineIndex,
    pub deprecations: Vec<Deprecation>,
    pub errors: Vec<Diagnostic>,
    claimed_headers: HashSet<String>,
}

impl<'a> Rewrite<'a> {
    pub fn new(file: Option<&'a Path>, lines: &'a mut LineIndex) -> Self {
        Self {
            file,
            lines,
            deprecations: Vec::new(),
            errors: Vec::new(),
            claimed_headers: HashSet::new(),
        }
    }

    fn deprecate(&mut self, old: &[Segment], new: &[Segment]) {
        self.deprecations.push(Deprecation {
            file: self.file.map(Path::to_path_buf),
            line: self.lines.line_of(old),
            old_key: render_path(old),
            new_key: render_path(new),
        });
        self.lines.alias(new, old);
    }

    fn fail(&mut self, at: &[Segment], message: String) {
        self.errors
            .push(Diagnostic::new(self.file, self.lines.line_of(at), message));
    }

    fn conflict(&mut self, old: &[Segment], new: &[Segment]) {
        let message = format!(
            "`{}` (deprecated) and `{}` both set the same thing; remove `{}`",
            render_path(old),
            render_path(new),
            render_path(old)
        );
        self.fail(old, message);
    }

    /// Move one value from `old` to `new`, converting it. Reports a conflict if `new` is set.
    fn relocate(
        &mut self,
        root: &mut Table,
        old: &[Segment],
        new: &[Segment],
        how: Convert,
    ) -> bool {
        let Some((leaf, old_parent)) = old.split_last() else {
            return false;
        };
        let (Segment::Key(leaf_name), Some((Segment::Key(new_leaf), new_parent))) =
            (leaf, new.split_last())
        else {
            return false;
        };
        if table_at(root, new_parent).is_some_and(|table| table.contains_key(new_leaf)) {
            self.conflict(old, new);
            return false;
        }
        let replaces_legacy_table = matches!(
            old_parent.first(),
            Some(Segment::Key(name)) if LEGACY_ROOTS.contains(&name.as_str())
        );
        let template = header_template(root, old_parent).filter(|_| replaces_legacy_table);
        let claim_key = render_path(old_parent);
        let following = if old_parent == new_parent {
            table_at_mut(root, old_parent)
                .map(|table| take_entries_after(table, leaf_name))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let Some((old_key, old_item)) =
            table_at_mut(root, old_parent).and_then(|table| table.remove_entry(leaf_name))
        else {
            return false;
        };
        let Item::Value(old_value) = &old_item else {
            self.fail(old, format!("`{}` must be a plain value", render_path(old)));
            return false;
        };
        let converted = match convert(old_value, how, &render_path(old)) {
            Ok(value) => value,
            Err(message) => {
                self.fail(old, message);
                return false;
            }
        };
        let claim = self.claimed_headers.insert(claim_key);
        let Some(parent) = ensure_table(root, new_parent, template.as_ref().filter(|_| claim))
        else {
            self.fail(
                old,
                format!(
                    "cannot move `{}` to `{}`: `{}` is not a table",
                    render_path(old),
                    render_path(new),
                    render_path(new_parent)
                ),
            );
            return false;
        };
        let mut new_key = Key::new(new_leaf);
        copy_key_decor(&old_key, &mut new_key);
        parent.insert_formatted(&new_key, Item::Value(converted));
        for (key, item) in following {
            parent.insert_formatted(&key, item);
        }
        self.deprecate(old, new);
        true
    }

    fn apply_rules(&mut self, root: &mut Table) {
        for entry in RULES {
            for found in find_matches(root, entry.old) {
                let new = instantiate(entry.new, &found.captures);
                self.relocate(root, &found.path, &new, entry.convert);
            }
        }
    }

    /// `kv_cache = "turboquant"` becomes `[models.kv_cache] mode = "turboquant"`.
    fn rewrite_kv_cache_modes(&mut self, root: &mut Table) {
        let rows = match root.get("models") {
            Some(Item::ArrayOfTables(rows)) => rows.len(),
            _ => return,
        };
        for index in 0..rows {
            let model_path = [Segment::key("models"), Segment::Index(index)];
            let Some(row) = table_at_mut(root, &model_path) else {
                continue;
            };
            let is_legacy = matches!(row.get("kv_cache"), Some(Item::Value(Value::String(_))));
            if !is_legacy {
                continue;
            }
            let Some((old_key, Item::Value(mode))) = row.remove_entry("kv_cache") else {
                continue;
            };
            let carried_comment = old_key
                .leaf_decor()
                .prefix()
                .and_then(|prefix| prefix.as_str())
                .unwrap_or_default()
                .to_owned();
            let mut table = Table::new();
            table.insert("mode", Item::Value(mode));
            *table.decor_mut() = Decor::new(format!("\n{carried_comment}"), "");
            row.insert("kv_cache", Item::Table(table));
            let old: Vec<Segment> = model_path
                .iter()
                .cloned()
                .chain([Segment::key("kv_cache")])
                .collect();
            let new: Vec<Segment> = old.iter().cloned().chain([Segment::key("mode")]).collect();
            self.deprecate(&old, &new);
        }
    }

    /// `[retention] enabled / minutes` collapse into the single key `observability.retention`.
    fn rewrite_retention(&mut self, root: &mut Table) {
        let Some(Item::Table(retention)) = root.get("retention") else {
            return;
        };
        let enabled_present = retention.contains_key("enabled");
        let minutes_present = retention.contains_key("minutes");
        if !enabled_present && !minutes_present {
            return;
        }
        let enabled = retention.get("enabled").and_then(Item::as_bool);
        let minutes = retention
            .get("minutes")
            .and_then(Item::as_integer)
            .and_then(|value| u64::try_from(value).ok());
        let enabled_path = [Segment::key("retention"), Segment::key("enabled")];
        let minutes_path = [Segment::key("retention"), Segment::key("minutes")];
        let new_path = [Segment::key("observability"), Segment::key("retention")];
        if enabled_present && enabled.is_none() {
            self.fail(
                &enabled_path,
                conversion_error(RETENTION_ENABLED, "true or false"),
            );
            return;
        }
        if minutes_present && minutes.is_none() {
            self.fail(
                &minutes_path,
                conversion_error(RETENTION_MINUTES, "a non-negative whole number of minutes"),
            );
            return;
        }
        let observability_path = [Segment::key("observability")];
        if table_at(root, &observability_path).is_some_and(|table| table.contains_key("retention"))
        {
            if enabled_present {
                self.conflict(&enabled_path, &new_path);
            }
            if minutes_present {
                self.conflict(&minutes_path, &new_path);
            }
            return;
        }
        let value: Value = match (enabled, minutes) {
            (Some(false), _) => Value::from(false),
            (_, Some(window_minutes)) => Value::from(
                HumanDuration::new(std::time::Duration::from_secs(
                    window_minutes.saturating_mul(60),
                ))
                .to_string(),
            ),
            (_, None) => Value::from(DEFAULT_RETENTION),
        };
        let retention_template = header_template(root, &[Segment::key("retention")]);
        let first_key_decor = {
            let Some(Item::Table(table)) = root.get("retention") else {
                return;
            };
            let first = if enabled_present {
                "enabled"
            } else {
                "minutes"
            };
            table.key(first).map(|key| key.leaf_decor().clone())
        };
        if let Some(Item::Table(table)) = root.get_mut("retention") {
            table.remove("enabled");
            table.remove("minutes");
        }
        let Some(parent) = ensure_table(root, &observability_path, retention_template.as_ref())
        else {
            return;
        };
        let mut key = Key::new("retention");
        if let Some(decor) = first_key_decor {
            *key.leaf_decor_mut() = decor;
        }
        parent.insert_formatted(&key, Item::Value(value));
        if enabled_present {
            self.deprecate(&enabled_path, &new_path);
        }
        if minutes_present {
            self.deprecate(&minutes_path, &new_path);
        }
    }

    /// Whatever remains in a legacy-only table is a key nobody defined.
    fn report_leftovers(&mut self, root: &mut Table) {
        for legacy_root in LEGACY_ROOTS {
            let mut leftovers: Vec<(Vec<Segment>, String, String)> = Vec::new();
            if let Some(Item::Table(table)) = root.get(legacy_root) {
                collect_leftovers(
                    table,
                    legacy_root,
                    &mut vec![Segment::key(legacy_root)],
                    &mut leftovers,
                );
            } else if let Some(item) = root.get(legacy_root) {
                if !item.is_table() {
                    self.fail(
                        &[Segment::key(legacy_root)],
                        format!("`{legacy_root}` must be a table"),
                    );
                    root.remove(legacy_root);
                    continue;
                }
            }
            for (path, pattern, name) in leftovers {
                let valid: Vec<Candidate> = legacy_keys_under(&pattern)
                    .into_iter()
                    .map(Candidate::same)
                    .collect();
                let section = render_path(path.split_last().map_or(&[][..], |(_, rest)| rest));
                let message = unknown_key_message(&name, &format!("[{section}]"), &valid);
                self.fail(&path, message);
            }
            root.remove(legacy_root);
        }
    }

    pub fn apply(&mut self, document: &mut DocumentMut) {
        let root = document.as_table_mut();
        tablify(root);
        self.rewrite_kv_cache_modes(root);
        self.apply_rules(root);
        self.rewrite_retention(root);
        self.report_leftovers(root);
        self.deprecations
            .sort_by_key(|deprecation| deprecation.line);
    }
}

/// The table pattern a leftover key is judged against: provider names are free, `metrics` is the
/// one table `logging` knows.
fn collect_leftovers(
    table: &Table,
    pattern: &str,
    path: &mut Vec<Segment>,
    out: &mut Vec<(Vec<Segment>, String, String)>,
) {
    for (name, item) in table {
        path.push(Segment::key(name));
        let container_pattern = match (pattern, name) {
            ("provider", _) => Some("provider.*".to_owned()),
            ("logging", "metrics") => Some("logging.metrics".to_owned()),
            _ => None,
        };
        match (container_pattern, item) {
            (Some(child_pattern), Item::Table(child)) => {
                collect_leftovers(child, &child_pattern, path, out);
            }
            _ => out.push((path.clone(), pattern.to_owned(), name.to_owned())),
        }
        path.pop();
    }
}

/// Turn inline tables and arrays of inline tables into standard tables, so the rest of the
/// rewrite sees one shape. Free-form maps (`env`, `headers`) are left alone.
fn tablify(table: &mut Table) {
    for (mut key, item) in table.iter_mut() {
        let _ = &mut key;
        if FREE_FORM_KEYS.contains(&key.get()) {
            continue;
        }
        if matches!(item, Item::Value(Value::InlineTable(_))) {
            let taken = std::mem::take(item);
            *item = match taken.into_table() {
                Ok(converted) => Item::Table(converted),
                Err(original) => original,
            };
        } else if matches!(item, Item::Value(Value::Array(array)) if !array.is_empty() && array.iter().all(Value::is_inline_table))
        {
            let taken = std::mem::take(item);
            *item = match taken.into_array_of_tables() {
                Ok(converted) => Item::ArrayOfTables(converted),
                Err(original) => original,
            };
        }
        match item {
            Item::Table(child) => tablify(child),
            Item::ArrayOfTables(rows) => {
                for row in rows.iter_mut() {
                    tablify(row);
                }
            }
            Item::Value(_) | Item::None => {}
        }
    }
}
