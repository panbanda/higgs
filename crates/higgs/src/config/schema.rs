//! The keys the configuration file accepts, and the check that rejects every other one.
//!
//! This catalog is the authority on spelling; the typed structs in `raw` only read values. A test
//! pins the two together so a key cannot be accepted here and then ignored.

use toml_edit::{Item, Table};

use super::document::{Segment, render_path};
use super::legacy::{legacy_key_table, legacy_model_keys, model_key_replacement};
use super::suggest::{Candidate, unknown_key_message};

#[derive(Clone, Copy)]
pub enum Shape {
    /// A value.
    Leaf,
    /// A table with these keys.
    Table(&'static [Field]),
    /// A table of user-named tables that all have the same shape.
    Map(&'static Self),
    /// An array of tables (`[[name]]`) that all have these keys.
    Rows(&'static [Field]),
    /// A table whose keys are chosen by the user (environment variables, HTTP headers).
    Free,
}

#[derive(Clone, Copy)]
pub struct Field {
    pub name: &'static str,
    pub shape: Shape,
}

const fn leaf(name: &'static str) -> Field {
    Field {
        name,
        shape: Shape::Leaf,
    }
}

const fn table(name: &'static str, fields: &'static [Field]) -> Field {
    Field {
        name,
        shape: Shape::Table(fields),
    }
}

const fn free(name: &'static str) -> Field {
    Field {
        name,
        shape: Shape::Free,
    }
}

const SERVER: &[Field] = &[
    leaf("host"),
    leaf("port"),
    leaf("api_key"),
    leaf("request_timeout"),
    leaf("max_body_size"),
    leaf("max_tokens"),
    leaf("rate_limit"),
    leaf("cors_origins"),
];

const MEMORY: &[Field] = &[
    leaf("guard"),
    leaf("limit"),
    leaf("max_loaded_models"),
    leaf("raise_wired_limit"),
];

const KV_CACHE: &[Field] = &[
    leaf("mode"),
    leaf("bits"),
    leaf("key_bits"),
    leaf("value_bits"),
    leaf("norm_correction"),
    leaf("adaptive_dense_layers"),
    leaf("seed"),
    leaf("mla_latent"),
];

const PREFIX_CACHE: &[Field] = &[
    leaf("disk_dir"),
    leaf("disk_budget"),
    leaf("max_entries"),
    leaf("checkpoint_interval_tokens"),
    leaf("max_checkpoints_per_entry"),
    leaf("checkpoint_budget"),
];

const PREFILL: &[Field] = &[leaf("yield_tokens")];

macro_rules! model_setting_fields {
    ($($identity:expr),*) => {
        &[
            $($identity,)*
            leaf("load"),
            leaf("idle_ttl"),
            leaf("batch"),
            leaf("profile"),
            leaf("thinking_budget"),
            leaf("pinned"),
            table("kv_cache", KV_CACHE),
            table("prefix_cache", PREFIX_CACHE),
            table("prefill", PREFILL),
        ]
    };
}

const DEFAULTS: &[Field] = model_setting_fields!();
const MODEL: &[Field] = model_setting_fields!(leaf("path"), leaf("name"), leaf("aliases"));

const PROVIDER: &[Field] = &[
    leaf("url"),
    leaf("format"),
    leaf("api_key"),
    leaf("strip_auth"),
    leaf("stub_count_tokens"),
];

const ROUTE: &[Field] = &[
    leaf("name"),
    leaf("description"),
    leaf("match"),
    leaf("provider"),
    leaf("model"),
];

const ROUTING_AUTO: &[Field] = &[
    leaf("enabled"),
    leaf("model"),
    leaf("timeout"),
    leaf("force"),
];

const ROUTING: &[Field] = &[leaf("default"), table("auto", ROUTING_AUTO)];

const METRICS_LOG: &[Field] = &[
    leaf("enabled"),
    leaf("path"),
    leaf("max_size"),
    leaf("max_files"),
];

const OBSERVABILITY: &[Field] = &[leaf("retention"), table("metrics_log", METRICS_LOG)];

const CHUNKED_PREFILL: &[Field] = &[leaf("threshold"), leaf("chunk")];

const EXPERIMENTAL: &[Field] = &[
    leaf("mtp"),
    leaf("mtp_draft_tokens"),
    leaf("mtp_adaptive_draft"),
    leaf("mtp_prompt_lookup"),
    leaf("prompt_lookup"),
    leaf("clear_cache_after_prefill"),
    table("chunked_prefill", CHUNKED_PREFILL),
];

const MCP_SERVER: &[Field] = &[
    leaf("enabled"),
    leaf("transport"),
    leaf("type"),
    leaf("command"),
    leaf("args"),
    free("env"),
    leaf("cwd"),
    leaf("url"),
    free("headers"),
    leaf("timeout"),
];

const MCP: &[Field] = &[
    leaf("max_result_size"),
    Field {
        name: "servers",
        shape: Shape::Map(&Shape::Table(MCP_SERVER)),
    },
];

const WEB: &[Field] = &[
    leaf("enabled"),
    leaf("search_provider"),
    leaf("searxng_url"),
    leaf("allow_private_addresses"),
    leaf("expose_as_tools"),
    leaf("timeout"),
];

const ROOT: &[Field] = &[
    leaf("include"),
    table("server", SERVER),
    table("memory", MEMORY),
    table("defaults", DEFAULTS),
    Field {
        name: "models",
        shape: Shape::Rows(MODEL),
    },
    Field {
        name: "providers",
        shape: Shape::Map(&Shape::Table(PROVIDER)),
    },
    Field {
        name: "routes",
        shape: Shape::Rows(ROUTE),
    },
    table("routing", ROUTING),
    table("observability", OBSERVABILITY),
    table("experimental", EXPERIMENTAL),
    table("mcp", MCP),
    table("web", WEB),
];

pub const fn root_fields() -> &'static [Field] {
    ROOT
}

/// Every key path the schema defines, with `[]` and `*` for rows and user-chosen names.
pub fn all_leaf_patterns() -> Vec<String> {
    fn walk(fields: &[Field], prefix: &str, out: &mut Vec<String>) {
        for field in fields {
            let name = if prefix.is_empty() {
                field.name.to_owned()
            } else {
                format!("{prefix}.{}", field.name)
            };
            match field.shape {
                Shape::Leaf | Shape::Free => out.push(name),
                Shape::Table(children) => walk(children, &name, out),
                Shape::Rows(children) => walk(children, &format!("{name}[]"), out),
                Shape::Map(Shape::Table(children)) => walk(children, &format!("{name}.*"), out),
                Shape::Map(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(ROOT, "", &mut out);
    out
}

/// A problem with a key found by the check, at the path of the offending key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKey {
    pub path: Vec<Segment>,
    pub message: String,
}

/// Report every key in `root` that the schema does not define.
pub fn find_unknown_keys(root: &Table) -> Vec<UnknownKey> {
    let mut found = Vec::new();
    check_table(root, ROOT, "", &mut Vec::new(), &mut found);
    found
}

fn section_label(path: &[Segment], is_row: bool) -> String {
    if path.is_empty() {
        return "the top level".to_owned();
    }
    let keys: Vec<Segment> = path
        .iter()
        .filter(|segment| matches!(segment, Segment::Key(_)))
        .cloned()
        .collect();
    let rendered = render_path(&keys);
    if is_row {
        format!("[[{rendered}]]")
    } else {
        format!("[{rendered}]")
    }
}

/// Names that used to be valid in this kind of table, with what replaced them.
fn renamed_candidates(pattern: &str) -> Vec<Candidate> {
    if pattern == "models[]" || pattern == "defaults" {
        return legacy_model_keys()
            .into_iter()
            .filter_map(|old| model_key_replacement(old).map(|new| Candidate::renamed(old, new)))
            .collect();
    }
    let prefix = format!("{pattern}.");
    legacy_key_table()
        .into_iter()
        .filter_map(|(old, new)| {
            let old_leaf = old
                .strip_prefix(&prefix)
                .filter(|rest| !rest.contains('.'))?;
            let new_leaf = new
                .strip_prefix(&prefix)
                .or_else(|| new.rsplit('.').next())?;
            Some(Candidate::renamed(old_leaf, new_leaf))
        })
        .collect()
}

fn check_table(
    table: &Table,
    fields: &'static [Field],
    pattern: &str,
    path: &mut Vec<Segment>,
    found: &mut Vec<UnknownKey>,
) {
    for (name, item) in table {
        let Some(field) = fields.iter().find(|field| field.name == name) else {
            if !is_legacy_root(pattern, name) {
                let is_row = path
                    .last()
                    .is_some_and(|segment| matches!(segment, Segment::Index(_)));
                let mut candidates: Vec<Candidate> = fields
                    .iter()
                    .map(|field| Candidate::same(field.name))
                    .collect();
                candidates.extend(renamed_candidates(pattern));
                if pattern.is_empty() {
                    candidates.extend(
                        [
                            "local",
                            "provider",
                            "default",
                            "auto_router",
                            "logging",
                            "retention",
                        ]
                        .map(Candidate::same),
                    );
                }
                let mut at = path.clone();
                at.push(Segment::key(name));
                found.push(UnknownKey {
                    path: at,
                    message: unknown_key_message(name, &section_label(path, is_row), &candidates),
                });
            }
            continue;
        };
        path.push(Segment::key(name));
        descend(item, field.shape, pattern, name, path, found);
        path.pop();
    }
}

fn is_legacy_root(pattern: &str, name: &str) -> bool {
    pattern.is_empty()
        && [
            "local",
            "provider",
            "default",
            "auto_router",
            "logging",
            "retention",
        ]
        .contains(&name)
}

fn child_pattern(pattern: &str, name: &str) -> String {
    if pattern.is_empty() {
        name.to_owned()
    } else {
        format!("{pattern}.{name}")
    }
}

fn descend(
    item: &Item,
    shape: Shape,
    pattern: &str,
    name: &str,
    path: &mut Vec<Segment>,
    found: &mut Vec<UnknownKey>,
) {
    let here = child_pattern(pattern, name);
    match (shape, item) {
        (Shape::Table(children), Item::Table(child)) => {
            check_table(child, children, &here, path, found);
        }
        (Shape::Rows(children), Item::ArrayOfTables(rows)) => {
            for (index, row) in rows.iter().enumerate() {
                path.push(Segment::Index(index));
                check_table(row, children, &format!("{here}[]"), path, found);
                path.pop();
            }
        }
        (Shape::Map(inner), Item::Table(entries)) => {
            for (entry_name, entry) in entries {
                path.push(Segment::key(entry_name));
                descend(entry, *inner, &here, "*", path, found);
                path.pop();
            }
        }
        _ => {}
    }
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use toml_edit::DocumentMut;

    fn unknown(text: &str) -> Vec<UnknownKey> {
        let document: DocumentMut = text.parse().unwrap();
        find_unknown_keys(document.as_table())
    }

    #[test]
    fn a_valid_document_has_no_unknown_keys() {
        assert!(
            unknown("[server]\nport = 1\n[[models]]\npath = \"a\"\n[models.kv_cache]\nbits = 3\n")
                .is_empty()
        );
    }

    #[test]
    fn a_typo_names_the_key_section_and_closest_key() {
        let found = unknown("[defaults]\nidel_ttl = \"1m\"\n");
        assert_eq!(found.len(), 1);
        let message = &found.first().unwrap().message;
        assert!(message.contains("`idel_ttl`"), "{message}");
        assert!(message.contains("[defaults]"), "{message}");
        assert!(message.contains("did you mean `idle_ttl`"), "{message}");
    }

    #[test]
    fn a_typo_of_an_old_name_suggests_the_new_name() {
        let found = unknown("[[models]]\npath = \"a\"\nkv_bitz = 3\n");
        let message = &found.first().unwrap().message;
        assert!(message.contains("[[models]]"), "{message}");
        assert!(
            message.contains("did you mean `kv_cache.bits`"),
            "{message}"
        );
    }

    #[test]
    fn every_section_reports_its_unknown_keys() {
        for (text, section) in [
            ("bogus = 1\n", "the top level"),
            ("[server]\nbogus = 1\n", "[server]"),
            ("[memory]\nbogus = 1\n", "[memory]"),
            ("[defaults]\nbogus = 1\n", "[defaults]"),
            ("[[models]]\npath = \"a\"\nbogus = 1\n", "[[models]]"),
            (
                "[[models]]\npath = \"a\"\n[models.kv_cache]\nbogus = 1\n",
                "[models.kv_cache]",
            ),
            (
                "[[models]]\npath = \"a\"\n[models.prefix_cache]\nbogus = 1\n",
                "[models.prefix_cache]",
            ),
            (
                "[[models]]\npath = \"a\"\n[models.prefill]\nbogus = 1\n",
                "[models.prefill]",
            ),
            ("[providers.x]\nbogus = 1\n", "[providers.x]"),
            ("[[routes]]\nbogus = 1\n", "[[routes]]"),
            ("[routing]\nbogus = 1\n", "[routing]"),
            ("[routing.auto]\nbogus = 1\n", "[routing.auto]"),
            ("[observability]\nbogus = 1\n", "[observability]"),
            (
                "[observability.metrics_log]\nbogus = 1\n",
                "[observability.metrics_log]",
            ),
            ("[experimental]\nbogus = 1\n", "[experimental]"),
            (
                "[experimental.chunked_prefill]\nbogus = 1\n",
                "[experimental.chunked_prefill]",
            ),
            ("[mcp]\nbogus = 1\n", "[mcp]"),
            ("[mcp.servers.fs]\nbogus = 1\n", "[mcp.servers.fs]"),
            ("[web]\nbogus = 1\n", "[web]"),
        ] {
            let found = unknown(text);
            assert_eq!(found.len(), 1, "{text}");
            let message = &found.first().unwrap().message;
            assert!(message.contains("unknown key `bogus`"), "{text}: {message}");
            assert!(message.contains(section), "{text}: {message}");
        }
    }

    #[test]
    fn free_form_tables_accept_any_key() {
        assert!(unknown("[mcp.servers.fs]\ncommand = \"x\"\nenv = { ANY_NAME = \"1\" }\n[mcp.servers.fs.headers]\nX-Whatever = \"1\"\n").is_empty());
    }

    #[test]
    fn leaf_patterns_cover_the_whole_schema() {
        let patterns = all_leaf_patterns();
        for expected in [
            "server.request_timeout",
            "models[].kv_cache.mla_latent",
            "providers.*.api_key",
            "routes[].match",
            "routing.auto.timeout",
            "observability.metrics_log.max_size",
            "experimental.chunked_prefill.chunk",
            "mcp.servers.*.timeout",
            "web.timeout",
            "defaults.prefix_cache.disk_budget",
        ] {
            assert!(
                patterns.iter().any(|pattern| pattern == expected),
                "{expected}"
            );
        }
    }
}
