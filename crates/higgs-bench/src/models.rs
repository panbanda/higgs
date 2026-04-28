//! Model manifest loader for `benchmarks/models.toml`.
//!
//! Each `[[models]]` entry describes one model that the benches can target.
//! Adding a new model is one TOML entry; benches can filter by tag.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One model entry from `benchmarks/models.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique key passed to `--model` on bench CLIs.
    pub key: String,
    /// Human-readable display name (used in summary tables).
    pub label: String,
    /// Model path: `HuggingFace` repo ID or absolute local path.
    pub path: String,
    /// Quantization format, e.g. `4bit`, `3bit`.
    pub quantization: String,
    /// Approximate model size in gigabytes.
    pub approx_size_gb: f64,
    /// Maximum context length in tokens.
    pub context: u32,
    /// Optional tags for filtering, e.g. `small`, `dense`, `moe`, `h2h`.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Local directory containing this model in oMLX's expected layout
    /// (parent of the model directory). oMLX's `--model-dir` walks one
    /// level deep and rejects `HuggingFace` repo IDs, so `bench_h2h`
    /// requires this for any model tagged `h2h`. Leading `~` is expanded
    /// against `$HOME` at load time.
    #[serde(default)]
    pub omlx_model_dir: Option<PathBuf>,
}

impl Model {
    /// Returns `omlx_model_dir` with a leading `~` expanded to `$HOME`.
    #[must_use]
    pub fn resolved_omlx_model_dir(&self) -> Option<PathBuf> {
        self.omlx_model_dir.as_ref().map(|p| expand_tilde(p))
    }
}

fn expand_tilde(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if s == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    p.to_path_buf()
}

/// Top-level structure of `benchmarks/models.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// All model entries declared in the manifest.
    #[serde(default)]
    pub models: Vec<Model>,
}

impl Manifest {
    #[must_use]
    pub fn find_by_key(&self, key: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.key == key)
    }

    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|m| m.tags.iter().any(|t| t == tag))
            .collect()
    }
}

pub fn load_manifest(path: &Path) -> Result<Manifest> {
    let body = fs::read_to_string(path)
        .with_context(|| format!("read model manifest at {}", path.display()))?;
    let manifest: Manifest = toml::from_str(&body).context("parse model manifest TOML")?;
    Ok(manifest)
}

/// Convenience for binaries: load the manifest at `path` and look up `key`.
pub fn find_by_key(path: &Path, key: &str) -> Result<Model> {
    let manifest = load_manifest(path)?;
    manifest
        .find_by_key(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model key '{key}' not found in {}", path.display()))
}

/// Convenience for binaries: load the manifest at `path` and filter by `tag`.
pub fn find_by_tag(path: &Path, tag: &str) -> Result<Vec<Model>> {
    let manifest = load_manifest(path)?;
    Ok(manifest.find_by_tag(tag).into_iter().cloned().collect())
}
