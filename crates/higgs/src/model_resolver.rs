use std::path::{Path, PathBuf};

/// Resolve a model specifier to a concrete directory path.
///
/// 1. If `path` is an existing directory, returns it directly.
/// 2. If it looks like `org/name`, resolves from the `HuggingFace` cache
///    (`~/.cache/huggingface/hub/models--org--name/snapshots/<hash>`).
pub fn resolve(path: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(path);

    let expanded = path.strip_prefix("~/").map_or_else(
        || as_path.to_path_buf(),
        |rest| {
            directories::BaseDirs::new()
                .map_or_else(|| as_path.to_path_buf(), |d| d.home_dir().join(rest))
        },
    );

    if expanded.is_dir() {
        return Ok(expanded);
    }

    // Tilde paths are explicit filesystem references, not HF model IDs.
    if path.starts_with("~/") {
        return Err(format!("model directory not found: {}", expanded.display()));
    }

    resolve_with_cache(path, default_hf_cache().as_deref())
}

/// Returns `true` if `s` looks like a `org/name` `HuggingFace` model ID.
pub fn is_hf_model_id(s: &str) -> bool {
    if s.starts_with("~/") || s.starts_with('/') {
        return false;
    }
    matches!(s.split_once('/'), Some((org, name)) if !org.is_empty() && !name.is_empty() && !name.contains('/'))
}

/// Testable resolver with explicit cache root.
fn resolve_with_cache(path: &str, cache_root: Option<&Path>) -> Result<PathBuf, String> {
    let as_path = Path::new(path);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    if is_hf_model_id(path) {
        if let (Some((org, name)), Some(cache)) = (path.split_once('/'), cache_root) {
            return resolve_hf_snapshot(cache, org, name);
        }
    }

    Err(format!(
        "model '{path}' is not an existing directory and was not found in the HuggingFace cache"
    ))
}

/// Read `refs/main` and resolve to the snapshot directory.
/// Only the default revision (`main`) is supported; models downloaded at a
/// specific revision or branch will not be found.
fn resolve_hf_snapshot(cache_root: &Path, org: &str, name: &str) -> Result<PathBuf, String> {
    let model_dir = cache_root.join(format!("models--{org}--{name}"));
    let ref_path = model_dir.join("refs").join("main");
    let hash = std::fs::read_to_string(&ref_path)
        .map_err(|e| format!("could not read HF cache ref for '{org}/{name}': {e}"))?
        .trim()
        .to_owned();

    if hash.is_empty() {
        return Err(format!("empty ref in HF cache for '{org}/{name}'"));
    }

    let snapshot_dir = model_dir.join("snapshots").join(&hash);
    if snapshot_dir.is_dir() {
        Ok(snapshot_dir)
    } else {
        Err(format!(
            "snapshot directory missing for '{org}/{name}' (hash: {hash})"
        ))
    }
}

fn default_hf_cache() -> Option<PathBuf> {
    let env = |key| std::env::var(key).ok();
    hf_cache_from_env(
        env("HF_HUB_CACHE").as_deref(),
        env("HUGGINGFACE_HUB_CACHE").as_deref(),
        env("HF_HOME").as_deref(),
    )
    .or_else(|| {
        directories::BaseDirs::new()
            .map(|d| d.home_dir().join(".cache").join("huggingface").join("hub"))
    })
}

/// Testable env var resolution without reading actual environment.
///
/// Resolution order matches the `HuggingFace` Python SDK:
/// 1. `HF_HUB_CACHE`          (direct cache path)
/// 2. `HUGGINGFACE_HUB_CACHE`  (legacy alias)
/// 3. `HF_HOME` + `/hub`       (home override)
///
/// Empty or whitespace-only values are treated as unset.
fn hf_cache_from_env(
    hub_cache: Option<&str>,
    legacy_cache: Option<&str>,
    hf_home: Option<&str>,
) -> Option<PathBuf> {
    fn non_empty(v: Option<&str>) -> Option<&str> {
        v.filter(|s| !s.trim().is_empty())
    }

    if let Some(cache) = non_empty(hub_cache) {
        return Some(PathBuf::from(cache));
    }
    if let Some(cache) = non_empty(legacy_cache) {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = non_empty(hf_home) {
        return Some(PathBuf::from(home).join("hub"));
    }
    None
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn create_hf_cache(root: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_dir = root.join(format!("models--{org}--{name}"));
        let refs_dir = model_dir.join("refs");
        let snapshot_dir = model_dir.join("snapshots").join(hash);
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(refs_dir.join("main"), hash).unwrap();
        snapshot_dir
    }

    #[test]
    fn test_resolve_existing_directory_returns_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let result = resolve_with_cache(dir.path().to_str().unwrap(), None);
        assert_eq!(result.unwrap(), dir.path());
    }

    #[test]
    fn test_resolve_hf_model_id_from_cache() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "mlx-community", "Qwen3-4bit", "abc123");
        let result = resolve_with_cache("mlx-community/Qwen3-4bit", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_resolve_hf_model_id_not_in_cache_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let result = resolve_with_cache("no-org/NoModel", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_empty_refs_main_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "  \n").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_snapshot_dir_missing_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "deadbeef").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_nonexistent_plain_path_is_err() {
        let result = resolve_with_cache("/nonexistent/path/to/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_no_cache_for_hf_id_is_err() {
        let result = resolve_with_cache("org/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_tilde_expansion_nonexistent_is_err() {
        let result = resolve("~/nonexistent_model_dir_12345");
        assert!(result.is_err());
    }

    // --- hf_cache_from_env tests ---

    #[test]
    fn test_hf_cache_from_env_hf_hub_cache_takes_priority() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(
            Some(dir.path().to_str().unwrap()),
            Some("/legacy"),
            Some("/home"),
        );
        assert_eq!(result, Some(dir.path().to_path_buf()));
    }

    #[test]
    fn test_hf_cache_from_env_legacy_over_hf_home() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(None, Some(dir.path().to_str().unwrap()), Some("/home"));
        assert_eq!(result, Some(dir.path().to_path_buf()));
    }

    #[test]
    fn test_hf_cache_from_env_hf_home_appends_hub() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(None, None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(dir.path().join("hub")));
    }

    #[test]
    fn test_hf_cache_from_env_none_when_all_unset() {
        let result = hf_cache_from_env(None, None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_hf_cache_from_env_empty_string_ignored() {
        let result = hf_cache_from_env(Some(""), Some(""), Some(""));
        assert!(result.is_none());
    }

    #[test]
    fn test_hf_cache_from_env_whitespace_only_ignored() {
        let result = hf_cache_from_env(Some("  "), None, None);
        assert!(result.is_none());
    }

    // --- is_hf_model_id tests ---

    #[test]
    fn test_is_hf_model_id_valid() {
        assert!(is_hf_model_id("org/model"));
        assert!(is_hf_model_id("mlx-community/Qwen3-4bit"));
    }

    #[test]
    fn test_is_hf_model_id_tilde_path_is_false() {
        assert!(!is_hf_model_id("~/models/foo"));
    }

    #[test]
    fn test_is_hf_model_id_absolute_path_is_false() {
        assert!(!is_hf_model_id("/some/absolute/path"));
    }

    #[test]
    fn test_is_hf_model_id_nested_slash_is_false() {
        assert!(!is_hf_model_id("org/name/extra"));
    }

    #[test]
    fn test_is_hf_model_id_no_slash_is_false() {
        assert!(!is_hf_model_id("justname"));
    }

    #[test]
    fn test_is_hf_model_id_empty_org_is_false() {
        assert!(!is_hf_model_id("/model"));
    }

    #[test]
    fn test_is_hf_model_id_empty_name_is_false() {
        assert!(!is_hf_model_id("org/"));
    }

    // --- tilde expansion error message test ---

    #[test]
    fn test_resolve_tilde_path_error_mentions_expanded_path() {
        let result = resolve("~/nonexistent_model_dir_12345");
        let err = result.unwrap_err();
        assert!(
            !err.contains("HuggingFace cache"),
            "tilde path error should not mention HF cache, got: {err}"
        );
    }

    #[test]
    fn test_resolve_hf_hash_with_trailing_newline() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "org", "model", "abc123");
        // Overwrite with trailing newline (common from `echo`)
        let ref_path = cache
            .path()
            .join("models--org--model")
            .join("refs")
            .join("main");
        std::fs::write(&ref_path, "abc123\n").unwrap();
        let result = resolve_with_cache("org/model", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }
}
