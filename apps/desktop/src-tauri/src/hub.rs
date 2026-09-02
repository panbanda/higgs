//! Hugging Face model hub browsing: search, repo metadata, cache-layout
//! downloads, and cache deletion. Downloads run as detached background jobs
//! polled by the frontend (`hub_download_status`) rather than streamed over
//! a `Channel`, so the same API shape works for the dev bridge too.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt as _;
use tokio_util::sync::CancellationToken;

const HUB_BASE: &str = "https://huggingface.co";
const DEFAULT_AUTHOR: &str = "mlx-community";

fn home_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

/// Root of the standard Hugging Face cache: `$HF_HOME/hub` or
/// `~/.cache/huggingface/hub`.
fn hub_cache_root() -> Result<PathBuf, String> {
    if let Ok(home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(home).join("hub"));
    }
    home_dir()
        .map(|home| home.join(".cache/huggingface/hub"))
        .ok_or_else(|| "could not determine home directory".to_owned())
}

fn repo_dir_name(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// A repo id must be `<author>/<name>`, each segment starting with an
/// alphanumeric and containing only alphanumerics, `.`, `_`, or `-`.
fn is_valid_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(author), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    is_valid_segment(author) && is_valid_segment(name)
}

fn is_valid_segment(segment: &str) -> bool {
    match segment.chars().next() {
        Some(first) if first.is_ascii_alphanumeric() => segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-'),
        _ => false,
    }
}

/// Restricts a hash-like token (a commit sha or a blob etag) to a charset
/// that can never introduce a path separator.
fn is_valid_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

/// A repo-listed `rfilename` must be a relative path made entirely of
/// normal components: no `..`, no absolute paths, no empty segments.
fn is_safe_relative_path(rfilename: &str) -> bool {
    let path = Path::new(rfilename);
    let mut saw_component = false;
    for component in path.components() {
        match component {
            std::path::Component::Normal(part) => {
                if part.is_empty() || part == ".." {
                    return false;
                }
                saw_component = true;
            }
            _ => return false,
        }
    }
    saw_component
}

/// Defense in depth: confirms a joined path actually landed inside the
/// directory it was meant to.
fn path_within(base: &Path, candidate: &Path) -> bool {
    candidate.starts_with(base)
}

fn client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| error.to_string())
}

fn no_redirect_client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .build()
        .map_err(|error| error.to_string())
}

fn apply_auth(request: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match token.filter(|value| !value.is_empty()) {
        Some(value) => request.bearer_auth(value),
        None => request,
    }
}

async fn error_body(response: reqwest::Response) -> String {
    let status = response.status();
    let message = match status.as_u16() {
        401 => "Unauthorized: the Hugging Face token was rejected".to_owned(),
        403 => {
            "Forbidden: this repo is gated or private; a token with access is required".to_owned()
        }
        429 => "Rate limited by huggingface.co; wait a moment and try again".to_owned(),
        _ => response.text().await.unwrap_or_default(),
    };
    format!("HTTP {status}: {message}")
}

#[derive(Debug, Serialize)]
pub struct HubModelSummary {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub tags: Vec<String>,
    pub gated: bool,
}

#[derive(Debug, Deserialize)]
struct RawModelSummary {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gated: serde_json::Value,
}

fn gated_flag(value: &serde_json::Value) -> bool {
    !matches!(
        value,
        serde_json::Value::Bool(false) | serde_json::Value::Null
    )
}

#[tauri::command]
pub async fn hub_search(
    query: Option<String>,
    author: Option<String>,
    pipeline_tag: Option<String>,
    token: Option<String>,
    limit: u32,
) -> Result<Vec<HubModelSummary>, String> {
    let author = author.unwrap_or_else(|| DEFAULT_AUTHOR.to_owned());
    let mut params: Vec<(&str, String)> = vec![
        ("sort", "downloads".to_owned()),
        ("direction", "-1".to_owned()),
        ("limit", limit.to_string()),
        ("expand[]", "downloads".to_owned()),
        ("expand[]", "likes".to_owned()),
        ("expand[]", "lastModified".to_owned()),
        ("expand[]", "tags".to_owned()),
        ("expand[]", "gated".to_owned()),
    ];
    if !author.is_empty() {
        params.push(("author", author));
    }
    if let Some(search) = query.filter(|value| !value.trim().is_empty()) {
        params.push(("search", search));
    }
    if let Some(pipeline) = pipeline_tag.filter(|value| !value.is_empty()) {
        params.push(("pipeline_tag", pipeline));
    }

    let request = client(Duration::from_secs(15))?
        .get(format!("{HUB_BASE}/api/models"))
        .query(&params);
    let response = apply_auth(request, token.as_deref())
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }
    let raw: Vec<RawModelSummary> = response.json().await.map_err(|error| error.to_string())?;
    Ok(raw
        .into_iter()
        .map(|item| HubModelSummary {
            id: item.id,
            downloads: item.downloads,
            likes: item.likes,
            last_modified: item.last_modified,
            tags: item.tags,
            gated: gated_flag(&item.gated),
        })
        .collect())
}

#[derive(Debug, Serialize, Clone)]
pub struct HubSibling {
    pub rfilename: String,
    pub size: Option<u64>,
}

#[derive(Debug, Deserialize, Clone)]
struct RawSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RawModelDetail {
    id: String,
    sha: String,
    #[serde(default)]
    siblings: Vec<RawSibling>,
    #[serde(default)]
    tags: Vec<String>,
}

async fn fetch_model_detail(repo: &str, token: Option<&str>) -> Result<RawModelDetail, String> {
    let request = client(Duration::from_secs(15))?
        .get(format!("{HUB_BASE}/api/models/{repo}"))
        .query(&[("blobs", "true")]);
    let response = apply_auth(request, token)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }
    response.json().await.map_err(|error| error.to_string())
}

/// Quantization hint inferred from the repo id or tags, e.g. "4-bit", "bf16".
fn quantization_hint(id: &str, tags: &[String]) -> Option<String> {
    const PATTERNS: &[(&str, &str)] = &[
        ("4-bit", "4-bit"),
        ("4bit", "4-bit"),
        ("8-bit", "8-bit"),
        ("8bit", "8-bit"),
        ("6-bit", "6-bit"),
        ("6bit", "6-bit"),
        ("3-bit", "3-bit"),
        ("3bit", "3-bit"),
        ("2-bit", "2-bit"),
        ("2bit", "2-bit"),
        ("bf16", "bf16"),
        ("fp16", "fp16"),
        ("fp32", "fp32"),
    ];
    let haystacks = std::iter::once(id).chain(tags.iter().map(String::as_str));
    for haystack in haystacks {
        let lower = haystack.to_lowercase();
        if let Some((_, label)) = PATTERNS.iter().find(|(needle, _)| lower.contains(needle)) {
            return Some((*label).to_owned());
        }
    }
    None
}

async fn fetch_config_model_type(repo: &str, sha: &str, token: Option<&str>) -> Option<String> {
    let request = client(Duration::from_secs(10))
        .ok()?
        .get(format!("{HUB_BASE}/{repo}/resolve/{sha}/config.json"));
    let response = apply_auth(request, token).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let value: serde_json::Value = response.json().await.ok()?;
    value
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
}

#[derive(Debug, Serialize)]
pub struct HubModelDetail {
    pub id: String,
    pub sha: String,
    pub siblings: Vec<HubSibling>,
    pub total_bytes: u64,
    pub config_model_type: Option<String>,
    pub quantization: Option<String>,
    pub tags: Vec<String>,
}

#[tauri::command]
pub async fn hub_model(repo: String, token: Option<String>) -> Result<HubModelDetail, String> {
    let detail = fetch_model_detail(&repo, token.as_deref()).await?;
    let total_bytes = detail.siblings.iter().filter_map(|s| s.size).sum();
    let config_model_type = fetch_config_model_type(&repo, &detail.sha, token.as_deref()).await;
    let quantization = quantization_hint(&detail.id, &detail.tags);
    Ok(HubModelDetail {
        id: detail.id,
        sha: detail.sha,
        siblings: detail
            .siblings
            .into_iter()
            .map(|s| HubSibling {
                rfilename: s.rfilename,
                size: s.size,
            })
            .collect(),
        total_bytes,
        config_model_type,
        quantization,
        tags: detail.tags,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct HubDownloadStatus {
    pub state: String,
    pub file: Option<String>,
    pub file_index: u32,
    pub file_count: u32,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub total_done: u64,
    pub total_bytes: u64,
    pub message: Option<String>,
    pub path: Option<String>,
}

impl Default for HubDownloadStatus {
    fn default() -> Self {
        Self {
            state: "idle".to_owned(),
            file: None,
            file_index: 0,
            file_count: 0,
            bytes_done: 0,
            bytes_total: 0,
            total_done: 0,
            total_bytes: 0,
            message: None,
            path: None,
        }
    }
}

struct HubJobHandle {
    status: Arc<Mutex<HubDownloadStatus>>,
    token: CancellationToken,
    /// Set while a file is streaming so cancellation can remove the partial write.
    incomplete_path: Arc<Mutex<Option<PathBuf>>>,
}

#[derive(Clone, Default)]
pub struct HubJobs(Arc<Mutex<HashMap<String, HubJobHandle>>>);

#[tauri::command]
pub fn hub_download_start(
    repo: String,
    token: Option<String>,
    jobs: tauri::State<'_, HubJobs>,
) -> Result<(), String> {
    if !is_valid_repo_id(&repo) {
        return Err(format!("invalid repo id: {repo}"));
    }
    {
        let map = jobs
            .0
            .lock()
            .map_err(|_| "hub job registry poisoned".to_owned())?;
        if let Some(handle) = map.get(&repo)
            && handle
                .status
                .lock()
                .map(|s| s.state == "running")
                .unwrap_or(false)
        {
            return Ok(());
        }
    }

    let status = Arc::new(Mutex::new(HubDownloadStatus {
        state: "running".to_owned(),
        ..HubDownloadStatus::default()
    }));
    let cancel_token = CancellationToken::new();
    let incomplete_path = Arc::new(Mutex::new(None));
    {
        let mut map = jobs
            .0
            .lock()
            .map_err(|_| "hub job registry poisoned".to_owned())?;
        map.insert(
            repo.clone(),
            HubJobHandle {
                status: status.clone(),
                token: cancel_token.clone(),
                incomplete_path: incomplete_path.clone(),
            },
        );
    }

    tauri::async_runtime::spawn(run_hub_download(
        repo,
        token,
        status,
        cancel_token,
        incomplete_path,
    ));
    Ok(())
}

async fn run_hub_download(
    repo: String,
    token: Option<String>,
    status: Arc<Mutex<HubDownloadStatus>>,
    cancel: CancellationToken,
    incomplete_path: Arc<Mutex<Option<PathBuf>>>,
) {
    let result = do_hub_download(&repo, token.as_deref(), &status, &cancel, &incomplete_path).await;
    let Ok(mut guard) = status.lock() else {
        return;
    };
    if cancel.is_cancelled() {
        guard.state = "cancelled".to_owned();
        return;
    }
    match result {
        Ok(path) => {
            guard.state = "done".to_owned();
            guard.path = Some(path);
        }
        Err(message) => {
            guard.state = "error".to_owned();
            guard.message = Some(message);
        }
    }
}

async fn do_hub_download(
    repo: &str,
    token: Option<&str>,
    status: &Arc<Mutex<HubDownloadStatus>>,
    cancel: &CancellationToken,
    incomplete_path: &Arc<Mutex<Option<PathBuf>>>,
) -> Result<String, String> {
    if !is_valid_repo_id(repo) {
        return Err(format!("invalid repo id: {repo}"));
    }
    let detail = fetch_model_detail(repo, token).await?;
    if !is_valid_token(&detail.sha) {
        return Err(format!("invalid sha returned by hub: {}", detail.sha));
    }
    let root = hub_cache_root()?;
    let repo_dir = root.join(repo_dir_name(repo));
    let blobs_dir = repo_dir.join("blobs");
    let snapshot_dir = repo_dir.join("snapshots").join(&detail.sha);
    let refs_dir = repo_dir.join("refs");
    std::fs::create_dir_all(&blobs_dir).map_err(|error| error.to_string())?;
    std::fs::create_dir_all(&snapshot_dir).map_err(|error| error.to_string())?;
    std::fs::create_dir_all(&refs_dir).map_err(|error| error.to_string())?;
    std::fs::write(refs_dir.join("main"), &detail.sha).map_err(|error| error.to_string())?;

    let total_bytes: u64 = detail.siblings.iter().filter_map(|s| s.size).sum();
    let file_count = u32::try_from(detail.siblings.len()).unwrap_or(u32::MAX);
    let mut total_done: u64 = 0;

    for (index, sibling) in detail.siblings.iter().enumerate() {
        if cancel.is_cancelled() {
            return Ok(snapshot_dir.to_string_lossy().into_owned());
        }
        if !is_safe_relative_path(&sibling.rfilename) {
            return Err(format!(
                "unsafe file path in repo listing: {}",
                sibling.rfilename
            ));
        }
        let file_index = u32::try_from(index + 1).unwrap_or(u32::MAX);
        {
            let mut guard = status
                .lock()
                .map_err(|_| "hub download status poisoned".to_owned())?;
            guard.file = Some(sibling.rfilename.clone());
            guard.file_index = file_index;
            guard.file_count = file_count;
            guard.bytes_done = 0;
            guard.bytes_total = sibling.size.unwrap_or(0);
            guard.total_done = total_done;
            guard.total_bytes = total_bytes;
        }

        let file_url = format!(
            "{HUB_BASE}/{repo}/resolve/{}/{}",
            detail.sha, sibling.rfilename
        );
        let etag = resolve_etag(&file_url, token, &detail.sha, &sibling.rfilename).await?;
        if !is_valid_token(&etag) {
            return Err(format!("invalid blob etag: {etag}"));
        }
        let blob_path = blobs_dir.join(&etag);
        if !path_within(&blobs_dir, &blob_path) {
            return Err(format!(
                "blob path escaped blobs dir: {}",
                blob_path.display()
            ));
        }

        let up_to_date = sibling.size.is_none_or(|expected| {
            std::fs::metadata(&blob_path)
                .map(|meta| meta.len() == expected)
                .unwrap_or(false)
        }) && blob_path.is_file();

        if !up_to_date {
            download_to_blob(
                &file_url,
                token,
                &blob_path,
                status,
                cancel,
                &mut total_done,
                total_bytes,
                incomplete_path,
            )
            .await?;
        } else {
            total_done += sibling.size.unwrap_or(0);
        }

        if cancel.is_cancelled() {
            return Ok(snapshot_dir.to_string_lossy().into_owned());
        }
        link_snapshot_file(&snapshot_dir, &sibling.rfilename, &blob_path)?;
    }

    Ok(snapshot_dir.to_string_lossy().into_owned())
}

async fn resolve_etag(
    file_url: &str,
    token: Option<&str>,
    sha: &str,
    rfilename: &str,
) -> Result<String, String> {
    let request = no_redirect_client(Duration::from_secs(30))?.head(file_url);
    let response = apply_auth(request, token)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let etag = response
        .headers()
        .get("x-linked-etag")
        .or_else(|| response.headers().get("etag"))
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_matches('"').to_owned());
    Ok(etag.unwrap_or_else(|| format!("{sha}-{}", rfilename.replace('/', "_"))))
}

#[allow(clippy::too_many_arguments)]
async fn download_to_blob(
    file_url: &str,
    token: Option<&str>,
    blob_path: &Path,
    status: &Arc<Mutex<HubDownloadStatus>>,
    cancel: &CancellationToken,
    total_done: &mut u64,
    total_bytes: u64,
    incomplete_path: &Arc<Mutex<Option<PathBuf>>>,
) -> Result<(), String> {
    use futures_util::StreamExt as _;

    let temp_path = blob_path.with_extension("incomplete");
    if let Ok(mut guard) = incomplete_path.lock() {
        *guard = Some(temp_path.clone());
    }

    let request = client(Duration::from_secs(60 * 60))?.get(file_url);
    let response = apply_auth(request, token)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }

    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| error.to_string())?;
    let mut stream = response.bytes_stream();
    let mut file_done: u64 = 0;
    loop {
        let next = tokio::select! {
            next = stream.next() => next,
            () = cancel.cancelled() => {
                drop(file);
                let _ = tokio::fs::remove_file(&temp_path).await;
                return Ok(());
            }
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| error.to_string())?;
        file.write_all(&chunk)
            .await
            .map_err(|error| error.to_string())?;
        file_done += chunk.len() as u64;
        *total_done += chunk.len() as u64;
        if let Ok(mut guard) = status.lock() {
            guard.bytes_done = file_done;
            guard.total_done = *total_done;
            guard.total_bytes = total_bytes;
        }
    }
    file.flush().await.map_err(|error| error.to_string())?;
    drop(file);
    tokio::fs::rename(&temp_path, blob_path)
        .await
        .map_err(|error| error.to_string())?;
    if let Ok(mut guard) = incomplete_path.lock() {
        *guard = None;
    }
    Ok(())
}

fn link_snapshot_file(
    snapshot_dir: &Path,
    rfilename: &str,
    blob_path: &Path,
) -> Result<(), String> {
    if !is_safe_relative_path(rfilename) {
        return Err(format!("unsafe file path in repo listing: {rfilename}"));
    }
    let link_path = snapshot_dir.join(rfilename);
    if !path_within(snapshot_dir, &link_path) {
        return Err(format!(
            "snapshot link path escaped snapshot dir: {}",
            link_path.display()
        ));
    }
    if let Some(parent) = link_path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    if link_path.exists() || link_path.is_symlink() {
        std::fs::remove_file(&link_path).map_err(|error| error.to_string())?;
    }
    std::os::unix::fs::symlink(blob_path, &link_path).map_err(|error| error.to_string())
}

#[tauri::command]
pub fn hub_download_status(
    repo: String,
    jobs: tauri::State<'_, HubJobs>,
) -> Result<HubDownloadStatus, String> {
    let map = jobs
        .0
        .lock()
        .map_err(|_| "hub job registry poisoned".to_owned())?;
    Ok(map
        .get(&repo)
        .and_then(|handle| handle.status.lock().ok())
        .map(|guard| guard.clone())
        .unwrap_or_default())
}

#[tauri::command]
pub fn hub_cancel(repo: String, jobs: tauri::State<'_, HubJobs>) -> Result<(), String> {
    let map = jobs
        .0
        .lock()
        .map_err(|_| "hub job registry poisoned".to_owned())?;
    if let Some(handle) = map.get(&repo) {
        handle.token.cancel();
        if let Ok(guard) = handle.incomplete_path.lock()
            && let Some(path) = guard.as_ref()
        {
            let _ = std::fs::remove_file(path);
        }
    }
    Ok(())
}

#[tauri::command]
pub fn hub_delete(repo: String) -> Result<(), String> {
    if !is_valid_repo_id(&repo) {
        return Err(format!("invalid repo id: {repo}"));
    }
    let root = hub_cache_root()?;
    let repo_dir = root.join(repo_dir_name(&repo));
    if !path_within(&root, &repo_dir) {
        return Err(format!(
            "repo path escaped cache root: {}",
            repo_dir.display()
        ));
    }
    if repo_dir.is_dir() {
        std::fs::remove_dir_all(&repo_dir).map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_dir_name_replaces_slash() {
        assert_eq!(
            repo_dir_name("mlx-community/Qwen3-4B-4bit"),
            "models--mlx-community--Qwen3-4B-4bit"
        );
    }

    #[test]
    fn quantization_hint_from_id() {
        assert_eq!(
            quantization_hint("mlx-community/Qwen3-4B-4bit", &[]),
            Some("4-bit".to_owned())
        );
        assert_eq!(
            quantization_hint("mlx-community/Qwen3-4B-bf16", &[]),
            Some("bf16".to_owned())
        );
        assert_eq!(quantization_hint("mlx-community/plain", &[]), None);
    }

    #[test]
    fn gated_flag_reads_variants() {
        assert!(!gated_flag(&serde_json::Value::Bool(false)));
        assert!(gated_flag(&serde_json::Value::String("auto".to_owned())));
        assert!(gated_flag(&serde_json::Value::Bool(true)));
    }

    #[test]
    fn rejects_traversal_filename() {
        assert!(!is_safe_relative_path("../../x"));
        assert!(!is_safe_relative_path("a/../../b"));
        assert!(is_safe_relative_path("a/b/c.safetensors"));
    }

    #[test]
    fn rejects_absolute_filename() {
        assert!(!is_safe_relative_path("/etc/passwd"));
    }

    #[test]
    fn rejects_bad_repo_id() {
        assert!(!is_valid_repo_id("no-slash-here"));
        assert!(!is_valid_repo_id("../etc/passwd"));
        assert!(!is_valid_repo_id("author/../name"));
        assert!(!is_valid_repo_id("author/name/extra"));
        assert!(is_valid_repo_id("mlx-community/Qwen3-4B-4bit"));
    }

    #[test]
    fn rejects_bad_etag() {
        assert!(!is_valid_token("../../etc/passwd"));
        assert!(!is_valid_token("has/slash"));
        assert!(!is_valid_token(""));
        assert!(is_valid_token("abc123.def-ghi_jkl"));
    }
}
