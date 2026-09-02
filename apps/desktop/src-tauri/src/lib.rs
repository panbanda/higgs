mod hub;
mod local;
mod paths;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::State;
use tauri::ipc::Channel;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct ActiveRequests(Mutex<HashMap<String, CancellationToken>>);

#[derive(Debug, Clone, Deserialize)]
struct Connection {
    base_url: String,
    api_key: Option<String>,
}

impl Connection {
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    /// True when it is safe to send the API key on the wire: the URL is
    /// `https:`, or `http:` restricted to a loopback host, so a bearer token
    /// is never sent to a network peer in clear text.
    fn allows_auth(&self) -> bool {
        let Ok(parsed) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };
        match parsed.scheme() {
            "https" => true,
            "http" => parsed.host_str().is_some_and(is_loopback_host),
            _ => false,
        }
    }

    /// Attaches the API key as a bearer token, refusing to do so unless
    /// [`Connection::allows_auth`] holds.
    fn apply_auth(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder, String> {
        match self.api_key.as_deref().filter(|key| !key.is_empty()) {
            Some(key) => {
                if !self.allows_auth() {
                    return Err("API key is only sent over HTTPS or loopback".to_owned());
                }
                Ok(request.bearer_auth(key))
            }
            None => Ok(request),
        }
    }
}

/// True for `localhost` and loopback IP literals (`127.0.0.1`, `::1`, with
/// or without the brackets a URL host puts around an IPv6 address).
fn is_loopback_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Pure decision behind the client's redirect policy: allow at most 5 hops,
/// and refuse any redirect that downgrades from `https` to `http` or that
/// targets a non-loopback `http` host, so a redirect can't be used to leak
/// an `Authorization` header in clear text.
fn evaluate_redirect(
    previous_scheme: &str,
    next: &reqwest::Url,
    hops_so_far: usize,
) -> Result<(), String> {
    if hops_so_far > 5 {
        return Err("too many redirects".to_owned());
    }
    if previous_scheme == "https" && next.scheme() == "http" {
        return Err("redirect from https to http is not allowed".to_owned());
    }
    if next.scheme() == "http" && !next.host_str().is_some_and(is_loopback_host) {
        return Err("redirect to a non-loopback http host is not allowed".to_owned());
    }
    Ok(())
}

fn redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        let previous_scheme = attempt
            .previous()
            .last()
            .map_or("https", reqwest::Url::scheme);
        match evaluate_redirect(previous_scheme, attempt.url(), attempt.previous().len()) {
            Ok(()) => attempt.follow(),
            Err(reason) => attempt.error(reason),
        }
    })
}

fn client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(timeout)
        .redirect(redirect_policy())
        .build()
        .map_err(|error| error.to_string())
}

#[derive(Debug, Serialize)]
struct HealthStatus {
    ok: bool,
    detail: String,
}

#[tauri::command]
async fn check_health(connection: Connection) -> Result<HealthStatus, String> {
    let response = client(Duration::from_secs(3))?
        .get(connection.url("/health"))
        .send()
        .await;
    Ok(match response {
        Ok(response) if response.status().is_success() => HealthStatus {
            ok: true,
            detail: "ok".to_owned(),
        },
        Ok(response) => HealthStatus {
            ok: false,
            detail: format!("HTTP {}", response.status()),
        },
        Err(error) => HealthStatus {
            ok: false,
            detail: error.to_string(),
        },
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelInfo {
    id: String,
    #[serde(default)]
    owned_by: String,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    data: Vec<ModelInfo>,
}

async fn error_body(response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("error"))
                .and_then(|m| m.as_str().map(ToOwned::to_owned))
        })
        .unwrap_or(body);
    format!("HTTP {status}: {message}")
}

#[tauri::command]
async fn list_models(connection: Connection) -> Result<Vec<ModelInfo>, String> {
    let request = client(Duration::from_secs(10))?.get(connection.url("/v1/models"));
    let response = connection
        .apply_auth(request)?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }
    let list: ModelList = response.json().await.map_err(|error| error.to_string())?;
    Ok(list.data)
}

#[tauri::command]
async fn fetch_metrics(connection: Connection) -> Result<serde_json::Value, String> {
    let request = client(Duration::from_secs(10))?.get(connection.url("/metrics"));
    let response = connection
        .apply_auth(request)?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }
    response.json().await.map_err(|error| error.to_string())
}

#[tauri::command]
async fn fetch_system(connection: Connection) -> Result<serde_json::Value, String> {
    let request = client(Duration::from_secs(10))?.get(connection.url("/v1/system"));
    let response = connection
        .apply_auth(request)?
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }
    response.json().await.map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    Chunk { data: serde_json::Value },
    Done,
    Cancelled,
    Error { message: String },
}

/// Streams a chat completion from the server and forwards each parsed SSE
/// `data:` payload to the frontend over `on_event`.
#[tauri::command]
async fn stream_chat(
    request_id: String,
    connection: Connection,
    body: serde_json::Value,
    on_event: Channel<StreamEvent>,
    active: State<'_, ActiveRequests>,
) -> Result<(), String> {
    let token = CancellationToken::new();
    active
        .0
        .lock()
        .map_err(|_| "request registry poisoned".to_owned())?
        .insert(request_id.clone(), token.clone());

    let result = run_stream(&connection, body, &on_event, &token).await;

    if let Ok(mut map) = active.0.lock() {
        map.remove(&request_id);
    }

    let final_event = match result {
        Ok(()) if token.is_cancelled() => StreamEvent::Cancelled,
        Ok(()) => StreamEvent::Done,
        Err(message) => StreamEvent::Error { message },
    };
    on_event
        .send(final_event)
        .map_err(|error| error.to_string())
}

async fn run_stream(
    connection: &Connection,
    body: serde_json::Value,
    on_event: &Channel<StreamEvent>,
    token: &CancellationToken,
) -> Result<(), String> {
    let request = connection.apply_auth(
        client(Duration::from_secs(60 * 60))?
            .post(connection.url("/v1/chat/completions"))
            .json(&body),
    )?;
    let response = tokio::select! {
        response = request.send() => response.map_err(|error| error.to_string())?,
        () = token.cancelled() => return Ok(()),
    };
    if !response.status().is_success() {
        return Err(error_body(response).await);
    }

    let mut stream = response.bytes_stream();
    // Frames are split on the byte level so a multibyte character straddling
    // two network chunks is never decoded in halves.
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        let next = tokio::select! {
            next = stream.next() => next,
            () = token.cancelled() => return Ok(()),
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| error.to_string())?;
        buffer.extend_from_slice(&chunk);

        while let Some((boundary, separator_len)) = find_frame_end(&buffer) {
            let frame_bytes: Vec<u8> = buffer.drain(..boundary + separator_len).collect();
            let frame = String::from_utf8_lossy(&frame_bytes[..boundary]).into_owned();
            match parse_sse_frame(&frame) {
                None => {}
                Some(SseData::Done) => return Ok(()),
                Some(SseData::Json(value)) => {
                    on_event
                        .send(StreamEvent::Chunk { data: value })
                        .map_err(|error| error.to_string())?;
                }
                Some(SseData::Invalid(raw)) => {
                    return Err(format!("malformed stream chunk: {raw}"));
                }
            }
        }
    }
    Ok(())
}

/// Finds the end of the next SSE frame, accepting both `\n\n` and `\r\n\r\n`
/// separators. Returns the boundary offset and the separator's byte length.
fn find_frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf_lf = buffer.windows(2).position(|pair| pair == b"\n\n");
    let crlf_crlf = buffer.windows(4).position(|quad| quad == b"\r\n\r\n");
    match (lf_lf, crlf_crlf) {
        (Some(lf), Some(crlf)) if crlf < lf => Some((crlf, 4)),
        (Some(lf), _) => Some((lf, 2)),
        (None, Some(crlf)) => Some((crlf, 4)),
        (None, None) => None,
    }
}

enum SseData {
    Done,
    Json(serde_json::Value),
    Invalid(String),
}

fn parse_sse_frame(frame: &str) -> Option<SseData> {
    let data: Vec<&str> = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect();
    if data.is_empty() {
        return None;
    }
    let payload = data.join("\n");
    if payload.trim() == "[DONE]" {
        return Some(SseData::Done);
    }
    Some(
        serde_json::from_str(&payload)
            .map_or_else(|_| SseData::Invalid(payload.clone()), SseData::Json),
    )
}

#[tauri::command]
fn cancel_chat(request_id: String, active: State<'_, ActiveRequests>) -> Result<(), String> {
    let map = active
        .0
        .lock()
        .map_err(|_| "request registry poisoned".to_owned())?;
    if let Some(token) = map.get(&request_id) {
        token.cancel();
    }
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(ActiveRequests::default())
        .manage(hub::HubJobs::default())
        .invoke_handler(tauri::generate_handler![
            check_health,
            list_models,
            fetch_metrics,
            fetch_system,
            stream_chat,
            cancel_chat,
            local::list_profiles,
            local::read_config,
            local::write_config_raw,
            local::write_config_structured,
            local::read_metrics_log,
            local::daemon_status,
            local::read_text_tail,
            local::run_higgs,
            local::higgs_binary_info,
            local::model_cache_info,
            hub::hub_search,
            hub::hub_model,
            hub::hub_download_start,
            hub::hub_download_status,
            hub::hub_cancel,
            hub::hub_delete
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connection(base_url: &str) -> Connection {
        Connection {
            base_url: base_url.to_owned(),
            api_key: Some("secret".to_owned()),
        }
    }

    #[test]
    fn allows_auth_over_https() {
        assert!(connection("https://api.example.com").allows_auth());
    }

    #[test]
    fn allows_auth_over_http_loopback() {
        assert!(connection("http://localhost:1234").allows_auth());
        assert!(connection("http://127.0.0.1:1234").allows_auth());
        assert!(connection("http://[::1]:1234").allows_auth());
    }

    #[test]
    fn refuses_auth_over_http_to_a_remote_host() {
        assert!(!connection("http://example.com").allows_auth());
        assert!(!connection("http://192.168.1.5:1234").allows_auth());
    }

    fn url(value: &str) -> reqwest::Url {
        reqwest::Url::parse(value).expect("valid url")
    }

    #[test]
    fn allows_a_plain_https_redirect() {
        assert!(evaluate_redirect("https", &url("https://example.com/next"), 1).is_ok());
    }

    #[test]
    fn allows_an_http_redirect_to_loopback() {
        assert!(evaluate_redirect("http", &url("http://127.0.0.1:8080/next"), 1).is_ok());
    }

    #[test]
    fn rejects_a_downgrade_from_https_to_http() {
        assert!(evaluate_redirect("https", &url("http://example.com/next"), 1).is_err());
    }

    #[test]
    fn rejects_an_http_redirect_to_a_remote_host() {
        assert!(evaluate_redirect("http", &url("http://example.com/next"), 1).is_err());
    }

    #[test]
    fn rejects_more_than_five_hops() {
        assert!(evaluate_redirect("https", &url("https://example.com/next"), 5).is_ok());
        assert!(evaluate_redirect("https", &url("https://example.com/next"), 6).is_err());
    }

    #[test]
    fn parses_json_frame() {
        match parse_sse_frame("data: {\"a\":1}") {
            Some(SseData::Json(value)) => assert_eq!(value["a"], 1),
            _ => panic!("expected json"),
        }
    }

    #[test]
    fn parses_done_frame() {
        assert!(matches!(
            parse_sse_frame("data: [DONE]"),
            Some(SseData::Done)
        ));
    }

    #[test]
    fn frame_end_is_found_on_bytes() {
        assert_eq!(find_frame_end(b"data: x\n\nrest"), Some((7, 2)));
        assert_eq!(find_frame_end(b"data: partial\n"), None);
    }

    #[test]
    fn frame_end_is_found_on_crlf_bytes() {
        assert_eq!(find_frame_end(b"data: x\r\n\r\nrest"), Some((7, 4)));
        assert_eq!(find_frame_end(b"data: partial\r\n"), None);
    }

    #[test]
    fn ignores_comment_frames() {
        assert!(parse_sse_frame(": keepalive").is_none());
    }
}
