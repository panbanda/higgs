use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::TryStreamExt;

use crate::error::ServerError;

fn is_hop_by_hop(name: &http::header::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

/// Build forwarding headers from the original request.
///
/// Strips hop-by-hop headers, optionally strips auth headers, and injects
/// the provider's API key as `x-api-key` if configured.
fn build_forwarding_headers(
    original: &HeaderMap,
    strip_auth: bool,
    api_key: Option<&str>,
    body_len: usize,
) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (key, value) in original {
        if key == http::header::HOST || is_hop_by_hop(key) {
            continue;
        }
        if strip_auth && (key == http::header::AUTHORIZATION || key.as_str() == "x-api-key") {
            continue;
        }
        headers.insert(key.clone(), value.clone());
    }

    if let Some(key) = api_key {
        if let Ok(value) = HeaderValue::from_str(key) {
            headers.insert(http::header::HeaderName::from_static("x-api-key"), value);
        } else {
            tracing::warn!("api_key contains invalid header characters, skipping");
        }
    }

    if body_len > 0 {
        if let Ok(val) = HeaderValue::from_str(&body_len.to_string()) {
            headers.insert(http::header::CONTENT_LENGTH, val);
        }
    }

    // Strip accept-encoding so provider doesn't compress; we need raw bytes for streaming
    headers.remove(http::header::ACCEPT_ENCODING);

    headers
}

/// Filter upstream response headers, removing hop-by-hop and content-encoding.
fn filter_response_headers(upstream: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (key, value) in upstream {
        if is_hop_by_hop(key) || key == http::header::CONTENT_ENCODING {
            continue;
        }
        headers.insert(key.clone(), value.clone());
    }
    headers
}

/// Rewrite the `model` field in a JSON body.
pub fn rewrite_model_in_body(body: &[u8], new_model: &str) -> Result<Bytes, ServerError> {
    let mut json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::InternalError(format!("JSON parse for model rewrite: {e}")))?;
    let obj = json
        .as_object_mut()
        .ok_or_else(|| ServerError::InternalError("expected JSON object".to_owned()))?;
    obj.insert(
        "model".to_owned(),
        serde_json::Value::String(new_model.to_owned()),
    );
    serde_json::to_vec(&json)
        .map(Bytes::from)
        .map_err(|e| ServerError::InternalError(format!("JSON serialize after rewrite: {e}")))
}

/// Return a stub response for `/count_tokens` when the provider doesn't support it.
pub fn stub_count_tokens_response() -> Response {
    let stub = serde_json::json!({"input_tokens": 0});
    let body = serde_json::to_vec(&stub).unwrap_or_default();
    (
        StatusCode::OK,
        [(http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// Send a request to an upstream provider and return the raw response.
///
/// Unlike [`proxy_request`], this returns the reqwest response directly so the
/// caller can process the body (e.g., for format translation).
pub async fn send_to_provider(
    client: &reqwest::Client,
    provider_url: &str,
    path: &str,
    body: Bytes,
    original_headers: &HeaderMap,
    strip_auth: bool,
    api_key: Option<&str>,
) -> Result<reqwest::Response, ServerError> {
    let url = format!("{}{path}", provider_url.trim_end_matches('/'));
    let headers = build_forwarding_headers(original_headers, strip_auth, api_key, body.len());

    tracing::debug!(url = %url, body_bytes = body.len(), "sending to provider");

    let upstream = client
        .post(&url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(url = %url, error = %e, "provider request failed");
            ServerError::ProxyError(format!("provider unreachable: {e}"))
        })?;

    let status = upstream.status().as_u16();
    if status >= 400 {
        let error_body = upstream
            .text()
            .await
            .unwrap_or_else(|_| String::from("(failed to read error body)"));
        return Err(ServerError::ProxyError(format!(
            "upstream returned HTTP {status}: {error_body}"
        )));
    }

    Ok(upstream)
}

/// Forward a request to an upstream provider and stream the response back.
pub async fn proxy_request(
    client: &reqwest::Client,
    provider_url: &str,
    path: &str,
    body: Bytes,
    original_headers: &HeaderMap,
    strip_auth: bool,
    api_key: Option<&str>,
) -> Result<Response, ServerError> {
    let url = format!("{}{path}", provider_url.trim_end_matches('/'));
    let headers = build_forwarding_headers(original_headers, strip_auth, api_key, body.len());

    tracing::debug!(url = %url, body_bytes = body.len(), "proxying request");

    let upstream = client
        .post(&url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|e| {
            tracing::error!(url = %url, error = %e, "provider request failed");
            ServerError::ProxyError(format!("provider unreachable: {e}"))
        })?;

    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    tracing::info!(status = %status, url = %url, "provider responded");

    let response_headers = filter_response_headers(upstream.headers());

    let stream = upstream.bytes_stream().map_err(std::io::Error::other);
    let response_body = Body::from_stream(stream);

    let mut response = Response::new(response_body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;
    Ok(response)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn hop_by_hop_headers_filtered() {
        assert!(is_hop_by_hop(&http::header::HeaderName::from_static(
            "connection"
        )));
        assert!(is_hop_by_hop(&http::header::HeaderName::from_static(
            "transfer-encoding"
        )));
        assert!(!is_hop_by_hop(&http::header::CONTENT_TYPE));
    }

    #[test]
    fn forwarding_headers_strip_host() {
        let mut original = HeaderMap::new();
        original.insert(http::header::HOST, HeaderValue::from_static("localhost"));
        original.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let headers = build_forwarding_headers(&original, false, None, 0);
        assert!(headers.get(http::header::HOST).is_none());
        assert!(headers.get(http::header::CONTENT_TYPE).is_some());
    }

    #[test]
    fn forwarding_headers_strip_auth_when_requested() {
        let mut original = HeaderMap::new();
        original.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-test"),
        );
        original.insert(
            http::header::HeaderName::from_static("x-api-key"),
            HeaderValue::from_static("sk-ant-test"),
        );

        let headers = build_forwarding_headers(&original, true, None, 0);
        assert!(headers.get(http::header::AUTHORIZATION).is_none());
        assert!(headers.get("x-api-key").is_none());
    }

    #[test]
    fn forwarding_headers_preserve_auth_when_not_stripping() {
        let mut original = HeaderMap::new();
        original.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-test"),
        );

        let headers = build_forwarding_headers(&original, false, None, 0);
        assert!(headers.get(http::header::AUTHORIZATION).is_some());
    }

    #[test]
    fn forwarding_headers_inject_api_key() {
        let original = HeaderMap::new();
        let headers = build_forwarding_headers(&original, false, Some("sk-provider"), 0);
        assert_eq!(
            headers.get("x-api-key").unwrap().to_str().unwrap(),
            "sk-provider"
        );
    }

    #[test]
    fn forwarding_headers_set_content_length() {
        let original = HeaderMap::new();
        let headers = build_forwarding_headers(&original, false, None, 42);
        assert_eq!(
            headers
                .get(http::header::CONTENT_LENGTH)
                .unwrap()
                .to_str()
                .unwrap(),
            "42"
        );
    }

    #[test]
    fn forwarding_headers_strip_accept_encoding() {
        let mut original = HeaderMap::new();
        original.insert(
            http::header::ACCEPT_ENCODING,
            HeaderValue::from_static("gzip"),
        );
        let headers = build_forwarding_headers(&original, false, None, 0);
        assert!(headers.get(http::header::ACCEPT_ENCODING).is_none());
    }

    #[test]
    fn rewrite_model_changes_field() {
        let body = br#"{"model":"old-model","messages":[]}"#;
        let result = rewrite_model_in_body(body, "new-model").unwrap();
        let json: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(json["model"].as_str().unwrap(), "new-model");
        assert!(json["messages"].is_array());
    }

    #[test]
    fn rewrite_model_invalid_json_errors() {
        let body = b"not json";
        assert!(rewrite_model_in_body(body, "x").is_err());
    }

    #[test]
    fn stub_count_tokens_returns_json() {
        let resp = stub_count_tokens_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn filter_response_removes_hop_by_hop() {
        let mut upstream = reqwest::header::HeaderMap::new();
        upstream.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        upstream.insert(
            reqwest::header::TRANSFER_ENCODING,
            reqwest::header::HeaderValue::from_static("chunked"),
        );
        upstream.insert(
            reqwest::header::CONTENT_ENCODING,
            reqwest::header::HeaderValue::from_static("gzip"),
        );

        let filtered = filter_response_headers(&upstream);
        assert!(filtered.get(http::header::CONTENT_TYPE).is_some());
        assert!(filtered.get(http::header::TRANSFER_ENCODING).is_none());
        assert!(filtered.get(http::header::CONTENT_ENCODING).is_none());
    }
}
