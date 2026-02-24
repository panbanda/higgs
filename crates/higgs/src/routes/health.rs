use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_returns_ok() {
        let Json(resp) = health().await;
        assert_eq!(resp.status, "ok");
    }
}
