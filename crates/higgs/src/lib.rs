pub mod anthropic_adapter;
pub mod attach;
pub mod auto_router;
pub mod cli_config;
pub mod config;
pub mod daemon;
pub mod doctor;
pub mod error;
pub mod metrics;
pub mod metrics_log;
pub mod model_download;
pub mod model_resolver;
pub mod proxy;
pub(crate) mod reasoning;
pub mod router;
pub mod routes;
#[doc(hidden)]
pub mod sse;
pub mod state;
pub mod translate;
pub mod tui;
pub mod types;

use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    extract::{ConnectInfo, Request},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, post},
};
use governor::{Quota, RateLimiter, clock::DefaultClock, state::keyed::DefaultKeyedStateStore};
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
    validate_request::ValidateRequestHeaderLayer,
};

use crate::state::SharedState;

type SharedRateLimiter = Arc<RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>>;

#[cfg(test)]
pub(crate) fn test_env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// Build the Axum router with all routes and middleware.
#[allow(clippy::needless_pass_by_value)]
pub fn build_router(
    state: SharedState,
    timeout_secs: f64,
    api_key: Option<String>,
    rate_limit: u32,
    max_body_size: usize,
    cors_origins: Option<Vec<String>>,
) -> Router {
    let timeout_duration = Duration::from_secs_f64(timeout_secs);

    let mut api_routes = Router::new()
        .route("/metrics", get(routes::metrics::metrics))
        .route(
            "/v1/models",
            get(routes::models::list_models).post(routes::models::load_model),
        )
        .route("/v1/models/{name}", delete(routes::models::unload_model))
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/messages", post(routes::anthropic::create_message))
        .route(
            "/v1/messages/count_tokens",
            post(routes::anthropic::count_tokens),
        );

    if let Some(rpm) = NonZeroU32::new(rate_limit) {
        let limiter: SharedRateLimiter = Arc::new(RateLimiter::keyed(Quota::per_minute(rpm)));
        api_routes = api_routes.layer(middleware::from_fn(move |req, next| {
            let limiter_clone = Arc::clone(&limiter);
            rate_limit_middleware(limiter_clone, req, next)
        }));
        tracing::info!(requests_per_minute = rate_limit, "Rate limiting enabled");
    }

    if let Some(ref key) = api_key {
        #[allow(deprecated)]
        // tower-http deprecated this as "too basic", but it's fine for a local inference server
        let auth_layer = ValidateRequestHeaderLayer::bearer(key);
        api_routes = api_routes.layer(auth_layer);
        tracing::info!("API key authentication enabled");
    }

    api_routes = api_routes.layer(DefaultBodyLimit::max(max_body_size));

    let mut router = Router::new()
        .route("/health", get(routes::health::health))
        .merge(api_routes)
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            timeout_duration,
        ));

    if let Some(cors) = build_cors_layer(cors_origins.as_deref()) {
        router = router.layer(cors);
    }

    router.with_state(state)
}

/// Build a CORS layer from the configured origin allow-list.
///
/// `None` (unset) sends no CORS headers; `["*"]` is fully permissive;
/// anything else is an explicit origin allow-list.
fn build_cors_layer(origins_opt: Option<&[String]>) -> Option<CorsLayer> {
    let origins = origins_opt?;
    if origins.iter().any(|o| o == "*") {
        return Some(CorsLayer::permissive());
    }
    let parsed: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| {
            origin.parse::<HeaderValue>().map_or_else(
                |_| {
                    tracing::warn!(origin = %origin, "ignoring invalid CORS origin");
                    None
                },
                Some,
            )
        })
        .collect();
    if parsed.is_empty() {
        return None;
    }
    Some(
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods(Any)
            .allow_headers(Any),
    )
}

async fn rate_limit_middleware(
    limiter: SharedRateLimiter,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let key = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map_or_else(|| "unknown".to_owned(), |ci| ci.0.ip().to_string());

    match limiter.check_key(&key) {
        Ok(()) => Ok(next.run(req).await),
        Err(_) => Err(StatusCode::TOO_MANY_REQUESTS),
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod cors_tests {
    use super::build_cors_layer;

    #[test]
    fn unset_origins_disable_cors() {
        assert!(build_cors_layer(None).is_none());
    }

    #[test]
    fn wildcard_enables_permissive_cors() {
        let origins = vec!["*".to_owned()];
        assert!(build_cors_layer(Some(&origins)).is_some());
    }

    #[test]
    fn explicit_origins_enable_cors() {
        let origins = vec!["https://example.com".to_owned()];
        assert!(build_cors_layer(Some(&origins)).is_some());
    }

    #[test]
    fn only_invalid_origins_disable_cors() {
        let origins = vec!["\u{7f}invalid".to_owned()];
        assert!(build_cors_layer(Some(&origins)).is_none());
    }

    #[test]
    fn empty_list_disables_cors() {
        let origins: Vec<String> = vec![];
        assert!(build_cors_layer(Some(&origins)).is_none());
    }
}
