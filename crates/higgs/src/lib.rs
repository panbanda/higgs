pub mod anthropic_adapter;
pub mod config;
pub mod error;
pub mod model_download;
pub mod model_resolver;
pub mod routes;
pub mod state;
pub mod types;

use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::{ConnectInfo, Request},
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use governor::{Quota, RateLimiter, clock::DefaultClock, state::keyed::DefaultKeyedStateStore};
use tower_http::{
    cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer,
    validate_request::ValidateRequestHeaderLayer,
};

use crate::state::SharedState;

type SharedRateLimiter = Arc<RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>>;

/// Build the Axum router with all routes and middleware.
#[allow(clippy::needless_pass_by_value)]
pub fn build_router(
    state: SharedState,
    timeout_secs: f64,
    api_key: Option<String>,
    rate_limit: u32,
) -> Router {
    let timeout_duration = Duration::from_secs_f64(timeout_secs);

    let mut api_routes = Router::new()
        .route("/v1/models", get(routes::models::list_models))
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

    Router::new()
        .route("/health", get(routes::health::health))
        .merge(api_routes)
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            timeout_duration,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state)
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
