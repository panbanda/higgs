use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use bytes::Bytes;

use crate::{
    config::ModelConfig,
    error::ServerError,
    model_resolver,
    state::{Engine, SharedState, build_engine},
    types::openai::{ModelList, ModelObject},
};

/// How long `DELETE /v1/models/{name}` waits for in-flight requests to release a
/// model before detaching the final drop to a background task.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// Poll cadence while draining references to an unloaded model.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// `GET /v1/models` -- list the locally loaded models.
pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let data = state
        .router
        .local_models_with_vlm()
        .into_iter()
        .map(|(name, vision)| model_object(name, vision))
        .collect();
    Json(ModelList {
        object: "list",
        data,
        runtime_model_load: state.config.local.allow_runtime_model_load,
    })
}

/// `POST /v1/models` -- load a model into GPU memory at runtime.
///
/// Opt-in via `local.allow_runtime_model_load`. The body mirrors a `[[models]]`
/// config entry (`path` required; `name`, `batch`, `mlx_profile`, `kv_*`
/// optional). Returns the created model object, `409` on a name collision, or
/// `403` when runtime loading is disabled.
pub async fn load_model(
    State(state): State<SharedState>,
    body: Bytes,
) -> Result<Json<ModelObject>, ServerError> {
    if !state.config.local.allow_runtime_model_load {
        return Err(ServerError::Forbidden(
            "runtime model loading is disabled; set local.allow_runtime_model_load = true to enable it"
                .to_owned(),
        ));
    }

    let model_cfg: ModelConfig = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("invalid model config: {e}")))?;

    // Validate the KV-cache config up front, mirroring `doctor::check_models`.
    model_cfg
        .kv_cache_config()
        .validate()
        .map_err(|e| ServerError::BadRequest(format!("invalid KV cache config: {e}")))?;
    if model_cfg.batch && model_cfg.kv_cache_config().is_turboquant() {
        return Err(ServerError::BadRequest(
            "unsupported combination: TurboQuant KV cache with batch=true".to_owned(),
        ));
    }

    // Cheap collision pre-check when the caller named the model, so we don't pay
    // for a full load just to reject it. `insert_engine` re-checks under the
    // write lock, so this is an optimization, not the source of truth.
    if let Some(ref name) = model_cfg.name {
        if state.router.contains_engine(name) {
            return Err(ServerError::Conflict(format!(
                "model '{name}' is already loaded"
            )));
        }
    }

    // Resolve the path without prompting -- the server has no interactive stdin.
    let resolved = model_resolver::resolve(&model_cfg.path).map_err(|e| {
        ServerError::BadRequest(format!(
            "model '{}' not found locally: {e}; pre-download it (e.g. `huggingface-cli download {}`)",
            model_cfg.path, model_cfg.path
        ))
    })?;

    // The weight load is blocking and GPU-bound; keep it off the async runtime.
    let local = state.config.local.clone();
    let cfg = model_cfg.clone();
    let (name, engine) = tokio::task::spawn_blocking(move || build_engine(&resolved, &cfg, &local))
        .await
        .map_err(|e| ServerError::InternalError(format!("model load task failed: {e}")))?
        .map_err(ServerError::BadRequest)?;
    let vision = engine.is_vlm();

    state
        .router
        .insert_engine(name.clone(), Arc::new(engine))
        .map_err(|n| ServerError::Conflict(format!("model '{n}' is already loaded")))?;

    tracing::info!(model_name = %name, "Model loaded at runtime");
    Ok(Json(model_object(name, vision)))
}

/// `DELETE /v1/models/{name}` -- unload a model and free its GPU memory.
///
/// Returns `204` once the model is fully unloaded, `202` if a request was still
/// in flight past the drain timeout (the final free is detached), `404` if no
/// such model is loaded, or `409` if the model backs the auto-router.
pub async fn unload_model(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    if state.router.auto_router_model_name() == Some(name.as_str()) {
        return Err(ServerError::Conflict(format!(
            "model '{name}' is bound to the auto-router and cannot be unloaded"
        )));
    }

    let engine = state
        .router
        .remove_engine(&name)
        .ok_or_else(|| ServerError::ModelNotFound(name.clone()))?;

    // The map entry is gone, so no new request can take a reference and the
    // strong count only decreases. Free GPU memory once the last in-flight
    // request releases its clone; detach past the timeout so a long generation
    // can't block the response.
    if drain_and_drop(engine, DRAIN_TIMEOUT).await {
        tracing::info!(model_name = %name, "Model unloaded");
        Ok(StatusCode::NO_CONTENT)
    } else {
        tracing::info!(model_name = %name, "Model unload deferred; request still in flight");
        Ok(StatusCode::ACCEPTED)
    }
}

/// Wait until `engine` is solely owned here, then drop it (freeing GPU memory).
///
/// Returns `true` if dropped within `timeout`; `false` if it timed out and the
/// final drop was handed to a detached task. Dropping is intentionally not gated
/// on the process-wide GPU gate: engine teardown frees MLX buffers but never
/// runs an `eval`, so it cannot race the cross-model output-array table that the
/// gate protects.
async fn drain_and_drop(mut engine: Arc<Engine>, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                drop(owned);
                // The engine's KV/prefix caches and weights are now freed, but
                // MLX parks the buffers in its allocator pool — return them to
                // the OS so a free-then-load swap actually reclaims memory. No
                // `eval`, so this is safe outside the GPU gate (same as `drop`).
                higgs_engine::simple::maybe_clear_mlx_cache(true, "model unload/swap");
                return true;
            }
            Err(shared) => {
                if start.elapsed() >= timeout {
                    tokio::spawn(drain_in_background(shared));
                    return false;
                }
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

/// Poll until the detached engine reference is sole-owned, then drop it.
async fn drain_in_background(mut engine: Arc<Engine>) {
    loop {
        match Arc::try_unwrap(engine) {
            Ok(owned) => {
                drop(owned);
                higgs_engine::simple::maybe_clear_mlx_cache(true, "model unload/swap (deferred)");
                return;
            }
            Err(shared) => {
                engine = shared;
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

fn model_object(name: String, vision: bool) -> ModelObject {
    ModelObject {
        id: name,
        object: "model",
        created: chrono::Utc::now().timestamp(),
        owned_by: "local".to_owned(),
        vision,
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::router::Router;
    use crate::state::AppState;

    fn build_state(toml: &str, engines: HashMap<String, Arc<Engine>>) -> SharedState {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        // A stub provider satisfies the "at least one [[models]] or [provider.*]"
        // config rule; it does not affect direct local-engine routing.
        let full = format!("[provider.stub]\nurl = \"http://127.0.0.1:1\"\n\n{toml}");
        std::fs::write(&path, full).unwrap();
        let config = crate::config::load_config_file(&path, None).unwrap();
        let router = Router::from_config(&config, engines).unwrap();
        Arc::new(AppState {
            router,
            config,
            http_client: reqwest::Client::new(),
            metrics: None,
        })
    }

    fn stub_engines(names: &[&str]) -> HashMap<String, Arc<Engine>> {
        names
            .iter()
            .map(|n| ((*n).to_owned(), Arc::new(Engine::test_stub(n))))
            .collect()
    }

    #[test]
    fn test_model_objects_have_correct_fields() {
        let obj = model_object("test-model".to_owned(), false);
        assert_eq!(obj.object, "model");
        assert_eq!(obj.owned_by, "local");
        assert!(obj.created > 0);
        assert!(!obj.vision);
    }

    #[tokio::test]
    async fn list_models_reports_capabilities() {
        // runtime_model_load mirrors the config flag; vision is per-model
        // (stub engines are text-only); names stay sorted.
        let on = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["zebra", "alpha"]),
        );
        let Json(list) = list_models(State(on)).await;
        assert!(list.runtime_model_load);
        let ids: Vec<&str> = list.data.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "zebra"]);
        assert!(list.data.iter().all(|m| !m.vision));

        let off = build_state(
            "[local]\nallow_runtime_model_load = false\n",
            HashMap::new(),
        );
        let Json(list) = list_models(State(off)).await;
        assert!(!list.runtime_model_load);
    }

    #[tokio::test]
    async fn load_disabled_returns_forbidden() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = false\n",
            HashMap::new(),
        );
        let err = load_model(State(state), Bytes::from_static(b"{\"path\":\"x\"}"))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::Forbidden(_)));
    }

    #[tokio::test]
    async fn load_existing_name_conflicts() {
        // allow_runtime on + an already-loaded "llama"; naming it again is a
        // conflict caught before any (impossible, in-test) GPU load.
        let state = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["llama"]),
        );
        let body = Bytes::from_static(b"{\"path\":\"some/path\",\"name\":\"llama\"}");
        let err = load_model(State(state), body).await.unwrap_err();
        assert!(matches!(err, ServerError::Conflict(_)));
    }

    #[tokio::test]
    async fn unload_unknown_returns_not_found() {
        let state = build_state("[local]\nallow_runtime_model_load = true\n", HashMap::new());
        let err = unload_model(State(state), Path("ghost".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::ModelNotFound(_)));
    }

    #[tokio::test]
    async fn unload_auto_router_model_is_refused() {
        let toml = r#"
            [[models]]
            path = "/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "router"
        "#;
        let state = build_state(toml, stub_engines(&["router"]));
        let err = unload_model(State(Arc::clone(&state)), Path("router".to_owned()))
            .await
            .unwrap_err();
        assert!(matches!(err, ServerError::Conflict(_)));
        // Still loaded.
        assert!(state.router.contains_engine("router"));
    }

    #[tokio::test]
    async fn unload_removes_engine_and_frees_it() {
        let state = build_state(
            "[local]\nallow_runtime_model_load = true\n",
            stub_engines(&["llama"]),
        );
        let status = unload_model(State(Arc::clone(&state)), Path("llama".to_owned()))
            .await
            .unwrap();
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(!state.router.contains_engine("llama"));
        assert!(state.router.local_model_names().is_empty());
    }

    #[tokio::test]
    async fn drain_and_drop_drops_sole_owner_immediately() {
        let engine = Arc::new(Engine::test_stub("solo"));
        assert!(drain_and_drop(engine, Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn drain_and_drop_waits_for_in_flight_clone() {
        let engine = Arc::new(Engine::test_stub("busy"));
        let clone = Arc::clone(&engine);
        // Release the simulated in-flight reference after a short delay.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(120)).await;
            drop(clone);
        });
        let start = Instant::now();
        assert!(drain_and_drop(engine, Duration::from_secs(5)).await);
        assert!(
            start.elapsed() >= Duration::from_millis(100),
            "should have waited for the clone to drop"
        );
    }

    #[tokio::test]
    async fn drain_and_drop_times_out_and_detaches() {
        let engine = Arc::new(Engine::test_stub("stuck"));
        let clone = Arc::clone(&engine); // held past the timeout
        let drained = drain_and_drop(engine, Duration::from_millis(80)).await;
        assert!(!drained, "should time out while a reference is held");
        drop(clone); // let the detached task finish
    }
}
