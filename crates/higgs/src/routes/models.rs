use axum::{
    Json,
    extract::{Path, Query, State},
};
use serde::{Deserialize, Serialize};

use crate::{
    error::ServerError,
    model_pool::{ModelState, ModelStatus},
    state::SharedState,
    types::openai::{ModelList, ModelObject},
};

/// Lists every configured model, loaded or not, so clients can discover what is available.
/// Never loads anything.
pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let statuses = state.router.pool().statuses();
    let data = model_objects_sorted(
        statuses
            .iter()
            .map(|status| (status.name.as_str(), status.state == ModelState::Loaded)),
    );
    Json(ModelList {
        object: "list",
        data,
    })
}

/// Build a sorted, stable list of [`ModelObject`]s from `(name, loaded)` pairs.
fn model_objects_sorted<'a>(models: impl Iterator<Item = (&'a str, bool)>) -> Vec<ModelObject> {
    let mut sorted: Vec<(&str, bool)> = models.collect();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|(name, loaded)| ModelObject {
            id: name.to_owned(),
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_owned(),
            loaded,
        })
        .collect()
}

#[derive(Debug, Serialize)]
pub struct ModelStatusList {
    pub object: &'static str,
    pub data: Vec<ModelStatus>,
}

/// `GET /v1/models/status`: lifecycle state of every configured local model. Never loads
/// anything.
pub async fn models_status(State(state): State<SharedState>) -> Json<ModelStatusList> {
    Json(ModelStatusList {
        object: "list",
        data: state.router.pool().statuses(),
    })
}

#[derive(Debug, Default, Deserialize)]
pub struct ModelActionQuery {
    /// Unload only: wait for in-flight requests to finish instead of refusing.
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelAction {
    Load,
    Unload,
}

/// Split `"<model id>/load"` or `"<model id>/unload"`. Model ids may contain slashes
/// (`org/name`), so this is matched on the whole remaining path.
fn parse_action(target: &str) -> Option<(&str, ModelAction)> {
    target
        .strip_suffix("/unload")
        .map(|id| (id, ModelAction::Unload))
        .or_else(|| {
            target
                .strip_suffix("/load")
                .map(|id| (id, ModelAction::Load))
        })
        .filter(|(id, _)| !id.is_empty())
}

/// `POST /v1/models/{id}/load` and `POST /v1/models/{id}/unload[?force=true]`.
pub async fn model_action(
    State(state): State<SharedState>,
    Path(target): Path<String>,
    Query(query): Query<ModelActionQuery>,
) -> Result<Json<ModelStatus>, ServerError> {
    let (model, action) = parse_action(&target).ok_or_else(|| {
        ServerError::ModelNotFound(format!(
            "no such endpoint: POST /v1/models/{target} (expected /load or /unload)"
        ))
    })?;
    let pool = state.router.pool();
    let status = match action {
        ModelAction::Load => pool.load(model).await?,
        ModelAction::Unload => pool.unload(model, query.force).await?,
    };
    Ok(Json(status))
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_model_list_is_sorted_alphabetically() {
        let names = ["zebra", "alpha", "middle"];
        let data = model_objects_sorted(names.iter().map(|name| (*name, false)));
        let ids: Vec<&str> = data.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "middle", "zebra"]);
    }

    #[test]
    fn test_model_list_empty_input() {
        let data = model_objects_sorted(std::iter::empty());
        assert!(data.is_empty());
    }

    #[test]
    fn test_model_list_single_entry() {
        let data = model_objects_sorted(std::iter::once(("only-model", false)));
        assert_eq!(data.len(), 1);
        assert_eq!(data.first().map(|m| m.id.as_str()), Some("only-model"));
    }

    #[test]
    fn test_model_objects_have_correct_fields() {
        let data = model_objects_sorted(std::iter::once(("test-model", true)));
        let obj = data.first().unwrap();
        assert_eq!(obj.object, "model");
        assert_eq!(obj.owned_by, "local");
        assert!(obj.created > 0);
        assert!(obj.loaded);
    }

    #[test]
    fn parse_action_splits_the_model_id_from_the_verb() {
        assert_eq!(parse_action("a/load"), Some(("a", ModelAction::Load)));
        assert_eq!(parse_action("a/unload"), Some(("a", ModelAction::Unload)));
        assert_eq!(
            parse_action("org/name/unload"),
            Some(("org/name", ModelAction::Unload))
        );
        assert_eq!(parse_action("a/restart"), None);
        assert_eq!(parse_action("/load"), None);
        assert_eq!(parse_action("load"), None);
    }

    // -- Endpoints --

    use axum::body::Body;
    use http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    use crate::model_pool::{ModelSpec, PoolLimits};
    use crate::state::test_support::{StubLoader, stub_spec, stub_state};

    struct Harness {
        app: axum::Router,
        state: SharedState,
        loader: Arc<StubLoader>,
    }

    fn harness_with(
        specs: Vec<ModelSpec>,
        limits: PoolLimits,
        api_key: Option<&str>,
        metrics: Option<Arc<crate::metrics::MetricsStore>>,
    ) -> Harness {
        let (state, loader) = stub_state(specs, limits, metrics);
        let app = crate::build_router(
            Arc::clone(&state),
            300.0,
            api_key.map(str::to_owned),
            0,
            1024 * 1024,
            None,
        );
        Harness { app, state, loader }
    }

    fn harness(names: &[&str]) -> Harness {
        harness_with(
            names.iter().map(|name| stub_spec(name)).collect(),
            PoolLimits::default(),
            None,
            None,
        )
    }

    impl Harness {
        async fn send(&self, method: &str, uri: &str) -> (StatusCode, Value) {
            self.send_with_auth(method, uri, None).await
        }

        async fn send_with_auth(
            &self,
            method: &str,
            uri: &str,
            bearer: Option<&str>,
        ) -> (StatusCode, Value) {
            let mut builder = Request::builder().method(method).uri(uri);
            if let Some(token) = bearer {
                builder = builder.header("authorization", format!("Bearer {token}"));
            }
            let response = self
                .app
                .clone()
                .oneshot(builder.body(Body::empty()).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            (
                status,
                serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            )
        }

        fn events(&self) -> Vec<String> {
            self.loader.events.lock().unwrap().clone()
        }
    }

    fn model<'a>(body: &'a Value, id: &str) -> &'a Value {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == id || entry["name"] == id)
            .unwrap_or_else(|| panic!("{id} not listed in {body}"))
    }

    #[tokio::test]
    async fn models_list_includes_unloaded_models_and_never_loads() {
        let fx = harness(&["beta", "alpha"]);
        fx.state.router.pool().load("beta").await.unwrap();

        let (status, body) = fx.send("GET", "/v1/models").await;

        assert_eq!(status, StatusCode::OK);
        let ids: Vec<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["alpha", "beta"]);
        assert_eq!(model(&body, "alpha")["loaded"], false);
        assert_eq!(model(&body, "beta")["loaded"], true);
        assert_eq!(fx.events(), ["load:beta"]);
    }

    #[tokio::test]
    async fn status_reports_every_model_without_loading() {
        let pinned = ModelSpec {
            pinned: true,
            idle_ttl: Some(std::time::Duration::from_secs(90)),
            decode: "batched",
            ..stub_spec("kept")
        };
        let fx = harness_with(
            vec![stub_spec("plain"), pinned],
            PoolLimits::default(),
            None,
            None,
        );

        let (status, body) = fx.send("GET", "/v1/models/status").await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        let plain = model(&body, "plain");
        assert_eq!(plain["state"], "unloaded");
        assert_eq!(plain["pinned"], false);
        assert_eq!(plain["in_flight"], 0);
        assert_eq!(plain["last_used_at"], Value::Null);
        assert_eq!(plain["engine"], "sequential");
        assert_eq!(plain["load"], "on_demand");
        let kept = model(&body, "kept");
        assert_eq!(kept["pinned"], true);
        assert_eq!(kept["idle_ttl_seconds"], 90);
        assert_eq!(kept["engine"], "batched");
        for field in [
            "last_error",
            "load_duration_ms",
            "estimated_bytes",
            "measured_bytes",
        ] {
            assert!(plain.get(field).is_some(), "missing {field}");
        }
        assert!(fx.events().is_empty());
    }

    #[tokio::test]
    async fn load_endpoint_loads_the_model_and_reports_its_status() {
        let fx = harness(&["a", "b"]);

        let (status, body) = fx.send("POST", "/v1/models/a/load").await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["name"], "a");
        assert_eq!(body["state"], "loaded");
        assert!(body["load_duration_ms"].is_u64());
        assert_eq!(fx.events(), ["load:a"]);

        let (_, again) = fx.send("POST", "/v1/models/a/load").await;
        assert_eq!(again["state"], "loaded");
        assert_eq!(fx.events(), ["load:a"], "loading a loaded model is a no-op");
    }

    #[tokio::test]
    async fn load_of_an_unknown_model_is_404() {
        let fx = harness(&["a"]);

        let (status, body) = fx.send("POST", "/v1/models/nope/load").await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["error"]["message"].as_str().unwrap().contains("nope"));
    }

    #[tokio::test]
    async fn failed_load_is_a_503_and_retryable() {
        let fx = harness(&["a"]);
        *fx.loader.fail_with.lock().unwrap() = Some("weights are corrupt".to_owned());

        let (status, body) = fx.send("POST", "/v1/models/a/load").await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "model_load_failed");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("weights are corrupt")
        );
        let (_, listed) = fx.send("GET", "/v1/models/status").await;
        assert_eq!(model(&listed, "a")["state"], "failed");
        assert!(
            model(&listed, "a")["last_error"]
                .as_str()
                .unwrap()
                .contains("weights are corrupt")
        );

        *fx.loader.fail_with.lock().unwrap() = None;
        let (retry_status, retry) = fx.send("POST", "/v1/models/a/load").await;
        assert_eq!(retry_status, StatusCode::OK);
        assert_eq!(retry["state"], "loaded");
    }

    #[tokio::test]
    async fn refused_load_names_the_model_and_the_limit() {
        let kept = ModelSpec {
            pinned: true,
            ..stub_spec("kept")
        };
        let fx = harness_with(
            vec![kept, stub_spec("wanted")],
            PoolLimits {
                max_loaded_models: Some(1),
                memory_ceiling_bytes: None,
            },
            None,
            None,
        );
        fx.send("POST", "/v1/models/kept/load").await;

        let (status, body) = fx.send("POST", "/v1/models/wanted/load").await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "insufficient_memory");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("'wanted'"), "{message}");
        assert!(message.contains("'kept'"), "{message}");
        assert!(message.contains("pinned"), "{message}");
    }

    #[tokio::test]
    async fn unload_endpoint_frees_the_model() {
        let fx = harness(&["a"]);
        fx.send("POST", "/v1/models/a/load").await;

        let (status, body) = fx.send("POST", "/v1/models/a/unload").await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["state"], "unloaded");
        assert_eq!(fx.events(), ["load:a", "unload:a"]);
    }

    #[tokio::test]
    async fn unload_with_requests_in_flight_is_409() {
        let fx = harness(&["a"]);
        let lease = fx.state.router.pool().acquire("a").await.unwrap();

        let (status, body) = fx.send("POST", "/v1/models/a/unload").await;

        assert_eq!(status, StatusCode::CONFLICT);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("in flight")
        );
        let (_, listed) = fx.send("GET", "/v1/models/status").await;
        assert_eq!(model(&listed, "a")["state"], "loaded");
        assert_eq!(model(&listed, "a")["in_flight"], 1);
        drop(lease);
    }

    #[tokio::test]
    async fn forced_unload_waits_for_the_stream_to_finish() {
        let fx = harness(&["a"]);
        let lease = fx.state.router.pool().acquire("a").await.unwrap();

        let app = fx.app.clone();
        let unloading = tokio::spawn(async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/models/a/unload?force=true")
                .body(Body::empty())
                .unwrap();
            app.oneshot(request).await.unwrap().status()
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        assert!(!unloading.is_finished());
        let (_, listed) = fx.send("GET", "/v1/models/status").await;
        assert_eq!(model(&listed, "a")["state"], "unloading");
        assert_eq!(fx.events(), ["load:a"]);

        drop(lease);
        assert_eq!(unloading.await.unwrap(), StatusCode::OK);
        assert_eq!(fx.events(), ["load:a", "unload:a"]);
    }

    #[tokio::test]
    async fn unload_of_an_unloaded_model_succeeds() {
        let fx = harness(&["a"]);

        let (status, body) = fx.send("POST", "/v1/models/a/unload").await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["state"], "unloaded");
    }

    #[tokio::test]
    async fn model_ids_with_slashes_work_raw_or_percent_encoded() {
        let fx = harness(&["org/model"]);

        let (raw_status, _) = fx.send("POST", "/v1/models/org/model/load").await;
        let (encoded_status, encoded) = fx.send("POST", "/v1/models/org%2Fmodel/unload").await;

        assert_eq!(raw_status, StatusCode::OK);
        assert_eq!(encoded_status, StatusCode::OK);
        assert_eq!(encoded["name"], "org/model");
        assert_eq!(encoded["state"], "unloaded");
    }

    #[tokio::test]
    async fn unknown_action_is_404() {
        let fx = harness(&["a"]);

        let (status, _) = fx.send("POST", "/v1/models/a/restart").await;

        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoints_require_the_api_key_like_the_rest_of_the_api() {
        let fx = harness_with(
            vec![stub_spec("a")],
            PoolLimits::default(),
            Some("secret"),
            None,
        );

        for (method, uri) in [
            ("GET", "/v1/models/status"),
            ("POST", "/v1/models/a/load"),
            ("POST", "/v1/models/a/unload"),
        ] {
            let (denied, _) = fx.send(method, uri).await;
            assert_eq!(denied, StatusCode::UNAUTHORIZED, "{method} {uri}");
        }
        assert!(fx.events().is_empty());

        let (allowed, _) = fx
            .send_with_auth("POST", "/v1/models/a/load", Some("secret"))
            .await;
        assert_eq!(allowed, StatusCode::OK);
    }

    #[tokio::test]
    async fn dashboard_and_control_endpoints_are_not_counted_as_traffic() {
        let metrics = Arc::new(crate::metrics::MetricsStore::new(
            std::time::Duration::from_secs(60),
        ));
        let fx = harness_with(
            vec![stub_spec("a")],
            PoolLimits::default(),
            None,
            Some(Arc::clone(&metrics)),
        );

        fx.send("GET", "/v1/models").await;
        fx.send("GET", "/v1/models/status").await;
        fx.send("GET", "/v1/system").await;
        fx.send("POST", "/v1/models/a/load").await;
        fx.send("POST", "/v1/models/a/unload").await;
        fx.send("POST", "/v1/models/missing/load").await;

        assert!(metrics.snapshot().is_empty());
    }

    #[tokio::test]
    async fn system_endpoint_reports_the_same_model_states() {
        let fx = harness(&["a", "b"]);
        fx.send("POST", "/v1/models/a/load").await;

        let (status, body) = fx.send("GET", "/v1/system").await;

        assert_eq!(status, StatusCode::OK);
        let models = body["models"].as_array().unwrap();
        let find = |name: &str| models.iter().find(|entry| entry["name"] == name).unwrap();
        assert_eq!(find("a")["state"], "loaded");
        assert_eq!(find("b")["state"], "unloaded");
        assert_eq!(find("b")["engine"], "sequential");
        assert_eq!(
            fx.events(),
            ["load:a"],
            "reading system state must not load"
        );
    }
}
