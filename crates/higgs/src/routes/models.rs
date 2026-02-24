use axum::{Json, extract::State};

use crate::{
    state::SharedState,
    types::openai::{ModelList, ModelObject},
};

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelList> {
    let data = model_objects_sorted(state.engines.keys().map(String::as_str));
    Json(ModelList {
        object: "list",
        data,
    })
}

/// Build a sorted, stable list of [`ModelObject`]s from an iterator of model names.
fn model_objects_sorted<'a>(names: impl Iterator<Item = &'a str>) -> Vec<ModelObject> {
    let mut sorted: Vec<&str> = names.collect();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|name| ModelObject {
            id: name.to_owned(),
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_owned(),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_model_list_is_sorted_alphabetically() {
        let names = ["zebra", "alpha", "middle"];
        let data = model_objects_sorted(names.iter().copied());
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
        let data = model_objects_sorted(std::iter::once("only-model"));
        assert_eq!(data.len(), 1);
        assert_eq!(data.first().map(|m| m.id.as_str()), Some("only-model"));
    }

    #[test]
    fn test_model_objects_have_correct_fields() {
        let data = model_objects_sorted(std::iter::once("test-model"));
        let obj = data.first().unwrap();
        assert_eq!(obj.object, "model");
        assert_eq!(obj.owned_by, "local");
        assert!(obj.created > 0);
    }
}
