use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

pub async fn list_models(
    axum::extract::State(state): axum::extract::State<crate::server::AppState>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model",
            owned_by: "vllm-omni",
        }],
    })
}
