//! Axum HTTP server setup.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;
use vllm_tokenizer::HuggingFaceTokenizer;

use crate::chat_template::ChatTemplateRenderer;
use crate::routes;
use crate::routing::PipelineRouter;

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub router: Arc<PipelineRouter>,
    pub tokenizer: Arc<HuggingFaceTokenizer>,
    /// `None` if the pipeline has no text-output stage, or if one exists
    /// but no chat template could be found for it -- either way,
    /// `/v1/chat/completions` returns 501 rather than failing startup.
    pub chat_template: Option<Arc<ChatTemplateRenderer>>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/audio/speech", post(routes::speech::create_speech))
        .route(
            "/v1/chat/completions",
            post(routes::chat::create_chat_completion),
        )
        .layer(CorsLayer::permissive())
        .with_state(state)
}
