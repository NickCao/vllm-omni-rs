//! Axum HTTP server setup.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;

use crate::routes;
use crate::routing::TtsRouter;

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
    pub router: Arc<TtsRouter>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/audio/speech", post(routes::speech::create_speech))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
