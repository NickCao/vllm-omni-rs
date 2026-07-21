//! Axum HTTP server setup.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use crate::engine::OmniEngine;
use crate::routes;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<OmniEngine>,
    pub model_name: String,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/audio/speech", post(routes::speech::create_speech))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
