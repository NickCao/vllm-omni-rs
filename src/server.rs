//! Axum HTTP server setup.
//! TODO: Wire routes to ZMQ-based engine core clients.

use axum::routing::get;
use axum::Router;
use tower_http::cors::CorsLayer;

use crate::routes;

#[derive(Clone)]
pub struct AppState {
    pub model_name: String,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::models::list_models))
        // TODO: .route("/v1/audio/speech", post(routes::speech::create_speech))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
