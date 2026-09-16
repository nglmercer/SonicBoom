pub mod index;

use crate::AppState;
use axum::{Router, routing::get};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index::get_index))
        .route("/health", get(index::get_health))
        .route("/ready", get(index::get_ready))
        .with_state(state)
}
