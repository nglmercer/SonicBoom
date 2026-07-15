pub mod index;

use crate::AppState;
use axum::{Router, routing::get};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index::get_index))
        .route("/health", get(index::get_health))
        .with_state(state)
}
