pub mod index;

use axum::{routing::get, Router};
use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index::get_index))
        .route("/health", get(index::get_health))
        .with_state(state)
}
