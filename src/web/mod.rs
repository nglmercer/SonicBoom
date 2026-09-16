pub mod index;
pub mod static_files;

use crate::AppState;
use axum::{Router, routing::get};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index::get_index))
        .route("/health", get(index::get_health))
        .route("/ready", get(index::get_ready))
        .route("/static/{file}", get(static_files::get_static))
        .with_state(state)
}
