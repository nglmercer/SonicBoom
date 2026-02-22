pub mod handlers;
pub mod lockout;
pub mod session;
pub mod templates;

use axum::{routing::{get, post}, Router};
use handlers::AdminState;

pub fn router(state: AdminState) -> Router {
    Router::new()
        .route("/admin", get(handlers::get_admin))
        .route("/admin/login", get(handlers::get_login).post(handlers::post_login))
        .route("/admin/logout", get(handlers::get_logout))
        .route("/admin/tokens", post(handlers::post_create_token))
        .route("/admin/tokens/{id}/revoke", post(handlers::post_revoke_token))
        .with_state(state)
}
