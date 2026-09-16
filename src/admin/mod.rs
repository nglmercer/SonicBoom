pub mod client_ip;
pub mod handlers;
pub mod lockout;
pub mod session;
pub mod templates;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use handlers::AdminState;

pub fn router(state: AdminState) -> Router {
    let body_limit = state.config.admin_max_body_bytes;
    Router::new()
        .route("/admin", get(handlers::get_admin))
        .route(
            "/admin/login",
            get(handlers::get_login).post(handlers::post_login),
        )
        // Logout changes session state, so it must be POST (with CSRF).
        .route("/admin/logout", post(handlers::post_logout))
        .route("/admin/tokens", post(handlers::post_create_token))
        .route(
            "/admin/tokens/{id}/revoke",
            post(handlers::post_revoke_token),
        )
        .route_layer(DefaultBodyLimit::max(body_limit))
        .with_state(state)
}
