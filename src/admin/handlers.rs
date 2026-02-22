use axum::{
    extract::{ConnectInfo, Form, Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect, Response},
};
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tower_sessions::Session;

use crate::{
    admin::{lockout::LoginAttemptTracker, session, templates},
    auth::{store::TokenStore, token::{generate_token_value, Token}},
    config::AppConfig,
};

#[derive(Clone)]
pub struct AdminState {
    pub token_store: Arc<TokenStore>,
    pub lockout: Arc<LoginAttemptTracker>,
    pub config: Arc<AppConfig>,
}

pub async fn get_login(session: Session) -> Response {
    if session::is_authenticated(&session).await {
        return Redirect::to("/admin").into_response();
    }
    Html(templates::login_page(None)).into_response()
}

#[derive(Deserialize)]
pub struct LoginForm {
    pub id: String,
    pub pw: String,
}

pub async fn post_login(
    State(state): State<AdminState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    session: Session,
    Form(form): Form<LoginForm>,
) -> Response {
    let ip = addr.ip();

    if state.lockout.is_locked(ip) {
        return Html(templates::login_page(Some("Too many failed attempts. Access blocked.")))
            .into_response();
    }

    if form.id == state.config.admin_id && form.pw == state.config.admin_pw {
        state.lockout.record_success(ip);
        session::set_authenticated(&session, true).await;
        Redirect::to("/admin").into_response()
    } else {
        state.lockout.record_failure(ip);
        Html(templates::login_page(Some("Invalid credentials."))).into_response()
    }
}

pub async fn get_logout(session: Session) -> Redirect {
    session::destroy(&session).await;
    Redirect::to("/admin/login")
}

pub async fn get_admin(
    State(state): State<AdminState>,
    session: Session,
) -> Response {
    if !session::is_authenticated(&session).await {
        return Redirect::to("/admin/login").into_response();
    }

    let tokens = state.token_store.list().await;
    Html(templates::admin_page(&tokens)).into_response()
}

#[derive(Deserialize)]
pub struct CreateTokenForm {
    pub expires_at: Option<String>,
}

pub async fn post_create_token(
    State(state): State<AdminState>,
    session: Session,
    Form(form): Form<CreateTokenForm>,
) -> Response {
    if !session::is_authenticated(&session).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let expires_at = form.expires_at.as_deref().filter(|s| !s.is_empty()).and_then(|s| {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M")
            .ok()
            .map(|ndt| DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc))
    });

    let token = Token::new(generate_token_value(), expires_at);
    if let Err(e) = state.token_store.add(token).await {
        tracing::error!("Failed to save token: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    Redirect::to("/admin").into_response()
}

pub async fn post_revoke_token(
    State(state): State<AdminState>,
    session: Session,
    Path(id): Path<String>,
) -> Response {
    if !session::is_authenticated(&session).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    if let Err(e) = state.token_store.revoke(&id).await {
        tracing::error!("Failed to revoke token: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    Redirect::to("/admin").into_response()
}
