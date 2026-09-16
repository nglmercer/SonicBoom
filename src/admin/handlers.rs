use axum::{
    extract::{ConnectInfo, Form, Path, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
};
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tower_sessions::Session;

use crate::{
    admin::{client_ip, lockout::LoginAttemptTracker, session, templates},
    auth::store::TokenStore,
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
    headers: HeaderMap,
    session: Session,
    Form(form): Form<LoginForm>,
) -> Response {
    let ip = client_ip::client_ip(&headers, addr, &state.config);

    if state.lockout.is_locked(ip) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Html(templates::login_page(Some(
                "Too many failed attempts. Try again later.",
            ))),
        )
            .into_response();
    }

    // Constant-time comparison to prevent timing attacks on credential validation
    let id_match =
        constant_time_eq::constant_time_eq(form.id.as_bytes(), state.config.admin_id.as_bytes());
    let pw_match =
        constant_time_eq::constant_time_eq(form.pw.as_bytes(), state.config.admin_pw.as_bytes());

    if id_match && pw_match {
        state.lockout.record_success(ip);
        session::rotate_id(&session).await;
        session::set_authenticated(&session, true).await;
        Redirect::to("/admin").into_response()
    } else {
        state.lockout.record_failure(ip);
        (
            StatusCode::UNAUTHORIZED,
            Html(templates::login_page(Some("Invalid credentials."))),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
pub struct CsrfForm {
    pub csrf_token: Option<String>,
}

pub async fn post_logout(session: Session, Form(form): Form<CsrfForm>) -> Response {
    let submitted = form.csrf_token.as_deref().unwrap_or("");
    if !session::validate_csrf(&session, submitted).await {
        return StatusCode::FORBIDDEN.into_response();
    }
    session::destroy(&session).await;
    Redirect::to("/admin/login").into_response()
}

pub async fn get_admin(State(state): State<AdminState>, session: Session) -> Response {
    if !session::is_authenticated(&session).await {
        return Redirect::to("/admin/login").into_response();
    }

    let tokens = state.token_store.list().await;
    let csrf = session::csrf_token(&session).await;
    Html(templates::admin_page(&tokens, &csrf, None)).into_response()
}

#[derive(Deserialize)]
pub struct CreateTokenForm {
    pub expires_at: Option<String>,
    pub csrf_token: Option<String>,
}

pub async fn post_create_token(
    State(state): State<AdminState>,
    session: Session,
    Form(form): Form<CreateTokenForm>,
) -> Response {
    if !session::is_authenticated(&session).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let submitted = form.csrf_token.as_deref().unwrap_or("");
    if !session::validate_csrf(&session, submitted).await {
        return StatusCode::FORBIDDEN.into_response();
    }

    let expires_at = match parse_token_expiry(form.expires_at.as_deref()) {
        Ok(expires_at) => expires_at,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Html(templates::error_page(message)),
            )
                .into_response();
        }
    };

    let (token, raw) = match state.token_store.create(expires_at).await {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("Failed to save token: {e}");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    let _ = token;

    // Show the raw token exactly once; it is never persisted or re-displayed.
    let tokens = state.token_store.list().await;
    let csrf = session::csrf_token(&session).await;
    Html(templates::admin_page(&tokens, &csrf, Some(&raw))).into_response()
}

/// Parse the admin `datetime-local` expiry field.
///
/// - Missing/blank → `None` (the token never expires).
/// - Valid `YYYY-MM-DDTHH:MM` in the future → `Some(expiry)`. The field
///   carries no timezone, so the server interprets it as UTC (documented
///   on the form and in `docs/admin.md`).
/// - Malformed or already past → `Err`, rendered as `400`. A typo must
///   never silently mint a perpetual token.
pub fn parse_token_expiry(raw: Option<&str>) -> Result<Option<DateTime<Utc>>, &'static str> {
    let Some(text) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let naive = NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M")
        .map_err(|_| "expiry must look like YYYY-MM-DDTHH:MM")?;
    let expiry = DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
    if expiry <= Utc::now() {
        return Err("expiry must be in the future");
    }
    Ok(Some(expiry))
}

#[derive(Deserialize)]
pub struct RevokeTokenForm {
    pub csrf_token: Option<String>,
}

pub async fn post_revoke_token(
    State(state): State<AdminState>,
    session: Session,
    Path(id): Path<String>,
    Form(form): Form<RevokeTokenForm>,
) -> Response {
    if !session::is_authenticated(&session).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let submitted = form.csrf_token.as_deref().unwrap_or("");
    if !session::validate_csrf(&session, submitted).await {
        return StatusCode::FORBIDDEN.into_response();
    }

    if let Err(e) = state.token_store.revoke(&id).await {
        tracing::error!("Failed to revoke token: {e}");
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    Redirect::to("/admin").into_response()
}

#[cfg(test)]
mod tests {
    use super::parse_token_expiry;

    #[test]
    fn blank_expiry_means_no_expiry() {
        assert_eq!(parse_token_expiry(None), Ok(None));
        assert_eq!(parse_token_expiry(Some("")), Ok(None));
        assert_eq!(parse_token_expiry(Some("   ")), Ok(None));
    }

    #[test]
    fn future_expiry_parses_as_utc() {
        let expiry = parse_token_expiry(Some("2999-01-02T03:04"))
            .unwrap()
            .unwrap();
        assert_eq!(expiry.to_rfc3339(), "2999-01-02T03:04:00+00:00");
    }

    #[test]
    fn malformed_expiry_is_rejected_not_perpetual() {
        for bad in [
            "banana",
            "2026-13-01T00:00",
            "2026-01-01",
            "2026-01-01T25:00",
            "2026-01-01 00:00",
            "2999-01-02T03:04:05",
            "2999-01-02T03:04+00:00",
        ] {
            assert!(
                parse_token_expiry(Some(bad)).is_err(),
                "expiry {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn past_and_present_expiry_is_rejected() {
        assert!(parse_token_expiry(Some("2000-01-01T00:00")).is_err());
        // Far past with valid syntax is still rejected.
        assert!(parse_token_expiry(Some("1970-01-01T00:00")).is_err());
    }
}
