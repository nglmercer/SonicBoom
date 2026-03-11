pub mod store;
pub mod token;

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};

use crate::AppState;

pub struct AuthenticatedToken(#[allow(dead_code)] pub String);

impl FromRequestParts<AppState> for AuthenticatedToken {
    type Rejection = StatusCode;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        // If auth is not required, allow all requests
        if !state.config.auth_required {
            return Ok(AuthenticatedToken("__no_auth__".to_string()));
        }

        // Allow without auth if Referer is same host
        if is_self_referer(parts) {
            return Ok(AuthenticatedToken("__self__".to_string()));
        }

        let TypedHeader(Authorization(bearer)) =
            TypedHeader::<Authorization<Bearer>>::from_request_parts(parts, state)
                .await
                .map_err(|_| StatusCode::UNAUTHORIZED)?;

        let token_value = bearer.token().to_string();

        if state.config.enable_sample_token && token_value == "SAMPLE_TOKEN" {
            return Ok(AuthenticatedToken(token_value));
        }

        if state.token_store.validate(&token_value).await {
            Ok(AuthenticatedToken(token_value))
        } else {
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Returns true if Referer header points to the same host as the request
fn is_self_referer(parts: &Parts) -> bool {
    let referer = match parts.headers.get("referer") {
        Some(v) => match v.to_str() {
            Ok(s) => s,
            Err(_) => return false,
        },
        None => return false,
    };

    let host = match parts.headers.get("host") {
        Some(v) => match v.to_str() {
            Ok(s) => s,
            Err(_) => return false,
        },
        None => return false,
    };

    // Check if referer is in "http(s)://<host>/..." form
    referer
        .strip_prefix("http://")
        .or_else(|| referer.strip_prefix("https://"))
        .map(|rest| rest == host || rest.starts_with(&format!("{host}/")))
        .unwrap_or(false)
}
