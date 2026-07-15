pub mod store;
pub mod token;

use axum::{
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
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

        let auth_header = parts
            .headers
            .get("authorization")
            .or_else(|| parts.headers.get("Authorization"));

        let token_value = match auth_header {
            Some(header) => {
                let s = header.to_str().map_err(|_| StatusCode::UNAUTHORIZED)?;
                if let Some(stripped) = s.strip_prefix("Bearer ") {
                    stripped.to_string()
                } else {
                    s.to_string() // Allow token without Bearer prefix
                }
            }
            None => return Err(StatusCode::UNAUTHORIZED),
        };

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


