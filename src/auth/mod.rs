pub mod store;
pub mod token;

use axum::{
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
};

use crate::AppState;

pub struct AuthenticatedToken(#[allow(dead_code)] pub String);

impl AuthenticatedToken {
    /// Stable non-secret identity for rate limiting. The raw bearer value
    /// is never used as a map key outside the authentication check itself.
    pub fn rate_limit_key(&self) -> String {
        crate::auth::token::hash_token_value(&self.0)
    }
}

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

        let auth_header = parts.headers.get(axum::http::header::AUTHORIZATION);

        let token_value = match auth_header {
            Some(header) => {
                let s = header.to_str().map_err(|_| StatusCode::UNAUTHORIZED)?;
                if let Some(stripped) = s.strip_prefix("Bearer ") {
                    stripped.to_string()
                } else {
                    s.to_string() // Allow token without Bearer [REDACTED]
                }
            }
            None => return Err(StatusCode::UNAUTHORIZED),
        };

        if token_value.is_empty() {
            return Err(StatusCode::UNAUTHORIZED);
        }

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
