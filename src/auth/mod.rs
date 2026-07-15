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
            None => {
                // Per spec: requests to /api/tts from our own pages (Referer == Host)
                // are allowed without Authorization (used by the web test UI).
                if parts.uri.path() == "/api/tts" && referer_is_self(parts) {
                    return Ok(AuthenticatedToken("__self_referer__".to_string()));
                }
                return Err(StatusCode::UNAUTHORIZED);
            }
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

/// Returns true when the `Referer` host matches the `Host` header, meaning the
/// request originated from our own web UI. Used to allow unauthenticated
/// `/api/tts` requests coming from the built-in test page.
fn referer_is_self(parts: &Parts) -> bool {
    let Some(host) = parts.headers.get("host").and_then(|h| h.to_str().ok()) else {
        return false;
    };
    let Some(referer) = parts
        .headers
        .get("referer")
        .or_else(|| parts.headers.get("Referer"))
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };

    // Extract host[:port] from the Referer (drop scheme and path).
    let after_scheme = referer.split_once("://").map(|(_, r)| r).unwrap_or(referer);
    let referer_host = after_scheme.split('/').next().unwrap_or("");
    if referer_host.is_empty() {
        return false;
    }

    // Compare exactly, but tolerate a missing port on either side.
    host == referer_host
        || host.split(':').next() == referer_host.split(':').next()
}


