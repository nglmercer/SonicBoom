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

/// Parse a strict `Authorization: Bearer <token>` header.
///
/// Only the standard bearer scheme is accepted (scheme name matched
/// case-insensitively per RFC 9110). Raw tokens, `Basic`, `Token`, empty
/// bearer values, and malformed schemes are all rejected.
fn parse_bearer(header_value: &str) -> Option<String> {
    let (scheme, credentials) = header_value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    // Exactly one space separates scheme and credentials; no leading
    // whitespace or additional tokens are allowed.
    if credentials.is_empty()
        || credentials.starts_with(' ')
        || credentials.contains(' ')
        || credentials.contains('\t')
    {
        return None;
    }
    Some(credentials.to_string())
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

        let token_value = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|header| header.to_str().ok())
            .and_then(parse_bearer)
            .ok_or(StatusCode::UNAUTHORIZED)?;

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

#[cfg(test)]
mod tests {
    use super::parse_bearer;

    #[test]
    fn strict_bearer_syntax() {
        assert_eq!(parse_bearer("Bearer abc123").as_deref(), Some("abc123"));
        // Scheme is case-insensitive.
        assert_eq!(parse_bearer("bearer abc123").as_deref(), Some("abc123"));
        assert_eq!(parse_bearer("BEARER abc123").as_deref(), Some("abc123"));
    }

    #[test]
    fn malformed_authorization_is_rejected() {
        for bad in [
            "",
            "abc123",
            "Basic YWJj",
            "Token abc123",
            "Bearer",
            "Bearer ",
            "Bearer  abc123",
            "Bearer abc123 ",
            "Bearer abc 123",
            "Bearer\tabc123",
            " Bearer abc123",
            "Bear abc123",
            "Bearerabc123",
        ] {
            assert_eq!(parse_bearer(bad), None, "accepted {bad:?}");
        }
    }
}
