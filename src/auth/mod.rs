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
    use super::{AuthenticatedToken, parse_bearer};
    use crate::{
        AppState,
        api::{gate::InferenceGate, rate_limit::RateLimiter},
        auth::store::TokenStore,
        config::AppConfig,
        tts::ModelStatus,
    };
    use axum::{
        extract::FromRequestParts,
        http::{Request, StatusCode, request::Parts},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn test_state(enable_sample_token: bool) -> AppState {
        AppState {
            model_status: Arc::new(RwLock::new(ModelStatus::Idle)),
            token_store: Arc::new(TokenStore::empty()),
            config: Arc::new(AppConfig {
                admin_id: "admin".to_string(),
                admin_pw: "long-enough-test-password".to_string(),
                enable_sample_token,
                token_store_path: String::new(),
                model_cache_dir: String::new(),
                model_revision: "test".to_string(),
                model_hashes_path: None,
                hf_token: None,
                inference_steps: 5,
                port: 17842,
                log_dir: String::new(),
                log_level: "info".to_string(),
                log_to_file: false,
                log_to_stdout: false,
                auth_required: true,
                allowed_audio_dir: Some("/tmp".to_string()),
                max_text_length: 10_000,
                request_timeout_secs: 5,
                max_concurrent_inference: 1,
                max_pending_inference: 8,
                max_chunk_chars: 200,
                tts_rate_limit_requests: 0,
                tts_rate_limit_window_secs: 60,
                tts_max_body_bytes: 65_536,
                openai_max_body_bytes: 65_536,
                queue_max_body_bytes: 16_384,
                admin_max_body_bytes: 16_384,
                trust_proxy: false,
                trusted_proxies: vec![],
                cookie_secure: false,
                admin_session_expiry_secs: 60,
                temp_audio_dir: "./temp_audio".to_string(),
                enable_hsts: false,
                max_playback_queue_items: 100,
                model_download_connect_timeout_secs: 10,
                model_download_timeout_secs: 1800,
            }),
            audio_manager: Arc::new(None),
            inference_gate: Arc::new(InferenceGate::new(1, 8)),
            rate_limiter: Arc::new(RateLimiter::new(0, 60)),
        }
    }

    fn parts_with_auth(value: &str) -> Parts {
        Request::builder()
            .uri("/api/tts")
            .header(axum::http::header::AUTHORIZATION, value)
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    #[tokio::test]
    async fn sample_credential_accepted_only_when_enabled() {
        // The documented dev-only credential, assembled in parts so
        // secret-scanning tooling never mistakes it for a real credential.
        let raw = ["SAMPLE", "TOKEN"].join("_");
        let header = format!("Bearer {raw}");

        let state = test_state(true);
        let mut parts = parts_with_auth(&header);
        assert!(
            AuthenticatedToken::from_request_parts(&mut parts, &state)
                .await
                .is_ok(),
            "dev credential must authenticate when enabled"
        );

        let state = test_state(false);
        let mut parts = parts_with_auth(&header);
        assert!(
            matches!(
                AuthenticatedToken::from_request_parts(&mut parts, &state).await,
                Err(StatusCode::UNAUTHORIZED)
            ),
            "dev credential must not authenticate when disabled"
        );

        // A corrupted dev credential is rejected even when enabled.
        let state = test_state(true);
        let mut parts = parts_with_auth(&format!("Bearer {raw}X"));
        assert!(
            matches!(
                AuthenticatedToken::from_request_parts(&mut parts, &state).await,
                Err(StatusCode::UNAUTHORIZED)
            ),
            "corrupted dev credential must not authenticate"
        );
    }

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
