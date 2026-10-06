pub mod store;
pub mod token;

use axum::{
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
};

use crate::AppState;
use crate::config::AuthMode;

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
        // The authentication mode is read live from the
        // configuration so `auth_mode` changes apply
        // without a restart (spec §30).
        let config = state.config.get().await;
        match config.server.auth_mode {
            // `local` and `none` do not require API
            // bearer authentication.
            AuthMode::Local => {
                let peer = parts
                    .extensions
                    .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>();
                if !peer.is_some_and(|peer| peer.0.ip().is_loopback()) {
                    return Err(StatusCode::FORBIDDEN);
                }
                return Ok(AuthenticatedToken("__no_auth__".to_string()));
            }
            AuthMode::None => {
                return Ok(AuthenticatedToken("__no_auth__".to_string()));
            }
            AuthMode::Token => {}
        }

        let token_value = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|header| header.to_str().ok())
            .and_then(parse_bearer)
            .ok_or(StatusCode::UNAUTHORIZED)?;

        if config.admin.enable_sample_token && token_value == "SAMPLE_TOKEN" {
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
        config::{AppConfig, AuthMode, ConfigManager, secrets::MemorySecretStore},
        runtime::{
            ModelService,
            subsystems::{InferenceGateManager, RateLimiterManager},
        },
        tts::ModelStatus,
    };
    use axum::{
        extract::FromRequestParts,
        http::{Request, StatusCode, request::Parts},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn test_state(enable_sample_token: bool) -> AppState {
        let mut config = AppConfig::default();
        config.server.auth_mode = AuthMode::Token;
        config.admin.enable_sample_token = enable_sample_token;
        config.paths.audio = Some("/tmp".to_string());
        AppState {
            model_status: Arc::new(RwLock::new(ModelStatus::Idle)),
            token_store: Arc::new(TokenStore::empty()),
            config: ConfigManager::in_memory(config),
            audio_manager: Arc::new(None),
            inference_gate: Arc::new(InferenceGateManager::new(InferenceGate::new(1, 8))),
            rate_limiter: Arc::new(RateLimiterManager::new(RateLimiter::new(0, 60))),
            model_service: Arc::new(ModelService::new(Arc::new(RwLock::new(ModelStatus::Idle)))),
            secrets: Arc::new(MemorySecretStore::default()),
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
    async fn local_auth_rejects_remote_peers_even_before_listener_restart() {
        let state = test_state(false);
        state
            .config
            .update(None, |c| c.server.auth_mode = AuthMode::Local)
            .await
            .unwrap();
        for address in ["127.0.0.1:1234", "[::1]:1234", "192.0.2.1:1234"] {
            let mut parts = parts_with_auth("Bearer unused");
            let peer: std::net::SocketAddr = address.parse().unwrap();
            parts.extensions.insert(axum::extract::ConnectInfo(peer));
            let result = AuthenticatedToken::from_request_parts(&mut parts, &state).await;
            assert_eq!(result.is_ok(), peer.ip().is_loopback());
        }
        let mut parts = parts_with_auth("Bearer unused");
        assert!(
            AuthenticatedToken::from_request_parts(&mut parts, &state)
                .await
                .is_err()
        );
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
