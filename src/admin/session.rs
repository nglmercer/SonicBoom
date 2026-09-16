use tower_sessions::Session;

const SESSION_KEY: &str = "admin_authenticated";
const CSRF_KEY: &str = "admin_csrf_token";

pub async fn is_authenticated(session: &Session) -> bool {
    session
        .get::<bool>(SESSION_KEY)
        .await
        .ok()
        .flatten()
        .unwrap_or(false)
}

pub async fn set_authenticated(session: &Session, value: bool) {
    let _ = session.insert(SESSION_KEY, value).await;
}

/// Regenerate the session id after login so a pre-login session id cannot be
/// reused (session fixation protection).
pub async fn rotate_id(session: &Session) {
    let _ = session.cycle_id().await;
}

pub async fn destroy(session: &Session) {
    let _ = session.delete().await;
}

/// Return the session's CSRF token, generating one on first use.
pub async fn csrf_token(session: &Session) -> String {
    if let Ok(Some(existing)) = session.get::<String>(CSRF_KEY).await {
        return existing;
    }
    let token = generate_csrf_token();
    let _ = session.insert(CSRF_KEY, token.clone()).await;
    token
}

/// Validate a submitted CSRF token against the session in constant time.
pub async fn validate_csrf(session: &Session, submitted: &str) -> bool {
    let expected: Option<String> = session.get(CSRF_KEY).await.ok().flatten();
    match expected {
        Some(expected) if !submitted.is_empty() => {
            constant_time_eq::constant_time_eq(expected.as_bytes(), submitted.as_bytes())
        }
        _ => false,
    }
}

fn generate_csrf_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}
