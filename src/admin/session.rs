use tower_sessions::Session;

const SESSION_KEY: &str = "admin_authenticated";

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

pub async fn destroy(session: &Session) {
    let _ = session.delete().await;
}
