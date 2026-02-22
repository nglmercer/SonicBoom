mod admin;
mod api;
mod auth;
mod config;
mod error;
mod tts;
mod web;

use std::{net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tower_sessions::{MemoryStore, SessionManagerLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use admin::{handlers::AdminState, lockout::LoginAttemptTracker};
use auth::store::TokenStore;
use config::AppConfig;
use tts::{download, model::ModelHandle, ModelStatus};

#[derive(Clone)]
pub struct AppState {
    pub model_status: Arc<RwLock<ModelStatus>>,
    pub token_store: Arc<TokenStore>,
    pub config: Arc<AppConfig>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            "SonicBoom=info,tower_http=debug".into()
        }))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Arc::new(AppConfig::from_env());

    let token_store = Arc::new(
        TokenStore::load(&config.token_store_path)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!("Could not load token store: {e}. Starting fresh.");
                // TokenStore::load에서 실패시 빈 store 반환하도록 수정 필요시 처리
                panic!("Failed to initialize token store: {e}");
            }),
    );

    let model_status = Arc::new(RwLock::new(ModelStatus::Idle));

    let app_state = AppState {
        model_status: Arc::clone(&model_status),
        token_store: Arc::clone(&token_store),
        config: Arc::clone(&config),
    };

    let lockout = Arc::new(LoginAttemptTracker::default());
    let admin_state = AdminState {
        token_store: Arc::clone(&token_store),
        lockout: Arc::clone(&lockout),
        config: Arc::clone(&config),
    };

    // 세션 스토어
    let session_store = MemoryStore::default();
    let session_layer = SessionManagerLayer::new(session_store);

    // 라우터 구성
    let app = axum::Router::new()
        .merge(web::router(app_state.clone()))
        .merge(api::router(app_state.clone()))
        .merge(admin::router(admin_state))
        .layer(session_layer)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let port = config.port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    // 모델 다운로드 및 로드를 백그라운드에서 병렬 실행
    let model_cache_dir = config.model_cache_dir.clone();
    let hf_token = config.hf_token.clone();
    let model_status_bg = Arc::clone(&model_status);
    tokio::spawn(async move {
        *model_status_bg.write().await = ModelStatus::Downloading { progress: 0.0 };

        match download::download_models(&model_cache_dir, hf_token.as_deref(), Arc::clone(&model_status_bg)).await {
            Ok(paths) => {
                *model_status_bg.write().await = ModelStatus::Loading;
                match tokio::task::spawn_blocking(move || ModelHandle::load(&paths)).await {
                    Ok(Ok(handle)) => {
                        tracing::info!("Model ready.");
                        *model_status_bg.write().await =
                            ModelStatus::Ready(Arc::new(handle));
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Model load error: {e}");
                        *model_status_bg.write().await =
                            ModelStatus::Failed(e.to_string());
                    }
                    Err(e) => {
                        tracing::error!("Model load task panic: {e}");
                        *model_status_bg.write().await =
                            ModelStatus::Failed(e.to_string());
                    }
                }
            }
            Err(e) => {
                tracing::error!("Model download error: {e}");
                *model_status_bg.write().await = ModelStatus::Failed(e.to_string());
            }
        }
    });

    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}
