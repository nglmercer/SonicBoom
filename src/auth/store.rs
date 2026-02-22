use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use crate::auth::token::Token;

pub struct TokenStore {
    tokens: Arc<RwLock<Vec<Token>>>,
    path: String,
}

impl TokenStore {
    pub async fn load(path: &str) -> Result<Self> {
        let tokens = if tokio::fs::try_exists(path).await.unwrap_or(false) {
            let data = tokio::fs::read_to_string(path).await?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            tokens: Arc::new(RwLock::new(tokens)),
            path: path.to_string(),
        })
    }

    pub async fn validate(&self, value: &str) -> bool {
        let tokens = self.tokens.read().await;
        tokens.iter().any(|t| t.value == value && t.is_valid())
    }

    pub async fn list(&self) -> Vec<Token> {
        self.tokens.read().await.clone()
    }

    pub async fn add(&self, token: Token) -> Result<()> {
        let mut tokens = self.tokens.write().await;
        tokens.push(token);
        self.save_locked(&tokens).await
    }

    pub async fn revoke(&self, id: &str) -> Result<bool> {
        let mut tokens = self.tokens.write().await;
        if let Some(t) = tokens.iter_mut().find(|t| t.id == id) {
            t.revoked = true;
            self.save_locked(&tokens).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn save_locked(&self, tokens: &[Token]) -> Result<()> {
        let data = serde_json::to_string_pretty(tokens)?;
        tokio::fs::write(&self.path, data).await?;
        Ok(())
    }
}
