use crate::auth::token::Token;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct TokenStore {
    tokens: Arc<RwLock<HashMap<String, Token>>>,
    path: String,
}

impl TokenStore {
    pub async fn load(path: &str) -> Result<Self> {
        let tokens = if tokio::fs::try_exists(path).await.unwrap_or(false) {
            let data = tokio::fs::read_to_string(path).await?;
            let token_list: Vec<Token> = serde_json::from_str(&data).unwrap_or_default();
            token_list.into_iter().map(|t| (t.value.clone(), t)).collect()
        } else {
            HashMap::new()
        };

        Ok(Self {
            tokens: Arc::new(RwLock::new(tokens)),
            path: path.to_string(),
        })
    }

    /// Create an empty token store (used as fallback when loading fails)
    pub fn empty() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            path: String::new(),
        }
    }

    pub async fn validate(&self, value: &str) -> bool {
        let tokens = self.tokens.read().await;
        tokens.get(value).is_some_and(|t| t.is_valid())
    }

    pub async fn list(&self) -> Vec<Token> {
        self.tokens.read().await.values().cloned().collect()
    }

    pub async fn add(&self, token: Token) -> Result<()> {
        let mut tokens = self.tokens.write().await;
        tokens.insert(token.value.clone(), token);
        self.save_locked(&tokens).await
    }

    pub async fn revoke(&self, id: &str) -> Result<bool> {
        let mut tokens = self.tokens.write().await;
        // Find token by id and revoke it
        if let Some(token) = tokens.values_mut().find(|t| t.id == id) {
            token.revoked = true;
            self.save_locked(&tokens).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn save_locked(&self, tokens: &HashMap<String, Token>) -> Result<()> {
        if self.path.is_empty() {
            return Ok(()); // No path set (empty store), skip saving
        }
        let token_list: Vec<&Token> = tokens.values().collect();
        let data = serde_json::to_string_pretty(&token_list)?;
        tokio::fs::write(&self.path, data).await?;
        Ok(())
    }
}
