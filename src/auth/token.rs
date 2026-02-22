use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: String,
    pub value: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked: bool,
}

impl Token {
    pub fn new(value: String, expires_at: Option<DateTime<Utc>>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            value,
            created_at: Utc::now(),
            expires_at,
            revoked: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at {
            return Utc::now() < exp;
        }
        true
    }
}

pub fn generate_token_value() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}
