use anyhow::Result;
use unicode_normalization::UnicodeNormalization;

pub struct TextProcessor {
    // 인덱스 = 유니코드 코드포인트, 값 = 모델 내부 ID (-1이면 미지원)
    indexer: Vec<i64>,
}

impl TextProcessor {
    pub fn load(unicode_indexer_path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read_to_string(unicode_indexer_path)?;
        let indexer: Vec<i64> = serde_json::from_str(&data)?;
        Ok(Self { indexer })
    }

    pub fn encode(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        let normalized: String = text.nfkd().collect();

        let ids: Vec<i64> = normalized
            .chars()
            .filter_map(|c| {
                let cp = c as usize;
                self.indexer.get(cp).copied().filter(|&id| id >= 0)
            })
            .collect();

        let mask: Vec<i64> = vec![1i64; ids.len()];
        (ids, mask)
    }

    pub fn split_sentences(text: &str) -> Vec<String> {
        const MAX_CHUNK: usize = 200;

        if text.len() <= MAX_CHUNK {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if (ch == '.' || ch == '?' || ch == '!' || ch == '。' || ch == '？' || ch == '！')
                && current.len() >= MAX_CHUNK / 2
            {
                chunks.push(current.trim().to_string());
                current = String::new();
            }
        }

        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }

        if chunks.is_empty() {
            chunks.push(text.to_string());
        }

        chunks
    }
}
