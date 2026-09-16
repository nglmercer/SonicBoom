use anyhow::Result;
use unicode_normalization::UnicodeNormalization;

/// Default hard maximum chunk size in Unicode characters.
pub const DEFAULT_MAX_CHUNK_CHARS: usize = 200;

pub struct TextProcessor {
    // Index = unicode codepoint, value = model internal ID (-1 means unsupported)
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
        Self::split_sentences_with_limit(text, DEFAULT_MAX_CHUNK_CHARS)
    }

    /// Split `text` into chunks of at most `max_chars` Unicode characters.
    ///
    /// Strategy: prefer sentence boundaries, then whitespace, then hard-split
    /// by character count. No returned chunk ever exceeds `max_chars`
    /// characters (measured with [`str::chars`], not UTF-8 bytes).
    pub fn split_sentences_with_limit(text: &str, max_chars: usize) -> Vec<String> {
        let max_chars = max_chars.max(1);
        let text = text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        if text.chars().count() <= max_chars {
            return vec![text.to_string()];
        }

        // 1. Split at sentence boundaries.
        let mut sentences = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if is_sentence_end(ch) {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                current = String::new();
            }
        }
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }

        // 2. Greedily pack sentences, splitting oversized ones.
        let mut chunks = Vec::new();
        let mut pending = String::new();
        let mut pending_len = 0usize;
        for sentence in sentences {
            for piece in split_oversized(&sentence, max_chars) {
                let piece_len = piece.chars().count();
                if pending_len > 0 && pending_len + 1 + piece_len > max_chars {
                    chunks.push(std::mem::take(&mut pending));
                    pending_len = 0;
                }
                if pending_len > 0 {
                    pending.push(' ');
                    pending_len += 1;
                }
                pending.push_str(&piece);
                pending_len += piece_len;
            }
        }
        if !pending.trim().is_empty() {
            chunks.push(pending.trim().to_string());
        }
        if chunks.is_empty() {
            chunks.push(hard_truncate(text, max_chars));
        }
        chunks
    }
}

fn is_sentence_end(ch: char) -> bool {
    matches!(
        ch,
        '.' | '?' | '!' | '。' | '？' | '！' | '…' | ';' | '；' | '\n'
    )
}

/// Split a single sentence that may exceed `max_chars`: first at whitespace,
/// then by hard character count. Every piece is <= `max_chars` chars.
fn split_oversized(sentence: &str, max_chars: usize) -> Vec<String> {
    if sentence.chars().count() <= max_chars {
        return vec![sentence.to_string()];
    }
    // Try whitespace splitting.
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;
    for word in sentence.split_whitespace() {
        let word_len = word.chars().count();
        if word_len > max_chars {
            // Single word exceeds the limit: flush and hard-split the word.
            if !current.is_empty() {
                pieces.push(std::mem::take(&mut current));
                current_len = 0;
            }
            pieces.extend(hard_split(word, max_chars));
        } else {
            if current_len > 0 && current_len + 1 + word_len > max_chars {
                pieces.push(std::mem::take(&mut current));
                current_len = 0;
            }
            if current_len > 0 {
                current.push(' ');
                current_len += 1;
            }
            current.push_str(word);
            current_len += word_len;
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    if pieces.is_empty() {
        // No whitespace at all: hard-split by character count.
        pieces.extend(hard_split(sentence, max_chars));
    }
    pieces
}

fn hard_split(text: &str, max_chars: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max_chars)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn hard_truncate(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bounded(chunks: &[String], max_chars: usize) {
        assert!(!chunks.is_empty());
        for chunk in chunks {
            let len = chunk.chars().count();
            assert!(
                len <= max_chars,
                "chunk of {len} chars exceeds limit {max_chars}: {chunk:?}"
            );
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn short_text_is_single_chunk() {
        assert_eq!(TextProcessor::split_sentences("Hello."), vec!["Hello."]);
    }

    #[test]
    fn normal_punctuation_splits() {
        let text = "Hello world. This is a test! How are you? Fine, thanks.";
        let chunks = TextProcessor::split_sentences_with_limit(text, 30);
        assert_bounded(&chunks, 30);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn long_punctuation_free_ascii_is_bounded() {
        let text = "lorem ipsum dolor sit amet ".repeat(40);
        let chunks = TextProcessor::split_sentences_with_limit(&text, 200);
        assert_bounded(&chunks, 200);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn long_single_word_without_whitespace_is_bounded() {
        let text = "a".repeat(1000);
        let chunks = TextProcessor::split_sentences_with_limit(&text, 200);
        assert_bounded(&chunks, 200);
        assert_eq!(chunks.len(), 5);
    }

    #[test]
    fn cjk_text_without_spaces_is_bounded() {
        let text = "日本語のテスト文章です。".repeat(60);
        let chunks = TextProcessor::split_sentences_with_limit(&text, 200);
        assert_bounded(&chunks, 200);
    }

    #[test]
    fn emoji_text_is_bounded_by_chars_not_bytes() {
        // Each emoji is 4 UTF-8 bytes but 1 char; 300 emojis = 1200 bytes.
        let text = "🎙️".repeat(150);
        let chunks = TextProcessor::split_sentences_with_limit(&text, 200);
        assert_bounded(&chunks, 200);
        // Must not split in the middle of nothing: rejoining covers input.
        let joined_len: usize = chunks.iter().map(|c| c.chars().count()).sum();
        assert_eq!(joined_len, text.chars().count());
    }

    #[test]
    fn mixed_unicode_long_sentence_is_bounded() {
        let text = "Héllo wörld, this is à ünïcodé sentence without any period ".repeat(20);
        let chunks = TextProcessor::split_sentences_with_limit(&text, 200);
        assert_bounded(&chunks, 200);
    }

    #[test]
    fn chunks_never_exceed_hard_limit_on_small_limit() {
        let text = "one two three four five six seven eight nine ten. ".repeat(10);
        for limit in [1, 7, 50] {
            let chunks = TextProcessor::split_sentences_with_limit(&text, limit);
            assert_bounded(&chunks, limit);
        }
    }
}
