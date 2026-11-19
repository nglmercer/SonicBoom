//! The pure-Rust unigram encoder must produce the same ids as the SentencePiece C++
//! library on the words Kyutai TTS feeds it. Skipped when the model is not present.
use kyutai_tts::spm::Spm;
use sentencepiece::SentencePieceProcessor;
use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/candle/kyutai-tts/tokenizer_spm_8k_en_fr_audio.model");
    p.is_file().then_some(p)
}

const TEXT: &str = "Hello, this is a short test. The quick brown fox jumps over the lazy dog. \
Speech synthesis should sound natural, with clear pronunciation and a steady rhythm throughout \
the whole sentence. Real time factor measures how long a system takes to generate audio compared \
with the duration of that audio! Bonjour, ceci est un court test de synthèse vocale. Où est la \
bibliothèque ? C'est l'été, il fait très chaud à Paris — n'est-ce pas ? Les élèves étudient \
l'histoire, la géographie et les mathématiques. Ça va coûter 1,250.75 euros (environ $1,400) le \
12/03/2025 à 14h30; l'adresse: 221B Baker Street, NW1. «Très bien», dit-il. Naïve façade, \
œuvre d'art, cœur, Ångström, Zürich, São Paulo, 東京, Привет, 😀 emoji, ALL CAPS WORDS, \
mixedCaseWords, hyphen-ated, under_score, e-mail@example.com, https://example.org/a?b=c#d, \
supercalifragilisticexpialidocious, antidisestablishmentarianism, x, I, a, 0, 007, 3.14159";

#[test]
fn matches_sentencepiece_cpp_on_words_and_sentences() {
    let Some(path) = model() else {
        eprintln!("skipped: kyutai tokenizer model not present");
        return;
    };
    let ours = Spm::open(&path).unwrap();
    let reference = SentencePieceProcessor::open(&path).unwrap();
    let mut checked = 0;
    let mut mismatches = Vec::new();
    let mut inputs: Vec<String> = TEXT.split_whitespace().map(String::from).collect();
    inputs.push(TEXT.to_string());
    inputs.extend(TEXT.split(". ").map(String::from));
    // Deterministic pseudo-random lowercase words.
    let mut state = 0x2545F4914F6CDD1Du64;
    for _ in 0..400 {
        let len = 2 + (state % 11) as usize;
        let mut w = String::new();
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            w.push((b'a' + (state % 26) as u8) as char);
        }
        inputs.push(w);
    }
    for text in &inputs {
        let want: Vec<u32> = reference.encode(text).unwrap().into_iter().map(|p| p.id).collect();
        let got = ours.encode(text);
        checked += 1;
        if want != got {
            mismatches.push((text.clone(), want, got));
        }
    }
    for (t, w, g) in mismatches.iter().take(10) {
        eprintln!("MISMATCH {t:?}\n  cpp : {w:?}\n  rust: {g:?}");
    }
    assert!(
        mismatches.is_empty(),
        "{} of {checked} inputs differ from the C++ encoder",
        mismatches.len()
    );
}
