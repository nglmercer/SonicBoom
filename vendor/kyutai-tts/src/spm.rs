//! Pure-Rust SentencePiece *unigram* encoder.
//!
//! Reads the `.model` protobuf and encodes text with the same Viterbi search as
//! SentencePiece's `Unigram` model. It exists so the binary does not link the
//! SentencePiece C++ library, whose bundled static protobuf collides with ONNX
//! Runtime's (duplicate `google::protobuf` symbols at link time).
//!
//! Scope: unigram models, whitespace escaping (`▁`), optional dummy prefix and
//! whitespace removal, NFKC normalization via `unicode-normalization` (instead of
//! the precompiled charsmap), byte fallback. Equivalence with the C++ encoder is
//! tested on a corpus in `tests/spm_equivalence.rs`.
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::path::Path;
use unicode_normalization::UnicodeNormalization;

const SPACE: char = '\u{2581}'; // ▁
const UNK_PENALTY: f32 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

pub struct Spm {
    scores: Vec<f32>,
    types: Vec<PieceType>,
    ids: HashMap<String, u32>,
    max_piece_chars: usize,
    unk_id: u32,
    min_score: f32,
    byte_ids: [Option<u32>; 256],
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos).ok_or_else(|| anyhow!("truncated varint"))?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                bail!("varint too long");
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.buf.len()).ok_or_else(|| anyhow!("truncated field"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Next `(field number, wire type)`.
    fn tag(&mut self) -> Result<(u32, u8)> {
        let t = self.varint()?;
        Ok(((t >> 3) as u32, (t & 7) as u8))
    }

    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                let n = self.varint()? as usize;
                self.take(n)?;
            }
            5 => {
                self.take(4)?;
            }
            w => bail!("unsupported wire type {w}"),
        }
        Ok(())
    }
}

impl Spm {
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let mut pieces: Vec<(String, f32, PieceType)> = Vec::new();
        let (mut add_dummy_prefix, mut remove_extra, mut escape) = (true, true, true);
        let mut model_type = 1u64; // UNIGRAM
        while !r.done() {
            let (field, wire) = r.tag()?;
            match (field, wire) {
                (1, 2) => {
                    let n = r.varint()? as usize;
                    let mut p = Reader { buf: r.take(n)?, pos: 0 };
                    let (mut piece, mut score, mut ty) = (String::new(), 0f32, PieceType::Normal);
                    while !p.done() {
                        let (f, w) = p.tag()?;
                        match (f, w) {
                            (1, 2) => {
                                let n = p.varint()? as usize;
                                piece = String::from_utf8(p.take(n)?.to_vec())?;
                            }
                            (2, 5) => score = f32::from_le_bytes(p.take(4)?.try_into()?),
                            (3, 0) => {
                                ty = match p.varint()? {
                                    1 => PieceType::Normal,
                                    2 => PieceType::Unknown,
                                    3 => PieceType::Control,
                                    4 => PieceType::UserDefined,
                                    5 => PieceType::Unused,
                                    6 => PieceType::Byte,
                                    t => bail!("unknown piece type {t}"),
                                }
                            }
                            (_, w) => p.skip(w)?,
                        }
                    }
                    pieces.push((piece, score, ty));
                }
                (2, 2) => {
                    // TrainerSpec: model_type = field 3.
                    let n = r.varint()? as usize;
                    let mut t = Reader { buf: r.take(n)?, pos: 0 };
                    while !t.done() {
                        let (f, w) = t.tag()?;
                        match (f, w) {
                            (3, 0) => model_type = t.varint()?,
                            (_, w) => t.skip(w)?,
                        }
                    }
                }
                (3, 2) => {
                    // NormalizerSpec: add_dummy_prefix=3, remove_extra_whitespaces=4, escape_whitespaces=5.
                    let n = r.varint()? as usize;
                    let mut t = Reader { buf: r.take(n)?, pos: 0 };
                    while !t.done() {
                        let (f, w) = t.tag()?;
                        match (f, w) {
                            (3, 0) => add_dummy_prefix = t.varint()? != 0,
                            (4, 0) => remove_extra = t.varint()? != 0,
                            (5, 0) => escape = t.varint()? != 0,
                            (_, w) => t.skip(w)?,
                        }
                    }
                }
                (_, w) => r.skip(w)?,
            }
        }
        if model_type != 1 {
            bail!("only SentencePiece unigram models are supported (model_type {model_type})");
        }
        if pieces.is_empty() {
            bail!("model has no pieces");
        }
        let mut ids = HashMap::new();
        let mut scores = Vec::new();
        let mut types = Vec::new();
        let mut byte_ids = [None; 256];
        let mut unk_id = None;
        let mut min_score = f32::MAX;
        let mut max_piece_chars = 1;
        for (i, (piece, score, ty)) in pieces.into_iter().enumerate() {
            match ty {
                PieceType::Unknown => unk_id = Some(i as u32),
                PieceType::Byte => {
                    // "<0xAB>"
                    if let Some(h) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                        if let Ok(b) = u8::from_str_radix(h, 16) {
                            byte_ids[b as usize] = Some(i as u32);
                        }
                    }
                }
                PieceType::Normal | PieceType::UserDefined => {
                    max_piece_chars = max_piece_chars.max(piece.chars().count());
                    if ty == PieceType::Normal {
                        min_score = min_score.min(score);
                    }
                    ids.insert(piece, i as u32);
                }
                _ => {}
            }
            scores.push(score);
            types.push(ty);
        }
        Ok(Self {
            scores,
            types,
            ids,
            max_piece_chars,
            unk_id: unk_id.ok_or_else(|| anyhow!("model has no <unk> piece"))?,
            min_score,
            byte_ids,
            add_dummy_prefix,
            remove_extra_whitespaces: remove_extra,
            escape_whitespaces: escape,
        })
    }

    /// NFKC, whitespace cleanup, dummy prefix and `▁` escaping.
    fn normalize(&self, text: &str) -> String {
        let mut s: String = text.nfkc().collect();
        if self.remove_extra_whitespaces {
            s = s.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
        }
        if s.is_empty() {
            return s;
        }
        if self.add_dummy_prefix {
            s.insert(0, ' ');
        }
        if self.escape_whitespaces {
            s = s.replace(' ', &SPACE.to_string());
        }
        s
    }

    /// Token ids for `text` (Viterbi over unigram scores).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalize(text);
        if norm.is_empty() {
            return vec![];
        }
        let chars: Vec<char> = norm.chars().collect();
        // Byte offset of every char boundary, to slice pieces.
        let mut offs = Vec::with_capacity(chars.len() + 1);
        let mut o = 0;
        for c in &chars {
            offs.push(o);
            o += c.len_utf8();
        }
        offs.push(o);

        let n = chars.len();
        // best[i] = (score, start of last piece, piece id or u32::MAX for unknown char)
        let mut best: Vec<(f32, usize, u32)> = vec![(f32::NEG_INFINITY, 0, 0); n + 1];
        best[0] = (0.0, 0, 0);
        let unk_score = self.min_score - UNK_PENALTY;
        for start in 0..n {
            let base = best[start].0;
            if base == f32::NEG_INFINITY {
                continue;
            }
            let mut has_single = false;
            for len in 1..=self.max_piece_chars.min(n - start) {
                let piece = &norm[offs[start]..offs[start + len]];
                if let Some(&id) = self.ids.get(piece) {
                    if self.types[id as usize] == PieceType::Unused {
                        continue;
                    }
                    if len == 1 {
                        has_single = true;
                    }
                    let score = base + self.scores[id as usize];
                    if score > best[start + len].0 {
                        best[start + len] = (score, start, id);
                    }
                }
            }
            if !has_single {
                // Unknown character: a one-character <unk> with a heavy penalty.
                let score = base + unk_score;
                if score > best[start + 1].0 {
                    best[start + 1] = (score, start, u32::MAX);
                }
            }
        }
        // Backtrack.
        let mut out = Vec::new();
        let mut end = n;
        while end > 0 {
            let (_, start, id) = best[end];
            if id == u32::MAX {
                let ch = &norm[offs[start]..offs[end]];
                let mut bytes: Vec<u32> = Vec::new();
                let all_bytes = ch.bytes().all(|b| {
                    if let Some(i) = self.byte_ids[b as usize] {
                        bytes.push(i);
                        true
                    } else {
                        false
                    }
                });
                if all_bytes && !bytes.is_empty() {
                    out.extend(bytes.into_iter().rev());
                } else {
                    out.push(self.unk_id);
                }
            } else {
                out.push(id);
            }
            end = start;
        }
        out.reverse();
        // Consecutive unknowns collapse into one, as SentencePiece does.
        out.dedup_by(|a, b| *a == self.unk_id && *b == self.unk_id);
        out
    }
}
