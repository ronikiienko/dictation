//! SentencePiece detokenizer for the GGUF vocab.
//! Ported from transcribe.cpp `src/transcribe-tokenizer.cpp` `decode_sentencepiece` and the Parakeet result builder in
//! `src/arch/parakeet/model.cpp` (MIT, (c) 2026 The transcribe.cpp authors): special pieces dropped, "<0xHH>" byte
//! tokens reassembled, U+2581 -> space, runs of spaces collapsed, ends trimmed.
use super::weights::Weights;

pub struct Tokenizer {
    pieces: Vec<String>,
    strip: Vec<bool>,
}

/// "<ll-RR>" language tag (2-3 lowercase letters, '-', 2-4 letters).
fn is_lang_tag(p: &str) -> bool {
    let Some(inner) = p.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else { return false };
    let Some((lang, region)) = inner.split_once('-') else { return false };
    (2..=3).contains(&lang.len())
        && lang.bytes().all(|b| b.is_ascii_lowercase())
        && (2..=4).contains(&region.len())
        && region.bytes().all(|b| b.is_ascii_alphabetic())
}

impl Tokenizer {
    pub fn load(g: &Weights) -> Tokenizer {
        let pieces = g.strs("tokenizer.ggml.tokens");
        let types = g.ints("tokenizer.ggml.token_type");
        let unk = g.int("tokenizer.ggml.unknown_token_id");
        // token_type: 1 normal, 2 unknown, 3 control.
        let strip = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| matches!(types.get(i), Some(2 | 3)) || i as i64 == unk || p == "<unk>" || is_lang_tag(p))
            .collect();
        Tokenizer { pieces, strip }
    }

    /// A piece's bytes ("<0xHH>" -> that byte, U+2581 -> space); None for special pieces.
    fn bytes(&self, id: usize) -> Option<Vec<u8>> {
        if id >= self.pieces.len() || self.strip[id] {
            return None;
        }
        let p = &self.pieces[id];
        if p.len() == 6 && p.starts_with("<0x") && p.ends_with('>') {
            if let Ok(b) = u8::from_str_radix(&p[3..5], 16) {
                return Some(vec![b]);
            }
        }
        Some(p.replace('\u{2581}', " ").into_bytes())
    }

    /// Words with the encoder frames of their first and last piece: a word starts at a piece beginning with U+2581
    /// (space); punctuation pieces stay with the word before but don't move its end (the model often emits a full
    /// stop only when the next speech starts). Same text as `decode`, split at its spaces.
    pub fn words(&self, toks: &[(usize, usize)]) -> Vec<(String, usize, usize)> {
        let mut out: Vec<(String, usize, usize)> = Vec::new();
        let (mut cur, mut first, mut last) = (Vec::<u8>::new(), 0, 0);
        let mut flush = |cur: &mut Vec<u8>, first, last| {
            let w = String::from_utf8_lossy(cur).trim().to_string();
            if !w.is_empty() {
                out.push((w, first, last));
            }
            cur.clear();
        };
        for &(id, frame) in toks {
            let Some(b) = self.bytes(id) else { continue };
            if b.first() == Some(&b' ') && !cur.iter().all(|&c| c == b' ') {
                flush(&mut cur, first, last);
            }
            if cur.iter().all(|&c| c == b' ') {
                first = frame;
            }
            cur.extend_from_slice(&b);
            if last < first || String::from_utf8_lossy(&b).chars().any(char::is_alphanumeric) {
                last = frame;
            }
        }
        flush(&mut cur, first, last);
        out
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.bytes(id).unwrap_or_default());
        }
        let s = String::from_utf8_lossy(&bytes);
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            if ch == ' ' && (out.is_empty() || out.ends_with(' ')) {
                continue;
            }
            out.push(ch);
        }
        out.truncate(out.trim_end_matches(' ').len());
        out
    }
}
