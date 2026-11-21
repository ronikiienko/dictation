//! Timed words (asr::Asr::run_words) -> SubRip subtitles (.srt): numbered cues, "HH:MM:SS,mmm --> HH:MM:SS,mmm",
//! at most two lines of ~42 characters. A cue ends at a sentence end, a pause, or when it gets too long to read.
use crate::asr::Word;

/// Characters per line (the usual subtitle limit), two lines per cue.
const LINE: usize = 42;
/// Longest a cue stays on screen, and how long a pause starts a new one, seconds.
const MAX_SECS: f64 = 6.0;
const PAUSE: f64 = 1.0;
/// A cue stays at least this long (short ones flash by), unless the next one starts sooner.
const MIN_SECS: f64 = 1.0;

struct Cue {
    start: f64,
    end: f64,
    text: String,
}

fn cue(words: &[&Word]) -> Cue {
    let text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
    Cue { start: words[0].start, end: words[words.len() - 1].end, text }
}

fn chars(words: &[&Word]) -> usize {
    words.iter().map(|w| w.text.chars().count() + 1).sum::<usize>().saturating_sub(1)
}

pub fn srt(words: &[Word]) -> String {
    let mut cues: Vec<Cue> = Vec::new();
    let mut cur: Vec<&Word> = Vec::new();
    for w in words {
        if let Some(last) = cur.last() {
            if w.start - last.end > PAUSE {
                cues.push(cue(&cur));
                cur.clear();
            } else if chars(&cur) + 1 + w.text.chars().count() > 2 * LINE || w.end - cur[0].start > MAX_SECS {
                // Too long: cut after the last comma (or similar) if there is one past the first two words, so the
                // rest of the phrase moves on with this word instead of a lone word starting the next cue.
                let at = cur.iter().rposition(|w| w.text.ends_with([',', ';', ':'])).filter(|&i| i >= 2);
                let keep = at.map_or(cur.len(), |i| i + 1);
                cues.push(cue(&cur[..keep]));
                cur.drain(..keep);
            }
        }
        cur.push(w);
        let n = chars(&cur);
        let sentence = w.text.ends_with(['.', '?', '!', '…']);
        // A sentence end closes the cue (unless it's only a word or two); a comma once the cue is fairly full.
        if (sentence && n >= 15) || (w.text.ends_with([',', ';', ':']) && n >= LINE + 10) {
            cues.push(cue(&cur));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        cues.push(cue(&cur));
    }
    for i in 0..cues.len() {
        let next = cues.get(i + 1).map_or(f64::INFINITY, |n| n.start);
        let c = &mut cues[i];
        c.end = c.end.max((c.start + MIN_SECS).min(next - 0.05)).max(c.start + 0.1);
    }
    let mut out = String::new();
    for (i, c) in cues.iter().enumerate() {
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, stamp(c.start), stamp(c.end), lines(&c.text)));
    }
    out
}

/// Seconds -> "HH:MM:SS,mmm".
fn stamp(s: f64) -> String {
    let ms = (s.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// One line if it fits, else two, split at the space nearest the middle.
fn lines(text: &str) -> String {
    let n = text.chars().count();
    if n <= LINE {
        return text.to_string();
    }
    let mid = n / 2;
    let split = text
        .char_indices()
        .filter(|&(_, c)| c == ' ')
        .map(|(b, _)| (b, text[..b].chars().count()))
        .min_by_key(|&(_, chars)| chars.abs_diff(mid))
        .map(|(b, _)| b);
    match split {
        Some(b) => format!("{}\n{}", &text[..b], &text[b + 1..]),
        None => text.to_string(),
    }
}
