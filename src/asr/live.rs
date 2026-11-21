//! Live mode ("paste while speaking"): the current window of the take (up to 30 s) is re-transcribed as audio comes
//! in; words that two passes in a row agree on are pasted, the rest waits (the newest words change while they are
//! still being spoken). Pasted text is never taken back.
use std::time::Instant;

const RATE: usize = 16_000;
/// Live mode: re-transcribe the current window once this much new audio is in (1 s: leaves the GPU idle most of the
/// time while still keeping it awake).
const LIVE_STEP: usize = RATE;
/// Live mode: a window is at most 30 s (the model's piece size); it ends at the latest pause before that.
const LIVE_WINDOW: usize = RATE * 30;
/// Lengths go to the model in whole encoder frames (80 ms): the lengths `dictation sweep` prepared.
const FRAME: usize = RATE * 8 / 100;

/// Live mode (paste while speaking): the current window [start, now) is re-transcribed every LIVE_STEP; words that
/// two passes in a row agree on are pasted.
#[derive(Default)]
pub struct Live {
    start: usize,
    /// Audio length of the last pass.
    seen: usize,
    /// Words pasted from the current window.
    done: Vec<String>,
    /// Words of the last pass after the pasted ones (candidates: pasted once the next pass agrees).
    pending: Vec<String>,
    /// Everything pasted in this take.
    pub text: String,
}

/// Word for comparing passes: letters and digits only, lowercase.
fn norm(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

impl Live {
    /// Where a new pass continues after the pasted words: just after the last pasted word found near its old
    /// position (passes rewrite earlier words, e.g. "two point five" -> "2.5", so positions shift).
    fn anchor(&self, w: &[String]) -> usize {
        let p = &self.done;
        for j in (p.len().saturating_sub(4)..p.len()).rev() {
            let target = norm(&p[j]);
            let (lo, hi) = (j.saturating_sub(4), (j + 5).min(w.len()));
            if let Some(i) = (lo..hi).rev().find(|&i| norm(&w[i]) == target) {
                return i + 1;
            }
        }
        p.len().min(w.len())
    }

    /// Add words to the take's text -> the string to paste (with a separating space).
    fn emit(&mut self, w: &[String]) -> Option<String> {
        if w.is_empty() {
            return None;
        }
        self.done.extend_from_slice(w);
        let s = w.join(" ");
        let p = if self.text.is_empty() { s } else { format!(" {s}") };
        self.text.push_str(&p);
        Some(p)
    }

    /// Index in `done` of the window's last word, if the pasted text already runs past the window's end (it ends at a
    /// pause, which can be seconds before the newest pasted words). Matched by its last two words near their position.
    fn end_in_done(&self, w: &[String]) -> Option<usize> {
        if w.len() < 2 {
            return None;
        }
        let (x, y) = (norm(&w[w.len() - 2]), norm(&w[w.len() - 1]));
        let c = w.len() - 1;
        let (lo, hi) = (c.saturating_sub(8).max(1), (c + 9).min(self.done.len()));
        (lo..hi).rev().find(|&k| norm(&self.done[k]) == y && norm(&self.done[k - 1]) == x)
    }

    /// A final pass over the current window: paste all of it that is new. Pasted words after the window's end carry
    /// over into the next window (they get transcribed again there and must not be pasted twice).
    fn close(&mut self, s: &str) -> Option<String> {
        let w = words(s);
        let p = match self.end_in_done(&w) {
            Some(k) => {
                self.done.drain(..=k);
                None
            }
            None => {
                let p = self.emit(&w[self.anchor(&w)..]);
                self.done.clear();
                p
            }
        };
        self.pending.clear();
        p
    }

    /// New audio (everything recorded so far) -> text to paste now, if any.
    pub fn check(&mut self, a: &mut super::Asr, x: &[f32]) -> Option<String> {
        if x.len() >= self.start + LIVE_WINDOW + FRAME {
            // Window full: end it at the latest pause before 30 s, paste its final text, start the next one there.
            let probe = &x[self.start..self.start + LIVE_WINDOW + FRAME];
            let cut = match a.first_cut(probe) {
                Ok(c) => (c / FRAME * FRAME).clamp(FRAME * 12, LIVE_WINDOW),
                Err(e) => {
                    eprintln!("live: {e}");
                    LIVE_WINDOW
                }
            };
            let end = self.start + cut;
            let r = a.run(&x[self.start..end]);
            println!("live window ends at {:.1} s", end as f64 / RATE as f64);
            self.start = end;
            self.seen = end;
            return match r {
                Ok((s, _)) => self.close(&s),
                Err(e) => {
                    eprintln!("live: {e}");
                    let rest = std::mem::take(&mut self.pending);
                    let p = self.emit(&rest);
                    self.done.clear();
                    p
                }
            };
        }
        let len = (x.len() - self.start) / FRAME * FRAME;
        if len < self.seen.saturating_sub(self.start) + LIVE_STEP {
            return None;
        }
        self.seen = self.start + len;
        let t = Instant::now();
        let s = match a.run(&x[self.start..self.start + len]) {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("live: {e}");
                return None;
            }
        };
        println!("live pass {:.1} s, {:.0} ms: {s}", len as f64 / RATE as f64, t.elapsed().as_secs_f64() * 1000.0);
        let w = words(&s);
        let tail = w[self.anchor(&w)..].to_vec();
        let agreed = tail.iter().zip(&self.pending).take_while(|(a, b)| norm(a) == norm(b)).count();
        self.pending = tail[agreed..].to_vec();
        self.emit(&tail[..agreed])
    }

    /// Stop: the rest of the current window -> what is left to paste.
    pub fn finish(&mut self, a: &mut super::Asr, x: &[f32]) -> Result<String, String> {
        let rest = &x[self.start.min(x.len())..];
        if rest.len() < RATE / 4 {
            let rest = std::mem::take(&mut self.pending);
            return Ok(self.emit(&rest).unwrap_or_default());
        }
        let (s, tm) = a.run(rest)?;
        println!("  final pass: mel {:.0} ms, pauses {:.0} ms, encoder {:.0} ms, decoder {:.0} ms", tm.mel, tm.vad, tm.enc, tm.dec);
        Ok(self.close(&s).unwrap_or_default())
    }
}
