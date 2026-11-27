//! Live subtitles (proof of concept): what the microphone or system audio hears, shown as text while it's said.
//! The model isn't a streaming one, so the current window (up to 30 s) is transcribed again about once a second and
//! its text replaces the last one on screen: the newest words can still change, earlier ones settle as more audio
//! comes after them. A full window ends at its latest pause (the model's pause head, as for long takes), is
//! transcribed once more as a whole and kept; the next window starts there. Windowing from the old live mode
//! (removed 2026-11-24), without its pasting.
use crate::RATE;
use crate::asr::Asr;
use crate::audio::Listener;

/// A window is at most 30 s (the model's piece size).
const WINDOW: usize = RATE * 30;
/// Lengths go to the model in whole encoder frames (80 ms).
const FRAME: usize = RATE * 8 / 100;
/// Texts of finished windows kept for the display (the overlay shows the last two lines).
const KEEP: usize = 3;

pub struct Live {
    listener: Listener,
    /// The current window's audio.
    window: Vec<f32>,
    /// Texts of finished windows, oldest first.
    done: Vec<String>,
    /// The current window's latest text.
    current: String,
}

impl Live {
    pub fn new(listener: Listener) -> Live {
        Live { listener, window: Vec::new(), done: Vec::new(), current: String::new() }
    }

    /// New audio -> transcribe the window again -> the text to show (finished windows + the current one).
    pub fn pass(&mut self, a: &mut Asr) -> Result<String, String> {
        self.window.extend(self.listener.take());
        if self.window.len() >= WINDOW + FRAME {
            // Full: end it at the latest pause before 30 s, keep its final text, start the next window there.
            let cut = match a.first_cut(&self.window[..WINDOW + FRAME]) {
                Ok(c) => (c / FRAME * FRAME).clamp(FRAME * 12, WINDOW),
                Err(e) => {
                    eprintln!("live: {e}");
                    WINDOW
                }
            };
            let (text, _) = a.run(&self.window[..cut])?;
            println!("live: window of {:.1} s: {text}", cut as f64 / RATE as f64);
            if !text.is_empty() {
                self.done.push(text);
                if self.done.len() > KEEP {
                    self.done.remove(0);
                }
            }
            self.window.drain(..cut);
            self.current.clear();
        }
        let len = self.window.len() / FRAME * FRAME;
        self.current = if len >= RATE / 4 { a.run(&self.window[..len])?.0 } else { String::new() };
        Ok(self.done.iter().chain([&self.current]).filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" "))
    }
}
