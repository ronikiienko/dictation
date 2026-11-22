//! Single words cut out of whole takes, transcribed on their own: does the model ever switch language/script on one
//! isolated word? Cut points come from the model's own word times on the whole take: a word runs from one frame
//! (80 ms) before its first token (but not before the previous word's end) to the next word's start; 0.5 s of
//! silence is added on both sides (like a short dictation take). One JSON line per word on stdout, summary on stderr.
//! usage: cargo run --release --example words -- <audio>...   (any format media.rs reads; uses the app's model)
use dictation::*;
use serde_json::json;

/// "latin" (ASCII letters only), "latin+" (Latin with accents, e.g. é), "cyrillic", "other", or "" (no letters).
fn script(s: &str) -> &'static str {
    let letters: Vec<char> = s.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        ""
    } else if letters.iter().any(|c| ('\u{0400}'..='\u{04FF}').contains(c)) {
        "cyrillic"
    } else if letters.iter().all(|c| c.is_ascii()) {
        "latin"
    } else if letters.iter().all(|c| (*c as u32) < 0x0250) {
        "latin+"
    } else {
        "other"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = &models::MODEL;
    asr::set_kernel_cache(&m.kernel_cache());
    let mut a = asr::Asr::load(&m.weights())?;
    let pad = vec![0.0f32; RATE / 2];
    let mut count = std::collections::BTreeMap::<&str, usize>::new();
    for arg in std::env::args().skip(1) {
        let x = media::read(std::path::Path::new(&arg))?;
        let (words, _) = a.run_words(&x)?;
        let at = |t: f64| ((t * RATE as f64).round() as usize).min(x.len());
        for (i, w) in words.iter().enumerate() {
            let prev_end = if i > 0 { words[i - 1].end } else { 0.0 };
            let start = at((w.start - 0.08).max(prev_end).max(0.0));
            let end = words.get(i + 1).map_or(x.len(), |n| at(n.start)).max(start);
            let clip: Vec<f32> = [&pad[..], &x[start..end], &pad[..]].concat();
            let (text, _) = a.run(&clip)?;
            let s = script(&text);
            *count.entry(if s.is_empty() { "empty" } else { s }).or_default() += 1;
            println!(
                "{}",
                json!({"file": arg, "word": w.text, "start": start as f64 / RATE as f64,
                       "end": end as f64 / RATE as f64, "alone": text, "script": s})
            );
        }
    }
    eprintln!("{count:?}");
    Ok(())
}
