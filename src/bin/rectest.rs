//! Recorder check: record for N seconds (default 2.3), print how much audio came back. No transcription, no paste.
fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2.3);
    let t = std::time::Instant::now();
    let r = dictation::audio::Recorder::start("").expect("mic");
    let opened = t.elapsed();
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    let x = r.stop().expect("stop");
    println!("stream opened in {:.0} ms; held {secs:.2} s (+0.15 s tail); got {:.3} s of audio",
        opened.as_secs_f64() * 1000.0, x.len() as f64 / 16000.0);
}
