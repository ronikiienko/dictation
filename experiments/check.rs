//! Check the app's transcription path on recorded takes: whole take in (transcribe.cpp segments long takes
//! itself), text out. One JSON line per take on stdout.
//! usage: cargo run --release --example check -- <wav>...   (uses the app's downloaded model)
use dictation::*;
use serde_json::json;
use std::time::Instant;

fn read_wav(path: &std::path::Path) -> Result<Vec<f32>, hound::Error> {
    let mut r = hound::WavReader::open(path)?;
    let s = r.spec();
    assert!(s.sample_rate as usize == RATE && s.channels == 1 && s.bits_per_sample == 16, "need 16 kHz mono 16-bit");
    r.samples::<i16>().map(|v| v.map(|v| v as f32 / 32768.0)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tc::init()?;
    let t = Instant::now();
    let mut asr = tc::Asr::load(&models::find("").path(), 4)?;
    asr.run(&vec![0.0; RATE])?;
    eprintln!("ready {:.2}s backend {}", t.elapsed().as_secs_f64(), asr.backend());
    for arg in std::env::args().skip(1) {
        let path = std::path::Path::new(&arg);
        let x = read_wav(path)?;
        let t = Instant::now();
        let text = asr.run(&x)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{}", json!({"wav": path.file_name().unwrap().to_string_lossy(), "seconds": x.len() as f64 / RATE as f64,
            "asr_ms": ms.round(), "text": text}));
    }
    Ok(())
}
