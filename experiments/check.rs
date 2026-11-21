//! Check the app's transcription path on recorded takes: whole take in (long takes are cut at pauses inside asr),
//! text out. One JSON line per take on stdout.
//! usage: cargo run --release --example check -- <wav>...   (uses the app's installed model)
//! LIVE=1: replay each take through the app's "paste while speaking" mode (asr::Live), audio fed in 300 ms steps.
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
    let m = &models::MODEL;
    if !m.installed() {
        // Same as the app: finish the download (or take an older version's GGUF), check, convert.
        let t = Instant::now();
        models::download(m, &std::sync::Arc::new(models::Progress::default()))?;
        eprintln!("model prepared in {:.1}s", t.elapsed().as_secs_f64());
    }
    asr::set_kernel_cache(&m.kernel_cache());
    let t = Instant::now();
    let mut a = asr::Asr::load(&m.weights())?;
    a.run(&vec![0.0; RATE])?;
    eprintln!("ready {:.2}s", t.elapsed().as_secs_f64());
    for arg in std::env::args().skip(1) {
        let path = std::path::Path::new(&arg);
        let x = read_wav(path)?;
        if std::env::var("LIVE").is_ok() {
            // Feed the audio in 300 ms steps, as the app's ticks do (each pass here runs instantly, not in real time).
            let mut l = asr::Live::default();
            for end in (RATE * 3 / 10..=x.len()).step_by(RATE * 3 / 10) {
                if let Some(p) = l.check(&mut a, &x[..end]) {
                    eprintln!("  at {:5.1} s paste: {p:?}", end as f64 / RATE as f64);
                }
            }
            let p = l.finish(&mut a, &x)?;
            eprintln!("  stop     paste: {p:?}");
            println!("{}", json!({"wav": path.file_name().unwrap().to_string_lossy(), "live_text": l.text}));
        }
        let t = Instant::now();
        let (text, s) = a.run(&x)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("{}", json!({"wav": path.file_name().unwrap().to_string_lossy(), "seconds": x.len() as f64 / RATE as f64,
            "asr_ms": ms.round(), "mel_ms": s.mel.round(), "pauses_ms": s.vad.round(), "encoder_ms": s.enc.round(),
            "decoder_ms": s.dec.round(), "text": text}));
    }
    // UNLOAD=1: drop the model, then report this process's VRAM and the GPU's runtime power state.
    if std::env::var("UNLOAD").is_ok() {
        let status = || {
            let smi = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).output().unwrap();
            let me = String::from_utf8_lossy(&smi.stdout).lines().find(|l| l.starts_with(&format!("{},", std::process::id()))).map(|l| l.to_string());
            let pm = std::fs::read_to_string("/sys/bus/pci/devices/0000:01:00.0/power/runtime_status").unwrap_or_default();
            format!("VRAM of this process: {:?} MiB; GPU power: {}", me.map(|l| l.split(',').nth(1).unwrap().trim().to_string()), pm.trim())
        };
        eprintln!("loaded:        {}", status());
        drop(a);
        for s in [1, 5, 20] {
            std::thread::sleep(std::time::Duration::from_secs(if s == 1 { 1 } else if s == 5 { 4 } else { 15 }));
            eprintln!("unloaded {s:2} s: {}", status());
        }
        let t = Instant::now();
        let mut a = asr::Asr::load(&m.weights())?;
        a.run(&vec![0.0; RATE])?;
        eprintln!("reload + warm-up in {:.2}s: {}", t.elapsed().as_secs_f64(), status());
    }
    Ok(())
}
