//! Check the app's transcription path on recorded takes: whole take in (long takes are cut at pauses inside asr),
//! text out. One JSON line per take on stdout.
//! usage: cargo run --release --example check -- <audio>...  (any format media.rs reads)   (uses the app's installed model)
//! MODEL=<id> (models.rs, e.g. parakeet-redux): that model, downloaded like the app does if missing.
//! WEIGHTS=<file>: another weights file (e.g. Parakeet Redux converted by experiments/redux.rs). CPU=1: encoder on
//! the CPU (src/asr/cpu.rs). ANE=<dir> (macOS): encoder on the Neural Engine with the CoreML models in <dir>
//! (ANE=1: the app's Neural Engine model, models/parakeet-ultra-ane/ane/).
use dictation::*;
use serde_json::json;
use std::time::Instant;

fn load(w: &std::path::Path) -> Result<asr::Asr, String> {
    if let Ok(d) = std::env::var("ANE") {
        let d = if d == "1" { models::ULTRA_ANE.ane_dir() } else { d.into() };
        return asr::Asr::load_ane(w, &d, |_| {});
    }
    if std::env::var("CPU").is_ok() { asr::Asr::load_cpu(w) } else { asr::Asr::load(w) }
}

/// LOG=1: CubeCL's log records (debug and up) on stderr.
struct Log;
impl log::Log for Log {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, r: &log::Record) {
        if r.target().starts_with("cubecl") {
            eprintln!("{} {}: {}", r.level(), r.target(), r.args());
        }
    }
    fn flush(&self) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("LOG").is_ok() {
        let _ = log::set_logger(&Log).map(|_| log::set_max_level(log::LevelFilter::Debug));
    }
    let m = models::by_id(&std::env::var("MODEL").unwrap_or_default()); // MODEL=parakeet-redux: that model
    if !m.installed() {
        // Same as the app: finish the download (or take an older version's GGUF), check, convert.
        let t = Instant::now();
        models::download(m, &std::sync::Arc::new(models::Progress::default()))?;
        eprintln!("model prepared in {:.1}s", t.elapsed().as_secs_f64());
    }
    // KERNELS=<dir>: use another kernel cache folder (tests that would otherwise add entries to the app's own).
    asr::set_kernel_cache(&std::env::var("KERNELS").map(std::path::PathBuf::from).unwrap_or(m.kernel_cache()));
    let t = Instant::now();
    let w = std::env::var("WEIGHTS").map(std::path::PathBuf::from).unwrap_or(m.weights());
    let mut a = load(&w)?;
    a.run(&vec![0.0; RATE])?;
    eprintln!("ready {:.2}s", t.elapsed().as_secs_f64());
    // GAP=<seconds>: idle this long before each take (GPU clocking down between takes vs back-to-back).
    let gap = std::env::var("GAP").ok().and_then(|g| g.parse::<f64>().ok());
    for arg in std::env::args().skip(1) {
        if let Some(g) = gap {
            std::thread::sleep(std::time::Duration::from_secs_f64(g));
        }
        let path = std::path::Path::new(&arg);
        let x = media::read(path)?;
        let t = Instant::now();
        let (text, s) = a.run(&x)?;
        if std::env::var("MEMSTEP").is_ok() {
            // MEMSTEP=1: this process's VRAM (nvidia-smi) and the GPU pool after each take.
            let smi = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).output().unwrap();
            let me = String::from_utf8_lossy(&smi.stdout).lines().find(|l| l.starts_with(&format!("{},", std::process::id()))).map(|l| l.split(',').nth(1).unwrap().trim().to_string());
            // No NVIDIA entry (AMD etc.): the kernel's per-client counters, /proc/self/fdinfo drm-memory-vram, once per client.
            let me = me.unwrap_or_else(|| {
                let mut seen = std::collections::HashSet::new();
                let (mut kib, mut gtt) = (0u64, 0u64);
                for e in std::fs::read_dir("/proc/self/fdinfo").into_iter().flatten().flatten() {
                    let t = std::fs::read_to_string(e.path()).unwrap_or_default();
                    let field = |k: &str| t.lines().find(|l| l.starts_with(k)).map(|l| l[k.len()..].trim().to_string());
                    let num = |v: Option<String>| v.map_or(0, |v| v.trim_end_matches("KiB").trim().parse::<u64>().unwrap_or(0));
                    if let Some(id) = field("drm-client-id:") {
                        if field("drm-memory-vram:").is_some() && seen.insert(id) {
                            kib += num(field("drm-memory-vram:"));
                            gtt += num(field("drm-memory-gtt:"));
                        }
                    }
                }
                format!("{} (+ GTT {})", kib / 1024, gtt / 1024)
            });
            eprintln!("{:6.1} s take -> VRAM {} MiB | {}", x.len() as f64 / RATE as f64, me, asr::mem_report());
        }
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
            format!("VRAM of this process: {:?} MiB; GPU power: {}\n    pool: {}", me.map(|l| l.split(',').nth(1).unwrap().trim().to_string()), pm.trim(), asr::mem_report())
        };
        eprintln!("loaded:        {}", status());
        drop(a);
        for s in [1, 5, 20] {
            std::thread::sleep(std::time::Duration::from_secs(if s == 1 { 1 } else if s == 5 { 4 } else { 15 }));
            eprintln!("unloaded {s:2} s: {}", status());
        }
        let t = Instant::now();
        let w = std::env::var("WEIGHTS").map(std::path::PathBuf::from).unwrap_or(m.weights());
    let mut a = load(&w)?;
        a.run(&vec![0.0; RATE])?;
        eprintln!("reload + warm-up in {:.2}s: {}", t.elapsed().as_secs_f64(), status());
    }
    Ok(())
}
