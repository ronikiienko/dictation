//! dictation — offline dictation (Parakeet Ultra, own Rust port: GPU via burn/Vulkan on Linux and Windows, Neural Engine
//! on macOS, or the CPU). Settings are in the tray menu.
//!   dictation            run the tray app
//!   dictation toggle     start / stop a take in the running app (for a custom shortcut or scripts)
//!   dictation transcribe [--srt] <file>...   text (or subtitles, .srt format) of audio/video files on stdout (by
//!                        the running app if there is one, else loads the model itself)
//!   dictation sweep      compile (and tune, if enabled) GPU kernels for every take length ahead of time. The tray
//!                        app does this by itself (once per GPU, again only after Cargo.lock or rustc change).
//!                        Not on macOS (no GPU path).
// Windows: a tray app, not a console program (no black console window next to it; closing that would kill the app).
#![cfg_attr(windows, windows_subsystem = "windows")]
use dictation::{RATE, asr, config::Config, daemon, media, models};
use std::path::{Path, PathBuf};

/// `full`: the old exhaustive sweep (every 80 ms to 30 s, full takes to 120 s), to check the short one misses nothing.
#[cfg(gpu)]
fn sweep(full: bool) -> Result<(), String> {
    let m = models::current();
    if !m.installed() {
        return Err("model not installed: start the app and download it first".into());
    }
    asr::set_kernel_cache(&m.kernel_cache());
    let t = std::time::Instant::now();
    let mut a = asr::Asr::load(&m.weights())?;
    eprintln!("model loaded in {:.1} s, sweeping on {}", t.elapsed().as_secs_f64(), asr::gpu());
    a.sweep(full, |done, s, ms| eprintln!("{:3.0}% {s:6.2} s audio: {ms:6.0} ms", done * 100.0))?;
    if !full {
        asr::mark_swept(&m.kernel_cache()).map_err(|e| e.to_string())?;
    }
    eprintln!("done in {:.0} s", t.elapsed().as_secs_f64());
    Ok(())
}

/// Each file's text on stdout (a `# name` line before each if there are several); progress and errors on stderr.
/// The running app does the work if it's there (its model, maybe loaded already; no second copy in memory).
fn transcribe(files: &[&str], srt: bool) -> Result<(), String> {
    let mut paths = Vec::new();
    for f in files {
        let p = Path::new(f).canonicalize().map_err(|e| format!("{f}: {e}"))?;
        if p.to_string_lossy().contains('\n') {
            return Err(format!("{f}: file names with line breaks are not supported"));
        }
        paths.push(p);
    }
    let mut local: Option<asr::Asr> = None;
    let mut failed = false;
    for (p, f) in paths.iter().zip(files) {
        if files.len() > 1 {
            println!("# {f}");
        }
        let cmd = if srt { "subtitles" } else { "transcribe" };
        let r = match daemon::request(&format!("{cmd} {}", p.display()), None) {
            Ok(line) => serde_json::from_str::<serde_json::Value>(&line).map_err(|e| e.to_string()).and_then(|v| match v["text"].as_str() {
                Some(t) => Ok(t.to_string()),
                None => Err(v["error"].as_str().unwrap_or(&line).to_string()),
            }),
            Err(_) => run_here(p, srt, &mut local),
        };
        match r {
            Ok(t) if srt => print!("{t}"),
            Ok(t) => println!("{t}"),
            Err(e) => {
                eprintln!("{e}");
                failed = true;
            }
        }
    }
    match (failed, files.len()) {
        (false, _) => Ok(()),
        (true, 1) => Err(String::new()), // its error is printed already
        (true, _) => Err("some files were not transcribed".into()),
    }
}

/// The app isn't running: decode and transcribe in this process (model loaded on the first file).
fn run_here(p: &PathBuf, srt: bool, local: &mut Option<asr::Asr>) -> Result<String, String> {
    let x = media::read(p)?;
    let a = match local {
        Some(a) => a,
        None => {
            let m = models::current();
            if !m.installed() {
                return Err("model not installed: start the app (`dictation`) and download it first".into());
            }
            let cfg = Config::load();
            if m.ane {
                local.insert(asr::Asr::load_ane(&m.weights(), &m.ane_dir(), |_| {})?)
            } else if cfg.cpu {
                local.insert(asr::Asr::load_cpu(&m.weights())?)
            } else {
                #[cfg(gpu)]
                {
                    asr::set_kernel_cache(&m.kernel_cache());
                    local.insert(asr::Asr::load(&m.weights())?)
                }
                #[cfg(not(gpu))]
                return Err("no GPU path in this build: choose the Neural Engine or the CPU in the tray menu".into());
            }
        }
    };
    let t = std::time::Instant::now();
    let text = if srt { dictation::subtitles::srt(&a.run_words(&x)?.0) } else { a.run(&x)?.0 };
    eprintln!("{:.1} s of audio in {:.1} s", x.len() as f64 / RATE as f64, t.elapsed().as_secs_f64());
    Ok(text)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.first().is_some_and(|a| a == "sweep" || a == "transcribe") {
        asr::choose_gpu(&Config::load().gpu); // first: sets the environment while single-threaded
    }
    let r = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => daemon::run(),
        [cmd @ "toggle"] => match daemon::send(cmd) {
            Ok(_) => Ok(()),
            Err(_) => Err("Dictation is not running (start it with `dictation`).".into()),
        },
        #[cfg(gpu)]
        ["sweep"] => sweep(false).map_err(Into::into),
        #[cfg(gpu)]
        ["sweep", "full"] => sweep(true).map_err(Into::into),
        ["transcribe", "--srt", files @ ..] if !files.is_empty() => transcribe(files, true).map_err(Into::into),
        ["transcribe", files @ ..] if !files.is_empty() => transcribe(files, false).map_err(Into::into),
        _ => Err("usage: dictation [toggle | sweep | transcribe [--srt] <file>...]".into()),
    };
    if let Err(e) = r {
        if !e.to_string().is_empty() {
            eprintln!("{e}");
        }
        std::process::exit(1);
    }
}
