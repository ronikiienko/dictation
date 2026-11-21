//! dictation — offline dictation for KDE Wayland (Parakeet Ultra, own Rust port on burn: Vulkan, macOS Metal). Settings are in the tray menu.
//!   dictation            run the tray app
//!   dictation toggle     start / stop a take in the running app (for a custom shortcut or scripts)
//!   dictation sweep      compile (and tune, if enabled) GPU kernels for every take length ahead of time. The tray
//!                        app does this by itself (once per GPU, again only after Cargo.lock or rustc change)
// Windows: a tray app, not a console program (no black console window next to it; closing that would kill the app).
#![cfg_attr(windows, windows_subsystem = "windows")]
use dictation::{asr, config::Config, daemon, models};

/// `full`: the old exhaustive sweep (every 80 ms to 30 s, full takes to 120 s), to check the short one misses nothing.
fn sweep(full: bool) -> Result<(), String> {
    let m = &models::MODEL;
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.first().is_some_and(|a| a == "sweep") {
        asr::choose_gpu(&Config::load().gpu); // first: sets the environment while single-threaded
    }
    let r = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => daemon::run(),
        [cmd @ "toggle"] => match daemon::send(cmd) {
            Ok(_) => Ok(()),
            Err(_) => Err("Dictation is not running (start it with `dictation`).".into()),
        },
        ["sweep"] => sweep(false).map_err(Into::into),
        ["sweep", "full"] => sweep(true).map_err(Into::into),
        _ => Err("usage: dictation [toggle | sweep]".into()),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
