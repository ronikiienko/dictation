//! dictation — offline dictation for KDE Wayland (Parakeet Ultra via transcribe.cpp). Settings are in the tray menu.
//!   dictation            run the tray app
//!   dictation toggle     start / stop a take in the running app (for a custom shortcut or scripts)
use dictation::daemon;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => daemon::run(),
        [cmd @ "toggle"] => match daemon::send(cmd) {
            Ok(_) => Ok(()),
            Err(_) => Err("Dictation is not running (start it with `dictation`).".into()),
        },
        _ => Err("usage: dictation [toggle]".into()),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
