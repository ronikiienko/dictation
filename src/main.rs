//! dictation — offline dictation for KDE Wayland (Parakeet Ultra via transcribe.cpp). Settings are in the tray menu.
//!   dictation            run the tray app
//!   dictation toggle     start / stop a take in the running app (for a custom shortcut or scripts)
//!   dictation reload     re-read config.toml after editing it by hand
//!   dictation unload     free the speech model now (it loads again on the next take)
use dictation::daemon;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => daemon::run(),
        [cmd @ ("toggle" | "reload" | "unload")] => match daemon::send(cmd) {
            Ok(_) => Ok(()),
            Err(_) => Err("Dictation is not running (start it with `dictation`).".into()),
        },
        _ => Err("usage: dictation [toggle | reload | unload]".into()),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
