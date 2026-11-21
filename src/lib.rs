pub mod asr;
pub mod audio;
pub mod config;
pub mod daemon;
pub mod models;
pub mod output;

use std::path::PathBuf;

/// The model takes 16 kHz mono f32. Takes longer than 30 s are split at pauses inside `asr` (Ultra's own pause head
/// + segmenter), so the app passes every take whole.
pub const RATE: usize = 16000;
/// The app's one folder, the OS per-user app data folder (`directories` crate; Linux: ~/.local/share/dictation):
/// config.toml, recordings/, models/.
pub fn root() -> PathBuf {
    directories::ProjectDirs::from("", "", "dictation").expect("no home directory").data_dir().to_path_buf()
}

/// Show the app folder in the file manager (created first, so there is something to open).
pub fn open_root() -> Result<(), String> {
    let d = root();
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    open::that_detached(&d).map_err(|e| format!("open {}: {e}", d.display()))
}
