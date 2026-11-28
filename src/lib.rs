pub mod asr;
pub mod audio;
pub mod config;
pub mod daemon;
pub mod icons;
pub mod login;
pub mod media;
pub mod models;
pub mod output;
pub mod overlay;
pub mod perms;
pub mod subtitles;
pub mod tray;

use std::path::PathBuf;

/// The model takes 16 kHz mono f32. Takes longer than 30 s are split at pauses inside `asr` (Ultra's own pause head
/// + segmenter), so the app passes every take whole.
pub const RATE: usize = 16000;
/// The app's one folder, the OS per-user local app data folder (`directories` crate; Linux: ~/.local/share/dictation,
/// macOS: ~/Library/Application Support/dictation, Windows: AppData\Local\dictation\data, not Roaming: the model
/// is ~1 GB): config.toml, recordings/, models/.
pub fn root() -> PathBuf {
    directories::ProjectDirs::from("", "", "dictation").expect("no home directory").data_local_dir().to_path_buf()
}

/// Show the app folder in the file manager (created first, so there is something to open).
pub fn open_root() -> Result<(), String> {
    let d = root();
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    open::that_detached(&d).map_err(|e| format!("open {}: {e}", d.display()))
}

/// The build ID CubeCL keys its kernel cache by (`buildid` with feature `buildid-custom-inject` calls this instead of
/// reading the OS-specific one from the binary): our build key (build.rs: Cargo.lock + rustc), so the cache survives
/// rebuilds that don't change the kernels. Same on every OS.
#[cfg(gpu)]
#[unsafe(no_mangle)]
pub extern "C" fn build_id__get(build_id: *mut *const u8, len: *mut usize) -> std::ffi::c_int {
    const KEY: &str = env!("DICTATION_BUILD_KEY");
    // SAFETY: buildid passes valid pointers to write the result to.
    unsafe {
        *build_id = KEY.as_ptr();
        *len = KEY.len();
    }
    1
}
