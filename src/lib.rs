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
/// Project root (the crate folder, which also holds recordings/ and config.toml).
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
