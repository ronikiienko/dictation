pub mod audio;
pub mod config;
pub mod daemon;
pub mod models;
pub mod output;
pub mod tc;

use std::path::{Path, PathBuf};

/// transcribe.cpp takes 16 kHz mono f32. Takes longer than 30 s are split at pauses inside transcribe.cpp (Ultra's
/// own VAD head + segmenter), so the app passes every take whole.
pub const RATE: usize = 16000;
/// Project root (the crate folder, which also holds models/, recordings/, config.toml).
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn read_wav(path: &Path) -> Result<Vec<f32>, hound::Error> {
    let mut r = hound::WavReader::open(path)?;
    let s = r.spec();
    assert!(s.sample_rate as usize == RATE && s.channels == 1 && s.bits_per_sample == 16, "need 16 kHz mono 16-bit");
    r.samples::<i16>().map(|v| v.map(|v| v as f32 / 32768.0)).collect()
}
