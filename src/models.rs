//! The model and its downloader (Hugging Face, resumable via .part files, SHA-256 checked).
//! Models live in the OS's per-user app data folder (`directories` crate):
//! Linux `~/.local/share/dictation/models`, macOS `~/Library/Application Support/dictation/models`,
//! Windows `%APPDATA%\dictation\data\models`. Each model has its own folder `<models>/<id>/` with
//! `model.weights` (converted from the downloaded GGUF, which is deleted afterwards) and `kernels/` (compiled GPU
//! kernels, filled as lengths are first seen).
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    /// File name of the download (a transcribe.cpp GGUF).
    pub file: &'static str,
    pub url: &'static str,
    pub size: u64,
    /// Hex SHA-256 of the file (Hugging Face's LFS oid).
    pub sha256: &'static str,
}

/// The app's one model (src/asr is a port of exactly this architecture).
pub const MODEL: ModelInfo = ModelInfo {
    id: "parakeet-ultra-q8",
    name: "Parakeet Ultra, 8-bit (Q8_0)",
    file: "parakeet-ultra-Q8_0.gguf",
    url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
    size: 740_363_168,
    sha256: "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59",
};

/// Folder holding the downloaded models (not created here).
pub fn dir() -> PathBuf {
    crate::root().join("models")
}

impl ModelInfo {
    pub fn folder(&self) -> PathBuf {
        dir().join(self.id)
    }
    pub fn weights(&self) -> PathBuf {
        self.folder().join("model.weights")
    }
    pub fn kernel_cache(&self) -> PathBuf {
        self.folder().join("kernels")
    }
    fn gguf_part(&self) -> PathBuf {
        self.folder().join(format!("{}.part", self.file))
    }
    /// A converted weights file this build can load.
    pub fn installed(&self) -> bool {
        crate::asr::is_current(&self.weights())
    }
    /// A finished (or partial) download is already on disk, or a GGUF from an older version of the app sits in
    /// the models folder: "downloading" then only checks and converts it.
    pub fn download_started(&self) -> bool {
        self.gguf_part().exists() || dir().join(self.file).exists()
    }
}

/// Progress shared with the UI thread.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
    /// Download finished: checking the SHA-256, then converting.
    pub verifying: AtomicBool,
}

fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Download the GGUF to `<id>/<file>.part` (resuming if it exists), check the SHA-256, convert it into
/// `model.weights`, delete the GGUF. Blocking; run it in a thread.
pub fn download(m: &ModelInfo, p: &Arc<Progress>) -> Result<(), String> {
    let part = m.gguf_part();
    std::fs::create_dir_all(m.folder()).map_err(|e| e.to_string())?;
    // A GGUF from an older version of the app (models/<file>): use it as a finished download.
    let old = dir().join(m.file);
    if old.exists() && !part.exists() {
        std::fs::rename(&old, &part).map_err(|e| e.to_string())?;
    }
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > m.size {
        have = 0;
    }
    p.total.store(m.size, Ordering::Relaxed);
    p.done.store(have, Ordering::Relaxed);
    if have < m.size {
        let mut req = ureq::get(m.url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut res = req.call().map_err(|e| format!("download failed: {e}"))?;
        let mut file = std::fs::OpenOptions::new().create(true).write(true).open(&part).map_err(|e| e.to_string())?;
        if res.status() == 206 {
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(have)).map_err(|e| e.to_string())?;
        } else {
            file.set_len(0).map_err(|e| e.to_string())?; // server ignored Range: start over
            have = 0;
            p.done.store(0, Ordering::Relaxed);
        }
        let mut body = res.body_mut().with_config().limit(u64::MAX).reader();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            if p.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let n = body.read(&mut buf).map_err(|e| format!("download interrupted: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            have += n as u64;
            p.done.store(have, Ordering::Relaxed);
        }
        file.sync_all().map_err(|e| e.to_string())?;
    }
    if have != m.size {
        return Err(format!("download incomplete: {have} of {} bytes (try again to resume)", m.size));
    }
    p.verifying.store(true, Ordering::Relaxed);
    let got = sha256_file(&part).map_err(|e| e.to_string())?;
    if got != m.sha256 {
        // Corrupt: resuming would keep the bad bytes, so start over next time.
        let _ = std::fs::remove_file(&part);
        return Err(format!("checksum mismatch (got {got}); the file was removed, try again"));
    }
    crate::asr::convert(&part, &m.weights()).map_err(|e| format!("converting the model: {e}"))?;
    std::fs::remove_file(&part).map_err(|e| e.to_string())
}
