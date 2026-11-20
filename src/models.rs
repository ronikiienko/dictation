//! Model catalog and downloader (Hugging Face, resumable via .part files, SHA-256 checked).
//! Models live in the OS's per-user app data folder (`directories` crate):
//! Linux `~/.local/share/dictation/models`, macOS `~/Library/Application Support/dictation/models`,
//! Windows `%APPDATA%\dictation\data\models`.
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub note: &'static str,
    /// File name inside the models folder.
    pub file: &'static str,
    pub url: &'static str,
    pub size: u64,
    /// Hex SHA-256 of the file (Hugging Face's LFS oid).
    pub sha256: &'static str,
}

pub const CATALOG: &[ModelInfo] = &[ModelInfo {
    id: "parakeet-ultra-q8",
    name: "Parakeet Ultra, 8-bit (Q8_0)",
    note: "recommended: best accuracy, ~0.7 GB VRAM",
    file: "parakeet-ultra-Q8_0.gguf",
    url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
    size: 740_363_168,
    sha256: "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59",
}];

pub fn find(id: &str) -> &'static ModelInfo {
    CATALOG.iter().find(|m| m.id == id).unwrap_or(&CATALOG[0])
}

/// Folder holding the downloaded models (not created here).
pub fn dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "dictation").expect("no home directory").data_dir().join("models")
}

/// Show the models folder in the file manager (created first, so there is something to open).
pub fn open_dir() -> Result<(), String> {
    let d = dir();
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    open::that_detached(&d).map_err(|e| format!("open {}: {e}", d.display()))
}

impl ModelInfo {
    pub fn path(&self) -> PathBuf {
        dir().join(self.file)
    }
    /// Cheap check (size only); the full SHA-256 check runs once, after downloading.
    pub fn installed(&self) -> bool {
        std::fs::metadata(self.path()).map(|m| m.len() == self.size).unwrap_or(false)
    }
}

/// Progress shared with the UI thread.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
    /// Download finished, checking the SHA-256.
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

/// Download to `<file>.part` (resuming if it exists), check the SHA-256, then rename into place.
/// Blocking; run it in a thread.
pub fn download(m: &ModelInfo, p: &Arc<Progress>) -> Result<(), String> {
    let dest = m.path();
    let part = dest.with_extension("gguf.part");
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
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
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())
}
