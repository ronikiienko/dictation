//! Model catalog and downloader (Hugging Face, resumable via .part files).
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub note: &'static str,
    /// Relative to the project root.
    pub file: &'static str,
    pub url: &'static str,
    pub size: u64,
}

pub const CATALOG: &[ModelInfo] = &[
    ModelInfo {
        id: "parakeet-ultra-q8",
        name: "Parakeet Ultra, 8-bit (Q8_0)",
        note: "recommended: best accuracy, ~0.7 GB VRAM",
        file: "models/parakeet-ultra-gguf/parakeet-ultra-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
        size: 740_363_168,
    },
];

pub fn find(id: &str) -> &'static ModelInfo {
    CATALOG.iter().find(|m| m.id == id).unwrap_or(&CATALOG[0])
}

impl ModelInfo {
    pub fn path(&self) -> PathBuf {
        crate::root().join(self.file)
    }
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
}

/// Download to `<file>.part` (resuming if it exists), then rename into place. Blocking; run it in a thread.
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
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())
}
