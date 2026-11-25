//! The model and its downloader (Hugging Face, resumable via .part files, SHA-256 checked).
//! Models live in the OS's per-user app data folder (`directories` crate):
//! Linux `~/.local/share/dictation/models`, macOS `~/Library/Application Support/dictation/models`,
//! Windows `%APPDATA%\dictation\data\models`. Each model has its own folder `<models>/<id>/` with
//! `model.weights` (converted from the download, which is deleted afterwards) and `kernels/` (compiled GPU
//! kernels, filled as lengths are first seen).
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// One file of a model's download.
pub struct File {
    pub name: &'static str,
    pub url: &'static str,
    pub size: u64,
    /// Hex SHA-256 (Hugging Face's LFS oid for big files).
    pub sha256: &'static str,
}

/// What the download is, i.e. how it becomes `model.weights`.
#[derive(PartialEq)]
pub enum Format {
    /// One transcribe.cpp GGUF (asr::convert).
    Gguf,
    /// Parakeet Redux's safetensors + tokenizer.json + config.json (asr::convert_redux): CPU only.
    Redux,
    /// Parakeet Ultra's original (fp16) safetensors + tokenizer.json + config.json: the encoder for the Neural
    /// Engine (asr::build_ane, into `ane/`) and the rest (asr::convert_ultra_head, `model.weights`).
    UltraAne,
}

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub format: Format,
    pub files: &'static [File],
    /// Runs on the GPU (else CPU only, or the Neural Engine).
    pub gpu: bool,
    /// Runs on the Neural Engine only (macOS).
    pub ane: bool,
}

/// The default model (src/asr is a port of exactly this architecture).
pub const MODEL: ModelInfo = ModelInfo {
    id: "parakeet-ultra-q8",
    name: "Parakeet Ultra, 8-bit (Q8_0)",
    format: Format::Gguf,
    files: &[File {
        name: "parakeet-ultra-Q8_0.gguf",
        url: "https://huggingface.co/handy-computer/parakeet-ultra-gguf/resolve/main/parakeet-ultra-Q8_0.gguf",
        size: 740_363_168,
        sha256: "007a59761e9258779f189df396b50d12b83bb5b79e66df4b955c230d2f2a0a59",
    }],
    gpu: true,
    ane: false,
};

/// Same architecture with a ternary encoder: ~4x less memory on the CPU, less accurate (NOTES.md "Encoder on the
/// CPU"). Pinned to one revision of the repo.
pub const REDUX: ModelInfo = ModelInfo {
    id: "parakeet-redux",
    name: "Parakeet Redux, ternary",
    format: Format::Redux,
    files: &[
        File {
            name: "model.safetensors",
            url: "https://huggingface.co/moondream/parakeet-redux/resolve/2bf128600aac4b16946f7ed8372e56117fe5e23b/model.safetensors",
            size: 177_774_490,
            sha256: "78ec25733ee0d0c1586d1346fc86db9d0c2e436e3a8ab1d32a82d1bb8f848d21",
        },
        File {
            name: "tokenizer.json",
            url: "https://huggingface.co/moondream/parakeet-redux/resolve/2bf128600aac4b16946f7ed8372e56117fe5e23b/tokenizer.json",
            size: 1_159_960,
            sha256: "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
        },
        File {
            name: "config.json",
            url: "https://huggingface.co/moondream/parakeet-redux/resolve/2bf128600aac4b16946f7ed8372e56117fe5e23b/config.json",
            size: 12_988,
            sha256: "503c653b2e3bb788adbcb04f5abdee532d958686564081baeed133ff10143f6e",
        },
    ],
    gpu: false,
    ane: false,
};

/// Parakeet Ultra on the Neural Engine (macOS): its own download, the original fp16 weights (as the Neural Engine
/// computes), pinned to one revision; nothing shared with the GGUF model's folder.
pub const ULTRA_ANE: ModelInfo = ModelInfo {
    id: "parakeet-ultra-ane",
    name: "Parakeet Ultra",
    format: Format::UltraAne,
    files: &[
        File {
            name: "model.safetensors",
            url: "https://huggingface.co/moondream/parakeet-ultra/resolve/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b/model.safetensors",
            size: 1_255_353_386,
            sha256: "c9608f36d0ab956c14bfcc525479b0746b3b42a56f6949ec85c14eb7466717dc",
        },
        File {
            name: "tokenizer.json",
            url: "https://huggingface.co/moondream/parakeet-ultra/resolve/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b/tokenizer.json",
            size: 1_159_960,
            sha256: "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
        },
        File {
            name: "config.json",
            url: "https://huggingface.co/moondream/parakeet-ultra/resolve/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b/config.json",
            size: 1_153,
            sha256: "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
        },
    ],
    gpu: false,
    ane: true,
};

pub const ALL: [&ModelInfo; 3] = [&MODEL, &REDUX, &ULTRA_ANE];

/// The model chosen in config.toml, fixed for this run (choosing another restarts the app).
pub fn current() -> &'static ModelInfo {
    static CURRENT: std::sync::OnceLock<&'static ModelInfo> = std::sync::OnceLock::new();
    CURRENT.get_or_init(|| by_id(&crate::config::Config::load().model))
}

/// The model with this id (config.toml `model`); unknown, "" or not runnable here (the Neural Engine one off macOS):
/// the default.
pub fn by_id(id: &str) -> &'static ModelInfo {
    ALL.into_iter().find(|m| m.id == id && (!m.ane || cfg!(target_os = "macos"))).unwrap_or(&MODEL)
}

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
    /// The Neural Engine model's folder (macOS): compiled CoreML model asr::ANE_MODEL (asr::build_ane).
    pub fn ane_dir(&self) -> PathBuf {
        self.folder().join("ane")
    }
    /// The Neural Engine model is there (not checked further: a broken one fails to load, with its error). An older
    /// build (other name) doesn't count: the model is downloaded and built again.
    pub fn ane_installed(&self) -> bool {
        self.ane_dir().join(crate::asr::ANE_MODEL).is_dir()
    }
    pub fn kernel_cache(&self) -> PathBuf {
        self.folder().join("kernels")
    }
    /// Download size, MB.
    pub fn size_mb(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum::<u64>() / 1_000_000
    }
    fn part(&self, f: &File) -> PathBuf {
        self.folder().join(format!("{}.part", f.name))
    }
    /// A converted weights file this build can load (and the Neural Engine model, for that one).
    pub fn installed(&self) -> bool {
        crate::asr::is_current(&self.weights()) && (!self.ane || self.ane_installed())
    }
    /// A finished (or partial) download is already on disk, or a GGUF from an older version of the app sits in
    /// the models folder: "downloading" then only checks and converts it.
    pub fn download_started(&self) -> bool {
        self.files.iter().any(|f| self.part(f).exists() || self.folder().join(f.name).exists())
            || (self.format == Format::Gguf && dir().join(self.files[0].name).exists())
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

/// Download every file to `<id>/<file>.part` (resuming if it exists), check its SHA-256, rename it to `<file>`;
/// then convert them into `model.weights` and delete them. Blocking; run it in a thread.
pub fn download(m: &ModelInfo, p: &Arc<Progress>) -> Result<(), String> {
    std::fs::create_dir_all(m.folder()).map_err(|e| e.to_string())?;
    // A GGUF from an older version of the app (models/<file>): use it as a finished download.
    let old = dir().join(m.files[0].name);
    if m.format == Format::Gguf && old.exists() && !m.part(&m.files[0]).exists() {
        std::fs::rename(&old, m.part(&m.files[0])).map_err(|e| e.to_string())?;
    }
    p.total.store(m.files.iter().map(|f| f.size).sum(), Ordering::Relaxed);
    let mut before = 0; // bytes of the files already handled
    for f in m.files {
        let done = m.folder().join(f.name);
        if !(done.exists() && std::fs::metadata(&done).is_ok_and(|x| x.len() == f.size)) {
            fetch(f, &m.part(f), before, p)?;
            std::fs::rename(m.part(f), &done).map_err(|e| e.to_string())?;
        }
        before += f.size;
        p.done.store(before, Ordering::Relaxed);
    }
    let files: Vec<PathBuf> = m.files.iter().map(|f| m.folder().join(f.name)).collect();
    let r = match m.format {
        Format::Gguf => crate::asr::convert(&files[0], &m.weights()),
        Format::Redux => crate::asr::convert_redux(&m.folder(), &m.weights(), false),
        Format::UltraAne => crate::asr::build_ane(&files[0], &m.ane_dir(), |_| {})
            .and_then(|()| crate::asr::convert_ultra_head(&m.folder(), &m.weights())),
    };
    r.map_err(|e| format!("converting the model: {e}"))?;
    for f in files {
        std::fs::remove_file(f).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One file into `part` (resuming), SHA-256 checked. `before`: bytes of earlier files, for the progress.
fn fetch(f: &File, part: &std::path::Path, before: u64, p: &Arc<Progress>) -> Result<(), String> {
    let mut have = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have > f.size {
        have = 0;
    }
    p.verifying.store(false, Ordering::Relaxed);
    p.done.store(before + have, Ordering::Relaxed);
    if have < f.size {
        let mut req = ureq::get(f.url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut res = req.call().map_err(|e| format!("download failed: {e}"))?;
        let mut file = std::fs::OpenOptions::new().create(true).write(true).open(part).map_err(|e| e.to_string())?;
        if res.status() == 206 {
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(have)).map_err(|e| e.to_string())?;
        } else {
            file.set_len(0).map_err(|e| e.to_string())?; // server ignored Range: start over
            have = 0;
            p.done.store(before, Ordering::Relaxed);
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
            p.done.store(before + have, Ordering::Relaxed);
        }
        file.sync_all().map_err(|e| e.to_string())?;
    }
    if have != f.size {
        return Err(format!("download incomplete: {have} of {} bytes (try again to resume)", f.size));
    }
    p.verifying.store(true, Ordering::Relaxed);
    let got = sha256_file(part).map_err(|e| e.to_string())?;
    if got != f.sha256 {
        // Corrupt: resuming would keep the bad bytes, so start over next time.
        let _ = std::fs::remove_file(part);
        return Err(format!("{}: checksum mismatch (got {got}); the file was removed, try again", f.name));
    }
    Ok(())
}
