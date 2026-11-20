//! Settings, stored as TOML. For now inside the project (`config.toml` in the project root), like everything else here;
//! a packaged build would use ~/.config/dictation/config.toml.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Config {
    /// Speech model id from models::CATALOG.
    pub model: String,
    /// CPU threads for the decoder (it runs on the CPU in transcribe.cpp).
    pub threads: i32,
    /// Microphone: cpal device id, or "" for the system default.
    pub microphone: String,
    /// What happens with the text.
    pub output: Output,
    /// Keys pressed to paste.
    pub paste_keys: PasteKeys,
    /// Start / stop sounds.
    pub sounds: bool,
    /// Desktop notifications (recording, transcribing, result).
    pub notifications: bool,
    /// Keep every take as wav + a line in recordings/log.jsonl (like gauge).
    pub save_recordings: bool,
    /// When to free the speech model (~0.7 GB VRAM). It is loaded again when the next take starts, while you speak
    /// (~1 s).
    pub keep_loaded: KeepLoaded,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum KeepLoaded {
    /// Unload as soon as the take is pasted.
    AfterTake,
    OneMinute,
    TenMinutes,
    Always,
}

impl KeepLoaded {
    /// Idle time before unloading; None = never.
    pub fn idle(self) -> Option<std::time::Duration> {
        match self {
            KeepLoaded::AfterTake => Some(std::time::Duration::ZERO),
            KeepLoaded::OneMinute => Some(std::time::Duration::from_secs(60)),
            KeepLoaded::TenMinutes => Some(std::time::Duration::from_secs(600)),
            KeepLoaded::Always => None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    /// Put on the clipboard and press the paste keys.
    Paste,
    /// Only put on the clipboard.
    Clipboard,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum PasteKeys {
    CtrlV,
    /// Terminals (Konsole etc.) paste with Ctrl+Shift+V.
    CtrlShiftV,
    /// Also pastes in most terminals and most other apps.
    ShiftInsert,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: "parakeet-ultra-q8".into(),
            threads: 4,
            microphone: String::new(),
            output: Output::Paste,
            paste_keys: PasteKeys::CtrlV,
            sounds: true,
            notifications: true,
            save_recordings: true,
            keep_loaded: KeepLoaded::OneMinute,
        }
    }
}

pub fn path() -> PathBuf {
    crate::root().join("config.toml")
}

impl Config {
    /// Missing file -> defaults (and writes them, so there is something to look at).
    pub fn load() -> Config {
        match std::fs::read_to_string(path()) {
            Ok(s) => toml::from_str(&s).unwrap_or_else(|e| {
                eprintln!("config.toml: {e}; using defaults");
                Config::default()
            }),
            Err(_) => {
                let c = Config::default();
                let _ = c.save();
                c
            }
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let tmp = path().with_extension("toml.tmp");
        std::fs::write(&tmp, toml::to_string_pretty(self).expect("serialize config"))?;
        std::fs::rename(tmp, path())
    }
}
