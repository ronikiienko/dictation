//! Settings, stored as TOML in the app folder (`config.toml` next to models/ and recordings/, see `crate::root`).
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Config {
    /// GPU (adapter name), or "" for the default: the fastest one ("high performance").
    pub gpu: String,
    /// Run the speech model on the CPU instead of the GPU (src/asr/cpu.rs): no GPU needed, nothing to tune.
    pub cpu: bool,
    /// Microphone: cpal device id, or "" for the system default.
    pub microphone: String,
    /// What happens with the text.
    pub output: Output,
    /// Keys pressed to paste.
    pub paste_keys: PasteKeys,
    /// Paste words while speaking (Output::Paste only), instead of the whole take after stop.
    pub live: bool,
    /// Start / stop sounds.
    pub sounds: bool,
    /// Desktop notifications (recording, transcribing, result).
    pub notifications: bool,
    /// Keep every take as wav + a line in recordings/log.jsonl (off by default: it stores the user's voice).
    pub save_recordings: bool,
    /// When to free the speech model (~0.7 GB VRAM). It is loaded again when the next take starts, while you speak
    /// (~1 s).
    pub keep_loaded: KeepLoaded,
    /// macOS/Windows: the global shortcut, e.g. "Ctrl+Alt+Space" ("" = none). Linux: set in the desktop's settings
    /// (portal), so not written to config.toml there.
    #[cfg_attr(target_os = "linux", serde(skip_serializing))]
    pub shortcut: String,
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
            gpu: String::new(),
            cpu: false,
            microphone: String::new(),
            output: Output::Paste,
            paste_keys: PasteKeys::CtrlV,
            live: false,
            sounds: true,
            notifications: true,
            save_recordings: false,
            keep_loaded: KeepLoaded::OneMinute,
            shortcut: "Ctrl+Alt+Space".into(),
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
        std::fs::create_dir_all(crate::root())?;
        let tmp = path().with_extension("toml.tmp");
        std::fs::write(&tmp, toml::to_string_pretty(self).expect("serialize config"))?;
        std::fs::rename(tmp, path())
    }
}
