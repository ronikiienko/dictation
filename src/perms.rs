//! Permissions the system can withhold, checked without prompting, so the tray can say what's missing (and open the
//! right settings page) instead of takes failing with a vague mark. macOS only: Accessibility (pressing the paste keys)
//! and the microphone. Not checkable there: system audio recording (no public API; a denied tap records silence, caught
//! after the take) and notifications. Linux/Windows: nothing to check (paste via /dev/uinput or SendInput fails loudly).

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Missing {
    /// Can't press the paste keys: text only lands on the clipboard.
    Accessibility,
    /// The microphone was denied: takes would record silence.
    Microphone,
}

impl Missing {
    /// Tray line (an action that opens the settings page).
    pub fn label(self) -> &'static str {
        match self {
            Missing::Accessibility => "⚠ Paste needs Accessibility: open settings…",
            Missing::Microphone => "⚠ Microphone not allowed: open settings…",
        }
    }

    /// The settings page where it's granted.
    pub fn open_settings(self) {
        #[cfg(target_os = "macos")]
        {
            // Accessibility: ask the system again first; that (re)adds Dictation to the list. Needed after the entry
            // was removed, or when it belongs to an older build (a new signature doesn't match it: the switch shows
            // on, but this build isn't allowed). The app asks by itself only once, at start.
            if self == Missing::Accessibility {
                let ask = enigo::Settings { open_prompt_to_get_permissions: true, ..Default::default() };
                let _ = enigo::Enigo::new(&ask);
            }
            let pane = match self {
                Missing::Accessibility => "Privacy_Accessibility",
                Missing::Microphone => "Privacy_Microphone",
            };
            if let Err(e) = open::that_detached(format!("x-apple.systempreferences:com.apple.preference.security?{pane}")) {
                eprintln!("open settings: {e}");
            }
        }
    }
}

/// What's missing now. `mic`: the input is a microphone (system audio needs no microphone permission).
pub fn missing(mic: bool) -> Vec<Missing> {
    let mut m = Vec::new();
    if !imp::accessibility() {
        m.push(Missing::Accessibility);
    }
    if mic && imp::microphone_denied() {
        m.push(Missing::Microphone);
    }
    m
}

/// The microphone is denied now (no prompt).
pub fn microphone_denied() -> bool {
    imp::microphone_denied()
}

/// Whether the microphone may be used; asks first (the system's prompt, blocking until answered) if it never was.
/// Asking here, not by opening the stream: when the audio library's open triggers the prompt, a "Don't Allow" left
/// this process reading "not asked yet" (measured: status stayed 0 after the denial), and the take recorded silence.
pub fn microphone_allowed() -> bool {
    imp::microphone_allowed()
}

#[cfg(target_os = "macos")]
mod imp {
    use block2::RcBlock;
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, Bool};
    use objc2_foundation::NSString;

    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> u8;
    }
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}

    pub fn accessibility() -> bool {
        // SAFETY: no arguments; only reads this process's trust state (never prompts).
        unsafe { AXIsProcessTrusted() != 0 }
    }

    /// AVAuthorizationStatus: 0 not asked yet, 1 restricted, 2 denied, 3 allowed.
    fn status() -> Option<isize> {
        let class = AnyClass::get(c"AVCaptureDevice")?;
        let audio = NSString::from_str("soun"); // AVMediaTypeAudio
        // SAFETY: class method taking an AVMediaType (NSString), returning AVAuthorizationStatus (NSInteger).
        Some(unsafe { msg_send![class, authorizationStatusForMediaType: &*audio] })
    }

    pub fn microphone_denied() -> bool {
        matches!(status(), Some(1 | 2))
    }

    pub fn microphone_allowed() -> bool {
        match status() {
            Some(0) => {}
            Some(s) => return s == 3,
            None => return true, // can't tell: let recording try
        }
        // Asking needs the usage text in the app's Info.plist (without it macOS ends the process): a bare binary
        // (development) leaves it to opening the stream, as before.
        if !std::env::current_exe().is_ok_and(|p| p.to_string_lossy().contains(".app/Contents/MacOS/")) {
            return true;
        }
        let Some(class) = AnyClass::get(c"AVCaptureDevice") else { return true };
        let (tx, rx) = std::sync::mpsc::channel();
        let done = RcBlock::new(move |granted: Bool| {
            let _ = tx.send(granted.as_bool());
        });
        let audio = NSString::from_str("soun");
        // SAFETY: class method taking an AVMediaType and a block called once (any thread) with the answer.
        let _: () = unsafe { msg_send![class, requestAccessForMediaType: &*audio, completionHandler: &*done] };
        rx.recv().unwrap_or(false)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    pub fn accessibility() -> bool {
        true
    }

    pub fn microphone_denied() -> bool {
        false
    }

    pub fn microphone_allowed() -> bool {
        true
    }
}
