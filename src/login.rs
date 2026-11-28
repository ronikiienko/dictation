//! "Open at login" (tray checkbox): the app adds or removes itself; the system keeps the setting, not config.toml.
//! macOS: SMAppService (macOS 13+), the login item of the app bundle; only when running as Dictation.app (a bare
//! binary has no bundle to register). It shows in System Settings → General → Login Items, where it can be switched
//! off too (then "requires approval": ticking the box again opens that page). Login items added by hand there are a
//! separate list the app can't read: not detected.
//! Linux: `~/.config/autostart/dictation.desktop` (XDG autostart; KDE: System Settings → Autostart), started with the
//! path of the binary that ticked the box; the path is updated at start if the binary moved.
//! Elsewhere: not offered.

/// Whether the app opens at login; None where it can't be set (no app bundle, other systems).
pub fn enabled() -> Option<bool> {
    imp::enabled()
}

/// Turn opening at login on or off. Ok(note): done, `note` (if any) is what the user still has to do.
pub fn set(on: bool) -> Result<Option<String>, String> {
    imp::set(on)
}

/// At start: keep an existing entry pointing at this binary (Linux; nothing elsewhere).
pub fn refresh() {
    #[cfg(target_os = "linux")]
    imp::refresh();
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2::rc::Retained;
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2_foundation::NSError;

    #[link(name = "ServiceManagement", kind = "framework")]
    unsafe extern "C" {}

    // SMAppServiceStatus
    const ENABLED: isize = 1;
    const REQUIRES_APPROVAL: isize = 2;

    /// SMAppService.mainAppService, if this process is an app bundle on macOS 13+.
    fn service() -> Option<Retained<AnyObject>> {
        let exe = std::env::current_exe().ok()?;
        if !exe.to_string_lossy().contains(".app/Contents/MacOS/") {
            return None;
        }
        let class = AnyClass::get(c"SMAppService")?;
        // SAFETY: a class method of SMAppService returning a retained-autoreleased object (or nil).
        unsafe { msg_send![class, mainAppService] }
    }

    fn status(s: &AnyObject) -> isize {
        // SAFETY: `status` is an NSInteger property of SMAppService.
        unsafe { msg_send![s, status] }
    }

    pub fn enabled() -> Option<bool> {
        service().map(|s| status(&s) == ENABLED)
    }

    pub fn set(on: bool) -> Result<Option<String>, String> {
        let s = service().ok_or("only when running as Dictation.app")?;
        // SAFETY: both take an NSError** out parameter and return BOOL (objc2's `_` form turns that into a Result).
        let r: Result<(), Retained<NSError>> =
            if on { unsafe { msg_send![&*s, registerAndReturnError: _] } } else { unsafe { msg_send![&*s, unregisterAndReturnError: _] } };
        let st = status(&s);
        if on && st == REQUIRES_APPROVAL {
            // Switched off in System Settings before: only the user can allow it again there.
            if let Some(class) = AnyClass::get(c"SMAppService") {
                // SAFETY: class method without arguments.
                let _: () = unsafe { msg_send![class, openSystemSettingsLoginItems] };
            }
            return Ok(Some("Allow Dictation under \"Open at Login\" in System Settings → General → Login Items.".into()));
        }
        if (st == ENABLED) == on {
            return Ok(None);
        }
        Err(match r {
            Err(e) => e.localizedDescription().to_string(),
            Ok(()) => format!("status {st} after the change"),
        })
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::path::PathBuf;

    fn file() -> Option<PathBuf> {
        Some(directories::BaseDirs::new()?.config_dir().join("autostart/dictation.desktop"))
    }

    /// Exec= value for this binary: quoted, with the characters the Desktop Entry spec reserves escaped.
    fn exec() -> Result<String, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut q = String::from("\"");
        for c in exe.to_string_lossy().chars() {
            if matches!(c, '"' | '`' | '$' | '\\') {
                q.push('\\');
            }
            q.push(c);
        }
        q.push('"');
        Ok(q)
    }

    fn entry() -> Result<String, String> {
        Ok(format!(
            "[Desktop Entry]\nType=Application\nName=Dictation\nComment=Offline dictation\nExec={}\nIcon=audio-input-microphone\nTerminal=false\nX-GNOME-Autostart-enabled=true\n",
            exec()?
        ))
    }

    pub fn enabled() -> Option<bool> {
        let f = file()?;
        Some(std::fs::read_to_string(f).is_ok_and(|s| !s.lines().any(|l| matches!(l.trim(), "Hidden=true" | "X-GNOME-Autostart-enabled=false"))))
    }

    pub fn set(on: bool) -> Result<Option<String>, String> {
        let f = file().ok_or("no home folder")?;
        if on {
            std::fs::create_dir_all(f.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::write(&f, entry()?).map_err(|e| format!("{}: {e}", f.display()))?;
        } else if f.exists() {
            std::fs::remove_file(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        }
        Ok(None)
    }

    /// Only the Exec line, so whatever else the desktop's settings wrote there stays.
    pub fn refresh() {
        let (Some(f), Ok(exec)) = (file(), exec()) else { return };
        let Ok(s) = std::fs::read_to_string(&f) else { return };
        let want = format!("Exec={exec}");
        if s.lines().any(|l| l.starts_with("Exec=") && l != want) {
            let new: Vec<&str> = s.lines().map(|l| if l.starts_with("Exec=") { want.as_str() } else { l }).collect();
            let _ = std::fs::write(&f, new.join("\n") + "\n");
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    pub fn enabled() -> Option<bool> {
        None
    }

    pub fn set(_: bool) -> Result<Option<String>, String> {
        Err("not available on this system".into())
    }
}
