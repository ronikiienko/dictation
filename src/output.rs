//! Getting text into the focused window: clipboard via arboard (cross-platform; on Wayland the data-control
//! protocol, on X11/Windows/macOS their native clipboards), paste keys via a virtual keyboard: on Linux our own on
//! /dev/uinput (works on X11 and Wayland; KDE gives the logged-in user write access), elsewhere enigo. Plus desktop
//! notifications.
pub use keyboard::{Keyboard, PASTE_HELP};

#[cfg(target_os = "linux")]
mod keyboard {
    use crate::config::PasteKeys;
    use evdev::{AttributeSet, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
    use std::time::Duration;

    pub struct Keyboard(VirtualDevice);

    /// Shown when the paste keys can't be pressed.
    pub const PASTE_HELP: &str = "No write access to /dev/uinput, so the app can't press the paste keys. To allow it \
        once (as root): a udev rule KERNEL==\"uinput\", TAG+=\"uaccess\", then log in again.";

    impl Keyboard {
        /// `_ask`: macOS only (permission prompt).
        pub fn new(_ask: bool) -> std::io::Result<Keyboard> {
            let mut keys = AttributeSet::<KeyCode>::new();
            for k in [KeyCode::KEY_LEFTCTRL, KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_V, KeyCode::KEY_INSERT] {
                keys.insert(k);
            }
            let dev = VirtualDevice::builder()?.name("dictation-keyboard").with_keys(&keys)?.build()?;
            Ok(Keyboard(dev))
        }

        fn key(&mut self, k: KeyCode, down: bool) -> std::io::Result<()> {
            self.0.emit(&[InputEvent::new(EventType::KEY.0, k.code(), down as i32)])
        }

        pub fn paste(&mut self, keys: PasteKeys) -> std::io::Result<()> {
            let combo: &[KeyCode] = match keys {
                PasteKeys::CtrlV => &[KeyCode::KEY_LEFTCTRL, KeyCode::KEY_V],
                PasteKeys::CtrlShiftV => &[KeyCode::KEY_LEFTCTRL, KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_V],
                PasteKeys::ShiftInsert => &[KeyCode::KEY_LEFTSHIFT, KeyCode::KEY_INSERT],
            };
            for &k in combo {
                self.key(k, true)?;
            }
            std::thread::sleep(Duration::from_millis(20));
            for &k in combo.iter().rev() {
                self.key(k, false)?;
            }
            Ok(())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod keyboard {
    use crate::config::PasteKeys;
    use enigo::{Direction, Enigo, Key, Keyboard as _, Settings};
    use std::time::Duration;

    pub struct Keyboard(Enigo);

    /// Shown when the paste keys can't be pressed.
    #[cfg(target_os = "macos")]
    pub const PASTE_HELP: &str = "Allow Dictation (or the terminal it runs in) in System Settings -> Privacy & Security \
        -> Accessibility, so it can press Cmd+V.";
    #[cfg(not(target_os = "macos"))]
    pub const PASTE_HELP: &str = "The system refused the simulated paste keys.";

    impl Keyboard {
        /// `ask`: macOS: show the system's permission prompt if Accessibility isn't allowed yet (once, at start).
        pub fn new(ask: bool) -> std::io::Result<Keyboard> {
            let settings = Settings { open_prompt_to_get_permissions: ask, ..Default::default() };
            Enigo::new(&settings).map(Keyboard).map_err(std::io::Error::other)
        }

        pub fn paste(&mut self, keys: PasteKeys) -> std::io::Result<()> {
            // macOS: Cmd+V pastes everywhere, terminals included (and there is no Insert key).
            #[cfg(target_os = "macos")]
            let combo: &[Key] = {
                let _ = keys;
                &[Key::Meta, Key::Unicode('v')]
            };
            #[cfg(not(target_os = "macos"))]
            let combo: &[Key] = match keys {
                PasteKeys::CtrlV => &[Key::Control, Key::Unicode('v')],
                PasteKeys::CtrlShiftV => &[Key::Control, Key::Shift, Key::Unicode('v')],
                PasteKeys::ShiftInsert => &[Key::Shift, Key::Insert],
            };
            for &k in combo {
                self.0.key(k, Direction::Press).map_err(std::io::Error::other)?;
            }
            std::thread::sleep(Duration::from_millis(20));
            for &k in combo.iter().rev() {
                self.0.key(k, Direction::Release).map_err(std::io::Error::other)?;
            }
            Ok(())
        }
    }
}

/// System clipboard. On Wayland/X11 the text is served by a thread in this process until something else is copied,
/// so it stays pasteable as long as the app runs. Opened lazily and reopened after an error.
#[derive(Default)]
pub struct Clipboard(Option<arboard::Clipboard>);

impl Clipboard {
    pub fn set(&mut self, text: &str) -> Result<(), arboard::Error> {
        let cb = match &mut self.0 {
            Some(cb) => cb,
            None => self.0.insert(arboard::Clipboard::new()?),
        };
        let r = cb.set_text(text);
        if r.is_err() {
            self.0 = None;
        }
        r
    }
}

/// Short notifications (result, errors). Recording state is shown by the tray icon and the start/stop sounds, not by
/// notifications. Linux: each replaces the previous one, 4 s, not kept in history, theme icon. macOS/Windows: plain
/// title + body (those features don't exist there: the OS decides how long, they stack, the icon is the app's).
#[derive(Default)]
pub struct Notice {
    #[cfg(target_os = "linux")]
    pub id: Option<u32>,
    pub enabled: bool,
}

impl Notice {
    /// `icon`: a theme icon name, Linux only.
    pub async fn show(&mut self, title: &str, body: &str, icon: &str) {
        if !self.enabled {
            return;
        }
        use notify_rust::Notification;
        let mut n = Notification::new();
        n.appname("Dictation").summary(title).body(body);
        #[cfg(target_os = "linux")]
        {
            use notify_rust::{Hint, Timeout};
            n.icon(icon).timeout(Timeout::Milliseconds(4000)).hint(Hint::Transient(true));
            if let Some(id) = self.id {
                n.id(id);
            }
            match n.show_async().await {
                Ok(h) => self.id = Some(h.id()),
                Err(e) => eprintln!("notification: {e}"),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = icon;
            // Only a blocking show() there: off the async thread.
            match tokio::task::spawn_blocking(move || n.show().map(|_| ())).await {
                Ok(Err(e)) => eprintln!("notification: {e}"),
                _ => {}
            }
        }
    }
}
