//! Getting text into the focused window: clipboard via arboard (cross-platform; on Wayland the data-control
//! protocol, on X11/Windows/macOS their native clipboards), paste keys via a virtual keyboard on /dev/uinput
//! (Linux; KDE gives the logged-in user write access). Plus desktop notifications.
use evdev::{AttributeSet, EventType, InputEvent, KeyCode, uinput::VirtualDevice};
use std::time::Duration;

use crate::config::PasteKeys;

pub struct Keyboard(VirtualDevice);

impl Keyboard {
    pub fn new() -> std::io::Result<Keyboard> {
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

/// Short notifications (result, errors), each replacing the previous one. Recording state is shown by the tray icon
/// and the start/stop sounds, not by notifications.
#[derive(Default)]
pub struct Notice {
    pub id: Option<u32>,
    pub enabled: bool,
}

impl Notice {
    pub async fn show(&mut self, title: &str, body: &str, icon: &str) {
        if !self.enabled {
            return;
        }
        use notify_rust::{Hint, Notification, Timeout};
        let mut n = Notification::new();
        n.appname("Dictation").summary(title).body(body).icon(icon).timeout(Timeout::Milliseconds(4000));
        n.hint(Hint::Transient(true));
        if let Some(id) = self.id {
            n.id(id);
        }
        match n.show_async().await {
            Ok(h) => self.id = Some(h.id()),
            Err(e) => eprintln!("notification: {e}"),
        }
    }
}
