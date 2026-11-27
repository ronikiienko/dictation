//! The small window shown while dictating: a dark pill at the bottom middle of the screen with bars following the
//! microphone while recording, then pulsing dots (or a progress line) while the take is transcribed. Drawing is the
//! same everywhere (paint.rs, tiny-skia); only the window differs: macOS a tao window run by the tray's event loop
//! (mac.rs), Linux a Wayland layer-shell surface on its own thread (wayland.rs). Windows: none yet.
//! Tray menu "Recording window" (config `overlay`) turns it off.
include!("paint.rs");

#[cfg(target_os = "macos")]
mod mac;
#[cfg(target_os = "macos")]
pub use mac::Window;

#[cfg(target_os = "linux")]
mod wayland;
#[cfg(target_os = "linux")]
pub use wayland::show;
