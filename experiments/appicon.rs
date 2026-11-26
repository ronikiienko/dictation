//! Writes the macOS app icon's pixels (the tray's microphone on red) as raw RGBA files for packaging/macos/icon.py:
//! `cargo run --release --example appicon -- <dir>` -> <dir>/<size>.rgba for 16..1024.
use dictation::icons::Icon;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: appicon <dir>");
    std::fs::create_dir_all(&dir).unwrap();
    for size in [16, 32, 64, 128, 256, 512, 1024] {
        std::fs::write(format!("{dir}/{size}.rgba"), Icon::Recording.rgba(size)).unwrap();
    }
}
