//! Stable linker build ID, so the GPU kernel cache survives rebuilds.
//!
//! CubeCL keys its compiled-kernel cache by the binary's GNU build ID (`buildid` crate), which normally changes on
//! every build, so every rebuild had to recompile all kernels (`dictation sweep`, ~2 min). All our kernels come from
//! burn/CubeCL (no `#[cube]` code in this crate), so they only change when the locked dependencies or the compiler
//! change: the build ID is a hash of `Cargo.lock` + the rustc version.

use std::hash::{Hash, Hasher};

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock = std::fs::read("Cargo.lock").expect("read Cargo.lock");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output().map(|o| o.stdout).unwrap_or_default();
    let mut h = std::hash::DefaultHasher::new();
    (lock, version).hash(&mut h);
    let key = h.finish();
    println!("cargo:rustc-link-arg=-Wl,--build-id=0x{key:016x}");
    // The same key in the code: a GPU's kernel cache is marked "swept" for this key (asr::mark_swept).
    println!("cargo:rustc-env=DICTATION_BUILD_KEY={key:016x}");
}
