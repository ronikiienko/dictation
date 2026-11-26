#!/bin/sh
# Builds the app (cargo build --release) and packs the stripped binary with the license files into
# dictation-linux-x86_64.tar.gz. The binary runs on glibc as old as the one it is built on, so release builds run on
# an old base (CI: ubuntu-22.04, glibc 2.35). The app writes its own .desktop file on first start (daemon.rs).
# Usage: packaging/linux/make-tar.sh [out dir, default: <target dir>/release]
# The cargo environment is yours (CARGO_HOME, CARGO_TARGET_DIR, ...).
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release --manifest-path "$root/Cargo.toml"
target=${CARGO_TARGET_DIR:-$root/target}
out=${1:-$target/release}
name=dictation-linux-x86_64
stage="$out/$name"
rm -rf "$stage"
mkdir -p "$stage"
cp "$target/release/dictation" "$stage/dictation"
strip "$stage/dictation"
cp "$root/LICENSE-MIT" "$root/LICENSE-APACHE" "$stage/"
(cd "$out" && rm -f "$name.tar.gz" && tar -czf "$name.tar.gz" "$name")
rm -rf "$stage"
echo "$out/$name.tar.gz"
