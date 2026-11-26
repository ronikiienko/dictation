#!/bin/sh
# Builds the app (cargo build --release), wraps it into Dictation.app (ad-hoc signed only) and zips it.
# Usage: packaging/macos/make-app.sh [out dir, default: <target dir>/release]
# The cargo environment is yours (CARGO_HOME, CARGO_TARGET_DIR, ...).
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
here="$root/packaging/macos"
cargo build --release --manifest-path "$root/Cargo.toml"
target=${CARGO_TARGET_DIR:-$root/target}
out=${1:-$target/release}
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/Cargo.toml" | head -1)
app="$out/Dictation.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$target/release/dictation" "$app/Contents/MacOS/dictation"
strip -x "$app/Contents/MacOS/dictation" # local symbols only; exported ones stay
sed "s/VERSION/$version/g" "$here/Info.plist" > "$app/Contents/Info.plist"
cp "$here/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
codesign --force --sign - "$app"
(cd "$out" && rm -f Dictation-macos-arm64.zip && ditto -c -k --keepParent Dictation.app Dictation-macos-arm64.zip)
echo "$out/Dictation-macos-arm64.zip"
