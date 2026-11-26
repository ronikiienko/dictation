#!/bin/sh
# Wraps a built binary into Dictation.app (unsigned except ad-hoc) and zips it.
# Usage: packaging/macos/make-app.sh <binary> <out dir>
set -eu
bin=$1 out=$2
here=$(cd "$(dirname "$0")" && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$here/../../Cargo.toml" | head -1)
app="$out/Dictation.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin" "$app/Contents/MacOS/dictation"
sed "s/VERSION/$version/g" "$here/Info.plist" > "$app/Contents/Info.plist"
cp "$here/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
codesign --force --sign - "$app"
(cd "$out" && rm -f Dictation-macos-arm64.zip && ditto -c -k --keepParent Dictation.app Dictation-macos-arm64.zip)
echo "$app"
