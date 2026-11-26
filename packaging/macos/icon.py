"""Raw RGBA files from `cargo run --release --example appicon -- <dir>` -> AppIcon.icns (via iconutil).
Usage: python3 icon.py <dir> <out.icns>"""
import os, struct, subprocess, sys, zlib

def png(path, size, rgba):
    rows = b"".join(b"\0" + rgba[y * size * 4:(y + 1) * size * 4] for y in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))

src, out = sys.argv[1], sys.argv[2]
iconset = os.path.join(src, "AppIcon.iconset")
os.makedirs(iconset, exist_ok=True)
raw = lambda s: open(os.path.join(src, f"{s}.rgba"), "rb").read()
for s in [16, 32, 128, 256, 512]:
    png(os.path.join(iconset, f"icon_{s}x{s}.png"), s, raw(s))
    png(os.path.join(iconset, f"icon_{s}x{s}@2x.png"), s * 2, raw(s * 2))
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
