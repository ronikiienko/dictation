"""Reference for the burn port: run transcribe.cpp (python bindings, Vulkan) on the first 25 s of a take with
TRANSCRIBE_DUMP_DIR set, so it writes its intermediate tensors (mel, pre-encode, block 0, last block).
Also saves the exact input pcm (f32 raw) and the text. usage: python ref_dump.py <wav> <outdir>"""
import os, sys, wave, json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / ".cache/build/transcribe.cpp"
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
os.environ["TRANSCRIBE_DUMP_DIR"] = str(out)
os.environ.setdefault("TRANSCRIBE_LIBRARY", str(BUILD / "build-vk-so/src/libtranscribe.so"))
sys.path.insert(0, str(BUILD / "bindings/python/src"))
import transcribe_cpp
w = wave.open(sys.argv[1]); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
x = x[:25 * 16000].copy()
x.tofile(out / "pcm.f32")
m = transcribe_cpp.Model(pathlib.Path.home() / ".local/share/dictation/models/parakeet-ultra-Q8_0.gguf")
s = m.session(n_threads=4)
r = s.run(x)
(out / "text.txt").write_text(r.text)
print(len(x) / 16000, "s:", r.text)
