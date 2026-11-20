"""transcribe.cpp timing on the same 25 s piece: whole run, and the decoder's own total from its log line,
so encoder (+ mel) = run - decoder. usage: python ref_time.py <ref dir> [runs]"""
import os, sys, time, pathlib, re, subprocess
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = ROOT / ".cache/build/transcribe.cpp"
os.environ.setdefault("TRANSCRIBE_LIBRARY", str(BUILD / "build-vk-so/src/libtranscribe.so"))
sys.path.insert(0, str(BUILD / "bindings/python/src"))
import transcribe_cpp
x = np.fromfile(pathlib.Path(sys.argv[1]) / "pcm.f32", np.float32)
m = transcribe_cpp.Model(pathlib.Path.home() / ".local/share/dictation/models/parakeet-ultra-Q8_0.gguf")
def vram(label):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    mib = sum(int(b) for a, b in (l.split(",") for l in out.splitlines() if l.strip()) if int(a) == os.getpid())
    tot = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout)
    print(f"VRAM {label}: process {mib} MiB, whole GPU {tot} MiB", flush=True)
vram("after load")
s = m.session(n_threads=4)
vram("after session")
s.run(x)
vram("after first run")
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
    t = time.perf_counter(); s.run(x); print(f"RUN {(time.perf_counter() - t) * 1000:.1f}", flush=True)
vram("after runs")
