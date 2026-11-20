"""transcribe.cpp (ggml, Vulkan build) Parakeet Ultra GGUF on the same gauge pieces as the current pipeline.

Build: .cache/build/transcribe.cpp/build-vk-so (shared libs, Vulkan + CPU). Cuts from .cache/tmp/cuts.json.
usage: .venv/bin/python experiments/bench/tcpp.py [gguf, default Q8_0] [pad]   (pad: pad every piece to 25 s like gauge)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ROOT, RATE, read_wav, wer  # noqa: E402

BUILD = ROOT / ".cache/build/transcribe.cpp"
os.environ.setdefault("TRANSCRIBE_LIBRARY", str(BUILD / "build-vk-so/src/libtranscribe.so"))
sys.path.insert(0, str(BUILD / "bindings/python/src"))
import transcribe_cpp  # noqa: E402

GGUF = ROOT / (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "pad" else "models/parakeet-ultra-gguf/parakeet-ultra-Q8_0.gguf")
PAD = "pad" in sys.argv[1:]
REFS = {"20261115-141151.wav": "experiments/bench/ref_en_int8.txt",
        "20261115-181554.wav": "experiments/bench/ref_en_story.txt",
        "20261115-203034.wav": "experiments/bench/ref_en_notes.txt",
        "20261115-222256.wav": "experiments/bench/ref_uk_mixed.txt"}


def mem():
    s = dict(l.split(":", 1) for l in open("/proc/self/status") if l.startswith(("VmRSS", "RssAnon", "RssFile")))
    v = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    vram = sum(int(b) for a, b in (l.split(",") for l in v.splitlines() if l.strip()) if int(a) == os.getpid())
    return " ".join(f"{k} {int(x.split()[0]) / 1024:.0f}MB" for k, x in s.items()) + f"  VRAM {vram}MB"


t = time.perf_counter()
model = transcribe_cpp.Model(GGUF)
session = model.session(n_threads=int(os.environ.get("THREADS", "0")))
session.run(np.zeros(RATE, dtype=np.float32))  # warm-up, like gauge
print(f"{GGUF.name}  ready {time.perf_counter() - t:.2f}s  pad={PAD}  | {mem()}")

cuts = json.loads((ROOT / ".cache/tmp/cuts.json").read_text())
for line in (ROOT / "recordings/log.jsonl").read_text().splitlines():  # takes not in cuts.json: gauge's logged pieces
    e = json.loads(line)
    if e["wav"] in REFS and e["wav"] not in cuts:
        cuts[e["wav"]] = [round(sum(e["pieces_s"][:i + 1]), 2) for i in range(len(e["pieces_s"]) - 1)]
times = []
for take, ref in REFS.items():
    x = read_wav(ROOT / "recordings" / take)
    b = [0.0] + cuts[take] + [len(x) / RATE]
    out = []
    for p, q in zip(b, b[1:]):
        piece = x[int(p * RATE):int(q * RATE)]
        if PAD:
            piece = np.concatenate([piece, np.zeros(int(25 * RATE) - len(piece), dtype=np.float32)])
        t = time.perf_counter()
        out.append(session.run(piece).text.strip())
        times.append((time.perf_counter() - t) * 1000)
    text = " ".join(s for s in out if s)
    r = (ROOT / ref).read_text()
    print(f"\n== {take} {len(x) / RATE:.0f}s {len(b) - 1} pieces  WER {wer(r, text):.1%}  numbers ignored {wer(r, text, True):.1%}")
    print("  ", text)
print(f"\nper piece ms: {' '.join(f'{v:.0f}' for v in times)}   | {mem()}")
