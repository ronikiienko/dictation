"""Moonshine (non-streaming, transformers, CPU) on saved English takes, cut into the same pieces as gauge.

Cuts come from .cache/tmp/cuts.json (gauge's Silero VAD + cut_points, computed in .venv because it needs sherpa-onnx).
usage: .venv-granite/bin/python experiments/bench/moonshine.py [model folder, default models/moonshine-tiny]
"""
import json
import sys
import time

import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ROOT, RATE, read_wav, wer, words  # noqa: E402

MODEL = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "models/moonshine-tiny")
REFS = {"20261115-141151.wav": ROOT / "experiments/bench/ref_en_int8.txt",
        "20261115-181554.wav": ROOT / "experiments/bench/ref_en_story.txt",
        "20261115-203034.wav": ROOT / "experiments/bench/ref_en_notes.txt"}
cuts = json.loads((ROOT / ".cache/tmp/cuts.json").read_text())
log = {}
for line in (ROOT / "recordings/log.jsonl").read_text().splitlines():
    e = json.loads(line)
    log[e["wav"]] = e

torch.set_num_threads(6)
proc = AutoProcessor.from_pretrained(MODEL)
model = MoonshineForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()


def run(x):
    inputs = proc(x, sampling_rate=RATE, return_tensors="pt")
    limit = max(8, int(len(x) / RATE * 6.5))  # model card: <= 6.5 tokens/s, avoids hallucination loops
    with torch.inference_mode():
        ids = model.generate(**inputs, max_length=limit)
    return proc.decode(ids[0], skip_special_tokens=True).strip()


run(read_wav(ROOT / "recordings/20261115-201147.wav"))  # warm-up
for take, cs in sorted(cuts.items()):
    x = read_wav(ROOT / "recordings" / take)
    b = [0.0] + cs + [len(x) / RATE]
    t = time.perf_counter()
    text = " ".join(s for s in (run(x[int(p * RATE):int(q * RATE)]) for p, q in zip(b, b[1:])) if s)
    ms = (time.perf_counter() - t) * 1000
    e = log.get(take, {})
    ultra = e["parakeet"] if e.get("final_model") == "ultra-gpu" else ""
    print(f"\n== {take}  {len(x) / RATE:.1f}s  {len(b) - 1} pieces  {MODEL.name} {ms:.0f} ms")
    if take in REFS:
        ref = REFS[take].read_text()
        print(f"   WER {wer(ref, text):.1%}   numbers ignored {wer(ref, text, True):.1%}")
    elif words(ultra):
        print(f"   word diff vs ultra: {wer(ultra, text):.1%}")
    print(f"   {MODEL.name}: {text}\n   ultra:   {ultra}")
