"""Granite Speech 5.0 470M TurboCTC (CPU, transformers) on saved English takes, next to Ultra's text from the log.

usage: .venv-granite/bin/python experiments/bench/granite.py [wav names...]   (default: every English take in the log)
"""
import json
import re
import sys
import time

import torch
from transformers import AutoModelForCTC, AutoProcessor

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ROOT, RATE, read_wav, wer, words  # noqa: E402

MODEL = ROOT / "models/granite-speech-5.0-470m-turboctc"
REFS = {"20261115-141151.wav": ROOT / "experiments/bench/ref_en_int8.txt",
        "20261115-181554.wav": ROOT / "experiments/bench/ref_en_story.txt",
        "20261115-203034.wav": ROOT / "experiments/bench/ref_en_notes.txt"}

log = {}
for line in (ROOT / "recordings/log.jsonl").read_text().splitlines():
    e = json.loads(line)
    log[e["wav"]] = e  # latest entry per wav wins
english = [w for w, e in log.items() if e.get("final_model") == "ultra-gpu" and not re.search("[а-яіїєґ]", e["parakeet"].lower())]
takes = sys.argv[1:] or sorted(english)

torch.set_num_threads(6)
t = time.perf_counter()
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCTC.from_pretrained(MODEL, dtype=torch.float32).eval()
print(f"load {time.perf_counter() - t:.1f}s")
with torch.inference_mode():  # warm-up
    model.generate(**proc([read_wav(ROOT / "recordings" / takes[0])[:RATE]], sampling_rate=RATE))

for take in takes:
    audio = read_wav(ROOT / "recordings" / take)
    t = time.perf_counter()
    with torch.inference_mode():
        inputs = proc([audio], sampling_rate=RATE)
        text = proc.batch_decode(model.generate(**inputs), skip_special_tokens=True)[0].strip()
    ms = (time.perf_counter() - t) * 1000
    e = log.get(take, {})
    ultra = e["parakeet"] if e.get("final_model") == "ultra-gpu" else ""
    print(f"\n== {take}  {len(audio) / RATE:.1f}s  granite {ms:.0f} ms")
    if take in REFS:
        ref = REFS[take].read_text()
        print(f"   WER granite {wer(ref, text):.1%}  "
              f"   numbers ignored {wer(ref, text, True):.1%}")
    else:
        print(f"   word diff granite vs ultra: {wer(ultra, text):.1%}" if words(ultra) else "")
    print(f"   granite: {text}\n   ultra:   {ultra}")
