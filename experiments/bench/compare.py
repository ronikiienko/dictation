#!/usr/bin/env python3
"""Run Parakeet v3 and Parakeet Ultra (int8, CPU) on saved takes and score them.

usage: .venv/bin/python experiments/bench/compare.py [wav ...]     (default: every take in recordings/)
Takes listed in REFS are scored against their reference text; the rest are just printed.
Results are appended to experiments/bench/results.jsonl.
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import onnxruntime as ort
import onnx_asr
import sherpa_onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RATE, REFS, read_wav, score  # noqa: E402

THREADS = 4


def sherpa_model(name):
    d = ROOT / "models" / name
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
        joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
        model_type="nemo_transducer", num_threads=THREADS, provider="cpu")

    def run(x):
        s = rec.create_stream()
        s.accept_waveform(RATE, x)
        rec.decode_stream(s)
        return s.result.text.strip()
    return run


def onnx_asr_model(name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", ROOT / "models" / name,
                            quantization="int8", sess_options=so, providers=["CPUExecutionProvider"])
    return lambda x: m.recognize(x, sample_rate=RATE).strip()


ALL_MODELS = {
    "parakeet-v3": lambda: sherpa_model("parakeet-tdt-0.6b-v3-int8"),
    "parakeet-ultra": lambda: onnx_asr_model("parakeet-ultra-int8"),
}


# MODELS=a,b env var picks a subset
MODELS = {k: v for k, v in ALL_MODELS.items()
          if not os.environ.get("MODELS") or k in os.environ["MODELS"].split(",")}


def main():
    wavs = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "recordings").glob("*.wav"))
    audio = {w.name: read_wav(w) for w in wavs}
    results = {}  # model -> take -> (text, seconds)
    for name, load in MODELS.items():
        t = time.perf_counter()
        run = load()
        print(f"loaded {name} in {time.perf_counter() - t:.1f}s", flush=True)
        results[name] = {}
        for take, x in audio.items():
            t = time.perf_counter()
            results[name][take] = (run(x), time.perf_counter() - t)

    log = open(ROOT / "experiments/bench/results.jsonl", "a")
    stamp = datetime.now().isoformat(timespec="seconds")
    for take, x in audio.items():
        ref = REFS.get(take)
        print(f"\n=== {take}  {len(x) / RATE:.1f}s" + (f"  (ref: {ref.name})" if ref else ""))
        for name in MODELS:
            text, secs = results[name][take]
            label, nums = score(take, text)
            entry = {"when": stamp, "take": take, "model": name, "text": text, "seconds": round(secs, 2), **nums}
            print(f"  {name:20s} {label}{secs * 1000:6.0f} ms   {text}")
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
