#!/usr/bin/env python3
"""Parakeet Ultra on the NVIDIA GPU (onnxruntime CUDA), timed on saved takes.

usage: .venv-gpu/bin/python experiments/bench/gpu.py [--model parakeet-ultra-fp16] [--runs 3] [wav ...]
Prints per-take best-of-N time, speed vs real time and WER; appends to experiments/bench/results.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import onnxruntime as ort

ort.preload_dlls()  # load the CUDA/cuDNN libs from the venv's nvidia-* packages
import onnx_asr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RATE, read_wav, score  # noqa: E402


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.strip() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--model", default="parakeet-ultra-fp16", help="folder under models/")
    ap.add_argument("--quantization", default=None, help="onnx-asr file suffix, e.g. int8")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per take (best is reported)")
    args = ap.parse_args()

    wavs = [Path(a) for a in args.wavs] or sorted((ROOT / "recordings").glob("*.wav"))
    audio = {w.name: read_wav(w) for w in wavs}

    before = vram_mib()
    t = time.perf_counter()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", ROOT / "models" / args.model,
                            quantization=args.quantization, sess_options=so,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    m.recognize(next(iter(audio.values()))[: RATE * 5], sample_rate=RATE)  # warm-up: CUDA init, kernels
    warm_s = time.perf_counter() - t
    print(f"{args.model}: loaded in {load_s:.1f}s, first run {warm_s:.1f}s, "
          f"VRAM {before} -> {vram_mib()} MiB")

    log = open(ROOT / "experiments/bench/results.jsonl", "a")
    stamp = datetime.now().isoformat(timespec="seconds")
    name = f"{args.model}/cuda"
    for take, x in audio.items():
        times, text = [], ""
        for _ in range(args.runs):
            t = time.perf_counter()
            text = m.recognize(x, sample_rate=RATE).strip()
            times.append(time.perf_counter() - t)
        best, secs = min(times), len(x) / RATE
        label, nums = score(take, text)
        print(f"\n=== {take}  {secs:.1f}s\n  {name}  {label}{best * 1000:6.0f} ms  "
              f"({secs / best:.0f}x real time)   {text}")
        log.write(json.dumps({"when": stamp, "take": take, "model": name, "text": text,
                              "seconds": round(best, 3), **nums}, ensure_ascii=False) + "\n")
    print(f"\npeak VRAM now {vram_mib()} MiB")


if __name__ == "__main__":
    main()
