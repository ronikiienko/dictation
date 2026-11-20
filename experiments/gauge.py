#!/usr/bin/env python3
"""Record a take, then transcribe it with Parakeet (Ultra on the GPU by default).

Live mic:   .venv/bin/python experiments/gauge.py [--final ultra-vk|ultra-gpu|ultra-cpu|v3] [--source <pactl source name>]
            Enter starts a take, Enter stops it, Ctrl+C quits.
            Nothing is transcribed while you speak. Silero VAD follows the audio as it is recorded; after you stop,
            the take is cut at pauses into pieces of at most 25 s and each piece is transcribed.
            Every take is saved to recordings/ (wav + log.jsonl).
Files:      .venv/bin/python experiments/gauge.py recordings/a.wav experiments/samples/en.wav

Cutting rule (see NOTES.md "Pause-aware cutting"): a piece ends at the latest pause >= 0.5 s that lies 10-25 s
into the piece; if there is none, at the longest pause there; if there is no pause at all, hard at 25 s.
--final ultra-vk (default): Parakeet Ultra Q8_0 GGUF through transcribe.cpp (ggml, Vulkan on the NVIDIA GPU, decoder
on the CPU); ~0.3 GB RAM, ~0.73 GB VRAM, ready in ~1 s; runs from .venv or .venv-gpu (no CUDA needed). Build:
.cache/build/transcribe.cpp/build-vk-so (see NOTES.md "transcribe.cpp").
--final ultra-gpu: Ultra fp16 via onnxruntime CUDA + fast_tdt, needs .venv-gpu; every piece is padded to 25 s (masked)
so the encoder always sees one input shape (warmed up at start). ultra-cpu and v3 run from .venv.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent  # project root (experiments/ lives in it)
PARAKEET = ROOT / "models/parakeet-tdt-0.6b-v3-int8"
ULTRA_FP16 = ROOT / "models/parakeet-ultra-fp16"
ULTRA_INT8 = ROOT / "models/parakeet-ultra-int8"
ULTRA_GGUF = ROOT / "models/parakeet-ultra-gguf/parakeet-ultra-Q8_0.gguf"
TRANSCRIBE_CPP = ROOT / ".cache/build/transcribe.cpp"
SILERO = ROOT / "models/silero-vad/silero_vad.onnx"
RECORDINGS = ROOT / "recordings"
RATE = 16000
CHUNK = RATE // 10  # 100 ms of audio per read

PIECE_S = 25.0      # longest piece; also the padded encoder length on the GPU
MIN_PIECE_S = 10.0  # a cut is never placed earlier than this into a piece
REAL_PAUSE_S = 0.5  # pauses at least this long are preferred cut points

DIM, BOLD, CYAN, GREEN, RESET = "\x1b[2m", "\x1b[1m", "\x1b[36m", "\x1b[32m", "\x1b[0m"


def load_final(kind, threads):
    """Returns a function: samples (<= PIECE_S) -> text."""
    if kind == "v3":
        d = PARAKEET
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            model_type="nemo_transducer", num_threads=threads, provider="cpu")

        def run(x):
            s = rec.create_stream()
            s.accept_waveform(RATE, x)
            rec.decode_stream(s)
            return s.result.text.strip()
        return run

    if kind == "ultra-vk":
        import os
        os.environ.setdefault("TRANSCRIBE_LIBRARY", str(TRANSCRIBE_CPP / "build-vk-so/src/libtranscribe.so"))
        sys.path.insert(0, str(TRANSCRIBE_CPP / "bindings/python/src"))
        # Show only warnings and errors: ggml (it lists the Vulkan devices while the library loads, so set before the
        # import; levels 3 warn, 4 error) and transcribe.cpp (logs at debug level by default; levels 2 warn, 3 error).
        import ctypes
        global _GGML_LOG  # keep the callback alive
        _GGML_LOG = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(
            lambda lvl, msg, _: lvl in (3, 4) and sys.stderr.write(msg.decode(errors="replace")))
        ctypes.CDLL(str(TRANSCRIBE_CPP / "build-vk-so/ggml/src/libggml-base.so")).ggml_log_set(_GGML_LOG, None)
        import transcribe_cpp
        transcribe_cpp.set_log_callback(lambda lvl, msg: lvl in (2, 3) and sys.stderr.write(msg.rstrip() + "\n"))
        session = transcribe_cpp.Model(ULTRA_GGUF).session(n_threads=threads)  # decoder threads: 4 measured best
        session.run(np.zeros(RATE, dtype=np.float32))  # warm-up (Vulkan pipelines)
        return lambda x: session.run(x).text.strip()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3  # hide the harmless "some nodes on CPU" warning
    if kind == "ultra-gpu":
        ort.preload_dlls()  # CUDA/cuDNN from the venv's nvidia-* packages
        # kSameAsRequested: grow the GPU memory pool by what is asked, not in doubling steps (default reserved ~2 GB
        # to load the 1.2 GB encoder). Same speed; fine now that every piece has the same padded shape.
        cuda = {"arena_extend_strategy": "kSameAsRequested"}
        args = dict(path=ULTRA_FP16, providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])
    else:  # ultra-cpu
        so.intra_op_num_threads = threads
        args = dict(path=ULTRA_INT8, quantization="int8", providers=["CPUExecutionProvider"])
    import onnx_asr
    m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", sess_options=so, **args)
    if kind == "ultra-gpu":
        import fast_tdt
        fast_tdt.install(m, ULTRA_FP16 / "decoder_joint-model.onnx", half=True)  # faster greedy decoding
        fast_tdt.pad_encoder(m, PIECE_S, RATE)  # one encoder input shape -> no first-run cost per new length
    m.recognize(np.zeros(RATE, dtype=np.float32), sample_rate=RATE)  # warm-up (CUDA init, the padded shape)
    # Loading leaves ~630 MB of freed memory that glibc keeps; hand it back to the OS (RAM 1.84 -> 1.18 GB own).
    import ctypes
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return lambda x: m.recognize(x, sample_rate=RATE).strip()


class Vad:
    """Silero VAD fed incrementally; .gaps() returns the pauses (start, end) in seconds."""

    def __init__(self):
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = str(SILERO)
        c.silero_vad.min_silence_duration = 0.1
        c.silero_vad.min_speech_duration = 0.1
        c.silero_vad.max_speech_duration = 1e4
        c.sample_rate = RATE
        # The buffer only has to hold one speech run until _collect() pops it (longest seen: 12.4 s without a 0.1 s
        # pause); it is preallocated, so 1200 s cost 77 MB of RAM, 60 s costs 4 MB (same cuts on all 39 recordings).
        self.vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)
        self.win = c.silero_vad.window_size
        self.buf = np.zeros(0, dtype=np.float32)
        self.segs = []

    def accept(self, x):
        self.buf = np.concatenate([self.buf, x])
        n = len(self.buf) // self.win * self.win
        for i in range(0, n, self.win):
            self.vad.accept_waveform(self.buf[i:i + self.win])
        self.buf = self.buf[n:]
        self._collect()

    def _collect(self):
        while not self.vad.empty():
            f = self.vad.front
            self.segs.append((f.start / RATE, (f.start + len(f.samples)) / RATE))
            self.vad.pop()

    def gaps(self):
        if len(self.buf):
            self.vad.accept_waveform(np.pad(self.buf, (0, self.win - len(self.buf))))
            self.buf = np.zeros(0, dtype=np.float32)
        self.vad.flush()
        self._collect()
        return [(self.segs[i][1], self.segs[i + 1][0]) for i in range(len(self.segs) - 1)]


def cut_points(duration, gaps):
    """Cut times (s) for a take: see the module docstring."""
    cuts, s = [], 0.0
    while duration - s > PIECE_S:
        window = [g for g in gaps if s + MIN_PIECE_S <= (g[0] + g[1]) / 2 <= s + PIECE_S]
        real = [g for g in window if g[1] - g[0] >= REAL_PAUSE_S]
        if real:
            g = real[-1]
        elif window:
            g = max(window, key=lambda g: g[1] - g[0])
        else:
            g = None
        c = (g[0] + g[1]) / 2 if g else s + PIECE_S
        cuts.append(c)
        s = c
    return cuts


def transcribe(final, x, gaps):
    """-> (text, piece lengths in s)."""
    bounds = [0.0] + cut_points(len(x) / RATE, gaps) + [len(x) / RATE]
    pieces = [x[int(a * RATE):int(b * RATE)] for a, b in zip(bounds, bounds[1:])]
    return " ".join(t for t in (final(p) for p in pieces) if t), [len(p) / RATE for p in pieces]


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, f"{path}: need 16-bit PCM"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    if rate != RATE:  # linear resample is fine for a quality check
        x = np.interp(np.arange(0, len(x), rate / RATE), np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


FINAL = "parakeet"  # label for the final transcript, set in main()


def report(audio_s, text, ms, pieces):
    sizes = " + ".join(f"{p:.1f}" for p in pieces)
    print(f"  {GREEN}{FINAL}{RESET} {DIM}({ms:.0f} ms for {audio_s:.1f}s audio, "
          f"{len(pieces)} piece{'s' if len(pieces) > 1 else ''}: {sizes} s){RESET}  {text}")


def run_files(paths, final):
    for p in paths:
        x = read_wav(p)
        print(f"{BOLD}{p}{RESET}  {DIM}{len(x) / RATE:.1f}s{RESET}")
        t = time.perf_counter()
        vad = Vad()
        vad.accept(x)
        gaps = vad.gaps()
        t_vad = time.perf_counter() - t
        t = time.perf_counter()
        text, pieces = transcribe(final, x, gaps)
        report(len(x) / RATE, text, (time.perf_counter() - t) * 1000, pieces)
        print(f"  {DIM}(VAD over the whole file: {t_vad * 1000:.0f} ms; live it runs while recording){RESET}\n")


class LiveLine:
    """Redraws a block of text in place, even when it wraps over several lines."""

    def __init__(self):
        self.rows = 0

    def show(self, text):
        cols = shutil.get_terminal_size().columns
        out = ""
        if self.rows > 1:
            out += f"\x1b[{self.rows - 1}A"
        out += "\r\x1b[J" + text
        self.rows = max(1, math.ceil(len(text) / cols)) if text else 1
        sys.stdout.write(out)
        sys.stdout.flush()


def run_live(final, source):
    RECORDINGS.mkdir(exist_ok=True)
    enter = threading.Event()

    def wait_enter():
        for _ in sys.stdin:
            enter.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    mic = source or subprocess.run(["pactl", "get-default-source"], capture_output=True,
                                   text=True).stdout.strip()
    print(f"mic: {BOLD}{mic}{RESET}   (Enter = start/stop, Ctrl+C = quit)\n")

    while True:
        print(f"{DIM}press Enter to record…{RESET}")
        enter.wait()
        enter.clear()
        sys.stdout.write("\x1b[1A\r\x1b[J")  # drop the prompt + the echoed newline
        rec = subprocess.Popen(
            ["parecord", "--raw", "--format=s16le", f"--rate={RATE}", "--channels=1",
             "--latency-msec=50"] + ([f"--device={source}"] if source else []),
            stdout=subprocess.PIPE)
        vad = Vad()
        chunks, line = [], LiveLine()
        line.show(f"{CYAN}● recording{RESET}")
        while not enter.is_set():
            data = rec.stdout.read(CHUNK * 2)
            if not data:
                break
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
            chunks.append(x)
            vad.accept(x)
            line.show(f"{CYAN}● recording{RESET} {DIM}{len(chunks) * CHUNK / RATE:.1f}s{RESET}")
        enter.clear()
        rec.terminate()
        rec.wait()
        line.show("")
        sys.stdout.write("\r\x1b[J")

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if len(audio) < RATE // 4:
            print(f"{DIM}(too short, skipped){RESET}\n")
            continue
        t = time.perf_counter()
        text, pieces = transcribe(final, audio, vad.gaps())
        ms = (time.perf_counter() - t) * 1000

        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        write_wav(RECORDINGS / f"{name}.wav", audio)
        with open(RECORDINGS / "log.jsonl", "a") as f:
            f.write(json.dumps({"wav": f"{name}.wav", "seconds": round(len(audio) / RATE, 2),
                                "parakeet": text, "final_model": FINAL, "parakeet_ms": round(ms),
                                "pieces_s": [round(p, 2) for p in pieces]}, ensure_ascii=False) + "\n")
        print(f"{BOLD}{name}{RESET}")
        report(len(audio) / RATE, text, ms, pieces)
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="wav files to transcribe instead of using the mic")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (ultra-vk decoder, ultra-cpu, v3)")
    ap.add_argument("--final", choices=["ultra-vk", "ultra-gpu", "ultra-cpu", "v3"], default="ultra-vk",
                    help="model for the transcript (default ultra-vk: Parakeet Ultra Q8_0 via transcribe.cpp/Vulkan)")
    ap.add_argument("--source", help="mic to record from (see: pactl list short sources); default mic if omitted")
    args = ap.parse_args()
    global FINAL
    FINAL = args.final
    t = time.perf_counter()
    final = load_final(args.final, args.threads)
    print(f"{DIM}model loaded in {time.perf_counter() - t:.1f}s  ({args.final}){RESET}")
    try:
        if args.files:
            run_files(args.files, final)
        else:
            run_live(final, args.source)
    except KeyboardInterrupt:
        print(RESET)


if __name__ == "__main__":
    main()
