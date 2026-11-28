# Dictation project — notes and findings

Goal: local, offline dictation for KDE Plasma (Wayland) on this laptop, Ukrainian + English
(often Ukrainian with English tech terms mixed in).

**Current state (2026-11-20):** the project root is the app, a pure Rust tray app with its own Parakeet Ultra
port on burn/CubeCL (Vulkan), see "Rust tray app" and "burn pipeline". In daily use.

**Cleanup 2026-11-20 (user: "all delete"):** everything not used by the app was deleted: the Python/ONNX research
(gauge.py, fast_tdt.py, dictate.py, bench scripts, `.venv*`, pip/hf caches), the transcribe.cpp build
(`.cache/build`), the `experiments/burn` crate (now `src/asr`), the root `models/` (all ONNX/GGUF test models,
Granite, Moonshine, Silero), `.cache/tmp`. Kept: `experiments/check.rs`, `experiments/bench/wer.py` + `ref_*.txt`,
`experiments/samples/`. The sections below keep the findings; files and paths they mention before this date are gone.

## Machine

| | |
|---|---|
| OS | CachyOS (Arch), kernel 7.3, KDE Plasma on Wayland, zsh in Claude's shell (user's login shell is fish) |
| CPU | Ryzen 5 5600H, 6c/12t, AVX2, **no AVX-512/VNNI**, no native fp16; L2 3 MB, **L3 16 MB** |
| RAM | 14 GB |
| GPU | RTX 3050 Ti Mobile, **4 GB** GDDR6 (~190 GB/s), driver 620.70 (CUDA ≤ 13.5) + AMD Vega iGPU |
| Displays | external 4K on HDMI-A-1 is wired to the **NVIDIA** GPU (so it stays awake, P5, ~7.6 W, ~154 MiB used by KWin); laptop panel eDP-1 on AMD. AMD is boot_vga, so KWin most likely composites on AMD and hands frames to NVIDIA for HDMI (not verified directly) |
| Mic | default source is a Bluetooth headset `bluez_input.XX:XX:XX:XX:XX:XX` (HFP, low quality). Internal mics: `alsa_input.pci-0000_07_00.6.HiFi__Mic1__source` / `…Mic2__source`. Recordings are clean: speech −20…−40 dB, silence ~−75…−80 dB |
| Audio capture | `parecord --raw --format=s16le --rate=16000 --channels=1` works (needs >~1 s before output arrives on very short runs) |

## Project rules

- **Everything stays inside this folder**: venvs, pip cache (`PIP_CACHE_DIR=$PWD/.cache/pip`), models, temp files
  (`.cache/tmp/`). No system installs. System CUDA was installed briefly and **removed** (`pacman -Rns cudnn cuda`,
  verified clean); CUDA comes from pip `nvidia-*` packages inside `.venv-gpu`.
- **Cap memory on anything heavy** (long audio, new models, model conversion):
  `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 <cmd>`. Test sizes in increasing order, stop at
  the first failure, never rerun a failed heavy command without finding out why (check
  `journalctl --user --since "-5 min" | grep -i oom`). A 5–10 min CPU test without a cap froze the laptop on 2026-11-15.
- **Only one GPU process at a time**: if the user has `gauge.py` open it holds the GPU memory; a second model load
  fails with `BFCArena … Failed to allocate`. Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  first and ask the user to quit gauge. Also: two model loads in one Python process don't free the first one's VRAM —
  run variants in separate processes.
- In zsh, `$var` with spaces is **not** word-split (`for`, `set -- $x`) — pass args explicitly or use a function.
- Quote globs in grep args (`--include='*.py'`) — zsh errors on unmatched globs.

## Layout (until 2026-11-20; deleted)

```
gauge.py            test harness: record (mic) or files → Silero VAD cuts → Parakeet per piece — see "How to run"
fast_tdt.py         GPU speed-ups for onnx-asr Parakeet TDT: install() = fast greedy decoder, pad_encoder() = fixed shape
bench/common.py     shared: read_wav, RATE, REFS (take -> reference text), wer(), score()  (numpy only)
bench/compare.py    CPU comparison: parakeet-v3 (sherpa) vs parakeet-ultra (int8, onnx-asr); MODELS=a,b env picks subset
bench/gpu.py        Ultra fp16 on CUDA, best-of-N timing + WER (calls the model directly, no cutting, old pipeline)
bench/granite.py    Granite Speech 5.0 470M on CPU (.venv-granite) vs Ultra text from log
bench/moonshine.py  Moonshine on CPU (.venv-granite), gauge VAD pieces from .cache/tmp/cuts.json
bench/pairs.py      recorded speech vs a Qwen TTS voice on the same 40 minimal-pair sentences, any of 3 models
bench/wer.py        older one-off scorer for recordings/log.jsonl (superseded by common.py)
bench/ref_en_int8.txt, ref_uk_int8.txt   technical texts read for takes 141151 (en) and 141800 (uk)
bench/ref_en_story.txt                   short story (written by Claude) read for take 181554 (en)
bench/ref_en_notes.txt                   weekend notes, read for take 203034 (en)
bench/ref_uk_mixed.txt                   uk with English terms (Python, GitHub, config.toml…) — NOT recorded yet
bench/ref_en_vowels_a.txt                8 sentences with cat/cot/cut/bet vowel traps — NOT recorded yet
bench/vad_pauses.png                     Silero speech/pauses + planned cuts for takes 141151 / 141800
recordings/         every take from gauge.py: <timestamp>.wav (16 kHz mono) + log.jsonl (texts, timings, pieces_s)
samples/            uk/en/de test clips (from the sherpa-onnx Nemotron repo)
models/             experiments' models, see below (the app downloads its own copy to ~/.local/share/dictation/models)
.venv               CPU: sherpa-onnx 2.0.1, onnx-asr 0.12.0, onnxruntime 1.32.0, numpy, matplotlib
.venv-granite       CPU: torch 2.15 + torchaudio + transformers 5.23 (Granite, Moonshine tests), 1.2 GB
.venv-gpu           GPU: onnxruntime-gpu 1.32.0 [cuda,cudnn] (CUDA 13 pip libs, cuDNN 9.28), onnx-asr, sherpa-onnx, onnx, sympy
.cache/tmp/         one-off measurement scripts (list in "Scripts" below)
```

Don't install plain `onnxruntime` and `onnxruntime-gpu` in the same venv: they share the `onnxruntime/` package
folder (onnx-asr pulls in the CPU one; it was uninstalled from `.venv-gpu` and onnxruntime-gpu force-reinstalled).

## Models (experiments; all in models/; deleted 2026-11-20)

| folder | what | source | size | runtime |
|---|---|---|---|---|
| `parakeet-ultra-fp16` | **main model on GPU**: Parakeet Ultra, encoder fp16 + decoder fp32 file (fast_tdt uses its weights as fp16) | `Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx` (`fp16/encoder-model.fp16.onnx` saved as `encoder-model.onnx`, `fp32/decoder_joint-model.onnx(+.data)`, vocab/config/nemo128) | 1.3 GB | onnx-asr, `quantization=None`, CUDA EP |
| `parakeet-ultra-int8` | Parakeet Ultra int8 (MatMulNBits 8-bit, block 64, weight-only; activations fp32) | same repo, `int8/` | 637 MB | onnx-asr, `quantization="int8"` (CPU) |
| `parakeet-tdt-0.6b-v3-int8` | original Parakeet v3 (reference only; deletion candidate) | `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | 640 MB | sherpa-onnx `model_type="nemo_transducer"` |
| `granite-speech-5.0-470m-turboctc` | IBM Granite Speech 5.0 470M (multilingual Oct 2026 rev, no uk); test only | ibm-granite HF | 905 MB | transformers CPU |
| `moonshine-tiny` | Moonshine tiny English 27M; test only | moonshine-ai HF | 106 MB | transformers CPU |
| `silero-vad` | Silero VAD (pause detection for cutting) | k2-fsa sherpa-onnx releases `asr-models/silero_vad.onnx` | 644 KB | sherpa-onnx VoiceActivityDetector |
| `tokenizer-v3` | `bpe.vocab` + `tokens.txt` (+ NOTICE) kept from Orukeet: same tokenizer as v3/Ultra, for sherpa-onnx hotwords | `oruk/orukeet` | 216 KB | — |

Deleted 2026-11-15 (user OK): `nemotron-3.5-streaming-560ms-int8` (Nemotron removed from gauge), `orukeet-int8`,
`parakeet-v3-onnxasr-int8`. onnx-asr model name for all Parakeet variants: `"nemo-parakeet-tdt-0.6b-v3"` with
`path=models/<folder>`. GPU: call `onnxruntime.preload_dlls()` before creating sessions.

## How to run gauge.py (deleted 2026-11-20)

```sh
.venv/bin/python gauge.py                          # live: Enter start/stop, Ctrl+C quit (ultra-vk: ~0.73 GB VRAM while open; --final ultra-gpu needs .venv-gpu, ~1.75 GB)
.venv-gpu/bin/python gauge.py recordings/x.wav    # files instead of mic
.venv-gpu/bin/python gauge.py --final ultra-cpu    # Ultra int8 on CPU (e.g. while gaming) — not re-tested since rewrite
.venv-gpu/bin/python gauge.py --final v3           # old Parakeet v3 — not re-tested since rewrite
.venv/bin/python bench/compare.py [wavs]           # CPU comparison on saved takes
```

**gauge.py now** (rewritten 2026-11-15 evening; user: "no workarounds, just let it be; remove Nemotron"):
- While recording: only a timer. Silero VAD is fed each 100 ms chunk (offline over a 126 s file it would cost ~540 ms
  after stop, so it runs during recording; per chunk ~0.4 ms).
- After stop: cut at pauses (rule below), each piece transcribed in sequence, texts joined. Log keys: wav, seconds,
  parakeet, final_model, parakeet_ms, pieces_s.
- ultra-gpu load: onnx-asr Ultra fp16 with CUDA option `arena_extend_strategy=kSameAsRequested`, then
  `fast_tdt.install(m, decoder_path, half=True)` and `fast_tdt.pad_encoder(m, 25.0)`, warm-up on 1 s of silence.
- No GPU keep-alive, no live text, no transcription during recording (all deliberately left out for now).

## Model selection — what was decided and why

Candidates researched (Nov 2026): Nemotron 3.5 Transcribe (30B MoE, too big), Granite Speech 5.0, Cohere Transcribe,
Qwen3-ASR (no Ukrainian in open models), Canary-1B-v2 (uk 10.6% FLEURS, worse), Whisper large-v3 (uk 6.3% but 1×
real time on CPU), Moonshine uk, w2v-bert-uk (uk only), Omnilingual ASR, TranscribePsy (too new).
Best independent source: Handy's per-language leaderboard https://models.handy.computer/languages/uk
(Ukrainian FLEURS: parakeet-ultra 5.71, whisper-large-v3 6.28, orukeet 6.53, parakeet-v3 6.84, nemotron-3.5-streaming 14.88).

**Decision: Parakeet Ultra** (moondream post-train of Parakeet TDT 0.6B v3, Sep 2026, CC-BY-4.0, 25 European
languages incl. uk, same architecture/tokenizer as v3).

Recorded takes, whole-take WER vs reference (early comparison; "numbers ignored" drops digit/number-word tokens):

| take | v3 (sherpa) | v3 (onnx-asr) | **Ultra int8 CPU** | Ultra fp16 GPU | Orukeet | Nemotron |
|---|---|---|---|---|---|---|
| 141151 en, 126 s | 31.2% | 40.1% | **6.2%** (4.3%) | 6.2% | 42.2% | ~ (no punctuation) |
| 141800 uk, 125.7 s | 33.5% | 36.4% | **16.2%** (11.7%) | 17.9% | 28.9% | 38.7% auto / 35.8% uk |

**Current pipeline (gauge, VAD pieces, GPU): story 181554 2.2%, en 141151 6.2%, uk 141800 15.6%.**

Findings:
- **Parakeet v3 silently skips chunks of long takes** (~20 s jumped in the 126 s en take) in *both* sherpa-onnx and
  onnx-asr → model weakness. Ultra transcribed all takes completely.
- Nemotron streaming (removed): dropped speech at t=0 (needed 0.5 s lead-in silence), leaked Russian on `auto`,
  its token timestamps lag the audio 0.3–0.7 s (unusable for cut points).
- Errors concentrate on English tech terms inside Ukrainian (int8 → "N8"/"intAid"/"and8", sherpa-onnx → "Sharpa").
  No model is trained for uk–en code-switching (searched). Dictated punctuation words come out as words ("comma").
- Hotwords: sherpa-onnx 2.0.1 supports them for NeMo transducers (`decoding_method="modified_beam_search"`,
  `hotwords_file`, `hotwords_score`), but words must be tokenized — plain words fail with "Cannot find ID for token".
  Fix: `models/tokenizer-v3/bpe.vocab` (`modeling_unit="bpe"`, `bpe_vocab=…`) or pre-tokenize. Not tested. onnx-asr
  (used for Ultra) has no hotwords → would need Ultra in sherpa-onnx format (convert) or replacement rules.

## Performance

Current gauge pipeline, ultra-gpu, GPU busy (best of 3): 22 s → 86 ms, 90 s → 412 ms (5 pieces), 126 s → 534–554 ms
(6–7 pieces). One 25 s-padded piece ≈ 80–95 ms (encoder ~50 ms + decoder). CPU int8 for comparison: 126 s ≈ 13 s.

Original (pre-optimization) numbers, whole take on GPU: 126 s 740 ms, 60 s 320 ms, 5–18 s 34–90 ms.
- Encoder ≈ 609 M params (1161 MiB fp16, 24 layers × 1024), decoder ≈ 18 M (72 MB fp32): 97% of params in encoder.
  Encoder outputs one vector per 80 ms (126 s → ~1580); TDT decoder took 502 steps, 417 of them emit a token.
- fp16 model: Cast fp32→fp16 at input, everything fp16, Cast back at output (activations fp16).
- int8 Ultra on GPU: 2× slower than fp16 (324 vs 161 ms for 30 s), fp32 activations.
- Encoder (fp16, CUDA) vs length: 15 s 34 ms, 30 s 62, 60 s 140, 90 s 236, 126 s 364 → ≈ 1.81 ms/audio-s linear
  + 0.0086 ms·s⁻² quadratic (attention). Attention ≈ 12% at 30 s, 37% at 126 s. Linear part ≈ 8 TFLOP/s, near what
  this GPU does in fp16 → no big encoder win left except shorter pieces (done) or maybe TensorRT.

### fast_tdt.py — decoder optimization (126 s en take, decoder part only)

| decoder variant | decoder ms | text vs base |
|---|---|---|
| onnx-asr original (502 calls of decoder_joint on CUDA) | 390 | — |
| same on CPU fp32 / CPU int8 1 thr / int8 4 thr | 1233 / 341 / 148 | int8 changes text (en WER 6.2→6.8%) |
| IO binding / + CUDA graph on original graph | 322 / 357 | same |
| **fast_tdt** step graph, window 8, cuDNN LSTM | 294 | same |
| + LSTM as plain ops (cuDNN LSTM has ~0.25 ms fixed cost/call) | 253 | same |
| + CUDA graph, 2 IO bindings swapping LSTM state on GPU | 205 | same |
| **+ fp16 weights in the 3 matmuls (`half=True`, used)** | **~160–210** | same text en + uk |

How it works: encoder-side joint projection for all frames once (numpy, ~20 ms/126 s); per call one ONNX "step"
graph built from the decoder file's weights: LSTM on last token + joint over next K=8 frames + ArgMax token/duration
(only [K,2] int64 copied back). The CPU walks the window with onnx-asr's exact rules. Per call ~0.38 ms is memory
bandwidth (fp32 ≈ 50 MB weights/call), hence fp16 helped. fp32 decoder on CPU is slow because 21 MB output matrix >
16 MB L3. After install, onnx-asr's own decoder_joint session is replaced by `_LazyCpuSession` (CPU, created only if
`need_logprobs` is ever used). `pad_encoder(m, 25.0)`: pads features to 2501 frames (25 s) and passes the real
length (masked); inputs longer than 25 s pass through unpadded.

Tried, no gain: `cudnn_conv_algo_search` EXHAUSTIVE/HEURISTIC same, DEFAULT 5× slower; ORT transformer optimizer
`model_type="conformer"` — full `optimize_model` OOMs at 6 GB cap (symbolic shape inference copies the 1.2 GB model;
needed `sympy`), `fuse_attention()` alone matches nothing (no q/k/v biases, custom Gather rel-shift). TensorRT not
tried (large pip download).

## Pause-aware cutting (Silero VAD) — in gauge.py

- Silero via sherpa-onnx VoiceActivityDetector: threshold 0.5, min_silence 0.1 s, min_speech 0.1 s.
- **Rule (user agreed)**: piece limit 25 s; cut in the middle of the **latest pause ≥ 0.5 s** whose middle lies
  10–25 s into the piece; if none, the longest pause there; if no pause, hard cut at 25 s. (No leftover merge: it can
  never fit in 25 s.) Every piece padded to 25 s on the GPU (masked).
- Why 0.5 s: sentence-end pauses in these takes are mostly ≥ 0.6 s (11/12 and 12/13 ≥ 0.5), mid-phrase hesitations
  0.3–0.4 s. In 126 s: 45 pauses, median 0.32 s, 15–17 pauses ≥ 0.5 s.
- Pieces on the takes: story 19.5/20.2/24.5/20.3/5.7 s; en 15.0/22.2/19.5/22.6/21.4/11.8/13.8; uk 20.3/22.3/23.7/
  19.6/18.9/20.9 — always the main rule, fallback never needed.
- WER per cutting rule (en / uk, 126 s takes; 1 take each, ±1–3 points noise): whole 6.2/17.9; strict 30 s
  7.3/15.0; strict 20 s 7.8/19.7; strict 10 s 23.4/26.0; VAD 30 s latest(any pause) 5.7/17.3; VAD 30 s longest
  5.7/14.5; VAD 20 s latest 7.8/16.8; VAD 20 s longest 5.2/17.9. → hard cuts damage seams; pause cuts ≈ whole or
  better; < ~15 s strict pieces lack context. "Latest any pause" picks 0.1–0.2 s gaps inside phrases → threshold.
- Masked padding check: 17/18 pieces word-identical to unpadded; 1 differs by one word ("a lantern" → "the lantern").
- Plain loudness would also find pauses here, but Silero ignores breaths (−60 dB bumps) and is robust to noise.

## Latency in gauge (why live numbers were worse than benchmarks)

- **First run at a new input length: +70–90 ms** (per-shape setup in the CUDA EP, not cudnn algo search). Fixed by
  padding to one 25 s shape + warm-up. Padded pieces: ~80–95 ms vs unpadded first runs ~130–140 ms.
- **GPU idle**: after a burst, SM clock falls from ~2 s, P3 at ~4 s, **P5 (memory 810 MHz) at ~5.7 s**. 21 s piece:
  idle ≤ 1 s → no penalty (~80 ms), 2 s ~100 ms, 4–8 s ~105–130 ms, **15 s → ~315 ms**.
- Keep-alive tests (20 s piece after 15 s "recording"): none 246–307 ms; tiny CUDA kernel every 0.2/0.5/1 s →
  134–299 ms (does NOT reliably help); medium matmul tick inconsistent; **re-transcribing the current piece every 1 s
  (~80 ms, ~8% GPU) → 85–97 ms** (every 2 s: 123–254 ms). That approach would also be a live Parakeet preview.
  User chose to leave it out for now.

## GPU memory (VRAM) and RAM

**Current gauge ultra-gpu: ~1.75 GB VRAM, flat for any take length** (was 2.6 GB before kSameAsRequested):

| stage | VRAM |
|---|---|
| sessions created (weights + CUDA context) | 1466 MiB |
| fast_tdt installed, old decoder session released | 1442 MiB (only −24: it never ran, so it held only weights) |
| warm-up (1 s silence → 25 s padded): encoder/preprocessor working memory | 1618 MiB (+176) |
| first real speech piece: decoder step buffers + CUDA graph capture | 1738 MiB (+120) |
| after 200 random pieces of 1–25 s (stress test) | **1738 MiB, flat** |

- The arena never returns memory while the process runs, but with one padded shape it doesn't grow. Calling the
  model directly with > 25 s audio (old bench scripts) grows it and keeps it.
- Default arena (kNextPowerOfTwo) reserved ~2 GB to load the 1.16 GB encoder → 2.6–2.8 GB total.
- Old measurements (whole takes, kSameAsRequested): working memory 5 s 112 MiB, 30 s 258, 60 s 426, 126 s 802; int8
  model 30 s 348, 60 s 662, weights 824. Single takes ≥ 5 min OOM on the GPU; on CPU they exhaust 14 GB RAM.
- **RAM ~2.4–2.5 GB** (peak 2.56): onnxruntime ~50 MB, CUDA libs +230 MB, CUDA context/cuBLAS +475 MB (≈ 750 MB any
  CUDA program), loading Ultra +1.46 GB, first run +~120 MB. ~620 MB of that is freed-but-unreturned malloc memory
  from loading the weights: `ctypes.CDLL("libc.so.6").malloc_trim(0)` after load → ~1.6 GB. **Applied in gauge
  (2026-11-15)**: gauge-like run after 6 takes 1.84 GB total (own 1.30 + files 0.44), was 2.54 (own 1.99).
- Breakdown (`.cache/tmp/ram_steps.py`, own = RssAnon): bare CUDA process with a tiny model after one run 472 MB
  (`ram_cuda_base.py`) = floor; Ultra after warm-up + trim 1.18 GB, flat over 20 pieces. `enable_cpu_mem_arena=False`:
  no change. Encoder with external data (`.cache/tmp/ultra-ext/`, converted copy): same 1.18 GB end, but peak while
  loading 1.70 instead of 2.35 GB; memory then appears at first runs (+360 fast_tdt, +480 encoder) → most of the
  remaining ~0.7 GB is CUDA loading kernel code per op type on first use (confirmed below).
- **Where the ~1.2 GB own memory goes** (2026-11-15; `ram_map.py`, `ram_parts.py`, `ram_ops.py`, `ram_variants.py`):
  992 MB is C/C++ heap still in use, Python/numpy only 70 MB. Encoder session after creation + trim holds ~50 MB
  (weights go to GPU); its **first run adds +626 MB**, preprocessor alone first run +225. Tiny single-op graphs, first
  run: MatMul fp16 **+261 MB** (cuBLAS init + kernels), Conv2d +142, depthwise Conv1d +54 (cuDNN), Softmax/LayerNorm
  ~+5; any CUDA session +97 (context); CUDA libs after preload_dlls 147. `CUDA_MODULE_LOADING=EAGER` → +2.68 GB, so
  lazy kernel loading (default) is what keeps it this low. Not reducible with settings.
- Applied: `fast_tdt` drops ORT wrapper's kept step-graph bytes (`_model_bytes`, −45 MB); gauge VAD buffer 1200 → 60 s
  (preallocated float buffer, −77 MB; cuts identical on all 39 recordings, longest speech run 12.4 s).
  gauge after two long takes: 1.77 GB total (own 1.23 + files 0.44); was 2.54 before today.
- Not applied: preprocessor on CPU (`preprocessor_config={"providers": ["CPUExecutionProvider"]}`) −100 MB,
  +3–5 ms per piece. Bigger cuts would need no CUDA (CPU int8) or another runtime with its own kernels (not tested).

## Pauses in read-aloud takes (2026-11-15, for the cutting rule)

Script: `.cache/tmp/speech_analysis.py <wav> <ref.txt>` (pauses placed into the reference text).
- **Measurement artifact**: Parakeet TDT stamps the first word after a pause INSIDE the silence (up to ~1 s early), so
  placing a pause "before the next word" shifts it one word late. Place pauses by the VAD pause **start**.
- Story (178 words, 90 s): pause at 12/12 sentence ends (0.3–1.5 s, mostly 0.8–1.4), 8/12 commas, 5 short mid-phrase
  (0.3–0.5).
- Technical texts: en 13/13 sentence ends paused, 11 mid-phrase pauses around file names/numbers; uk 13/14,
  sentence-end pauses up to 2.2–2.5 s.

## Granite Speech 5.0 470M TurboCTC (checked 2026-11-15, English only — no Ukrainian)

IBM, CTC conformer, released 2026-08-25 English-only, multilingual update 2026-10-21 (en fr de es pt it ja,
punctuation). Run: `.venv-granite/bin/python bench/granite.py [wavs]` (CPU torch 2.15 + transformers 5.23 +
torchaudio, fp32, 6 threads, whole takes; model in `models/granite-speech-5.0-470m-turboctc`, 946 MB).
- WER: en tech 141151 **7.8%** (numbers ignored 6.5%) vs Ultra pipeline 6.2%; story 181554 **2.8%** (1.7%) vs Ultra 2.2%.
- Other English takes (no ref): short sentences mostly identical; Granite lowercases acronyms (cpu, gpu, yaml, mcp)
  and garbles more names (Namanotron, Prakeet, Sailero, "post-proing"); Ultra better on tech dictation.
- Speed CPU: 126 s → 6.4 s, 90 s → 4.5–5.2 s, 5 s → ~0.5 s (≈ 2× faster than Ultra int8 CPU). Peak RAM 3.5 GB.
- Note: log.jsonl entries for 141151 (and other pre-15:30 takes) have v3/Nemotron text, not Ultra.

## Moonshine tiny (English, non-streaming; checked 2026-11-15)

27M params, Oct 2024, `models/moonshine-tiny` (108 MB). `.venv-granite/bin/python bench/moonshine.py [model dir]`:
CPU transformers fp32, same gauge VAD pieces (cuts precomputed in `.cache/tmp/cuts.json` with .venv), max 6.5 tok/s.
- WER: en tech 141151 **20.3%** (Ultra 6.2%, Granite 7.8%); story **9.0%** (2.2 / 2.8); notes 203034 **10.0%** (4.3 / 4.3).
- Plain short sentences fine; tech terms and names collapse ("Orenna Nixes" = ONNX); one 3.7 s clip looped
  ("bird" ×12), music-note output on a silent take. CPU speed: 126 s → 2.6 s, 80 s → 1.7 s, short clips 60–120 ms.
- Take 203034: Granite and Moonshine both give "says around" and "seats seats" where Ultra differs → Ultra's
  "forecast around stays around" is likely Ultra's error, and the reading probably did repeat "seats" (not listened).

## Recorded speech vs TTS and LibriSpeech clips (2026-11-15)

**Result:** recorded read text scores like LibriSpeech audiobook speech (story 2.2% vs LibriSpeech test-clean ~2.3%);
what remains is a few minimal pairs.
- Minimal-pair sentences (40, ship/sheep, live/leave, full/fool, hat/hut, th, w/v; recording vs Qwen3-TTS "ryan";
  `bench/pairs.py ultra|granite|moonshine`, Ultra = int8 CPU): WER recording / tts: Ultra 6.6 / 2.2%, Granite 9.4 / 2.2,
  Moonshine 17.9 / 4.1. Of Ultra's 14 sentences with an error only **~5–6 are real mishearings** (leave→live,
  full→fall, hut→hood, skin→scan, rites→rides); the rest are repeats in the reading, 1 recording artifact (0.9 s of
  exact zeros at the start), 1 spelling (colour).
- 20 LibriSpeech test-clean sentences read cold in one take (215 s) vs the original clips: Ultra 5.6 / 1.9%,
  Granite 7.5 / 1.2%, Moonshine 12.5 / 4.1%. Almost all of the gap is reading slips the other models also show
  ("to to", "Capn Captain", "protested, proceeded"); **0–3 real mishearings in ~300 words** (Henley→Hanley,
  chiaroscurists). Ultra on the original clips: 0 real mishearings (6 spellings).
- Granite dropped 6–8 words at the start of piece 1 ("how cheerfully he seems to grin how … spread his").

## Startup time, ultra-gpu (2026-11-15, `.cache/tmp/load_time.py`, separate process each run, GPU idle at P5)

Ready (sessions + fast_tdt + warm-up) **2.6–3.2 s** with files in page cache, **3.35 s cold** (model + .venv-gpu libs
evicted with `.cache/tmp/evict.py` = posix_fadvise DONTNEED, verified with `fincore`; NVMe reads the 1.2 GB encoder in
0.45 s, so disk adds only ~0.3–0.6 s). Steps: import ort 0.1, preload CUDA libs 0.05 (cold 0.25), create sessions
**1.75–2.2**, fast_tdt 0.36–0.56, warm-up 0.35–0.47 s. First real speech piece right after: 69–95 ms = same as the
second → open item 1 (first-take decoder setup) not visible here.
→ Loading on hotkey press while recording hides it for anything spoken longer than ~3 s.

## transcribe.cpp (built + tested 2026-11-15, Vulkan, Q8_0) — candidate to cut RAM/VRAM

github.com/handy-computer/transcribe.cpp: C/C++ STT library on ggml (same tensor lib as llama.cpp); hand-written
model code per family (encoder graphs, TDT/CTC/RNNT/attention decoders, mel front-end, Silero VAD v6.2), GGUF
weights, backends CUDA / **Vulkan** / Metal / CPU (tinyBLAS), CPU weights mmapped, **Python binding**, 16 families
(Parakeet incl. **parakeet-ultra**, Granite 5.0, Moonshine, Whisper, Nemotron 3.5, Qwen3-ASR, Canary, Cohere…).
- `handy-computer/parakeet-ultra-gguf` (2026-10-05): F16 1.26 GB, **Q8_0 740 MB**, Q6_K 611, Q4_K_M 486 MB;
  LibriSpeech test-clean 1.80 / 1.80 / 1.81 / 1.82% (their numbers). Also `vladkens/parakeet-tdt-0.6b-v3-uk-gguf`
  (uk fine-tune of v3, 2026-11-10, untested).
- Why: onnxruntime CUDA RAM is mostly NVIDIA libs (see GPU memory); ggml has its own small kernels, no cuDNN; quantized
  weights use ggml's own matmul (cuBLAS maybe still loaded on CUDA build). Vulkan build uses no CUDA at all.
- Vulkan build here: system has libvulkan, NVIDIA+AMD ICDs, glslc, gcc; missing cmake (→ pip in a venv) and Vulkan
  headers (→ download into project). CUDA build needs nvcc (pip nvidia nvcc package — not checked).
- Build (all in `.cache/build/`): cmake+ninja in `.cache/build/venv`, Vulkan-Headers + SPIRV-Headers cloned and
  installed to `.cache/build/prefix`; `build-vk` (static, CLI) and `build-vk-so` (shared, for the ctypes Python
  binding, used via TRANSCRIBE_LIBRARY + PYTHONPATH, no pip install). ~2 min with -j4. No OpenBLAS on the system →
  host-side TDT decoder runs the scalar fallback (README: BLAS makes it ~10–15× faster).
- Model: `models/parakeet-ultra-gguf/parakeet-ultra-Q8_0.gguf` (740 MB). Auto-picks Vulkan1 = the NVIDIA GPU.
- `bench/tcpp.py` (same gauge pieces from cuts.json, one process):
  | | current (ORT CUDA fp16 + fast_tdt) | transcribe.cpp Vulkan Q8_0 |
  |---|---|---|
  | RAM after load / after takes | 1.77 GB | **0.26 / 0.29 GB** |
  | VRAM | 1.75 GB | **0.72–0.73 GB** |
  | ready (warm) | 2.6–3.2 s | **0.95 s** (first ever run +14 s: Vulkan shader compile, cached by driver in ~/.cache/nvidia) |
  | per piece | ~80 ms | ~200–250 ms (of which decoder pred+joint 100–160 ms, scalar) |
  | WER int8-talk / story / notes | 6.2 / 2.2 / 4.3% | 6.2 / 1.7 / 5.0% unpadded; 7.3 / 2.2 / 3.6% padded to 25 s |
  WER differences are a few words either way (noise level on 3 takes). Padding not needed (shapes are dynamic).
  Whole notes take in one call (its own long-form): 6.4%.
- OpenBLAS 0.3.30 built into `.cache/build/prefix` (single-thread, static-linked into libtranscribe): **no speedup** —
  BLAS is only used by the CTC head; the TDT decoder runs every step (LSTM `pred` ~0.8 ms + `joint` ~0.7 ms) as a small
  ggml graph on the CPU threadpool (src/arch/parakeet/decoder.cpp decode_tdt_greedy). README's "10–15×" isn't for TDT.
- Decoder threads (session n_threads, env THREADS in tcpp.py), ms per piece: 1 → 250–330, 2 → 220–260,
  **4 → 200–230 (best)**, 6 / default 8 → 200–250. Encoder ≈ 70 ms of it, decoder 100–150 ms (fast_tdt does the
  decoder on the GPU, hence ~80 ms total). Same text at every thread count.
- Tried fast_tdt's window trick in their C++ (joint over K frames per call, cached until a token is emitted;
  patch saved as `.cache/build/tdt_window.patch`, reverted): text identical, but **slower** (K=8 ~225 ms/piece, K=16
  ~290 vs ~200). Reason: TDT already jumps frames by its duration and ~2/3 of steps emit a token (e.g. 87 steps,
  58 tokens for 278 frames), so a window is used ~1.5 frames before it goes stale; on CPU there is no round trip to
  save (on GPU in fast_tdt there is). Real cost is memory bandwidth: decoder weights are fp32 (joint out_w 21 MB read
  every step, LSTM ~26 MB every token, ~0.5–0.8 ms each). Ideas not tried: f16 decoder weights (half the bytes, tiny
  numeric change), decoder graphs on the Vulkan device.
- **In gauge since 2026-11-15 as the default `--final ultra-vk`** (runs from `.venv`; `--final ultra-gpu` still
  there). Session n_threads = --threads (default 4). Logs filtered to warnings/errors (ggml's log hook must be set
  before importing transcribe_cpp: the Vulkan device list is printed while the library loads). Same text as
  bench/tcpp.py. Startup measured: ready 1.08–1.10 s warm, 1.32 s cold (model load 0.85 s, imports 0.2, session +
  warm-up 0.04); first 20 s piece 217–251 ms.
- **Ukrainian** (take 20261115-222256, ref_uk_mixed, 121 words): ultra-gpu fp16 10.7%, ultra-vk Q8_0 9.9% — same 7
  errors (гауч, hard words, Sharpa, що, вкладає, вручно, "2 ГБ"), only "Вчора ввечері" differs (у цьому вечорі /
  увечері). No loss from Q8 or from transcribe.cpp in Ukrainian.
- **Q4_K_M** (486 MB, bench/tcpp.py, 4 threads): VRAM **493 MB** (Q8_0 733), RAM same 0.29 GB, speed same (~200 ms;
  decoder is fp32 either way), first run after download +5.8 s (shader compile, cached after). WER int8-talk / story /
  notes / uk: Q4 4.7 / 3.4 / 4.3 / 9.1% vs Q8 6.2 / 1.7 / 5.0 / 9.9% — ~10 words differ in both directions, noise.
  Sentence sets (bench/pairs.py Q8_0|Q4_K_M [set]): pairs recorded 6.3 → **7.2%** (13 → 15/40 sentences wrong), pairs
  tts 1.6 → 2.2%, LibriSpeech 1.9 → 2.5%. Of 7 sentences that differ, Q4 is worse in 6: vest→west,
  full→fall, grosser rites→grocer rides, chiaroscurists→chioscurists, eight/ten→8/10, honourable→honorable; better
  in 1 (hut: heard → hood). → **Q4 slightly worse (~+0.6–0.9 WER), mostly on close calls (close vowels, rare
  words)**; kept Q8_0 in gauge.
- Not done yet: CUDA build, more Ukrainian takes, f16/Q8 decoder weights,
  decoder on Vulkan.

## dictate.py — POC "type anywhere" app (2026-11-15, written, NOT yet confirmed working end-to-end)

`.venv/bin/python dictate.py` = server (loads ultra-vk via gauge.load_final, keeps it loaded, ~0.73 GB VRAM);
`dictate.py toggle` = client over unix socket `$XDG_RUNTIME_DIR/dictation.sock` (bind to a KDE shortcut).
- Toggle 1: parecord + gauge.Vad in a thread, sound (ocean audio-volume-change), notification "● Recording…"
  (notify-send -u critical = stays; replaced in place via -p/-r). Toggle 2: sound (completion-success),
  "Transcribing…", gauge.transcribe (same cuts), text → clipboard via `qdbus6 org.kde.klipper /klipper
  setClipboardContents`, then Ctrl+V from a uinput virtual keyboard ("dictation-keyboard", raw ioctls, no deps;
  /dev/uinput is user-writable via logind ACL), notification "Pasted" + text (4 s). Takes saved to recordings/ with
  `"app": "dictate"` in log.jsonl. SIGINT/SIGTERM handled (clean socket + device removal).
- Why clipboard+paste: uinput sends keycodes → typing letters depends on the active layout (breaks uk/en mix).
- Tested by me: virtual keyboard registers/unregisters, socket replies, model loads 1.5 s, clean exit,
  notification replace works. NOT tested: a real take + paste (would record mic / paste into user's window).
- Shortcut: user picked **Meta+Y** (Meta+D = KDE Show Desktop, don't use) but it was **not saved** — nothing in
  ~/.config/kglobalshortcutsrc or ~/.local/share/applications. Either user adds it (System Settings → Keyboard →
  Shortcuts → Add New → Command or Script → command above → set Meta+Y → Apply) or I write the .desktop +
  kglobalshortcutsrc entry (outside project dir → only with user's OK; offered, not answered).
- Known POC limits: terminals need Ctrl+Shift+V (nothing pasted there); clipboard overwritten (not restored);
  toggle not push-to-talk; runs in a terminal (no autostart/tray yet).
- Q4_K_M file (486 MB) still in models/parakeet-ultra-gguf/: asked whether to delete, answer unclear → ask again.

## Rust tray app (project root; was `app/` until 2026-11-20; in daily use by the user; replaces dictate.py)

Build (in the root): `CARGO_HOME=$PWD/.cache/cargo CMAKE_BUILD_PARALLEL_LEVEL=4 cargo build --release -j4` (cap
memory 6G). Run: `target/release/dictation` (tray); `dictation toggle` talks to it over
`$XDG_RUNTIME_DIR/dictation.sock` (the socket's `status` = single-instance check). One process, no helpers.
Settings = tray menu → `config.toml`. Since 2026-11-20 all app data is in one folder, `~/.local/share/dictation/`
(`crate::root()`, OS app data folder): `config.toml`, `recordings/`, `models/` (before: config and recordings in the
project root).
- Cleanup 2026-11-20 (cruft removed): `reload`/`unload` CLI commands (tray does both); "Recording…"/"Transcribing…"
  notifications + KDE critical-urgency hack (tray icon + sounds show state; notifications = result and errors only,
  4 s, transient); `threads` setting → const 4; `save_recordings` default off (stores voice; the user's config.toml
  keeps it on); recordings log.jsonl now `{wav, seconds, text, ms}` (gauge fields `parakeet`, `final_model`,
  `parakeet_ms`, … dropped → bench scripts reading `parakeet` won't see new app takes); timestamps via `jiff`
  instead of libc; `malloc_trim` only on Linux/glibc; `src/bin/rectest.rs` deleted; `src/bin/check.rs` →
  `experiments/check.rs` as a Cargo example (`cargo run --release --example check -- <wav>…`, uses the app's model).
- Full-quality copy of every take (2026-11-25): before, recordings were only the 16 kHz wav asr got (the original
  was resampled in the audio thread, gone; takes from before are 16 kHz only, not converted: they are
  exactly what asr heard, benchmarks use them). Now also `<name>.opus` = mono at the mic's own rate (MacBook mic:
  44.1 kHz -> resampled to 48 kHz, Opus takes 8/12/16/24/48 kHz only), 48 kbit/s fullband, ~21 MB/hour (wav 16 kHz
  110 MB/h, 48 kHz wav would be 345 MB/h); log line + `"opus"`, `"opus_rate"` (16000 = Bluetooth headset mode).
  Why: keep full quality for later use (e.g. TTS or fine-tuning data, which want >= 24 kHz); 16 kHz loses 8-12 kHz
  (s/f/sh, "air").
  Encoder `opus-pure` 0.2.5 (pure Rust, BSD, no deps/build script; vs libopus 1.6.1: 433/440 streams > 100 dB SNR,
  pure SILK bit-identical, their numbers; young: 3 months, 7 dependents). Skimmed for malware: no build.rs, no
  net/process/fs/FFI/asm, unsafe only in SIMD kernels. Alternatives: `opus` crate (libopus C), rusty-opus (no Ogg
  writer), opus-rs (self-labelled LLM-generated). Encoded on a thread after the paste (`save_take`).
  Checked on a 3.7 s take: ffmpeg reads it (opus, 48 kHz, 3.70 s = wav length, 46 kbit/s), energy up to 20 kHz
  (8-12 kHz -49 dB, 12-20 kHz ~-53 dB vs total; nothing above 8 kHz in the wav), `dictation transcribe` on the
  .opus = same text.
- Takes saved as .opus only (2026-11-26, user's decision): the 16 kHz .wav (5-6x the size) only gave bit-exact
  re-runs; for those, turn the .opus into one 16 kHz wav once and give every run that file. Log line: `opus`,
  `opus_rate`, `seconds`, `text`, `ms` (no `wav`). Older takes keep their .wav. `hound` is a dev-dependency now
  (experiments/rec.rs). The 230 .wav that had an .opus copy (Nov 25-26) were deleted; the older ones (16 kHz only)
  stay.
- Neural Engine: a new app binary = one slow first load (33-36 s instead of ~1 s; 2026-11-26), our "prepared" check
  not involved: macOS's own compiled-model cache seems tied to the exact binary (a rebuild giving the identical
  binary loaded in 1.4 s). Read from the log over three rebuilds, macOS' cache not inspected. Users: once per update.
- **ASR = `src/asr` (since 2026-11-20)**: our Rust port of transcribe.cpp's Parakeet path on burn 0.22 / CubeCL
  (Vulkan), see "burn pipeline" below. Replaced the `transcribe-cpp` crate (C++/CMake, static); `src/tc.rs` and the
  CMake env in `.cargo/config.toml` are gone (file deleted). Model folder
  `models/parakeet-ultra-q8/`: `model.weights` (converted once from the downloaded GGUF, which is then deleted) +
  `kernels/` (CubeCL kernel cache). Native code left: `aegis` (C via cc) from turso (CubeCL's cache DB); runtime
  dlopen of Vulkan/NVML. App numbers (2026-11-20): ready 1.1 s warm (first run with empty kernel cache ~4 s);
  836 MiB VRAM loaded, ~1 GB peak on long takes; after unload 129 MiB stays (device context + pool; freeing it
  needs a separate process). Live takes: 13 s in 176 ms right after a load, but 6.8 s in 301 ms after idle and the
  first takes after start 1–3 s (cause not verified: GPU clock after idle vs kernel compile for new lengths).
  Before (transcribe.cpp) numbers in the bullets below.
- **No own VAD/cutting any more**: whole take → `session.run()`. Ultra GGUF carries a VAD head + segmenter
  (`stt.parakeet.vad.*`, `segmenter.*`); transcribe.cpp `run_longform` (port of NVIDIA "kestrel") cuts takes > 30 s at
  the midpoint of the latest ≥ 0.2 s pause in (1, 30) s. vs our Silero + gauge cuts on the 4 refs: WER 4.7/5.6/6.4/9.1%
  vs 5.7/4.5/2.9/9.9% (word-weighted 6.2 vs 5.5%, few-word differences both ways = noise level). Cost: VAD head scans
  120 s blocks → +45 MB VRAM, +45–100 MB RAM (781 vs 735 MB VRAM; piece size 25→30 s alone changes nothing).
  297 s take: 488–528 MB RAM, 781 MB VRAM, 3.6 s. Silero gguf no longer used (models/silero-vad/*.gguf, ask to delete).
- Model memory: **load on start press (in parallel with recording, 1.0 s warm), unload after idle**: tray "Keep model
  loaded" = after each take / 1 min (user's choice, default) / 10 min / always. Unloaded: 0 MB VRAM, ~23 MB RAM;
  loaded 715 MB VRAM / ~250 MB RAM; after unload ~160 MB RSS of which ~130 MB = mapped NVIDIA/Vulkan libs (shared,
  reclaimable). Idle loaded model costs **no GPU power** here (A/B/A 7.30/7.49/7.36 W): the dGPU never runtime-suspends
  (KWin uses it, ~28 % busy).
- Mic: cpal (PulseAudio protocol), **fixed 50 ms blocks + 150 ms tail after stop** — default ~1.024 s blocks cut up to
  1 s off every take's end (take lengths were exact multiples of 1.024 s). Stream opens in ~70 ms.
- Build: `cargo build --release -j4` under the 6G cap; cold ~10 min (≈500 crates: burn, CubeCL, wgpu), so run it in
  the background. (transcribe.cpp era: CMake shader compile OOMed at 6G without CMAKE_BUILD_PARALLEL_LEVEL=4.)
- Models (2026-11-20): downloaded into the OS app data folder via `directories` (Linux `~/.local/share/dictation/models`,
  macOS `~/Library/Application Support/dictation/models`, Windows `%APPDATA%\dictation\data\models`) — no longer the project's
  `models/` (which experiments still use). Resumable `.part`, then SHA-256 check against HF's LFS oid (Q8_0 =
  007a5976…; the project copy matches); mismatch → .part removed. Tray: top-level "Open app folder…" (`open` crate). One model only (models::MODEL); the Model submenu and `model` setting were removed 2026-11-20 (other architectures would each need their own port)
  so users can delete files; a deleted model is noticed within 10 s (back to "No model", offers download).
- Output: clipboard via arboard (cross-platform; Wayland data-control, text served by our process until replaced; replaced Klipper D-Bus 2026-11-20) + uinput virtual keyboard (evdev); paste keys Ctrl+V / Ctrl+Shift+V / Shift+Insert / clipboard
  only. Sounds: two sine notes generated in code (rising = start, falling = stop), played with rodio (playback only, no decoders) — no sound files (user: don't depend on the desktop for minor stuff), notifications notify-rust, tray ksni. egui settings window was built then removed (user prefers
  tray menu). 4-bit model removed from the menu (file still on disk, ask to delete).
- **Global shortcut not working yet**: XDG GlobalShortcuts portal refuses unregistered host apps ("App info not found
  for 'local.dictation.Dictation'") → needs `~/.local/share/applications/local.dictation.Dictation.desktop` (outside
  project; asked, not answered). Clicking the tray icon moves focus to the panel → paste lands nowhere.
- Portability (not done): binary needs glibc ≥ 2.43 (the decoder picks AVX2 at runtime, plain fallback otherwise), uinput access here only via
  KDE Connect's udev rule, paths baked in at compile time.
- Discussed next: pure-Rust Parakeet. First step = own CPU TDT decoder with int8 weights from the GGUF (joint + LSTM ≈
  12 MB fits the 16 MB L3; transcribe.cpp dequantizes to fp32 ≈ 47 MB/step → 100–150 ms/piece). Encoder: burn
  (CubeCL Vulkan/SPIR-V backend) prototype, measure vs ~70 ms. candle = CUDA/Metal only on GPU.

## burn encoder experiment (2026-11-20, `experiments/burn/`)

Question: is a pure-Rust encoder on burn (CubeCL, Vulkan) slower than transcribe.cpp's ggml encoder?
(Measurements below are **with autotune**; since 2026-11-20 the experiment builds without it, see "burn autotune
and startup" below: encoder 100–126 ms, no spikes, first run 0.14 s.)
Setup: burn 0.22.1 (`std, vulkan, fusion, autotune`), own GGUF reader (Q8_0/F16 → f32/f16 on the CPU), NeMo
FastConformer written out by hand (`src/main.rs`, ~300 lines): dw_striding ×3 subsampling, 24 × [½FF, rel-pos MHSA
(Transformer-XL, pos_bias_u/v, rel_shift), conv module (GLU, dw 9, BN folded, SiLU), ½FF, LN]. Input = transcribe.cpp's
own mel (`ref_dump.py` with `TRANSCRIBE_DUMP_DIR` dumps mel, pre-encode, block 0 steps, block 12/23 = final, decoder
steps). Piece: first 25 s of take 141151 (2501 mel frames → 313 encoder frames). RTX 3050 Ti Laptop 4 GB.
- **Speed (warm, 25 s piece)**: burn f16 encoder **55–75 ms** (min / median over 30 runs). transcribe.cpp (same
  piece, `ref_time.py`): run 219–240 ms, of which decoder 116–138 ms (its own log) → mel + encoder + overhead
  **98–102 ms**. So burn is not slower (burn number excludes mel; not yet ported). burn f32: 87 ms min but median 290
  (erratic). Without `fusion`/`autotune` features: f32 410–526 ms.
- **Accuracy vs transcribe.cpp**: final encoder output rel. L2 diff 1.1% (f32) / 1.2% (f16); 2% already at
  pre-encode, where the only quantized op is the Q8_0 output linear → likely ggml's int8 activation quantization, not
  a port bug (not proven until text is compared with a decoder).
- **f16 pitfall**: pure f16 gives garbage (rel. err 100%): residual stream reaches ±6800, LayerNorm's x² overflows
  f16 (max 65504). Fix = LayerNorm + softmax in f32 (as ggml does), rest f16.
- **Costs**: VRAM steady **~1.5 GB** (f16 weights 1.2 GB) vs 0.73 GB (Q8_0); **3.4 GB spikes during the first run**
  (autotune trying kernels). First run per process **18–35 s** (shader compile + autotune; no persistent cache was
  written anywhere — results stay in memory, so every start re-tunes). Load 6.5 s (naive loader: reads whole file,
  dequantizes + transposes on the CPU) vs ~1 s.
- **int8 weights (`Q8=1 F16=1`)**: Q8_0 blocks uploaded as they are (burn `QuantValue::Q8S`, `per_block([32], F16)`,
  `TensorData::quantized`, GGUF layout [out, in] + `.transpose()` view; burn's quantized matmul streams them).
  Same accuracy (1.2%, bit-identical weights). VRAM: **no creep** — `MEMLOG=1` (burn `memory_pool_usage()` + driver
  per run, 60 runs): flat 964 MB driver = 719.5 MB weights in use + 165 MB pool for activations (transcribe.cpp's
  compute buffer: 108 MB) + ~80 MB driver context; `memory_cleanup()` → 864 MB. The earlier "0.81 → 1.02 GB creep"
  was an artifact: 1 s samples fell almost all inside the 42 s autotune first run. vs transcribe.cpp 0.73 GB;
  encoder **46 ms min / 80 ms median**; load **1.7 s** (no CPU dequant/transpose). Autotune spikes (2.9 GB) and
  18–42 s first run remain.
- **VRAM vs transcribe.cpp, same method** (whole-GPU used minus 177 MiB idle; both include their own context):
  transcribe.cpp **739 MiB** = 705.8 MiB weights (the *whole* GGUF incl. decoder/VAD, though the decoder runs on the
  CPU) + ~33 MiB (its logged 107.8 MiB compute buffer is evidently not all backed). burn steady **~980 MiB**
  (PERSIST=1) = 37 MiB Vulkan/wgpu context + 686 MiB weights (encoder only) + ~80 MiB pool for activations +
  **~165 MiB outside burn's pool, reserved once per kernel *type*** (`BISECT=1`: subsampling +0, feed-forward +36,
  attention +75, conv module +25; the other 23 blocks, warm runs and a new input length (26 new kernels) add 0 →
  not per kernel). **Corrected below (Startup): it was tuning — absent when the tuning cache is filled.** A lower
  autotune level didn't change it (`[autotune] level = "minimal"`: same memory, first run 19 s), not wgpu block size (cubecl already uses
  `MemoryHints::MemoryUsage`).
- Reductions found: `PERSIST=1` (weights via `dev.memory_persistent_allocations`, own buffers instead of pool pages)
  −63 MiB after load / −22 MiB steady; `dev.memory_cleanup()` after a take −43 MiB. Biggest remaining lever: the
  attention kernels' one-time 75 MiB (try burn's fused attention op / other matmul layout).
- Mel, decoder, tokenizer, long takes: done, see "burn pipeline" below.

### burn autotune and startup (investigated 2026-11-20) — first decision: no autotune (superseded the same day: tuning on, see "Tuning in the app" below)

**Decision**: build burn **without the `autotune` feature** (fusion and `persistence` kept). Reason: the only
concern with tuning is its VRAM spike (~3.1 GB) and what happens if VRAM runs out mid-tune (untested). The
encoder is ~2× slower without tuning but as fast as transcribe.cpp's mel + encoder. Revisit tuning later via a
pre-tune sweep in a child process (plan at the end).

**What autotune is**: for each operation (matmul, fused matmul, conv, reduce), cubecl benchmarks several kernel
variants on this GPU the first time it sees a "key" and keeps the fastest. Without the feature it uses a fixed
default variant per operation. The tuning itself compiles + runs every candidate, which is the time and VRAM cost.

**Persistent caches** (both need burn feature `persistence`; with `default-features = false` it was off, so every
start re-tuned/recompiled):
- `cubecl.toml` (searched from cwd upward, or a `[cubecl]` section in `burn.toml`): `[compilation] cache = true`
  (SPIR-V of compiled kernels) and `[environment] path = { directory = "..." }`. Default path `target/environment`
  when a Cargo.toml is above cwd, else `~/.cache/cubecl` (`global`; `local` = cwd). One sqlite file `default.db`
  (+ wal) holds autotune results, compiled kernels, throughput numbers.
- SPIR-V cache key includes the GPU (vendor/device id) and the binary's build id: every rebuild invalidates it
  (~3.5 s once). The autotune results survived rebuilds.
  **Fixed 2026-11-20:** `build.rs` sets the linker build ID to a hash of `Cargo.lock` + rustc version
  (`-Wl,--build-id=0x…`; CubeCL reads it via the `buildid` crate). Since 2026-11-20 (portable): no linker flag;
  `buildid` feature `buildid-custom-inject` + our `build_id__get` in lib.rs returns the key, same on every OS
  (checked in the disassembly: buildid::build_id calls it; switching the feature rebuilt the GPU crates, ~5 min). We have no own `#[cube]` kernels, so kernels only
  change with locked deps/compiler. Rebuilds and `check` now share the cache: after a rebuild, the first `check`
  clip ran in 124 ms with no compiling. `dictation sweep` is needed once per model install and again only after a
  dependency update or new rustc.
  **Per-GPU cache folder (2026-11-20):** CubeCL files compiled kernels per GPU (vendor/device id), but autotune
  results under the device *kind* (`device-100-8192-wgpu<spirv>` = DefaultDevice), so a second GPU reused the
  first one's picks: NVIDIA's tuned matmul ("No tile size is available") crashed the AMD iGPU. Now
  `asr::set_kernel_cache` uses `kernels/<GPU name>/`; the name comes from the same choice CubeCL makes
  (`CUBECL_WGPU_DEFAULT_DEVICE` if set, else wgpu's high-performance Vulkan adapter). Existing caches were moved
  in: `NVIDIA-GeForce-RTX-3050-Ti-Laptop-GPU/`, `AMD-Radeon-Graphics--RADV-RENOIR-/`.
- **AMD iGPU (Radeon Vega, Renoir) measured 2026-11-20**, 23.75 s take, `CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(0)`:
  untuned encoder ~2735 ms / total ~2820 ms; tuned (`--features autotune`) encoder ~1272 / total ~1355 ms (2.1x;
  tuning cost 27 s at load + 12 s first take for this one length). Same text. NVIDIA, same take: 250 ms. One-shot
  dictation would work on such hardware (~18x real time), live mode not (~1.7 s per 30 s pass).
- Env overrides: `CUBECL_AUTOTUNE_LEVEL`, `CUBECL_AUTOTUNE_CACHE=0/1`, `CUBECL_ENVIRONMENT` (name), `CUBECL_DEBUG_LOG`.
- For the app (agreed with user): cache goes **in the model's folder** (`models/<model>/` = model file + cache),
  so deleting the folder removes both. The app has no cubecl.toml, so the path must be set in code before device
  init (not checked yet how: `CubeClRuntimeConfig` set/override API).

**Numbers** (RTX 3050 Ti 4 GB, 25 s reference, F16=1 Q8=1, PERSIST=1):

| | no autotune | autotune, cache filled | autotune, nothing cached | transcribe.cpp |
|---|---|---|---|---|
| encoder, 25 s audio | **100–126 ms** | 42–50 ms | 42–50 ms (after tuning) | ~100 ms mel+enc |
| device init + weights | 0.3 + 1.7 s | same | same | ~1.0 s total |
| first run | 0.8–3.3 s cold, **0.14 s** with SPIR-V cache | 0.09 s | 19–69 s | — |
| unseen length, 1st time | 0.1–2 s compile (then cached) | 0 if pre-tuned | 1–40 s tuning | — |
| VRAM during take / after cleanup | ~845 / ~785 MB | ~850 / ~790 MB | spikes ~3.1 GB | 739 MiB |
| accuracy enc.final vs transcribe.cpp | rel. L2 1.08% | 1.19% | | |

- **Correction of the earlier VRAM breakdown**: the ~165 MiB "outside burn's pool" and the 2.9–3.4 GB spikes were
  **tuning**, not driver scratch. Absent when nothing tunes in the process (cache filled, or no autotune).
- Weight loading 1.7 s is our experiment loader (reads the whole 740 MB file, converts on CPU); mmap + skipping
  decoder tensors should get near transcribe.cpp's 1.0 s. Not tried.
- The first-run 0.8 s vs 3.3 s difference without autotune = NVIDIA's own driver shader cache warm vs cold.

**How tuning keys depend on length** (from cubecl source, `cubecl-runtime/src/tune/util.rs` `anchor()`):
- Each dimension is rounded **up** to a power of a base: level `minimal` 2.5 (bands 1, 3, 7, 16, 40, 98, 245,
  611… encoder frames; 1 frame = 80 ms → bands end at 1.3, 3.2, 7.8, 19.6, 49 s), `balanced` (default) 2,
  `extensive` 1.5, `full` = no rounding. The pre-encode works on mel-length dims with their own bands.
- Plus **alignment class** ("how divisible" the shape/stride is; keys show `lhs_shape_align` etc.): a multiple of
  8 can be read 8 numbers at a time with no edge checks, an odd length needs a variant with bounds checks /
  narrower reads. Different variants → separate keys even inside one band (1500 vs 1501 mel frames: tuned separately, 1.4 s).
- So without padding the number of keys ≈ bands × alignment classes × per-dim combos. Padding the mel length to a
  multiple of 8 would leave ~1 class per band (relies on internals; not tested).
- Tuned per key: MatmulAutotuneKey, FusedMatmulAutotuneKey, ConvAutotuneKey (pre-encode convs),
  FusedReduceAutotuneKey, ReduceDimAutotuneKey, FusedBroadcastedReduce. Log: `[autotune.logger] level="minimal", file=…`.

**Lazy tuning measured** (`LENS=…`, autotune on, cache partly filled): new lengths 1–40 s on first use (the first
unseen one in a process the worst, it also compiles shared parts); once tuned, fast in later processes too. Lengths
in an already-tuned band and class: no tuning (850 frames after 700/1234: 83 ms).

**Pre-tune sweep measured** (`SWEEP=1`: every length 8…3000 mel frames = 0.08–30 s in 80 ms steps, 375 runs,
reference mel repeated to reach 30 s, empty cache, autotune on):
- **139 s total** (load 2 s + sweep 137 s); 44 lengths tuned (233 tuning decisions), 1–13 s each. Worst: the
  first lengths (8 and 16 frames, 13 s each), powers of two (64, 128, 256, 512: ~4 s; 1024: 9 s; 1960–2048 ~20 s
  together; 2560: 3.6 s).
- Peak whole-GPU VRAM 3.65 GB (~3.1 GB ours, with 505 MiB of other use). Cache file 12 MB.
- **2nd sweep, new process: 0 tuned, 22 s, no slow length** → brute force covers everything ≤ 30 s, no burn
  internals needed. Segments never exceed 30 s if we port transcribe.cpp's cutter (cuts > 30 s takes at the latest
  ≥ 0.2 s pause, see "Rust tray app"); its exact max still to confirm in the GGUF `segmenter.*` keys.

**If we turn tuning back on — plan discussed**:
1. Pre-tune once after model download with the brute-force sweep (~2.3 min, tray "Optimizing for your GPU…").
2. Run it in a **child process** so an out-of-memory crash can't kill the app; progress is cached, so a retry resumes.
3. Check free VRAM first (~3.5 GB needed), postpone if a game etc. holds it.
4. Re-run when the cache is invalidated (other GPU; maybe driver/burn update — unchecked).
- Open questions: what an out-of-memory during tuning actually does on Linux/NVIDIA (likely a wgpu panic: NVIDIA
  Vulkan doesn't spill VRAM to RAM; Windows WDDM would page instead); where in the sweep the spike happens and
  whether `memory_cleanup()` between lengths lowers it.
- Alternatives considered: lazy tuning only (cache fills by itself; bad first takes, spikes mid-dictation);
  padding every segment to one 30 s shape (one key, needs an attention mask — NeMo-style: padded frames get −inf
  scores before softmax and are zeroed before each conv; masked padding gave 17/18 identical pieces in the old
  ORT pipeline).

**Experiment switches** in `experiments/burn` `burn-enc`: `LENS=300,700,…` (mel frames: 1st/2nd run time at each
length), `SWEEP=1` (pre-tune sweep), `MEMLOG=1`, `PERSIST=1`. Caches/logs in `.cache/tmp/burn/`: `env/` (filled
autotune cache from the sweep), `env-partial/`, `env-notune/` (older), `autotune.log`, `compile.log`, `sweep1.txt`,
`sweep2.txt`. To turn tuning back on: add `"autotune"` to the burn features in `experiments/burn/Cargo.toml`.

### Tuning in the app: one-time optimization + short sweep (2026-11-20, decided and measured)

- **Tuning is on** (Cargo feature `autotune`, default; `--no-default-features` builds without). Encoder ~40% faster
  on the RTX 3050, 2.1x on the AMD iGPU. Level: CubeCL default `balanced` (sizes rounded up to powers of 2 per
  dimension; it also stops trying candidates once one reaches 80% of estimated peak; `minimal` = powers of 2.5 + 60%).
- **App**: when idle and `!asr::swept(kernels)`, the tray runs `Asr::sweep(false)` once per GPU (state
  "Speeding up dictation for <GPU>… N%", notifications say it is one-time, for this GPU, and heavier than dictation).
  Then `mark_swept` writes `kernels/<GPU>/swept` = build key (+ " tuned"). GPU menu (Automatic = wgpu high
  performance, or a named adapter) restarts the app (device and cache folder are fixed per process).
- **GPU asleep matters more than tuning on NVIDIA**: tuned encoder 378–401 ms at P5 (memory clock 810 MHz) vs
  99–132 ms awake; untuned awake 167–210 ms. Live mode keeps it only partly awake (see "Live mode and the iGPU").
- **80 ms padding** (`pipeline::Model::mel_tensor`): the mel is padded with zero frames to 8k+1 frames (whole
  encoder frames). Before, takes ending partway into an 80 ms block gave the subsampler odd sizes = new kernels
  compiled during takes (+0.3–2 s on 14.7/31.9/41.3 s takes even after the full sweep). Same results: the subsampler
  masks past the real length (`valid_lengths` from the true sample count), valid frames read only up to the last real
  one; the conformer gets the true frame count as before. 48 real takes: texts identical to unpadded.
- **Pause head lengths**: takes > 120 s are scanned in 120 s blocks, the last block can be short; a take <= 30 s
  never runs the pause head, so the sweep never tuned it at short lengths (126.3 s take: +2.6 s). The sweep now also
  runs the pause scan alone (`Model::scan`) at the short lengths, and at every second 31..120 s (the 41/60/120 s
  points alone missed ~1 s of compiling on 41–52 s takes).
- **Short sweep** (`dictation sweep`, the app's): 220 of the 375 lengths <= 30 s (every 80 ms up to 1.92 s, every
  0.4 s, every 80 ms just after 32, 48, 64, 96, 112, 128, 192, 256, 288, 320 frames and in the last 1.2 s; the spots
  where a from-scratch full sweep showed new work), each as full model + pause scan; pause scan at 30.08 s (live
  window end), 31..120 s and 240 s. Every tuning group (>= 5 frames wide) contains a grid point, so a missed spot on
  another GPU can only cost a small one-time compile, not a worse kernel. `dictation sweep full` = every 80 ms + full
  31..120 s takes (for checking).
- **Measured** (fresh cache folders under `.cache/tmp`, then 48 real takes twice with `check`; "no compiles" =
  kernel cache file not written during the takes):

  | | sweep time | compiles in real takes | encoder sum, 48 takes |
  |---|---|---|---|
  | NVIDIA, full sweep, before padding | 265 s | ~2 s (3 takes) | |
  | NVIDIA, short sweep, before padding | 81–87 s | ~6 s | |
  | NVIDIA, full, padded + pause head | 151 s | none | 9209 ms |
  | NVIDIA, short, padded, old 5 pause lengths | 86 s | ~1 s (41–52 s takes) | 9225 ms |
  | NVIDIA, short, final | 90 s | none (pass 1 18.2 s, pass 2 19.0 s) | |
  | AMD iGPU, short, final, on top of old cache | 224 s | none (122.6 vs 122.7 s) | |
  | AMD iGPU, short, final, from scratch (in the app) | 268 s (old full: 775 s) | | |

  The NVIDIA/Mesa driver shader caches (outside the project, not cleared) were warm for most runs, so from-scratch
  times on a new machine are likely longer (NVIDIA full sweep 265 → 151 s: that, plus padding removing the
  off-grid pieces of its 31–120 s takes; not separated).
  iGPU speed unchanged by padding (10.75 s take 858 ms, 22.85 s 1334 ms). All 48 texts identical on both GPUs.
- **Not done**: the "tuned" marker holds only the build key, not a sweep version, so a changed sweep doesn't
  re-tune existing caches (caches were deleted by hand this time). Progress weights overweight the warm-up.
  Group-only sweep (one run per tuning group, ~70 runs) not tried; would save little now.

### Live mode and the iGPU in use (2026-11-20, measured in the app + `check`)

- **How live mode works** (`src/asr/live.rs`): every ~1 s the current window [start, now) is re-transcribed; words
  two passes in a row agree on (after the last pasted word) are pasted, never taken back. At 30.08 s of window audio:
  pause scan on exactly the first 30.08 s, cut at the latest pause before 30 s, final pass to the cut pastes the
  rest, the next window starts at the cut. A pass is queued only when the previous one finished, so a slow GPU just
  runs passes back to back (delay grows, nothing piles up; a window never exceeds 30 s). Delay: 1-2 s per word.
- **Quality**: a word is pasted with only ~1-2 s of audio after it (normal mode: the whole take), so live mode is
  somewhat less accurate (e.g. a filler at the start came out as Cyrillic "Соуэ" and stuck). Not measured (`check`
  with LIVE=1 vs normal on the saved takes would). Tray now says so under "Paste while speaking".
- **Rewriting a window when it ends (tried, reverted)**: at the window's end, backspace over what differed and paste
  the window's full-context text. Correct in `check` replay (edits reproduced the text on 5 takes, 35-92 s), but in
  the app, typing into Zed failed twice: 216 backspaces at once lost 21 (likely the kernel's 64-event buffer per
  reader of a small uinput device); paced at 1/ms, all backspaces landed but the paste after them never appeared
  (cause unknown). Blind editing of another app's text can't be checked and fails per app: dropped. Also every
  window end changed something (70-270 chars retyped), and the final pass is not always better ("130 to to 200").
- **Window lengths** (27 takes over 30 s, 31 min): live windows (not the last) median 25.8 s, 7 of 60 under 20 s;
  full-mode pieces median 24.0 s. Same cuts as full mode in 22 of 27 takes (within ~0.1 s); the rest differ by a few
  seconds (full mode's pause scan sees 120 s blocks, live mode only the window's first 30.08 s).
- **Windows while recording for normal takes (tried, reverted)**: transcribe each 30 s window as it fills (live
  mode's window end, no pasting), so at stop only the last window is left. AMD: wait after stop 0.5-1.3 s for any
  length (whole take: 2-12 s). But on the 5 reference takes 68 vs 63 errors of 804 words (worse on 3, equal on 2,
  never better), most likely the cut points (no look-ahead). User: keep the whole take at once.
- **NVIDIA asleep during live mode**: one pass per second does NOT keep it fully awake: it still drops to P5 between
  passes; slow passes (287 ms) line up with P5, the final pass at full clock: encoder 99 ms. Non-live takes: P0 but
  core ~907 of 1897 MHz -> ~1.3x slower than awake. A tiny keep-alive kernel was already tested and doesn't help
  ("Latency in gauge"). Without root, `nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'` (prefer max performance)
  is queryable here (reads 0); not tested whether setting it holds.
- **AMD iGPU (Renoir, 7 CUs), tuned**: takes 3-8 s 0.54-0.60 s, 12-20 s 0.85-0.95 s, 24-29 s 1.34-1.40 s; 41 s take
  2.39 s (NVIDIA 0.46 s). Clock pinned at 1800 MHz, ~97% busy: compute-bound. Steps follow the matmul cost in the
  tuning db (q8 1024x4096 matmul: 64 rows 3.6 ms, 128 3.7, 256 5.0, 512 7.6 ms: short takes leave most CUs idle,
  ~0.5 s floor). Live mode on AMD works: passes 0.5 s (early in a window) to 1.43 s (near 30 s), back to back past
  ~21 s; window switch 1.5 s. `.cache/tmp/tunedb.py <default.db>` decodes the autotune picks (CBOR).

## burn pipeline: full Rust port of transcribe.cpp's Parakeet path (2026-11-20, `experiments/burn/`)

Ported from transcribe.cpp (MIT; files keep its copyright line): `mel.rs` (transcribe-mel.cpp, NeMo path),
`decoder.rs` (parakeet/decoder.cpp TDT greedy), `tokenizer.rs` (transcribe-tokenizer.cpp), `longform.rs`
(parakeet/longform.cpp + run_longform), pause head `Vad` in `encoder.rs` (build_vad_graph), `pipeline.rs` (order of
steps from parakeet/model.cpp). `burn-enc check <gguf> <refdir>` compares every stage with the dumps;
`burn-enc wav <gguf> <wav>…` transcribes (env `SEGS=1` prints the cut points, `F16=0` f32 activations).

Details that mattered (all read from transcribe.cpp, not guessed):
- **Mel padding is "constant" (zeros), not reflect** (`model.cpp` sets `cfg.pad_mode = "constant"`, as NeMo's
  FilterbankFeatures): normalize stats over the first n/160 frames, the last frame set to 0. With reflect our mel
  was 2% off (all in the last frame); now rel. err 2e-7.
- **Kestrel length masking** (`stt.parakeet.encoder.length_masking = kestrel`): valid length n/160 (one less than
  the mel frames); zero frames past it after relu0, conv2 and conv5 of the subsampler (stride-2 output lengths
  (L-1)/2+1); encoder output cut to the valid length (= their key-padding + conv masks). Subsampler convs in f32.
  This also explained most of the old "2% at pre-encode": now 0.3% (the rest is the same in f32 and f16, so it's
  not our precision; probably ggml's int8 activation rounding, unproven). enc.final 0.76% (f16) / 0.62% (f32).
- Decoder: start state = zero embedding; LSTM gates i,f,g,o; joint relu; **global symbol budget** (max_symbols × T
  steps in total, a blank with duration 0 advances one frame). On transcribe.cpp's own encoder output our LSTM /
  joint match to ~1e-7 and the text is identical.
- Weights: decoder int8 (Q8_0 blocks as in the GGUF, ~13 MB, stays in CPU cache) with an explicit AVX2 matvec
  (runtime-detected; plain fallback). f32 weights were memory-bound (21 MB joint matrix > 16 MB L3): ~200 ms;
  int8 with auto-vectorization was worse (3.5 ms/step); AVX2 intrinsics: **0.6 ms/step, 64–116 ms per 25 s**
  (transcribe.cpp's decoder 116–138 ms with 4 threads). The joint's encoder projection runs on the GPU.
- Pause head works on the **subsampler output only** (not the full encoder): mel → pre-encode → k1 conv 1024→128,
  silu, k5 conv, silu, k1 conv → sigmoid; per 120 s scan block. Cutting rules ported 1:1.

**Results** (Q8_0 weights int8 on GPU, f16 activations, no autotune, RTX 3050 Ti):
- 25 s reference piece: text identical in f32; in f16 only the cut-off last word differs ("onx" vs "on").
- **WER on the 5 reference takes (804 words): burn 7.84% vs transcribe.cpp 7.71%**; 4 takes word-for-word
  identical, 1 word differs ("energy bars bars" vs "energy bars,", take 203034; same in f32; cut points identical
  to transcribe.cpp's → a borderline decode from the small encoder difference).
- Speed, 25 s piece: mel 8 ms + encode step ~170 ms (encoder 105–165, varies run to run) + decoder 65–116 ms ≈
  250–300 ms vs transcribe.cpp 219–240 ms.
- Speed, 68–126 s takes (warm): **1.0–2.3 s vs transcribe.cpp 0.7–1.5 s**. 2 min take ≈ pauses 0.3 s + encoder
  0.95–1.3 s (untuned) + decoder 0.5–0.6 s. First take of new lengths compiles kernels once (+0.5–2 s).
- Load: device 0.3 s + model 1.4–1.6 s (from the GGUF; now 0.8 s from our converted file, see "Model load" below).
- Open: "NVVM compilation failed: 3" (NVIDIA driver failed compiling one shader). First seen once during the
  first pass over the 5 takes; with the chunked pause scan it prints in every 5-take run (not for 222256 alone,
  not before chunking); results unaffected (cuts + text identical). Which kernel: unknown.

**Sanity check 2026-11-20 (after tuning + padding + app integration)**, 5 reference takes, 804 words: app path (Q8_0
int8 weights, f16 activations, tuned) 63 errors = 7.84% on NVIDIA and on AMD, same as right after the port;
f32 activations + weights dequantized to f32 (`F32=1 DEQ=1 KERNELS=<copy> cargo run --release --no-default-features
--example check`): **texts word-for-word identical**. So f16 and the int8 kernels cost nothing here. Found on the way:
the f32 path was broken (position table and the rel_shift zeros were created in the device's default f16).

### VRAM vs transcribe.cpp (2026-11-20, nvidia-smi per process, sampled every 50 ms)

| MiB | burn | transcribe.cpp |
|---|---|---|
| model loaded | ~770 (runtime 38 + weights 719 MB in use) | 709 |
| 26 s take (one piece, no pause scan) | 942 | 843 |
| 5 long takes (68–126 s), before fix | **1632** (pool stays at the peak) | 1259 |
| same, subsampler chunked in the scan | **1015–1044** (944 after cleanup) | — |

Earlier "845 / 785 MB" and transcribe.cpp "720–730" were short-run / idle numbers. Encoder bench (25 s piece):
device 38 → weights 765 → after a run 919 (pool reserved 898 MB: +170 MB work buffers) → `memory_cleanup()` 826
(84 MB of empty pages stay reserved).

**The spike was the pause scan**: the subsampler (pre-encode) ran in f32 over a whole 120 s scan block; its first
conv output alone is 6000 frames × 64 × 256 ch × 4 B ≈ 390 MB, next stage similar → ~+600 MB (transcribe.cpp ~+400).
The 120 s block comes from kestrel (transcribe.cpp ports kestrel 0.9.1 segment.py / vad.py 1:1, cut points sample
for sample) and only matters for the mel normalization (per feature over the block). **Fix**: mel still over the
whole block; the subsampler runs 3000 mel frames (30 s) at a time with a 32-frame margin, chunk starts on multiples
of 8, masks with global indices (output frame j depends only on mel 8j-7..8j+7) — `PreEncode::fwd_chunked`,
`Model.scan_chunk` (env SCAN_CHUNK, 0 = old). Speech probabilities **identical (max |diff| 0)** on all 5 takes,
cut points + text identical, pause-scan time unchanged (~300 ms per 120 s block). `memory_cleanup()` after each take
(`Model.cleanup`, env CLEANUP) frees ~70 MB more, no measurable time cost.

**Subsampler in 5 s chunks everywhere (2026-11-20)**. nvidia-smi per process every 50 ms, `UNLOAD=1` check, pool
readout `asr::mem_report()`. One take per process, whole piece: peak 785 (4 s) / 835 / 895 / 953 (30 s) / 987 (215 s)
MiB, loaded after it 764-854, after unload 57-147. The pool keeps one page sized to the largest single buffer =
the subsampler's first conv output in f32 (256 ch x 64 freq per 20 ms: 98 MB for a 30 s piece), two at the peak.
**All 8 takes (4-215 s) in one process** (as the app does): peak **1092** -> **951** MiB with `PRE_CHUNK` = 500
mel frames (5 s; 10 s: 983), loaded 951 -> 900, after unload 358 -> 217, largest buffer 100 -> 18 MB. Texts
identical (8 takes NVIDIA, 3 AMD), speed same (NVIDIA encoder 115 vs 118 ms on 30 s; AMD 10/30/68 s 626/1396/3951
vs 621/1404/3982 ms). The sweep key now has a shapes version (`asr::SWEEP_KEY`), so the app re-tunes once.
transcribe.cpp for comparison: 843 on a 26 s take, 1259 on long takes.
**Memory outside the pool** (`MEMSTEP=1` check: VRAM + pool after every take): the same 30 s take x8 stays flat
(773 MiB, 66 after unload); 8 different lengths grow in jumps on the long takes (+12/+22/+112 MiB, more in a 2nd
round, pattern differs run to run) while burn's pool (staging + uniforms included) stays at 749 MB: after unload
217-289 MiB. Not wgpu's allocator either: on Vulkan CubeCL allocates its buffers itself (one vkAllocateMemory per
page, freed with it; `cubecl-wgpu backend/vulkan.rs create_storage_buffer`) and wgpu's own allocator report holds
only ~20 MB. So it is inside the NVIDIA driver (compiled pipelines, command/descriptor memory, ...), freed only
with the Vulkan device, i.e. the process. Only fix: run the model in a worker process (also frees the ~38 MiB base).
Long run (8 lengths x 10 rounds, one process): between takes 764 -> 900 -> 926 -> **1094 MiB by round 3**
(jumps of ~110-120 MiB on the 126/215 s takes), then flat for 7 rounds; after unload 387. So after many takes the
driver part (~330 MiB) outweighs what the 5 s chunks saved. Left as is (user, 2026-11-20); the idle unload frees the
model but not this, only a worker process would.
AMD iGPU (`MEMSTEP=1`, /proc/self/fdinfo drm-memory-vram + gtt): 8 lengths x 2 rounds: 765 -> 786-788 MiB, flat in
round 2 (~100 VRAM carve-out + rest GTT): no such build-up there.

**Kernel cache stuck read-only (2026-11-20)**: from ~18:49 every process (app, check) opened both GPUs' caches
read-only, so nothing was saved (the app's re-tune "optimized in 35 s" was lost, `swept` written anyway). Only
visible as WARN "database is readonly" in the app log; `LOG=1` check shows the cause: "short read on WAL frame ...
got 0": `default.db-tshm` (Turso's WAL index) listed frames of a WAL that was empty. Likely left by our experiments
(several processes on one cache, runs stopped midway). Fix: app closed, `default.db-tshm` deleted (rebuilt on open,
all 2272 entries kept) + NVIDIA `swept` deleted -> re-tune 34 s, saved. Possible guard (not done): on a read-only
open at start, delete the -tshm and reopen.

### Model load (2026-11-20): 1.57 s -> 0.80 s, GGUF converted once into our own file

Profiled `Model::load` (device init 0.25–0.3 s not included):
| step | before | |
|---|---|---|
| read the GGUF into a Vec (`read_to_end`, page cache) | 260–450 ms | gone: mmap |
| 48 conv pointwise weights (f16 in the GGUF, 151 MB): f16->f32, strided CPU transpose, f32->f16 | **865 ms** | gone: uploaded as stored [out, in], transposed as a view (like Q8_0) |
| Q8_0 unpack (scale + 32 i8 interleaved -> separate arrays) + burn's `TensorData::quantized` repack | ~240 + 150–300 ms | gone: stored in burn's layout |
| the upload itself (~718 MB) | — | **~740 ms = all that's left** |
Step 1 (f16 as stored) alone: 1.57 -> 1.11–1.22 s. With the converted file: **0.80–0.84 s** (+ device 0.25 s ≈ 1.05 s;
transcribe.cpp ready 0.95–1.09 s). Accuracy unchanged (every check number identical, 5 takes: cuts + text identical);
encode step 167–171 ms (same). The transposed-f16 matmuls are new kernels: compiled once per length (cache).

**Upload is ~1 GB/s and that's the floor**: one 740 MB tensor straight from the mmap takes 775 ms (forced with a sum;
reading back 4 elements does NOT wait for the whole upload — earlier "12 ms" was that artifact). Pre-faulting the
mmap (`populate`) changes nothing; per-call overhead isn't it (median upload call ~0 ms, the queue stalls 30–60 ms at
random calls when it flushes). GPU idle at P5 with PCIe link gen 2 (max 3) x8 — likely the limit; driver's business.

**Weights file** (`weights.rs`, `burn-enc convert <gguf> <out>`, 1.8 s, 740 MB = same size as the GGUF): magic
"PQWEIGHT", format u32, JSON header (GGUF metadata + tensor table), data 4096-aligned, tensors 64-aligned. Kinds: f32,
f16 (as in the GGUF), q8 (burn's quantized bytes: int8 values, then f16 scales per 32, aligned 4 — the GGUF's Q8_0
de-interleaved; the converter checks the round trip). Loaded with memmap2 + `Bytes::from_shared(.., File)` (zero copy;
burn's `shared-bytes` feature is on by default). `LAYOUT` string names burn's q8 layout: a mismatch refuses the file
-> the app would re-download the GGUF and convert again (plan: keep only the converted file, delete the GGUF).
The CPU decoder reads its Q8 weights from the same q8 bytes.

## App design discussed (2026-11-15)

Always-on small app (hotkey, recording, typing) + model in a **separate worker process** (only way to free all
VRAM/RAM: ORT doesn't free in-process). Load on hotkey press **while recording** (startup ~3 s hides under speech
> 3 s); unload after idle timeout (~5–10 min); optional later: unload when a game/other GPU app starts, shorter
timeout on battery, manual "unload now".

## Scripts in .cache/tmp (deleted 2026-11-20)

gpu_opt.py (decoder variants), fast_prof.py / step_prof.py / step_parts.py (per-call profiling), enc_prof.py /
enc_scale.py (encoder ops / scaling), enc_fuse*.py (failed fusion), first_run.py / idle_run.py / gpu_sleep.py /
keepalive.py (latency), fixed_cut.py / vad_cut.py / vad_show.py / vad_plot.py (cutting), pad_check.py (padding
identity), gauge_wer.py / gauge_check.py (pipeline WER), mem_now.py / mem_after.py / workmem_now.py / pool_stress.py
(memory), load_time.py / evict.py (startup), ram_steps.py / ram_cuda_base.py / ram_map.py / ram_parts.py /
ram_ops.py / ram_variants.py (RAM breakdown), score_take.py (long take vs LibriSpeech refs), pairs/ librispeech/ cuts.json
(test sets), speech_analysis.py / ts_check.py / pause_text.py (speech analysis).

### Tray: menu described once, tray on the main thread (2026-11-20)
`src/tray.rs`: platform-neutral menu (`tray::Item`: Label, Separator, Action, Check, Radio, Sub; clicks are closures
sending events) + `tray::Model` (icon, tooltip, menu, left click, menu opening); `daemon::Tray` implements it.
Backend Linux = ksni (D-Bus, KDE draws the menu, no GTK). `tray::run` keeps the tray on the main thread and runs the
app (tokio runtime, engine thread) on a thread "app": macOS requires on-screen things on the main thread.
Mac/Windows tray (2026-11-20): tray-icon 0.26 + muda on a tao 0.37 event loop (`run_return` on the main thread;
updates arrive as user events through the loop proxy). Radio = checkmarks; menu rebuilt
only when its text/states change; menu_opening = mouse down on the icon; left click = activate, right
click = menu (also macOS, like Linux); macOS: no Dock icon (Accessory). Type-checked for Windows only (tray.rs + icons.rs as a mini crate,
`.cache/tmp/traycheck`), never run; macOS not compiled at all (no target installed). Rest of the plan: global-hotkey,
burn Metal (Mac); no extra system packages anywhere (tray-icon on Linux would need libayatana-appindicator -> ksni stays).
Paste (2026-11-20): Linux keeps uinput (evdev, X11 + Wayland), macOS/Windows use enigo 0.6.1 (macOS: always Cmd+V,
no Insert key there). Not compiled for Windows: `cargo check --target x86_64-pc-windows-gnu` stops earlier, at
turso_sdk_kit's build script (needs a Windows resource compiler). Adding enigo changed Cargo.lock -> the app re-tuned
(build key = Cargo.lock + rustc; narrowing it to the kernel crates was offered, user: leave it).
Tray icons (2026-11-20): drawn in code (`src/icons.rs`: coloured disc + white symbol, 4x4 supersampled, any size;
ksni gets 22/32/48/64 px pixmaps), same on every platform. Menu items have no icons (removed 2026-11-20: theme names only work on Linux; user chose a plain menu everywhere over drawing 9 more). Notification icons stay theme names on Linux only (macOS can't show per-notification icons at all; Mac/Windows show the app icon).
GPU API: Vulkan everywhere except macOS = Metal (see below).
Notifications (2026-11-20): Linux unchanged (show_async, replace previous by id, transient, 4 s, theme icon: all
XDG-only). macOS/Windows: plain title + body via blocking show() in spawn_blocking; they stack in the notification
center and use the app icon. Windows labels them "Windows PowerShell" until the app has an app ID (needs a Start-menu
shortcut, i.e. an installer); user: leave the Windows things for now. Type-checked for Windows (mini crate).
Checked: menu over D-Bus identical item for item, clicks change config.toml, Quit exits cleanly.
Timings in the menu (2026-11-25): while recording the status line counts the take's length live ("● Recording
0:12", from the start press; a 1 s task redraws the tray while `rec_since` is set); after a take a line "Last take:
8.2 s, transcribed in 170 ms" (stop -> text ready, before the paste; same number as the log). Not checked whether
macOS redraws the open menu every second (it may only update on reopening).

## Open items / next steps (2026-11-20)

1. Slow takes after idle: the GPU asleep (P5, see "Tuning in the app"); first-use compiles are gone (padding +
   short sweep). Open: keep the GPU awake while recording (non-live)?
2. Load policy: keep loaded vs load per take; unload leaves 129 MiB VRAM (needs a worker process to free).
3. Global shortcut: portal needs `~/.local/share/applications/local.dictation.Dictation.desktop` (outside project, ask).
4. "NVVM compilation failed: 3" message in multi-take runs (outputs unaffected).
5. Binary 870 MB with debug info (152 MB without).
6. Hotwords / replacement rules for recurring tech terms.
7. More uk reference recordings (`ref_uk_mixed.txt`, `ref_en_vowels_a.txt` not recorded yet).


### Global shortcut works on KDE Wayland (2026-11-20)
It never worked before: the portal refused with "App info not found for 'local.dictation.Dictation'" (KDE identifies
the app by its .desktop file, we had none). Now the app writes `~/.local/share/applications/local.dictation.Dictation.desktop`
(Exec = its own path, quoted; Icon audio-input-microphone) when missing or different, then registers; right after
writing, KDE needs ~1 s to notice (first try fails, a 1 s retry works). First registration shows KDE's "Global
Shortcuts Requested" dialog (our preferred Meta+Y was not pre-filled: shows None). "Set shortcut…" in the tray opens
System Settings -> Shortcuts with Dictation selected; several keys per action allowed, KDE warns about conflicts with
other global shortcuts. Checked: Ctrl+I bound there -> toggles recording. The .desktop file also adds a launcher
entry and stays if the program is deleted (in-app Uninstall discussed, not built).

### Global shortcut on macOS/Windows (2026-11-20, not run there yet)
No system place for apps' global shortcuts there (first app to ask gets the key), so the app picks it:
`shortcut` in config.toml (default Ctrl+Alt+Space; not written on Linux, where the portal/KDE owns it) and a tray
submenu "Shortcut (…)" with Ctrl+Alt+Space, Ctrl+Shift+Space, Ctrl+Alt+D, F9, None (+ the config.toml one if
different). Registered by global-hotkey 0.10 inside the tray backend (needs the same main-thread loop);
tray::Model::hotkey / hotkey_pressed / hotkey_result. Taken or invalid key -> notification "Shortcut not set",
menu shows none. Linux unchanged (portal, KDE dialog). ashpd and ksni are now Linux-only dependencies.
Type-checked for Windows (tray mini crate); daemon.rs's non-Linux branches are not compiled anywhere yet.

### Control socket on every OS (2026-11-20)
`dictation toggle` + the "already running" check now use interprocess 2.4 local sockets (tokio): Linux/macOS the
same socket file (XDG_RUNTIME_DIR, else the per-user temp dir: macOS $TMPDIR), Windows a named pipe
"dictation-<USERNAME>" (pipe names are machine-wide). Leftover socket file is replaced (try_overwrite). Checked: new
client vs the running app -> "already running"; Windows type-check of the same code (mini crate). The new listener
runs after the next app restart.

### macOS GPU: Metal (2026-11-20, not compiled or run on a Mac yet)
burn's `vulkan` option = wgpu plumbing + CubeCL writing SPIR-V, passed straight to Vulkan (no WGSL). macOS has no
Vulkan: burn's `metal` option (CubeCL writes MSL, passed straight through). The generic `wgpu` option (WGSL, translated
by naga, adds bounds checks, no tensor-core/cooperative-matrix path) was not chosen; generic vs native never measured.
Cargo.toml: burn "vulkan" on cfg(not macos), "metal" on macos; asr::device() / asr::BACKENDS pick it. Linux build
unchanged (burn not recompiled). Unknown until tried on a Mac: our Q8 int8 + f16 kernels on MSL, autotune, speed.

### Decoder NEON path (2026-11-20, never run)
`decoder.rs q8_neon`: same as q8_avx2 for aarch64 (Apple M chips etc.; NEON is always there, no runtime check):
8 int8 -> 2x f32x4, vfmaq, per-block scale, vaddvq. Type-checked for aarch64-unknown-linux-musl in a mini crate
(`.cache/tmp/neoncheck`, a planted error was caught, so the code is really checked); no ARM machine/emulator here, so
neither correctness nor speed is tested. AVX-512 skipped: this CPU lacks it, decoder is ~6 ms/take, memory-bound.

### Paste unavailable is no longer fatal (2026-11-20)
Before: no /dev/uinput access -> the app refused to start. Now: the virtual keyboard is optional; each paste retries
making it (permission may have been granted since). If it fails: notification "Not pasted: the text is on the
clipboard" + how to fix (output::PASTE_HELP): Linux = udev rule KERNEL=="uinput", TAG+="uaccess" + log in again;
macOS = allow in Privacy & Security -> Accessibility (enigo shows the system prompt once at start, not on retries).
Clipboard failure also gets its own notice now (was a silent "Pasted"). Checked: Linux build, Windows type-check of
the enigo side. Not tested: the failure path itself (this machine has uinput access).

### First run on a Mac (2026-11-20, MacBook M1, 8 GB, macOS 26)
Cold build 11 min 49 s (`-j4`; no systemd-run on macOS, so no memory cap; Zed's rust-analyzer took 7.7 GB indexing
in parallel, stopped). Everything compiled for macOS first try (Metal, NEON decoder, tao/tray-icon, global-hotkey,
enigo, CoreAudio); one warning: `futures_util::StreamExt` unused off Linux. Model download + convert worked
(`~/Library/Application Support/dictation/`), Ctrl+Alt+Space registered, one-time tuning runs on Metal (GPU ~91%).
Fixed: (1) first notification popped "Where is use_default?" (mac-notification-sys asks AppleScript for an app of
that name when none is set) -> `set_application("com.apple.Finder")` once (own name needs an .app bundle);
(2) right-click menu closed right away during tuning: every progress change replaced the menu (`set_menu`), which
closes an open menu on macOS -> same shape = texts/states/clicks changed in place (Mac + Windows backend);
(3) tuning hint split over two lines (one long line made the menu too wide, all platforms).
Tuning on the M1: 115 s. First take: encoder 264 ms, decoder 23 ms (length not noted).
(4) Paste hung the app ("Transcribing…" forever): enigo's `Key::Unicode('v')` looks up the keyboard layout
(TSMGetInputSourceProperty), which macOS allows only on the main thread; our paste runs on the app thread ->
`dispatch_assert_queue_fail`, thread stuck. Now Cmd + raw key code 9 (kVK_ANSI_V), no lookup; Ukrainian layout fine
(Cmd shortcuts use the Latin key positions), a layout that moves V (Dvorak) would paste the wrong key. Windows keeps
Key::Unicode (no main-thread rule there), Linux uses uinput raw codes.
Paste works after the fix (user dictated into Zed for the rest of the session).

### Mac: speed varies per take = Metal kernels compiled per process (2026-11-20, M1, measured)
App takes: 140–450 ms usually (4.2 s take 162 ms, 7.9 s 455 ms), but 0.7–1.7 s for some (5.1 s 1694 ms, 19.1 s
1290 ms), mostly right after app starts. Decoder always 9–51 ms: it's the encoder. `check` with new `GAP=<s>` (idle
before each take), takes A = 19.1 s, B = 4.2 s, fresh process each run:
| | A | B | A | B | A | B |
|---|---|---|---|---|---|---|
| back-to-back | 2249 | 1352 | 403 | 114 | 394 | 112 |
| 10 s idle before each | 1813 | 1008 | 443 | 170 | | |
(encoder ms). Idle costs little (+40–60 ms); **first run of each length in a process** costs ~1–2 s. Autotune results do
load from disk ("Loaded 160 autotune cached entries"), compiled kernels don't: CubeCL 0.11.4's kernel disk cache is
SPIR-V only (`cubecl-wgpu/src/compute/server.rs:162`, store "vulkan"; other paths `CompilationCache::unbound()`,
in-memory). The MSL path writes kernels anew per process. wgpu 30's PIPELINE_CACHE is Vulkan-only (Metal
`create_pipeline_cache` is an empty stub), WGSL path saves nothing either. macOS's own Metal cache
(`$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal/`, libraries.data 146 MB, written during our runs) most likely already
keeps Apple's compile step, so the ~1 s is probably CubeCL's side (building kernels / writing MSL) — **not verified**
(profile a first-time take to split CubeCL vs Metal compiler). Within one app run each length compiles once (kernels
belong to the device, survive model unload); app restarts start from zero.
Options: precompile all sweep lengths at start (user: no); coarser padding (fewer lengths, slower takes); save MSL to
disk in a patched `cubecl-wgpu` (`[patch.crates-io]`, crate 752 KB / 12.7k lines, change a few dozen lines; or upstream PR).
Undecided; measure the split first.
**Measured and fixed 2026-11-21, see "Mac kernel cache fixed" below: the guess above was wrong, most of the time was
Apple's compiler, re-run every process because CubeCL's MSL text differed per run.**

### Mac memory (2026-11-20, M1, unified memory: `footprint` = all of it, GPU included)
App: model loaded **841 MB** (712 MB GPU buffers, 107 MB heap); unloaded **154 MB** (27 MB GPU, the same 107 MB heap:
probably in-memory kernels + allocator; malloc_trim is Linux-only; not examined). Peak while transcribing (`check`,
`/usr/bin/time -l` peak footprint, takes built by repeating saved takes): 30 s 847 MB, 60 s 872, 120 s 900, 240 s 913.
Flat-ish since > 30 s is cut into ≤ 30 s pieces. System stayed at 54–55% free.
Open: two app takes came out empty (14.0 s and 6.2 s, text ""): silence or a bug not checked (wavs in recordings/).

### Mac tray icon (2026-11-20)
macOS: the symbol alone (no disc), zoomed to fill the menu bar height, black + alpha as a template image
(`Icon::glyph_rgba`, `set_icon_templated`: macOS tints it for light/dark bars); Recording = red symbol, not a template.
Linux/Windows unchanged (coloured disc, `set_icon`). tray-icon's `set_icon_with_as_template` is deprecated and a
no-op off macOS, so not used.

### Transcribing files (2026-11-20)
`dictation transcribe <file>...` → each file's text on stdout (`# name` line before each when several); errors on
stderr, exit 1 if any file failed. If the tray app runs, the command sends it the file path over the control socket
(`transcribe <abs path>` → `{"text"|"error"}` line, no reply timeout); the app decodes and runs it on its engine
(loads the model if unloaded; refuses during tuning / with no model). Otherwise the command loads the model itself.
`src/media.rs`: symphonia 0.6 (`all` features) reads the first audio track — wav, mp3, flac, ogg/vorbis, m4a/aac, alac,
aiff, caf, sound of mp4/mov/mkv/webm; channels averaged; rubato 5 FFT resampler to 16 kHz while decoding (a 1 h file
holds only its 16 kHz copy, 230 MB). Not supported, clean error: AC-3, .avi (user: no need for niche files).
Opus (.opus, most .webm, voice messages; added 2026-11-20, user: Rust only, no C library to build): symphonia has no
Opus decoder yet → `symphonia-opus` 0.1.3 (pure Rust, MPL-2.0, deps only log + thiserror; registered at
Tier::Fallback so a future symphonia decoder takes over). Compared on en.opus against Apple's decoder (`afconvert` →
f32): 133.8 dB SNR, i.e. identical up to rounding; it keeps the 312-sample (6.5 ms) encoder pre-skip at the start
(symphonia 0.6 has no gapless trimming) — harmless. The alternative `symphonia-adapter-opus-rs` (ruopus, MIT) also
matched (107 dB) and trims pre-skip, but pulls in a spectrograms crate (ndarray, regex) via non-optional default
features. en.opus / en_opus.webm through the app: same text as en.flac; re-tune after the Cargo.lock change 140 s. Long files go through asr's long-take path (pieces ≤ 30 s), same as long takes.
Measured on the M1 through the app: 7 s files 0.2 s per call (wav 16/22 kHz, mp3, m4a, flac, mp4, mkv: right text);
240 s m4a (48 kHz stereo) 6.4 s, the CLI process itself 7 MB. App not running: 6.3 s for two short files (model load +
kernel compiles). Adding the crates changed Cargo.lock → one re-tune at the next app start (107 s here).
Windows not tried: the binary is a GUI-subsystem app, so output in a console probably needs `> file.txt`.
Tray menu "Transcribe a file…" (user: "just make it into the app"): rfd 0.18 file picker (macOS NSOpenPanel via the
tao event loop, Linux xdg portal, Windows native), several files allowed. All are transcribed, then the app asks where to save (user, 2026-11-20: "let user choose
destination", after transcribing, as "it also helps to signal when it's done"): one file → save dialog preset to its
folder + `<name>.txt` (the dialog asks before replacing); several → one folder picker, each `<name>.txt` there
(`<name> (2).txt` if taken, never overwritten); cancel → not saved. Failed files are reported as they fail. Then
opened in the default editor (.srt: shown in the file manager), notification.
Dialogs are rfd's blocking ones (in spawn_blocking; rfd puts them on the main thread itself), not AsyncFileDialog: on
macOS the async ones without a parent attach as a sheet to `app.windows().firstObject()`, in a tray app an invisible
window in the menu bar → the dialog hung from the top edge, no title bar, cut off at the right, unmovable (user's
screenshots). The blocking ones are free-standing windows (checked in a test program: normal title bar, movable). Same engine path (Ev::File) as the command.
Adding rfd changed Cargo.lock → another one-time re-tune (110 s). Picker not yet clicked through by the user.

### Mac notifications were never shown (2026-11-20)
Test: the app's notifications (notify-rust → mac-notification-sys, NSUserNotification posing as com.apple.Finder)
returned no error but never appeared; `osascript -e 'display notification …'` did (user saw only that one, twice).
Finder isn't even listed in com.apple.ncprefs. Workaround: macOS notifications through `/usr/bin/osascript` (title/body
as argv, no quoting), shown as Script Editor. Windows has the same root cause (no own app id → shown as PowerShell,
but displayed). Real fix for both: install as a real app (Dictation.app / Start menu shortcut with an AUMID).
Also seen: "sound Start: Error opening the stream with the OS" on several takes (start cue silent) — not looked at yet.

### Mac: what tuning gives (2026-11-20, M1, `check` tuned vs `--no-default-features`, each file twice per process)
Own cache copies in `.cache/tmp/tune/` (tuned = sqlite `.backup` of the app's cache; the app's cache untouched).
total ms, first run (compiles) / second run (warm):

| audio | tuned | untuned | warm speed-up |
|---|---|---|---|
| 7.2 s | 2114 / **193** | 1264 / 274 | 1.4x |
| 30 s | 2319 / **707** | 1628 / 1106 | 1.6x |
| 60 s | 4608 / **1493** | 3220 / 2998 | 2.0x |
| 240 s | 6879 / **5863** | 11861 / 12175 | 2.1x |

Tuning helps more the longer the audio (encoder and pause scan both; pause scan 323 vs 3220 ms on 240 s). First runs
of a new length are *slower* tuned (likely bigger fused kernels compile longer; not verified) — short takes pay that
once per length per app run. Same texts.

### Subtitles (.srt) (2026-11-20)
Tray "Transcribe a file" → "As text (.txt)…" / "As subtitles (.srt)…"; command `dictation transcribe --srt <file>`
(socket command `subtitles <path>`). The TDT decoder now returns each token's encoder frame (80 ms); `Tokenizer::words`
groups pieces into words (a word = from a U+2581 piece; punctuation pieces don't move a word's end: the model often
emits the full stop only when the next speech starts, which made cues hang on ~2 s); long audio: piece offsets added.
`src/subtitles.rs`: a cue ends at a sentence end (≥ 15 chars), a pause > 1 s, or before passing 2 × 42 chars / 6 s
(then cut after its last comma, so no lone word starts the next cue); ≥ 1 s on screen unless the next starts sooner;
2 lines split at the space nearest the middle. .srt saved next to the file and shown in Finder (not opened).
Checked: srt words == plain text words (240 s file, 57 cues). Timing vs loudness of en.wav: pause after "boy,"
3.24–3.72 s, cues end 3.52 / start 3.76; end of speech ~6.4 s, cue end 6.64; first cue starts 0.64 s, audible speech
~1.0 s (soft "The" may be under the loudness threshold; not checked further).

### Start beep never played with Bluetooth headphones (2026-11-20)
Log: "sound Start: Error opening the stream with the OS" on ~every take (31×). Full error chain: "Buffer size 2048 is
not in the supported range 14..=1024": rodio's `open_default_sink` asks for a fixed ~50 ms buffer (2048 frames at
44.1 kHz); Bluetooth headphones (default in + out) switches to headset mode as soon as its mic records,
and then allows at most 1024. Retrying didn't help (still refused after 1.5 s). Fix (`audio::play`): rodio's default
first, on failure open with `BufferSize::Default` (device picks) — Linux/Windows unchanged. 6 short takes after: 0 errors.
Also: no more "Pasted" / "Copied to clipboard" notification after each take (user: distracting); problems, "Nothing
recognized", file jobs and setup still notify.

### Parakeet vs Apple's own speech-to-text (2026-11-21, M1, macOS 26)
Apple has (`.cache/tmp/apple/stt.swift`, SpeechAnalyzer API, no permission prompt needed):
- **DictationTranscriber**: the Siri ASR model (`siri.asr.assistant.<locale>`, 1.2 GB asset + `siri.asr.hammer`). The
  **F5 key / system Dictation uses this same model** (checked with lsof while the user dictated: it opened
  `siri.asr.assistant.en_CA`; the user's dictation language is en-CA). en + uk.
- **SpeechTranscriber** (new in macOS 26, Sept 2025): a separate model (`speech.asr.transcription.en`, 392 MB),
  **English only** (no uk). Runs in the `localspeechrecognition` XPC service, which exits when idle. Memory while
  transcribing: own 28 MB (peak 46 MB phys_footprint) + ~200 MB of the model mapped from disk ≈ 250 MB; 0 when idle
  (app: ~840 MB loaded, 154 MB unloaded).
English takes (Bluetooth mic, `experiments/bench/ref_en_*.txt`), WER with number words = digits
(`.cache/tmp/apple/score.py`); other formatting ("int 8", "megabytes", "we will") still counts:

| take | Parakeet | SpeechTranscriber | Dictation (F5) |
|---|---|---|---|
| story 132 s | 2.2% | 2.2% | 6.7% |
| int8 (tech) 129 s | 6.1% (real ~1: "Sharpa") | 18.3% (int8→"an eight", onnx→"nx", CPU→"scipue"/"GPU") | 19.3% |
| notes 80 s | 2.1% | 3.6% ("Martha") | 7.9% |
| vowels_a 79 s | 5.7% | 9.8% (dropped "and a box of pens") | 9.8% |
| **all 637 words** | **4.1%**, 12.0 s | **8.9%**, 6.7 s | 11.5%, 15.3 s |

→ Plain English: a tie. Technical English: Parakeet far ahead. Apple ~1.8x faster, ~1/3 the RAM. Ukrainian tech take
(ref_uk_int8, 144 s): Parakeet 14.5% (~8 of the errors number style), Dictation uk-UA 32.4% (drops English terms
inside Ukrainian: CPU, FP32, ONNX missing; "encoder.int8.onnx … joiner.int8.onnx" → "and and").
LibriSpeech (public benchmark, `.cache/tmp/librispeech/`: run.py, score.py, results.json; 250 random clips per half,
seed 0, same normalization): test-clean Parakeet 2.29% / Apple 2.40% (clips better 33/31, 186 equal); test-other
3.42% / 4.47% (55/36); all 9697 words 2.80% / 3.34%. Most common differences alike for both (a/the, and/in, British
vs American spelling). Per short clip (5–20 s): Apple 325 ms (per-request startup), Parakeet 224 ms via the CLI.
Caveat: Parakeet was likely trained on LibriSpeech train (same audiobook domain); Apple's data unknown.
Apple's model files: readable but compiled (encoder = M1 Neural Engine program `model.hwx` 159 MB, decoder BNNS
`model.bnnsir` 38 MB); usable via Core ML/BNNS on the Mac only, no portable weights; macOS license forbids reverse
engineering. The API (SpeechAnalyzer) is the open way.
Degraded copies of the 4 English takes (`.cache/tmp/degrade/`: degrade.swift = Apple's AVAudioUnit reverb/EQ/
distortion rendered offline + pink noise at a set SNR, speech level kept; score.py), WER over 637 words,
Parakeet / Apple SpeechTranscriber: clean 4.1/8.9, noise 10 dB 7.1/14.9, noise 0 dB 24.8/53.8, bad mic (300–3400 Hz
band, light distortion, hiss 20 dB) 5.2/15.1, far (medium room reverb 60% wet + noise 15 dB) 8.3/21.5, far in a
large hall (75% wet) 11.5/35.0. Apple's extra errors are wrong words, not dropped audio (word counts ≈ reference).
Synthetic: approximates real rooms/mics. User: reverb sounds realistic, "bad mic" (Speech Radio Tower preset) sounds
alien/warbling → added realistic ones: phone band only (300–3400 Hz + hiss 30 dB) 4.1/8.9 (no effect on either);
phone call (that, then AAC 8 kHz 12 kbps and back; this Mac can't encode AMR/GSM) 5.0/15.1 (tech take 8%/30%).
Unscripted English takes on the Mac (71 takes, ~1200 words, no reference; `.cache/tmp/apple/own/`,
own_diffs.txt = the 82 disagreement spots): judged by context, ~28 Parakeet clearly right (WGSL→"wsl", MSL→"a masala",
Burn→"Byrne", cache→"cash", C→"see", commit→"comment comment", tray→"tree"), ~2 Apple right ("pre-compiling at
start"), ~19 unclear, ~33 style (contractions, fillers, repeats). Calls made on plausibility from context.
Conclusion: plain clean English a tie; technical words, degraded audio and messy speech clearly Parakeet.
Live mode issues seen in these takes (not yet examined): en story dropped "and the night before"; uk take pasted its
last sentence twice (once while speaking, once after stop); worse on the opening terms than the whole-take pass.

### Text-to-speech side note (2026-11-21, not part of the app)
Apple's built-in voices (`say`, AVSpeechSynthesizer) come in Compact / Enhanced / Premium (+ Siri voices, not
available to apps). Only Compact ships installed; Enhanced/Premium are downloads (Read & Speak → System voice ⓘ).
User listened to Alex (2007, 885 MB) and the Premium voices: clearly below Kokoro-82M and Qwen3-TTS. Ukrainian: only
Lesya (Enhanced at best). Kokoro in burn: not tried. Reference from another project (M1): Kokoro whole fp32 on CPU ~0.6 s/sentence; in Firefox WebGPU ~2.0 s
(LSTM/STFT have no WebGPU kernels → CPU fallbacks + 100 ms readbacks); fp16 decoder crackled. Obstacles for a native
port: phonemizer (espeak-ng is C), the LSTMs, no Ukrainian voice.
Start beep with Bluetooth headphones: sounds played while opening the mic switches the headset A2DP → HFP come late or
not at all; user turned Sounds off (config) for now. Possible fix, not done: play the start cue before opening the mic.

### Mac kernel cache fixed (2026-11-21, M1): first takes after an app start no longer compile
**Obsolete since 2026-11-25: macOS has no GPU path any more ("No GPU path on macOS" below), `patches/` was deleted.
The two CubeCL bugs (random MSL text per process, no MSL disk cache) are still upstream; the fixes are in git history.**
Symptom: after an app restart every take length compiled again (10 s take ~1 s, 33.5 s take 2.6–4 s instead of
0.25/0.9 s). Profiled a fresh process on the 33.5 s take (`/usr/bin/sample`): Apple's MSL compile
(`newLibraryWithSource`) 2.1 s, Apple pipeline build 1.0 s, CubeCL generating the MSL ~0.8 s. A second fresh process
still spent 1.9 s in Apple's compiler: macOS's own shader cache never hit, because **CubeCL's MSL text differed every
run** (dumped with CubeCL's compilation logger, `diff`: same kernels, variable numbers/argument order shuffled) —
Rust HashMap/HashSet iteration order is random per process.
Fixes, as full crate copies in `patches/` (`[patch.crates-io]` in Cargo.toml; diff against
`.cache/cargo/registry/src/*/cubecl-*-0.11.4` to see them):
- `cubecl-cpp` `src/metal/builtin.rs` (`append_msl_builtins`): built-in kernel arguments collected in Vecs in visit
  order instead of HashSets.
- `cubecl-opt` `src/passes/alloc_shared_memory.rs`: shared-memory slices emitted sorted by offset, not HashMap order.
  With both, two runs' kernel dumps are byte-identical.
- `cubecl-wgpu`: an MSL disk cache like the existing SPIR-V one (CubeCL persists only SPIR-V; the MSL path was
  in-memory only). `compute/server.rs`: store "metal" (`MslCacheEntry`: entry point, MSL text, buffer visibilities,
  buffer IO), looked up before compiling, written after; key = device properties + kernel id + this binary's build
  id (a rebuild never reuses another build's kernels; Apple's cache still hits since the text is the same).
  `backend/base.rs`: `create_pipeline_from` (bindings given), `backend/metal.rs`: `bindings_from`. Cargo.toml: serde +
  cubecl-ir "serde". macOS + `msl` only, Linux/Vulkan untouched.
Fresh process each run (`check`), same take twice after the first run:
| take | before | identical MSL | + MSL cache |
|---|---|---|---|
| 33.5 s | 3.4–4.1 s | 1.4 s | 0.84–0.90 s |
| 22.8 s | ~1.4 s | 0.9 s | 0.57 s |
| 10.1 s | ~1.0 s | 0.5 s | 0.24 s |
App (empty kernel folder → tuning 186 s instead of ~115 s, cache 29 MB instead of 5 MB; then app restarted, takes it
never saw): 10.1 s 253 ms, 22.8 s 557 ms, 33.5 s 868 ms = the warm numbers, nothing compiled. Text unchanged.
Cargo: moving a patched crate or editing it rebuilds burn/CubeCL (~6 min); only our code ~25 s.
**Read-only cache again (Mac)**: while testing, saving failed with "database is readonly" (only visible by printing the
store's error; CubeCL logs it as a warning we don't show). Same as Linux 2026-11-20: `default.db-wal` 0 bytes,
`default.db-tshm` (Turso's WAL index) still listing frames; files last written 2026-11-20 22:41, so nothing (tuning
included) was saved on the Mac after that. Cause unknown; both times during experiments with several processes on one
cache and runs stopped midway. Fixed by moving the -tshm aside, then emptying the kernel folder (user's call).
Guard (2026-11-21, `asr::repair_stale_wal_index`, first `set_kernel_cache` call, before CubeCL opens the cache): log
empty (0 bytes / missing) + index present + no other process using the database (fcntl F_GETLK on the index's byte 0,
where Turso holds a lock while a process has it open; unix only) → remove the index, Turso rebuilds it. Tested on
copies (`.cache/tmp/guardtest`): the real broken index → removed, the run then saved its kernels; another process
holding the database → left alone; same with that process gone → removed.
Repro attempt (`.cache/tmp/tursorepro`: turso 0.8.3 opened like CubeCL, `experimental_multiprocess_wal(true)`):
the saved broken -tshm next to a 0-byte WAL gives exactly "short read on WAL frame at offset 2430832 ... got 0" (its
header says ~600 frames in the WAL) → the read-write open fails, CubeCL ends up read-only. How the two got out of
sync is NOT reproduced: 20 processes in a row (exit without close / clean close), kill -9 mid-write, 4 concurrent
writers, a long-running holder + 10 short writers, holder killed while another writes, WAL emptied externally (with
and without a live holder): all stayed writable. Turso empties the WAL to 0 bytes only in a TRUNCATE checkpoint, which
it runs when the last process closes the database (the app and check exit without closing, so normally never).
Later the same day: first takes of lengths the tuning didn't run (it runs 220 sample lengths, not all) still compile
once (11 s take 2.6 s), are saved, and are fast from then on, also after restarts (same 3 takes after an app restart:
311/186/104 ms). Apple's compile cache is per program: kernels compiled by the app don't help `check` and vice versa
(each compiles once for itself). The saved MSL survives app rebuilds (key = build.rs hash of Cargo.lock + rustc,
not the binary): after editing a patched crate (Cargo.lock unchanged) empty the kernel folder, or stale MSL is reused.
Upstream: both ordering fixes and the MSL cache would suit a CubeCL PR (not opened).

### Source: microphone or system audio; recording never changes a device (2026-11-21, M1, measured)
Tray "Source" (was "Microphone") also offers "System audio (what's playing)" on macOS/Windows: cpal records the
default output device (macOS 14.6+: Core Audio process tap, permission "System Audio Recording"; Windows: WASAPI
loopback). Linux: pick the "Monitor of …" source. A silent system-audio take shows a notice (denied permission = silence).
Bug found on the way: the app asked cpal for 16 kHz mono, and on macOS cpal applies the requested format to the
device itself, for every app, and never restores it. System audio -> the headphones' output was set to 16 kHz mono,
which on Bluetooth exists only in headset mode: music in call quality, stuck after the app quit. On a mic the same
request would switch e.g. the MacBook mic to 16 kHz for every app. Fix: always record in the device's current format
(cpal still sets it, but to what it already is) and convert in the app with rubato (the same FFT resampler as
media.rs; the old linear resampler had no low-pass). Block size requested only on Linux (a stream property there;
the 50 ms fix for cut-off ends), on macOS it's a device setting -> device default.
Measured (`experiments/rec.rs` records a source to wav; Core Audio format snapshot of all devices before/during/after):
no device changed, system audio take of en.wav -> exact text, MacBook mic 5 s -> 5.13 s.
System audio records only while the output device runs: nothing playing -> 0 samples; silence before playback
starts is not in the take (1.2–1.6 s shorter than wall time), the sound itself is complete. Mic takes: exact length.

### Single isolated words switch language, also for native speakers (2026-11-21, M1, measured)
One dictated word ("silverback") came out in Cyrillic ("Силвербаг", "Сивербек"); full sentences are fine.
Test (`experiments/words.rs`): first LibriSpeech test-clean clip of each of the 40 speakers (native English), cut
into single words at Ultra's own word times (1 frame before the first token .. next word's start; +0.5 s digital
silence each side), each cut transcribed alone. 941 cuts: Cyrillic 58 (27 of 40 speakers): 1–3 letter words 37 of
273 non-empty (13.6%), 4+ letters 21 of 497 (4.2%); empty 171. Mostly Ukrainian-looking ("А що?", "Та", "Так"),
some Russian (blue -> "Блин", the -> "Это"); other languages too: and -> "sedm", at -> "devět", it -> "Tschüss",
this -> "Des choses", our -> "Bewaard". Report: `python3 experiments/words_report.py <out.jsonl>`.
Caveat: cut points are rough (token times off by ~1 frame), some cuts hold part of a neighbour word or miss part of
the word. → Not the speaker: on a lone word the model guesses the language; short English words sound like
Ukrainian particles (a/А, and/Та, the/Так). Fix would be a language hint (open: does Ultra accept one?).

### Parakeet Redux, Photon, MOSS (2026-11-21, M1, measured)
**Redux** (moondream/parakeet-redux, Sep 2026, CC-BY-4.0): Ultra's architecture/tokenizer/VAD head, encoder linears
ternary (-1/0/+1, f16 scale per 128, packed 5 per byte), 178 MB. `experiments/redux.rs` converts it into our weights
file (exact: codes -> int8, group scale copied to its four 32-blocks; pointwise convs f16; rest as in Ultra's file);
check: zero fraction of all 264 ternary matrices = their ternary.json exactly; Photon's own Redux gives the same texts.
`WEIGHTS=<file>` makes check/words load another weights file. Converted Redux runs as int8 = Ultra's speed/memory.
| | Ultra (ours) | Redux (ours) | Apple SpeechTranscriber (earlier) |
|---|---|---|---|
| 4 en takes, 637 words | 4.1% | 6.8% | 8.9% |
| uk tech take | 15.6% | 22.0% | — (Dictation 32.4%) |
| LibriSpeech clean / other / 500 clips | 2.29 / 3.42 / 2.80% | 2.63 / 5.05 / 3.73% | 2.40 / 4.47 / 3.34% |
Redux's extra errors: tech terms (int8 -> "end8", onnx -> "onx"), uk words broken (запуску -> "записку").
**Photon** (Moondream's engine, `pip install moondream`, Python + PyTorch + proprietary precompiled kernels
`.kstlc`, license: use only under a written agreement; CPU/Metal/CUDA, no Vulkan; needed Python <= 3.14 and torch 2.14,
not 2.15). 5 long takes (564 s), second run, peak footprint:
| | time | WER | peak memory |
|---|---|---|---|
| ours Ultra | 15.1 s (37x) | 6.5% | 934 MB |
| ours Redux (int8) | 15.2 s | 10.1% | 930 MB |
| Photon Ultra (full precision) GPU | 14.2 s (40x) | 6.5% | 2729 MB |
| Photon Redux GPU / CPU | 14.3 s / 21.7 s (26x) | 10.1% | 1975 / 1945 MB |
Short clips (6–7 s): Photon Ultra 152 ms vs ours ~200 ms. Photon's first run per new length 3–5.6 s, ours no penalty.
→ Our runtime ≈ Photon on real takes with 1/3 the memory; int8 Ultra = full-precision accuracy. Ternary doesn't speed
up the GPU (compute-bound); its win is CPU (+ memory). Redux only makes sense as a future CPU mode (own NEON/AVX2
ternary kernels; other PC: Ryzen 5600H AVX2, no AVX-512) — not built. Switching to a CPU path needs no tuning.
**MOSS-Transcribe-Diarize 0.9B** (Open ASR mean 5.17 vs Ultra 5.32; better on SPGI/VoxPopuli/AMI, worse on
LibriSpeech): Whisper-medium encoder + Qwen3-0.6B LLM decoder (token by token, 30 s windows), card tags en/zh, no
Ukrainian numbers → not for dictation. Larger LLM-style ASR (ARK-ASR-3B, MOSS-2B): too big for 8 GB / 4 GB VRAM at
8-bit, ~5–10x slower (estimate, not measured).

### Encoder on the CPU (2026-11-21): src/asr/cpu.rs, Redux 35x real time on the M1
`Asr::load_cpu` (check.rs `CPU=1`): subsampler, 24 conformer blocks, pause head and joint.enc on the CPU, same maths as
encoder.rs; mel/decoder unchanged. Pure Rust, no new crates outside the tree (rayon, matrixmultiply already via burn).
- Linears: int8 weights x int8 activations (quantized per row and block at each call, like ggml CPU Q8_0), NEON `sdot`
  by lane (`vdotq_laneq_s32`, stable Rust), tile 4 rows x 8 outputs, two accumulator sets (one set = 8 chains was
  latency-bound: 130 -> 158 GMAC/s on one P-core, peak ~204). Blocks of 128 when a matrix's scales come in groups of
  128 (Redux), else 32. Ternary matrices (all codes -1/0/1, i.e. Redux) stored 2-bit, one 8-output panel unpacked
  (NEON) per thread task: same speed, 4x less memory. Redux's f16 pointwise convs re-quantized exactly (ternary x scale).
- f32: attention via matrixmultiply, tasks = head x 64 query rows (8 head-tasks left E-cores as the tail), only the
  needed rel-pos columns; first subsampler conv as im2col + sgemm (scalar loop was 0.39 s of a 132 s take -> 0.1 s).
- In the app (2026-11-21): tray menu "Processor" (was "GPU") = Automatic / each GPU / "CPU (no GPU)", config.toml
  `cpu = true`; switching restarts the app (like a GPU change). On the CPU no sweep/tuning runs; `dictation
  transcribe` follows the setting too. ARM without `sdot` (checked at run time) falls back to the plain loop; unit
  test `cargo test --lib cpu` checks packing/2-bit/kernels against a reference and the plain loop against NEON.
- Still scalar (until 2026-11-24, see below: AVX2 kernel): the int8 linear kernel (`tile_plain`, ~70% of the CPU time) on x86 (no AVX2/AVX-512/VNNI kernel
  yet; the Ryzen PC would need AVX2 `maddubs`, AVX-512 VNNI `dpbusd` where present) and on ARM without `sdot`. Also
  plain Rust loops everywhere (auto-vectorized at best): the 2-bit unpack off ARM, layer norm, softmax, GLU, depthwise
  convs, activation quantizing. Attention and subsampler matrix products use matrixmultiply, which has its own SIMD
  kernels (AVX/FMA, NEON). User chose to keep x86 scalar for now (2026-11-21).
- Redux in the app (2026-11-21), CPU only: tray "Model" menu (Ultra / "Parakeet Redux, ternary — CPU only"), config
  `model = "parakeet-redux"`, switching restarts the app. Download (models.rs `REDUX`, pinned to revision 2bf1286,
  SHA-256 per file): model.safetensors 178 MB + tokenizer.json + config.json -> src/asr/redux.rs -> model.weights
  198 MB (ternary modules as new kind "t2": 2-bit codes + f32 scale per 128), downloads deleted. Metadata built from
  Redux's own files (vocab identical to Ultra's 8192 pieces, + "<blank>") and Ultra's frontend/pause/segmenter
  constants: the GPU q8 variant (experiments/redux.rs) gives texts identical to the earlier Ultra-metadata file.
  GPU + Redux: load refuses ("GPU: not implemented, choose Processor -> CPU"), no sweep. Checked: CPU t2 file 15.4 s
  on the 5 takes, 380 MB peak, WER 10.25% (3 borderline spellings vs the earlier 2-bit run, from exact instead of
  round-tripped pointwise scales; deterministic run to run); LibriSpeech 2.63% / 5.10%. Download path tested via
  check.rs `MODEL=parakeet-redux` (downloaded, converted, cleaned up in 23 s).
- 2026-11-24: Processor + Model menus merged into one tray "Model" list of combinations that run (each GPU model
  on each GPU, "— Automatic" only with 2+ GPUs, and on the CPU; Redux only "— CPU"). The old menus let Redux be
  picked with a GPU: load failed, takes then did nothing. config.toml unchanged; Config::load moves a CPU-only model
  set to a GPU to the CPU and saves. Speed on the Ryzen 5600H (x86 int8 kernel still scalar): Redux CPU 6.0 s for 7.7 s
  of audio, 3.8 s for 4.7 s (GPU Ultra ~0.25-0.3 s for the same lengths).
- 2026-11-24: recording accepts every cpal sample format (cpal FromSample), not only f32/i16: the laptop's built-in
  mic on PipeWire runs at s32, "unsupported sample format I32" since recording uses the device's own format (5d4e94e).
- 2026-11-24: x86 int8 kernel `tile_avx2` (cpu.rs; AVX2+FMA checked at run time, else tile_plain): per 4 values of k
  one 256-bit panel row = 8 outputs, x broadcast, maddubs(|w|, x with w's sign) + madd(ones). 2-bit unpack in SSE2
  (x86-64 baseline). Ryzen 5600H, samples en.wav 7.2 s / uk.wav 6.0 s, encoder: Redux 6132 / 5270 ms -> 258 / 228 ms
  (~24x; texts identical), Ultra on the CPU 293 / 259 ms (= the GPU's ~0.25-0.3 s). Unit test checks it against
  tile_plain. No AVX-512 / VNNI kernel: this CPU has neither (AVX-512 CPUs run the AVX2 one).
- 2026-11-24 CPU profile, Ultra on the CPU, Ryzen 5600H, 126 s take (asr 4.9 s; temporary timers per stage, wall
  time): int8 linears 2700 ms (~65%), decoder 409, activation quantizing 355, attention core (f32 sgemm + softmax)
  297, subsampler (encoder + pause scan) 238, silu 135, layer norm 121, depthwise conv 84, rest (allocations, bias,
  adds) ~400. The kernel runs ~1 T int8 MACs at ~16 MACs/cycle/core: its inner loop compiles to the minimal 4 x
  (sign, maddubs, madd, add) per panel row, and that is about the multiply throughput of Zen 3 without VNNI (same
  method as llama.cpp's AVX2 q8_0). No change from task sizes (TASK_ROWS 64-256, TASK_COLS 16-64), 6 vs 12 threads,
  or glibc malloc keeping big buffers (all within the +-3% noise) -> compute-bound, not memory. Fixed: quantizing
  used `f32::round` (a library call per value without SSE4.1); AVX2 version with the same rounding (half away from
  zero, unit test: identical bytes incl. ties) 355 -> 100 ms. Whole runs vary ~10% (laptop heat), so ~5% total gain
  isn't visible in one run. Left: the rest outside the kernel is ~1 s of 4.9 s, a few % each (exp in silu/softmax
  would need a vectorized approximation = different numerics).
- Restart on macOS fixed (2026-11-21): switching GPU/CPU restarted via `exec`, and on macOS the exec'd process came
  back without its menu-bar icon (it still ran and answered `toggle`). Now macOS/Windows spawn a new process with
  `DICTATION_RESTARTED=1`, which waits (<= 10 s) for the old instance to quit before the single-instance check;
  Linux keeps exec. Checked on the M1: CPU -> Automatic restarted with the icon back.
Results, M1 MacBook Air (fanless, so long runs throttle; numbers vary ~10-20% run to run), 8 rayon threads:
| | 5 takes 564 s: time | WER | peak footprint | LibriSpeech 500 clean / other |
|---|---|---|---|---|
| GPU Ultra (ours) | 15.7 s | 6.54% | 983 MB | 2.29% / 3.42% |
| GPU Redux (ours) | 15.3 s | 10.0% | 965 MB | 2.63% / 5.05% |
| CPU Ultra | 20.1 s (28x) | 6.54% | 891 MB | 2.21% / 3.40% |
| CPU Redux (2-bit) | 16.3 s (35x) | 10.0% | 383 MB | 2.63% / 5.15% |
| Photon Redux CPU (earlier) | 21.7 s (26x) | 10.1% | 1945 MB | |
Texts differ from the GPU by a word or two per take (int8 activations vs f16), WER the same. Short clips: 7 s ~245 ms,
2.8 s ~115 ms (GPU ~200 ms, no kernel compiling/tuning needed on the CPU). Of the 16.3 s: encoder 12.7 s (linears
~70%), TDT decoder 2.5 s (same code as the GPU path), pause scan 1.0 s. Run tools: `.cache/tmp/cpu/` (wer.py, jsonl).

### Phonon-2 (2026-11-21, M1, measured; not added to the app)
FermionResearch/Phonon-2 (Oct 2026, CC-BY-4.0): English-only, retrained from Parakeet TDT 0.6B v3, encoder weights
five-valued (0, ±a, ±b per block), rest 6-bit, own packed 164 MB file (`phonon-2.bps.tar.zst`, needs their engine);
the model inside their Mac app Detta. Their table (full Open ASR sets, own scoring): LS clean/other 1.72/3.92, mean
5.21 vs Redux 5.69 and v3 4.96; claims 174x real time on an M5 Air. No pause head.
Tested via cstr/phonon2-GGUF `phonon2-q8_0.gguf` (674 MB, CrispASR names; their check: 19/21 clips identical to full
precision) -> `experiments/phonon.rs` (names mapped to ours, tokenizer checked identical to Ultra's, Ultra's metadata
and pause head copied in) -> int8 weights file like Ultra's (741 MB). Files in `.cache/tmp/phonon2/`.
| | 4 en takes | uk take | LS 500 clean / other (CPU) | 5 takes GPU / CPU | LS 500 CPU time | peak (GPU / CPU) |
|---|---|---|---|---|---|---|
| Ultra | 4.1% | 15.6% | 2.21 / 3.40% | 15.7 / 20.1 s | 147 s | 983 / 891 MB |
| Redux (t2, CPU) | 7.1% | 22.0% | 2.63 / 5.10% | — / 16.2 s | 114 s | — / 380 MB |
| Phonon-2 (int8) | 10.7% | 97% (Latin gibberish) | 2.35 / 4.44% | 15.6 / 20.8 s | 160 s | 973 / 889 MB |
Reading test (a read-aloud text, 70 s, 113 words, ref `.cache/tmp/phonon2/ref_en_reading.txt`,
not in experiments/bench; ~3-4 errors shared by all = reading differences): Ultra 10.6%, Phonon-2 19.5%,
Redux 21.2%.
- LibriSpeech ranking matches their table (between Redux and full), so their benchmark numbers look real. On the
  real dictation (technical, unscripted) it drops to Redux level or below: most of the gap is the int8
  take (int8 -> "end eight"/"nt8", onnx -> "o and nx"). Possibly tuned to leaderboard-like audio; one speaker,
  ~750 words, so not proven.
- It writes numbers as digits and often skips the separate "▁" token before them ("about650", "of2 .4",
  "phonon2"): digit pieces have no "▁" variant in this vocab, Ultra emits it; ~3-6 errors per test.
- Speed: none gained here (stored int8 = Ultra's speed and memory). A native five-value (3-bit) CPU path would be about
  Redux's speed/memory (estimate, not built). English-only -> not for this app.

### Kokoro-82M text-to-speech on burn (2026-11-22, M1, measured; experiment only, not in the app)
hexgrad/Kokoro-82M (Apache-2.0, StyleTTS2 + iSTFTNet vocoder, 24 kHz; revision f3ff357). No engine like Photon: the
reference is the `kokoro` Python package (PyTorch) + `misaki` (text -> phonemes; English uses dictionaries + espeak-ng
fallback, a C library). Languages en/es/fr/hi/it/ja/pt/zh, no Ukrainian. Not in transcribe.cpp (STT only); C++ ports:
CrispASR (ggml), TTS.cpp, kokoro.cpp.
- Reference setup `.cache/tmp/kokoro/`: venv (python3.14; kokoro 0.9.4 + misaki installed `--no-deps
  --ignore-requires-python`, they want <3.13; spacy doesn't build on 3.14, so misaki's English G2P can't run here),
  `hf/` weights + 2 voices, `scripts/ref.py` (exports `ref/kokoro.safetensors` with weight norm folded,
  `voices.safetensors`, intermediates for `ref/phonemes.json` with random source noise off), `ref_gen.py` (vocoder
  intermediates), `cmp.py` (compare), `mps.py` (PyTorch speed).
- Port `experiments/kokoro.rs` (phonemes in, wav out). GPU (burn): ALBERT, convs, style blocks, vocoder nets; weights
  uploaded once and cached. CPU: the 5 bidirectional LSTMs (step by step, tiny), durations, sine source (9 harmonics),
  both STFTs (n_fft 20, hop 5). Env: NOISE=0, DUMP=, HAR=, PROF=1 (per-stage times with device sync), BENCH=1.
- Exactness: durations identical, F0/text encoder ~1e-6 rel; with the reference's source spectrum (HAR=) audio
  1.4e-4 rel. Without it ~10%: a quarter of the source STFT bins are silent and their phase is float noise on
  both sides (the vocoder takes it as input), so no exact match is possible; in normal use random noise is added
  anyway. torch's CPU cumsum adds in f64 (phase drifted until we did too). Our ASR reads the output back correctly.
- Speed with burn's ops only (warm, noise on; OWN=0): 4.45 s sentence 0.51 s (8.7x), 11.6 s sentence 1.30 s (8.9x).
  PyTorch MPS 0.62 / 1.41 s (7.2x / 8.2x), PyTorch CPU 0.82 / 2.08 s (5.4x / 5.6x). Of the 1.30 s: vocoder ~1.15 s,
  nearly all of it the 128/256-channel convs of the AdaINResBlock1 blocks (burn conv1d ~520-650 GFLOPS).
- Tried before own kernels: CPU dot products in 8 lanes + parallel STFT/source (CPU parts 450 -> 120 ms); AdaIN as one
  per-channel scale/shift from E[x], E[x^2] (vocoder 1.36 -> 1.16 s); f16 (no faster on the M1, and f16 ALBERT changed
  the durations: dropped); stride-1 convs as stacked shifted copies + matmul (1.93 s vs 1.23: dropped).
- Own CubeCL kernels (2026-11-22), `mod kern` in kokoro.rs, through burn 0.22's `#[backend_extension(Cube, Fusion)]`
  (feature `extension`, burn-cubecl as dev-dependency; examples only, the app's build is unchanged). Two kernels per
  block step: per-channel sum/sum of squares (one read), then a conv that applies AdaIN + Snake while loading its
  input and adds bias + residual when storing. The conv is an implicit matrix product on cmma 8x8x8 f32 fragments
  (Metal simdgroup matrices; tensor cores on NVIDIA where CubeCL offers f32 cmma, unmeasured there): cube = 128 out
  channels x 128 steps, 8 planes x (4 x 8 fragments); weights re-laid per tap [k][co][ci] and read as fragments
  from memory; input chunk (32 ch for k3, else 16) in shared memory, a dilated tap = an offset into it.
  Same results: conv vs burn 4e-7..1.2e-6 rel; audio vs PyTorch with the reference source (HAR=) 3.0e-5 rel
  (was 1.4e-4); our ASR reads the long sentence back right.
  - Speed per conv (KTEST=1, M1): k11 ~1.3-1.4 TFLOPS vs burn ~0.62 (2.2x), k7 ~1.3, k3 ~1.0 (input fill weighs more
    with few taps). Whole thing: 4.45 s sentence 0.27-0.30 s (~15x), 11.6 s sentence 0.74-0.78 s (~15x real time),
    vs 0.53 / 1.36 s with OWN=0 in the same session. Now ~160 ms of the 0.76 s is ALBERT + LSTMs + CPU parts,
    decoder blocks (burn, 1024 ch) ~70 ms.
  - Lessons: fusing AdaIN/Snake into the conv saved little by itself (burn's fusion already runs them in ~2 ms per
    step); a plain-FMA register-tiled conv reached only ~380-550 GFLOPS; weights staged in shared memory per tap
    was slower (extra syncs); padding the shared row stride did nothing. The big one: the tap loop fully unrolled
    (#[unroll]) ran k11 at ~550 GFLOPS (d=2 even ~350); as a plain loop 1.4 TFLOPS for every dilation.
  - CubeCL's disk kernel cache keys kernels by our build key (Cargo.lock + rustc), blind to own kernels: an edited
    kernel loaded the old binary. kokoro.rs uses a cache folder per hash of `mod kern`'s source
    (target/kokoro-kernels/<hash>/); the app would need the same if it ever gets own kernels.
- First use per new length: CubeCL autotunes per shape band: 47 s on the very first runs (burn-only), ~2 s for a new
  length once most bands are tuned. The app would need a warm-up sweep like ASR's.
- Left: the k3 convs (~1.0 TFLOPS), the decoder blocks and the CPU parts (~160 ms) are what remains; not tried.
  Missing for real use: text -> phonemes (port misaki's English dictionaries; out-of-vocabulary words without
  espeak).
- Next (user, 2026-11-22): "read selection" = select text anywhere, press a shortcut, Kokoro speaks it (Mac first).
  Order agreed: phonemizer as an experiment first (typed text on the command line), then the tray app (shortcut,
  selected text via simulated Cmd+C + clipboard restore, sentence-by-sentence synthesis + rodio playback, press
  again = stop, model download + Rust conversion of weights/voices). English only.
- Phonemizer plan: Rust port of misaki 0.9.4 `en.py` (~700 lines; read in .cache/tmp/kokoro/venv/.../misaki/en.py).
  Kokoro v1.0 uses misaki's phoneme style (diphthongs as one capital: A I O W Y, plus ᵊ ᵻ; v0.19 used espeak), so
  raw espeak output wouldn't match. misaki: subtokenize regex -> per word: special cases (a/an/the/to/in/I/used,
  context = next word starts with a vowel), gold lexicon (us_gold.json, 90,201 entries, 790 with per-POS variants
  DEFAULT/VBD/...) then silver (us_silver.json, 93,361), both "grown" with Capitalized/lowercase copies;
  suffix stemming -s/-ed/-ing (phonetic rules _s/_ed/_ing); all-caps words spelled letter by letter (get_NNP);
  numbers via num2words (cardinal/ordinal/year/currency $ £ €, decimals "point"); stress adjustment (apply_stress,
  caps get extra stress); punctuation kept (;:,.!?—…"); final step ɾ -> T, ʔ -> t (v1.0). Lost without its
  helpers: spacy POS tags (read/live/used get DEFAULT; tag-dependent a/by/in rules use defaults) and the espeak
  fallback for unknown words (plan: letter-by-letter or simple rules).
- Phonemizer found instead of porting (2026-11-22): crate `misaki-rs` 0.11 (MIT, github.com/MicheleYin/misaki-rs):
  Rust port of misaki 0.9.4's English pipeline incl. a port of spacy's tokenizer + POS tagger, data embedded
  (us/gb gold+silver lexicons + tagger weights, crate 9.5 MB); claims parity with Python misaki on 1,722 fixture
  sentences (US). `default-features = false` = no espeak (pure Rust; 11 new crates, none with a build script).
  Dev-dependency of experiments/kokoro.rs: input is now text (PH=1 = phonemes as before). Load 60-110 ms, a
  sentence 7-18 ms. Checked by hand: same phonemes as my hand-written ref/phonemes.json up to small stress/vowel
  choices; "wind/wind" and "read" by POS right; dates, $12.50, Dr., p.m., 20% right. Our ASR reads both test wavs
  back exactly (g2p1: the fox sentence; g2p2: "On March 3, 2025, Dr. Smith paid $12.50 for three books at
  7:30 p.m. That's 20% off."). Gaps: unknown words become ❓ and are silently dropped (e.g. "Kokoro"; names and jargon
  in real selections will hit this), "7:30" gave "sevən❓ θˈɜɹTi" (colon -> ❓, still sounds right). Needs a fallback
  (misaki-rs has a `Fallback` trait). Kokoro takes max 510 phonemes per pass (asserted): longer text must be split.
  After the Cargo.lock change the first run re-tuned all kernels (38 s); then the fox sentence 0.88 / 1.24 s
  (0.74-0.78 s before; not looked into).
- Read-selection try-out (`kokoro <dir> --listen`, macOS): Ctrl+Option+R -> wait for the modifiers to be let go,
  simulated Cmd+C (clipboard text restored after), misaki-rs per paragraph, phonemes cut per sentence (>500 cut at
  a comma/space), each piece synthesized and appended to a rodio player while the previous plays; press again =
  stop. Worked first try. Tries: copy 130-220 ms; first sound 0.43-0.87 s for a length seen before, 2.3-5.9 s
  for a new length (autotune; a one-word "Yes." 5.85 s). Ideas not done: a warm-up sweep over lengths (once, cached)
  and a short first piece (up to the first comma). More misaki-rs gaps: "→" -> ❓, "0.3–0.5 s" -> ")))".
- **Shelved (user, 2026-11-22): no text-to-speech for now.** experiments/kokoro.rs is committed but not built: its
  dev-dependencies (burn `extension`, burn-cubecl, misaki-rs) and `[[example]]` were taken back out of Cargo.toml so
  the app's Cargo.lock (= kernel cache key) stays as it was; the file's header says what to add back.

### Cactus Needle 3 + Whistle (2026-11-22, M1, measured; voice-commands test was in the app, removed the same day)
Cactus Compute (Apache 2.0): weights open (safetensors + own 2-bit `.cact`), model code readable (JAX,
`cactus/model/architecture.py`), but the engine is published only as prebuilt binaries (`libcactus.{a,dylib,so}` +
a 1.7 KB C header, huggingface.co/Cactus-Compute/cactus `releases/4.1.4/`), no source. Python package
`cactus-compute` (no JAX for inference, anonymous telemetry: `CACTUS_TELEMETRY=0`). Test venv/scripts/caches in
`.cache/tmp/needle/`, `.cache/tmp/whistle/`.
**Needle 3** (tool calling, 20 layers = the full/biggest model, 35 MB): 35–70 ms per command on the CPU. English
commands mostly right after tool wording was tuned (tab actions needed "next/previous/close/new tab" in the
description); app names as free text beat an enum of the 85 installed apps (enum: lower confidence, wrong apps).
Ukrainian: 1/10 (only "відкрий Safari"), others nonsense at confidence 0.02–0.12 → no Ukrainian.
Built-in refusal: empty call list for off-topic input; the engine holds back calls below confidence 0.1.
App test (removed again, user 2026-11-22: `src/commands.rs`, tray "Voice commands (test, English)", config `commands`): take text -> Needle via
dlopen of libcactus (no new crates, Cargo.lock unchanged) from `models/needle3/` -> open/quit app, volume, browser tab
keys, open URL (macOS only). App names resolved to installed apps (exact, contains, else edit distance <= 1/3).
In use: errors came from speech recognition on 2–3 s commands ("quit Steam" -> "With Steam" -> opened at 0.38;
"Close Stream" written as "Клоуз Стрим").
**Whistle** (Cactus ASR, en de fr es it nl pl, keyword biasing, 4-bit): on the 6 short command takes not better than
Parakeet, keywords barely helped. Overall (same refs/scorers as Redux/Phonon-2):
| | Ultra | Whistle |
|---|---|---|
| 4 en takes, 637 words | 4.1% | 9.7% (int8 tech take 18.8%) |
| reading test, 113 words | 10.6% | 34.5% ("parliament speeches" -> "parliaments beaches") |
| uk take | 15.6% | 98% (detected as German) |
| LibriSpeech clean / other / 500 | 2.29 / 3.42 / 2.80% | 4.73 / 10.13 / 7.19% |
| 5 long takes (564 s) | 15.7 s GPU / 20.1 s CPU | 7.4 s CPU |
| LibriSpeech 500 clips | 147 s CPU | 37.7 s CPU |
| peak memory | 891 MB CPU | 247 MB |
→ ~3x faster and 1/4 the memory, but 2.5x the errors, no Ukrainian: not for dictation, no help for commands.
Short speech (20 short takes, 2–4 s, 57 words, commands + short replies, CPU; files in `.cache/tmp/whistle/`
short_*): takes fully right / WER: Ultra 14/20, 15.8% (2 of its misses = right words in Cyrillic: "Некст таб",
"Мьют"); Redux 15/20, 10.5%; Whistle 12/20, 22.8%; Whistle + keywords (14 command words incl. Zed, commit, push)
15/20, 14.0%. Nobody got "Quit Steam" (all "With/Wit Steam"). 100–110 ms vs 19 ms per take (both instant).
LibriSpeech by clip length (≤4 s, 157 clips): Ultra 2.87%, Redux 4.05%, Whistle 9.79%; Whistle's gap is largest on
the shortest clips. → Whistle with keywords only ties Redux on short commands; tiny sample.
**Language hint for Parakeet (answers the open question in "Single isolated words switch language"):** Ultra's vocab
has no language tags (none at all; tokenizer.rs strips them only for other models), so no way to say "English".
What works instead: ban every vocab piece with Cyrillic letters in the decoder (logit -> -inf; tested with temporary
env hooks in decoder.rs, removed again). 20 short takes: Ultra 14/20 -> 16/20 ("Некст таб" -> "Next tab", "Мьют" ->
"Mute"), nothing else changed; Redux 15/20 unchanged. Earlier commands: "Клоуз стрим" -> "Clous Stream", "Open Steam,
плиз" -> "Open Steam, please", "Кло три" -> "" (empty). The 4 long English takes: text identical. Not in the app yet
(would be for voice-command mode only; Ukrainian dictation needs Cyrillic).

### Short phrases: language flip, and a dead mic at the start (2026-11-22, M1, measured)
- Kokoro (experiments/kokoro.rs, the built binary in target/) read the 20 short command phrases in af_heart and
  am_michael (`.cache/tmp/kshort/`): Ultra 20/20 both voices (counting "30" for "thirty", leaving out "Zed"), Redux
  19/20 and 17/20 ("Sended", "tasks"). "Switch to Zed" -> "Z" in all four (letter name; fix in text, not audio).
  Recorded takes: 14/20. So short length alone isn't the problem.
- The list re-read carefully (built-in mic): 15/20, all five misses Cyrillic spellings of the right English
  ("Квит Стим", "Мьют", "Оупен Ютьюб", "Сандет", "Копи дат") -> a Latin-only ban in a command mode would fix them.
- **Dead mic at the start:** the default input was Bluetooth headphones (16 kHz). Opening its mic switches the
  headphones to headset mode; until then macOS delivers exact zeros: 0.4-0.7 s (up to 1.6 s) in nearly every take
  since 2026-11-21 that started cold (takes within a few seconds of the previous one: none). The app turned red
  and beeped at once, so the first syllable went into the gap ("Commit and push" -> "Mitt and Bush", "Quit Steam"
  -> "Wet Steam": speech jumps from exact zero to full level within 50 ms).
- Fix (src/audio.rs, src/daemon.rs): new state Starting (normal icon, "Starting the microphone…"); the recorder
  drops samples until the first non-zero one and calls `on_live`, then the app shows Recording + start beep.
  System audio is live at once; after 2.5 s of zeros (muted, no permission) a mic counts as live anyway. Log line
  "microphone live after N ms": Bluetooth headphones cold 450-720 ms, warm ~25 ms; built-in mic 13-38 ms. Linux/Windows: same
  code, not run there (a mic with a noise gate that sends exact zeros would go red only when speech starts).
- The wait itself can't be shortened (Bluetooth profile switch). Better: record with the built-in mic while
  wearing BT headphones (instant, and the headphones stay in music quality); user switched Source to it.
- Seen once: cpal "A buffer underrun or overrun occurred." = Core Audio kAudioDeviceProcessorOverload, right when
  switching from the headset mic to the built-in one; takes around it intact, not reproduced in 3 rec.rs switches.
  Headset takes also show one ~20 ms zero gap ~0.1 s after waking (before speech).

### Decoder on all cores; long takes decode while the GPU encodes (2026-11-24, Ryzen 5600H + RTX 3050 Ti, measured)
- **Decoder matvecs split over the cores** (decoder.rs `Q8::matvec`: rayon tasks of 64 rows, each the same
  AVX2/NEON/plain f32-activation kernel as before, so the same arithmetic). GPU mode, 126 s take: decoder 462 -> ~180
  ms, asr 1133 -> 867; 10 s 31 -> 14 ms. CPU mode: 126 s 372 -> 142 ms, 10 s 43 -> 12. 230 local takes (71 min):
  decoder sum 11.1 -> 4.4 s, **texts all identical**. Decoder share of GPU-mode time was ~1/3 on 10-90 s takes (41% on
  126 s), now 15-20%. Not built or run on ARM yet (NEON kernel only got the row-range argument); the M1's 4+4 cores
  may gain less.
- Tried first: the M1 branch's way (m1-gpu-kernels, decoder through cpu::QLin = int8 activations, all cores): a bit
  faster (126 s: 166 ms) but 4 of 5798 words changed over the 230 takes ("thing" -> "thank", "Dambas" -> "Damas",
  "optimized" -> "optimize", one added "the"). Dropped for the identical-text version.
- **Overlap** (pipeline.rs `pieces_tokens`, from the M1 branch): takes over 30 s, GPU mode: piece i decodes on a CPU
  thread while the GPU encodes piece i + 1. With the decoder on all 12 threads the GPU encoder slowed down (it needs
  a CPU thread to build and launch its work; 215 s take: encoder 0.86 -> 1.9-2.2 s). Decoder pool size while
  overlapping, 9 takes 80-215 s x3 (median sum): no overlap 7434 ms, overlap with 12 threads 7964, 8 threads 7313,
  **4 threads 6957**, 1 thread 7041. Kept: own pool of a third of the threads (4 here). 80-126 s takes 7-15% faster;
  200+ s takes jitter +-10% run to run in every variant (pause scan too, which the overlap doesn't touch). Under 30 s
  (one piece): unchanged. CPU mode: no overlap (encoder and decoder want the same cores).
- **First-run cost per new length on NVIDIA was the driver, not CubeCL**: a fresh sweep into a scratch data folder
  (`XDG_DATA_HOME=.cache/tmp/xdg`, 171 s), then the 9 test takes twice: no first-run cost up to 30 s; 1-4 s on the
  first long takes, but CubeCL's cache wasn't written, while NVIDIA's own shader cache
  (`~/.cache/nvidia/GLCache/e2fa...`, 299 MB) was, at that time only. Repeats (same and other orders, new processes):
  no spikes. So the first run of a newly built or copied binary compiles shaders in the driver once; the
  0.4-3.4 s "first runs" in the earlier GPU-vs-CPU table were that. Nothing to fix for the app.
- **GPU encoder, where the time goes** (temporary `Block::bench`: block 0's parts, 20 runs queued, one sync, best of 4;
  per block, x24 per pass): 19 frames (1.5 s) 0.93 ms, 63 frames 0.99, 128 frames 1.52, 375 frames (30 s) 3.95.
  Up to ~10 s it is nearly all matmuls, and those are far from the GPU's limit: with few rows a q8 matmul should cost
  reading its weights (~25 MB per block = ~0.13 ms at ~190 GB/s), burn's takes ~0.9 ms (~40 GB/s). At 375 frames
  matmuls run 5-9 TFLOPS (cooperative matrix f16 is available and used), attention core (scores, rel shift, softmax)
  ~0.6 ms per block, LayerNorms ~0.1 ms each. Own kernels (small-M q8 matmul, fused attention) would need burn's
  `extension` feature = a burn rebuild + Cargo.lock change (app re-sweeps once); not started.

### Live mode removed (2026-11-24, user's decision)
"Paste while speaking" (src/asr/live.rs: every ~1 s the current window re-transcribed, words pasted once two passes
agreed, never corrected) is gone with everything that only served it: tray item, config `live` (an old config.toml
line is ignored on load and dropped on the next save), daemon's 300 ms tick and Live jobs/events, the recorder's
mid-take `snapshot`, check.rs `LIVE=1`. Takes are always transcribed whole after stop (full context). The sweep's
30.08 s pause-scan run stays (shortest take that is cut at pauses). Earlier notes about live mode describe the old app.

### Neural Engine (2026-11-24, M1, measured): Parakeet Ultra's encoder on the ANE, ~3x faster than the GPU
Tray menu Model -> "Parakeet Ultra — Neural Engine" (macOS only, model `parakeet-ultra-ane`, its own folder and
download). The encoder (subsampler + 24 blocks) runs on the ANE through CoreML (src/asr/ane.rs, objc2-core-ml, in
process); pause scan (subsampler + pause head), joint projection and decoder on the CPU (cpu.rs without its blocks).
No GPU, no tuning.
- **Results, 99 takes / 1630 s** (`.cache/tmp/ane/cmp`, WER on the 10 reference takes): GPU (main) 5.44%, 46.6 s
  total, medians <=5 s 116 ms, 5-15 s 231, 15-30 s 431, >30 s 1435. ANE (as shipped) 5.37%, 13.9 s, 32 / 70 / 175 /
  487 ms; 1.0-1.3% of words differ from the GPU, both ways. Encoder alone per window: 5 s 21 ms, 15 s 47, 30 s 125.
  (These are the 3-window model; since 2026-11-24 evening it is one 30 s window, see "Load": every piece ~130 ms of
  encoder, so short takes ~+0.1 s.) **30 s model, same 99 takes (2026-11-25)**: WER 5.37% (same), 22.6 s total,
  encoder 17.2 s. Back to back (`check`, warm), 36 takes 1.5-165 s: up to 30 s the encoder is flat ~125-137 ms for
  any length (decoder 12-84 ms grows with words): 1.5-4 s takes 145-160 ms total, 6-10 s 161-186, 10-30 s 173-224;
  each further 30 s piece +~130 ms encoder (33-51 s 332-436 ms, 81 s 696, 145-165 s ~1.25 s). Texts fine, incl.
  "Hello." (1.5 s, 28.5 s of padding). The app's long takes were slower (encoder 200-250 ms) because of the idle
  wake-up, see "Idle" below; it was not the CPU tail (joint projection etc. are a few ms).
- **Model**: moondream/parakeet-ultra is fp16 as published (608.6M of 608.9M encoder params) = the ANE's own format.
  Exported once (experiments/ane/export.py: transformers + coremltools, fp16; attention mask -1e4 instead of
  finfo.min, which is -inf in fp16 and gave NaN rows), one fixed 30 s window (3001 mel frames; shorter pieces padded,
  exact), a plain model (1.2 GB). Was: 3 windows (5 / 15 / 30 s) as one multifunction model sharing the weights, dropped
  because macOS recompiled it on every load (see "Load").
- **The app builds it itself** from the download (pinned revision, SHA-checked) as `ane/enc-30s-noshift.mlmodelc` (was `enc-30s.mlmodelc` before 2026-11-27; an older
  build under another name counts as not installed: downloaded and built again): assets/parakeet-ultra-ane (0.8 MB:
  compiled model without weight.bin + recipe.json + consts.bin, experiments/ane/recipe.py) + src/asr/ane_build.rs
  writes weight.bin in ~5 s: 540 tensors copied, 48 depthwise convs with batch norm folded (same bytes as CoreML),
  24 relative_k_proj x position table (CoreML folds those; ours from the f64 table: 1 fp16 step from CoreML's, values
  < 0.06 differ by < 1e-4; with 3 windows: 3 one-word diffs in 99 takes). `model.weights` for the CPU parts from the same download (redux.rs
  `convert_ultra_head`). Then a one-time "Preparing for the Neural Engine" (macOS compiles it, ~40-120 s).
- **Tried and dropped**: FluidInference/parakeet-ultra-coreml (int8 per channel, fixed 15 s): 7.21%, 15 s pieces
  spell out numbers/terms. 8-bit from our GGUF (CoreML blockwise int8 = the GGUF's bytes exactly): quality fine
  (5.51%) but M1's ANE does not run that format natively: 99.7 s total, slower than the GPU. CoreML on the GPU: 1.2 s
  per window.
- **Load**: two caches. CoreML's per app (`~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache/`: how the
  model is split CPU / ANE, small) points into aned's (macOS's Neural Engine daemon, root-only
  `/Library/Caches/com.apple.aned/`: the compiled ANE program, weights baked in, ~1.2 GB per window). First load of a
  new binary or model folder compiles (~40 s for the 30 s window), later loads 0.2-0.4 s (check: 39 s, then 0.44 s).
  **Recompiles (a take waits 40-100 s), cause found: low disk space.** aned's compiled programs are purgeable: when
  free space is under macOS's target (`deleted` kept asking for ~4.3 GB more at ~10 GB free, so ~15 GB), it purges
  them (log: `deleted_helper` "fsPurge: purged" in chunks of exactly one program, 1.18-1.22 GB) and the next load
  compiles again. Seen 22:36-23:04 at 6-10 GB free (our experiment copies in .cache/tmp/ane: 21 GB, deleted; free
  now 18 GB). Each compile writes ~1.2 GB more, so it feeds itself. The app cannot prevent it (macOS's root-owned
  cache). With enough space: 6 reloads in one process 0.2 s each, separate processes 0.2-0.4 s. Debug: CoreML's
  per-app entries (mtimes of the e5bundlecache folders), aned's log (`/usr/bin/log`, zsh's `log` is a builtin: every
  recompile shows "Calling ANECompilerService"), `deleted`'s log.
  The multifunction model (3 windows) looked like the cause (recompiled on every load at the time) and was replaced
  by one plain 30 s model: 1 program instead of 3 to lose and to recompile (40 s, not 100). One model with enumerated
  input shapes (5/15/30 s) was tried too: the ANE compiler rejects it (ANECCompile() FAILED), CoreML runs it on the
  CPU (30 s: 5 s). Other processes loading the model (`check`, Python) compiled it again (~35 s) at times; whether
  aned keeps a program per process was not established. Experiments here pushed free disk under 15 GB again
  (2026-11-25: 7.6 GB, purges and app recompiles); CoreML's Python cache reached 2.2 GB.
- **Idle (2026-11-25, measured)**: the ANE winds down when unused; the next call pays it once (bare CoreML call,
  warm 130 ms): idle 3 s +10 ms, 8 s +30, 16 s +95, 25 s +100-140, then flat (beyond 25 s not measured). In the app
  the load + warm-up run at the start press, so the idle that counts is the take itself: 7 s take +30 ms, 16-25 s
  takes +85-150 ms (app logs: encoder 200-250 ms vs 130 warm). **Keep-awake** (daemon.rs `KEEP_AWAKE`, user's
  choice): while a take is recorded, the engine thread runs one encode of silence after 4 s without work
  (`Asr::keep_awake`, Neural Engine only; GPU/CPU: nothing). Cost: ~130 ms of ANE per 4 s while recording only; a
  stop during a run waits for its rest (~3.5% of takes, ~65 ms on average, at worst ~140). A cheaper keep-awake (a
  tiny dummy model) is untested: only helps if the whole ANE sleeps, not if our program is put aside (on NVIDIA a
  tiny kernel did not help, "Latency in gauge"). App after the change: 37.1 s take encoder 275 ms (2 pieces, ~137
  each; before: 31.1 s 387 ms), 39.6 s take 392 ms total (encoder 257, decoder 100, pause scan 94).
- **Memory (2026-11-25, measured)**: loaded model = +1.2 GB wired (vm_stat: 1.22 -> 2.42 GB loaded idle, 2.58-2.67
  while encoding back to back, back down after unload); none of it in our process (85-94 MB) or aned (5 MB), so
  Activity Monitor doesn't show it (`footprint` lists it as 1.22 GB of device mappings). The app holds it from start
  press to text (unloads after each take unless "keep model loaded"). Each window size = its own program and copy.
- **Faster encoder? (2026-11-25, tried, dropped)**: CoreML's compute plan: all on the ANE except the mask (a few
  casts). Op costs: linear 47%, reshape + transpose 12%, slice + pad 7%. Peak of this ANE on plain 1x1 convs:
  ~7.2-7.9 TFLOPS; the encoder (~470 GFLOP per 30 s window) runs at ~3.7. Per layer (6-layer cut-down models):
  attention ~2.8 ms (of which the relative-position shift, pad + reshape, ~1.5 ms), feed-forwards ~1.7 ms (near
  peak), conv module ~0.3 ms; subsampler ~11.5 ms once. Tried, all exact in fp32 (vs transformers, 6e-7): the ANE's
  own layout (B, C, 1, T) with 1x1 convs: no gain (130-133 ms, the compiler already handles it); heads split
  (Apple's ml-ane-transformers style): 2x slower; shift as 9 slice + select steps: 1.7x slower (big elementwise ops
  are slow on the ANE); minimal subsampler masks: no gain; **shift-free position term** (sin/cos of i - j split into
  i and j, as RoPE: one plain matmul, no shift): 112 vs 128 ms (-13%, 99 takes encoder 14.9 vs 17.2 s) but worse on
  the reference texts, 75 vs 73 errors of 1360 words (2 words, both wrong: "energy bar bars", "deck" -> "duck"; fp16
  in the rotation). User: worse = not used. Scripts were deleted with the experiment.
  **Retried 2026-11-27** (.cache/tmp/ane/sf/): the precision theory was wrong. Position term of layer 17 on the real
  ANE (compute plan: all on the ANE) vs fp32: today's matmul + shift rms 0.085, shift-free (A_i . cos jw + B_i .
  sin jw, 1024-long dot per head) rms 0.079, so no fp16 loss on the ANE (it sums in higher precision; the CPU in
  fp16 does not: 0.45 vs 1.41). Full model, 10 reference takes: 73 = 73 errors (2 words flip, one each way: noise of
  any equivalent change). But this version is not faster: encoder 4733 vs 4790 ms (-1%); the 1024-long dot (~2.5
  GFLOP more per layer) eats what the shift saved. How the 2026-11-25 version got 112 ms is not known (script gone).
  **Found (`sfp`)**: sin and cos halves of relative_k_proj as two separate weights (`Wh[..., 0::2]`, `[..., 1::2]`)
  instead of one interleaved output split on the ANE. 6-layer cut: 33.5 vs 38.5 ms. Full model, 10 reference takes:
  encoder 4266 vs 4823 ms (-11.5%), 74 vs 73 errors. vs fp32 (CPU, transformers), 33 windows of the same takes: rel rms
  3.862% vs 3.876% (today), further in 17 of 33 windows, so equal. All 1751 takes (5.6 h; check stops at the unreadable
  20261127-012452.opus): encoder 215.5 vs 248.0 s (-13.1%), 20 spots / 24 words differ (0.08%). User listened blind
  (.cache/tmp/ane/sf/listen/index.html, clips + comments): today right 9, new right 7, 4 ties. My reading-only verdict
  (8 worse, 1-3 better) was wrong on several (fillers and repeats the user really said). Scripts: .cache/tmp/ane/sf/
  (export_sf.py, bench6.py, vsfp32b.py, alldiff.py, listen/make.py).
  **Adopted 2026-11-27**, with one fix: CoreML had folded the sin/cos tables (built from torch ops in the graph) with a
  poor sin for large angles (sin(374) stored as -0.113, true -0.150; 22% of the table's values off). export.py now
  computes them (f64 -> f32) and passes finished constants. Exact tables: vs fp32 closer than today in 20 of 33
  windows (max abs 0.108 vs 0.115), 10 reference takes: only "energy bar bars" differs (a stumble, user: either fits;
  ONNX/int8 flips gone), encoder 4168 vs 4823 ms (-13.6%). Recipe kinds `rkp` (relative_k_proj's sin/cos half, copied)
  and `sincos` (tables, same bytes as CoreML's) replace `pos`; consts.bin 57 KB; the app's build (anebuild) gives
  CoreML's weight.bin byte for byte and the same texts. New folder name, so existing installs download and build again.
  The listening test above was on the inexact tables.
  **More after adopting (2026-11-27, tried, nothing gained)** (.cache/tmp/ane/sub/: abl.py, rot.py, fast.py; 6-layer
  cuts, base 32.1-32.8 ms). CoreML's cost estimate for the shipped model: linear 43%, add 13%, mul 10%, matmul 8%,
  conv 5.5%, transpose 5%. Ablations (ops dropped, wrong math, timing only): sin/cos rotation (A, B: 6 elementwise
  ops on [8, 376, 512] per layer) -3.4 ms = ~0.57 ms/layer, ~14 ms (~12%) of the full encoder; score scaling, attention
  mask, bias_u/v adds ~0; all masks (subsampler + attention + conv) -3.5 ms. Exact rewrites, none faster: rotation
  tables pre-expanded to 8 heads 33.3; [A|B] fused from P = qv@[Ws|Wc], Q = qv@[Wc|Ws] + one K=1024 matmul 33.2;
  5-op rotation (t = SI * qv@(Ws+Wc), A = t + uc*(CI-SI), B = t - us*(SI+CI)) 32.2. The rotation's cost is its
  volume of elementwise work, not broadcasting. Subsampler: dropping the redundant masks (after the depthwise convs:
  the pointwise mask redoes them; after pw2: padded rows are ignored downstream) 33.1; conv1's mask (24.6M values)
  folded into conv1 as a 2nd input channel (-1e4 on padded mel rows, centre tap only, relu zeroes them): exact, but
  CoreML puts that conv on the CPU (49.6 ms; with the weights as a constant still CPU in the plan, 32.2 ms = no
  gain). A standalone 2-channel conv runs on the ANE, so it's the stacked input (stack/tile on the CPU). Left at
  that: at most ~2-3 ms per window there.
- **Smaller window for short takes (discussed, not done)**: takes are short (median 7.3 s, 63% under
  10 s, 31% under 4 s; Silero VAD: 34% of the audio is silence, median 0.7 s before the first word). A 10 s window
  (~35 ms encoder, between the old 5 s 21 ms and 15 s 47 ms) would bring short takes from ~150-200 to ~60-110 ms.
  Cost: a second program (~1.2 GB more aned disk, one more ~40 s compile, purgeable); RAM can stay at 1.2 GB by
  loading the 10 s model at start and switching to the 30 s one once a take passes ~8-9 s (both loaded only during
  the switch). Takes of 10-30 s and leftover pieces of long takes would still use the 30 s window.
  **Tried in the app (2026-11-25, reverted)**: a 10 s model (export.py 1001 frames) in place of the 30 s one, pieces
  capped at the window. Takes <= 10 s: 38-76 ms total (encoder 27-35 ms), felt instant. Takes over 10 s cut at
  pauses: 13.5 s -> 8.2 + 5.3 s 115 ms, 18.6 s -> 9.4 + 9.2 s 146 ms, 21.7 s -> 7.8 + 9.2 + 4.7 s 169 ms (pause scan
  27-56 ms, ~30-35 ms encoder per piece); text quality of the 10 s pieces not measured. Back to 30 s only: one
  compiled model, no switching (the user's choice; 10 + 30 or 10 + 20 + 30 with switching = 2-3 compiled programs).
- **Silence** (Silero VAD, all takes): 34% silence = 7.0% before the first word, 6.2% after the last, 1.0%
  takes without speech, 20% pauses inside (under 0.75 s 5.4%, 0.75-1.5 s 5.1%, over 1.5 s 9.6%). Capping inner
  pauses at 0.75 s + 0.3 s at the edges: ~20% shorter audio; takes fitting 10 s 61% -> 68%, 20 s 85% -> 89%. No gain
  with the fixed 30 s window (encoder time is flat), not done.
- **Purges seen again** (2026-11-25 15:25, ~16-17 GB free): `deleted` asked for 2.8 GB, purged 3.6 GB (~3 compiled
  programs), the next take recompiled (26 s). macOS's Storage "available" (~40 GB) counts purgeable space, incl.
  these programs; `diskutil info /` "Container Free Space" is the real free space. `check` (a separate, freshly built
  binary) compiled its own copy of the 10 s model again (~27 s) although the app had it compiled.
- **Why 30 s pieces**: the GPU with 15 s pieces (as the 15 s FluidInference model forces) drops from 5.44% to 7.43%
  (numbers and terms spelled out: "FP thirty two", "int 8"); short takes unaffected. So the ANE model has a 30 s window.
- **User's decisions**: the ANE option has its own download and folder, never reuses the GGUF model's files; the
  weights are the fp16 original (Q8 vs fp16 costs no quality, see above, but fp16 is the ANE's format and needs no
  conversion). Stalls on reload: find the cause, no "always keep loaded" workaround.
- **Not done yet**: a comparison against the m1 kernels branch (the GPU numbers above are main's GPU path; m1 kernels
  ~26% faster; its build dir .cache/target-m1 was deleted 2026-11-25 for disk space), a Linux build of this
  commit (ANE code is macOS-gated, not compiled there), re-running experiments/ane/*.py as committed, a clean-slate
  test of the whole download -> build -> prepare flow in the app.
- `check`: `ANE=<dir>` (a folder with the model) or `ANE=1` (the app's). `cargo run --release --example anebuild --
  <model.safetensors> <out dir>` builds the model like the app.
- **Decoder on the ANE (discussed, not done)**: not worth it. The TDT decoder runs one tiny step per token (~13 MB of
  int8 weights, cached on the CPU), each step depends on the previous (argmax, duration jump); a CoreML call alone is
  ~0.1 ms+, hundreds of steps per take. Decoder = 15-30% of a take with the 30 s window (3-8 s takes 19-34 ms, 38.9 s
  128 ms). If ever: on the CPU (rayon per step may cost more than it saves on the M1; Accelerate/AMX), measure first.
- **Newer chips (Apple's figures, not measured)**: Neural Engine M1 11 TOPS, M2 15.8, M3 18, M4 38 (int8; fp16 ~1.5-2x
  the M1), M5 16-core + a Neural Accelerator in each GPU core, M6 (Aug 2026) two 16-core Neural Engines + GPU
  accelerators, FP8. Our CoreML model runs as is and only gets faster (macOS compiles it per chip). A17 Pro/M4+:
  native int8 (W8A8 needs activations quantized too, quality to check); no CoreML API reports it (MLComputePlan shows
  the device per op only). M5+: the GPU (via Metal/MLX, not our burn/wgpu path) may beat the ANE, any length.

### No GPU path on macOS (2026-11-25, user's decision)
The Neural Engine beats the M1's GPU on everything but the shortest takes (~30 ms), so on macOS the GPU path is gone:
burn/CubeCL is a dependency only off macOS (Cargo.toml target tables), and build.rs sets `cfg(gpu)` for the code
that uses it (src/asr/encoder.rs, src/asr/gpu.rs = GPU choice + kernel cache + sweep bookkeeping, the GPU engine in
pipeline.rs, sweep jobs in daemon.rs/main.rs). Sizes and helpers the CPU/ANE paths share moved to src/asr/shapes.rs;
weights.rs writes the q8 bytes itself (int8 values, then f16 scales; GPU builds assert they equal burn's at
conversion), so the Mac writes the same files. `patches/` (Metal kernel cache fixes) deleted. Mac menu: Neural Engine
(default), Ultra on the CPU, Redux on the CPU; a config with a GPU choice is moved to the Neural Engine.
Measured on the M1: stripped release binary 78 MB -> 10 MB; Neural Engine take (15.5 s) gives the same text as the
app before (176 ms); CPU Ultra works; existing weights files' q8 layout matches. Linux build not compiled yet
(removing the patches changes Cargo.lock, so the GPU sweep runs once more there).


### Own q8 matmul kernel on Vulkan/NVIDIA: measured, not kept (2026-11-24)
Tried on the 3050 Ti: an own CubeCL kernel for the encoder's 9 q8 linears per block (burn backend extension op,
from the M1 branch's kern.rs with the fragment size as a parameter: Vulkan cooperative matrix on NVIDIA is 16x16x16
f16 with f32 accumulate, M1 Metal 8x8x8). int8 weights + f16 scales uploaded straight from the weights file; split K
for few rows (grid z = slices, f32 partial sums, then a sum kernel that also does silu); tiles by row count (<= 32
rows 32x128, <= 64 rows 64x128 so weights are read once, K split until ~32 cubes on the 20 SMs; more rows 64x64).
- Isolated vs burn's q8 matmul: 19 rows ~2x faster (ff linear1 0.09 -> ~0.05 ms, linear2 ~0.13 -> 0.048, linear_q
  0.034 -> 0.018); 63 rows ~1.5x; 375 rows (30 s piece) about equal, both ~10 TFLOPS, compute-bound. Error ~3e-4.
- All 235 recordings, same binary with and without it: texts identical; encoder 20.9 -> 18.5 s (-12%), asr
  29.1 -> 26.8 s (-8%). By take length: 0-3 s -4%, 3-8 s -9%, 8-60 s -12%, 60 s+ -14%.
- Why not kept: ~10 ms per take on average, a few ms on typical short takes (not noticeable), for ~420 lines of own
  GPU code tied to the burn/CubeCL version and tiles tuned on this one GPU. Short takes gained far less than the
  isolated speedup predicts (~1.3 ms instead of ~10 ms per 2 s take); unverified guess: CPU-side launch/fusion
  overhead of the many small ops. Worth revisiting only for long-file transcription on the GPU, together with fused
  residual + LayerNorm and fused relative attention (M1's `res_ln` / `att_kernel`, estimated 10-20% more on long takes).
- Pitfalls: zsh doesn't word-split `$var` (sweeps silently ran defaults; use bash); CubeCL `terminate!()` did not end
  the kernel, so "empty kernel" tests are invalid; burn's `x + 1` on a big f16 tensor ran ~57 GB/s vs ~192 GB/s
  spec (not investigated).

### Long-file speed and other Parakeet-on-Apple numbers (2026-11-27, M1, measured)
A Reddit comment claimed Parakeet v3 at ~415x on an M3 Max (Argmax Playground, encoder + decoder on the Neural Engine;
its "Transcribe" 7.15 s leaves out time its own "Total" 11.59 s counts = ~256x) and "M1 Air only 50% slower". Published:
Argmax "over 100x" on an M1 Air (v2), FluidAudio ~110-145x on an M4 Pro. Argmax ships compressed weights (~0.4-0.5 GB,
probably palettized; older ANEs don't run those natively, as our int8 test). So nobody is far ahead on the M1.
Ours, 3.4 h podcast (12,120 s mp3) through the app (`dictation transcribe`, old encoder): 104 s transcription = 116x,
119 s end to end with mp3 decoding (102x); mel 4.4 s, pause scan 22.9 s, encoder 71.1 s, decoder 69.2 s, which runs
on CPU threads while the ANE encodes the next piece, so it costs only ~6 s of waiting. The critical path is pause
scan + encoder. The same file through `check` while the user was dictating: 157 s (77x; the app and check share the
ANE/CPU). With the ~13% faster encoder (below in "Faster encoder?"): ~125x estimated, not measured.
New encoder, all 1751 takes: under 5 s median 117 ms (~25x), 5-10 s 127 ms (~57x), 10-30 s 142 ms (~114x), 30-60 s
339 ms, over 60 s 566 ms (~132x); short takes are low because every take costs one full 30 s window.
Dictation: takes <= 30 s have no pause scan and decode after the encode (decoder 15-30% of a take); over 30 s the scan
runs after the stop, before the first encode.
Ideas not done (long files only): pause scan ahead of the encoder (scan a few minutes, start encoding; same text, up to
~20 s of 104 saved); padding: pieces cut at the last pause before 30 s are padded to 30 s (encoder 176 ms per window
on average assumes 404 full windows; more, shorter pieces at ~130 ms fit as well; count pieces before changing cuts).
Attention tiled for the ANE's 32 MB on-chip memory (Orion paper, arXiv 2603.06728: ~30% slower past it): untested.
Tooling found on the way: two big CoreML models used alternately in one Python process hung inside
`MLE5ExecutionStream cancelPendingReset` on the second window: one model per process (vsfp32b.py). `check` stops at
the first file it can't read: 20261127-012452.opus ("not a format this app reads"), not looked into.

### "A buffer underrun or overrun occurred" in the log (2026-11-27)
cpal's macOS backend reports Core Audio's `kAudioDeviceProcessorOverload` (the device's I/O thread missed a cycle) as
`Xrun`; we print every cpal error with `audio:` (audio.rs). In the log: 101 of 861 takes (~12%), always as the mic
stream opens (MOMENTUM 4 switching to headset mode restarts its I/O), never mid-take, since the first day. Before the
mic counts as live (25-38 ms), so no speech lost. Not changed (user OK): hiding it would be cosmetic, keeping the mic
open is the kind of audio workaround ruled out.

### Recording window: smooth bars and a done mark (2026-11-27, M1, measured)
- Smoothness measured from screen recordings, not by eye: `.cache/tmp/pill/record.sh <label>` (app on system audio +
  clipboard-only output, config backed up and restored, `dictation toggle`, a generated burst sound, `screencapture
  -v -R` of the pill), `motion.py <mov>` (bars' horizontal shift per frame by profile matching).
- The stutter ("runs, then stops") was the bar step clock restarting at the frame time: frames rarely land on the
  50 ms mark, so one frame in ~7 showed the row in the same place (54 of 366 frames). Steps now on a fixed 50 ms
  grid: 0 stalled frames. Not the cause (measured): frame timing (16.7 ms steady, 0 late) and the mic (512 samples
  every 11.6 ms). Remaining double jumps match frames missing in the recording (recorder or a skipped refresh).
- 60 fps; the row slides between steps, the incoming bar follows the mic's peak live. macOS frames come from a
  CoreVideo display link on the pill's screen (timer fallback); everything moves by time, so 120 Hz should work
  (not tested, the M1 Air is 60 Hz). Linux keeps its timer (its frame callbacks not used); Linux side not compiled
  here (no Linux target installed), CI builds it.
- Done mark: green check (text pasted or copied), dim dash (nothing recognized), red cross (copy/paste/transcription
  failed), 650 ms then the pill fades. User after the fix: "much smoother".

### Open at login, permission warnings, version (2026-11-27, M1)
- **Version**: tray menu line "Dictation <version>" above Quit, `dictation --version`; both from Cargo.toml (Info.plist
  already had it). A release is still the pushed `v*` tag; nothing checks the tag matches Cargo.toml (user: fine).
- **Open at login** (tray checkbox, src/login.rs; the system keeps the setting, not config.toml). macOS: SMAppService
  main app (13+), only as Dictation.app (a bare binary has no bundle: the checkbox is hidden); tested register ->
  enabled -> unregister. Login items added by hand are another list the app can't read (user: assume the app does
  it). Linux: `~/.config/autostart/dictation.desktop`, one file whatever binary ticks it; at start only its Exec line
  follows a moved binary; Hidden=true (switched off in KDE) reads as off. Linux logic run against a temp home on the
  Mac, the app's Linux build not compiled here.
- **Permissions** (src/perms.rs, macOS): Accessibility (AXIsProcessTrusted) and microphone (AVCaptureDevice status)
  checked without prompting; missing -> tray icon "!" in orange (not a template) and a ⚠ line at the top of the menu
  that opens the settings page (user: never open settings by itself). Rechecked every event / 10 s tick, so granting
  clears it. Not checkable: system audio recording, notifications.
  - Microphone: when the first take's prompt came from opening the audio stream, "Don't Allow" left this process
    reading "not asked yet" (measured: status 0 after the denial) and takes recorded silence. Now the app asks via
    AVFoundation before opening the mic (needs Info.plist's usage text: .app only; the bare binary keeps the old way).
  - Accessibility: the app asks once at start; the ⚠ line asks again (re-adds Dictation to the list) and opens it.
  - Every build is ad-hoc signed anew: macOS keeps one entry per bundle ID per permission, with the signature it was
    granted to; a new build doesn't match ("Failed to match existing code requirement" in tccd's log), the switch still
    shows on but doesn't apply. Users hit this on every update: remove it with "-" and add again (paste notice says
    so). Real fix: a Developer ID signature (same identity every build), user's decision.
- **Pill**: amber check = meant to paste, couldn't (text on the clipboard); red cross only for copy or transcription
  failures.
