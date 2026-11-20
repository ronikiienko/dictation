"""Shared bits for the bench scripts: audio loading, references, WER. Only needs numpy."""
import re
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
RATE = 16000
REFS = {  # take -> reference text it was read from
    "20261115-141151.wav": ROOT / "experiments/bench/ref_en_int8.txt",
    "20261115-141800.wav": ROOT / "experiments/bench/ref_uk_int8.txt",
}

# Number words, so a model writing "650" isn't punished against a reference saying "six hundred fifty"
# (and vice versa). Only used for the "numbers ignored" score.
NUMBER_WORDS = re.compile(
    r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|point|"
    r"нуль|один|одна|два|дві|двох|три|трьох|чотири|чотирьох|чотирма|п'ять|шість|шістсот|сім|вісім|"
    r"восьми|дев'ять|десять|п'ятдесят|цілих|ціла)$")


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


def words(text, drop_numbers=False):
    t = text.lower().replace("’", "'").replace("ʼ", "'")
    w = re.findall(r"[\w']+", t)
    if drop_numbers:
        w = [x for x in w if not x.isdigit() and not NUMBER_WORDS.match(x)]
    return w


def wer(ref, hyp, drop_numbers=False):
    r, h = words(ref, drop_numbers), words(hyp, drop_numbers)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def score(take, text):
    """'' for takes without a reference, else a printable WER string + dict of numbers."""
    ref = REFS.get(take)
    if not ref:
        return "", {}
    t = ref.read_text()
    raw, nonum = wer(t, text), wer(t, text, drop_numbers=True)
    return (f"WER {raw:5.1%}  (numbers ignored {nonum:5.1%})  ",
            {"wer": round(raw, 4), "wer_no_numbers": round(nonum, 4)})
