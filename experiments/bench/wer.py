"""Word error rate of each take in recordings/log.jsonl against a reference text.
usage: python3 experiments/bench/wer.py <reference.txt> [wav name, default: last take]"""
import json, re, sys
from pathlib import Path

def words(t):
    t = t.lower().replace("’", "'").replace("ʼ", "'")
    return re.findall(r"[\w']+", t)

def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r), len(r)

ref = Path(sys.argv[1]).read_text()
takes = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent.parent / "recordings/log.jsonl")]
take = next(t for t in takes if t["wav"] == sys.argv[2]) if len(sys.argv) > 2 else takes[-1]
print(f'{take["wav"]}  {take["seconds"]}s  nemotron_lang={take["nemotron_lang"]}')
for m in ("nemotron", "parakeet"):
    w, n = wer(ref, take[m])
    print(f"  {m:9s} WER {w:6.1%}  ({n} reference words)")
