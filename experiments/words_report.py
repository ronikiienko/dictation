"""Summary of words.rs output: Cyrillic / accented / non-dictionary outputs. usage: python3 experiments/words_report.py <words.rs output .jsonl>"""
import json, re, sys, glob, collections
eng = {w.strip().lower() for w in open("/usr/share/dict/words")}
for t in glob.glob(".cache/tmp/librispeech/LibriSpeech/*/*/*/*.trans.txt"):
    for l in open(t): eng |= {w.lower() for w in l.split()[1:]}
rows = [json.loads(l) for l in open(sys.argv[1])]
spk = lambda r: r["file"].split("/")[-1].split("-")[0]
cyr = [r for r in rows if r["script"] == "cyrillic"]
acc = [r for r in rows if r["script"] in ("latin+", "other")]
def foreign(r):
    ws = re.findall(r"[a-z']+", r["alone"].lower())
    return [w for w in ws if w.strip("'") not in eng and w.strip("'").removesuffix("'s") not in eng]
lat = [(r, foreign(r)) for r in rows if r["script"] == "latin" and foreign(r)]
print(f"{len(rows)} cuts, {len({spk(r) for r in rows})} speakers; cyrillic {len(cyr)} from {len({spk(r) for r in cyr})} speakers")
print(f"words/cut s: median {sorted(r['end']-r['start'] for r in rows)[len(rows)//2]:.2f}")
for name, rs in (("CYRILLIC", cyr), ("ACCENTED/OTHER", acc)):
    print(f"\n{name}"); [print(f"  {spk(r):5} {r['word']:14} -> {r['alone']}") for r in rs]
print("\nLATIN, NOT IN ENGLISH DICTIONARY"); [print(f"  {spk(r):5} {r['word']:14} -> {r['alone']}   {f}") for r, f in lat]
