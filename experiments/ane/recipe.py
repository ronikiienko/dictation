"""Step 2 of making assets/parakeet-ultra-ane (step 1: export.py; NOTES.md "Neural Engine"): the 30 s window package
-> compiled -> what the app ships to rebuild it from moondream/parakeet-ultra's model.safetensors (src/asr/ane_build.rs):
the compiled model folder without weights/weight.bin, plus recipe.json for weight.bin: a list of segments, each one of
  ["bytes", start, len]              literal bytes from consts.bin (file/blob headers, architecture-only constants)
  ["tensor", name]                   a safetensors tensor as fp16 (F32 ones rounded), same element order
  ["bn_w", layer] / ["bn_b", layer]  depthwise conv with its batch norm folded in (weight / bias)
  ["rkp", layer, half]               relative_k_proj's sin (half 0: even inputs) or cos (1: odd) columns, as [1, 8, 128, 512]
  ["sincos", "sin"|"cos", T, tr]      sin/cos(j w) of the attention's position term, [T, 512] (tr 1: [512, T])
Checks that numpy's version of every computed blob matches CoreML's (prints the worst difference).
One plain 30 s model, not several windows: one compiled program for macOS to keep instead of 3 (NOTES.md "Neural Engine").
usage: python recipe.py <hf dir> <enc3001.mlpackage> <out dir, e.g. assets/parakeet-ultra-ane>"""
import sys, os, re, json, struct, shutil, hashlib, subprocess, tempfile, numpy as np
from safetensors import safe_open
ST, PK, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/'), sys.argv[3]
TMP = tempfile.mkdtemp()
subprocess.run(['xcrun', 'coremlcompiler', 'compile', PK, TMP], check=True, stdout=subprocess.DEVNULL)
M = f'{TMP}/' + os.path.basename(PK).replace('.mlpackage', '.mlmodelc')
cfg = json.load(open(ST + 'config.json'))
ecfg = cfg.get('encoder_config', cfg)
mil = open(M + '/model.mil').read()
wb = open(M + '/weights/weight.bin', 'rb').read()
# both MIL text forms: `string("x")` / `uint64(n)` and the older `tensor<string, []>("x")` / `tensor<uint64, []>(n)`
S, U = r'(?:string|tensor<string, \[\]>)', r'(?:uint64|tensor<uint64, \[\]>)'
pat = re.compile(r'tensor<(\w+), \[([\d, ]*)\]> (\w+) = const\(\)\[name = ' + S + r'\("([^"]+)"\), val = tensor<[^>]+>\(BLOBFILE\(path = ' + S + r'\("[^"]+"\), offset = ' + U + r'\((\d+)\)\)\)\]')
blobs = {}
for dt, shape, var, name, off in pat.findall(mil):
    off = int(off)
    sentinel, mdt, size, doff = struct.unpack_from('<IIQQ', wb, off)
    assert sentinel == 0xDEADBEEF
    blobs.setdefault(off, dict(name=name, dt=dt, shape=tuple(int(s) for s in shape.split(',') if s.strip()), size=size, doff=doff))

f = safe_open(ST + 'model.safetensors', 'np')
src = {}
for k in f.keys():
    if k.startswith('encoder.'):
        a = f.get_tensor(k)
        if a.dtype in (np.float16, np.float32):
            src.setdefault(hashlib.sha1(a.astype(np.float16).tobytes()).digest(), k)
g = lambda k: f.get_tensor(k).astype(np.float32)

L = sum(1 for k in f.keys() if k.endswith('.self_attn.q_proj.weight'))
D = g('encoder.layers.0.self_attn.q_proj.weight').shape[0]
H = ecfg.get('num_attention_heads', 8)
eps = 1e-5

def bn(i):
    w = g(f'encoder.layers.{i}.conv.depthwise_conv.weight')
    p = f'encoder.layers.{i}.conv.norm.'
    s = g(p + 'weight') / np.sqrt(g(p + 'running_var') + np.float32(eps))
    return (w * s[:, None, None]).astype(np.float16), (g(p + 'bias') - g(p + 'running_mean') * s).astype(np.float16)

def sincos(fn, T):
    """export.py's sincos(), rounded to fp16 (as CoreML stores it)."""
    ang = np.arange(T, dtype=np.float64)[:, None] / 10000 ** (np.arange(0, D, 2, dtype=np.float64) / D)
    return getattr(np, fn)(ang).astype(np.float32).astype(np.float16)

for i in range(L):   # relative_k_proj's halves, exact copies
    w = f.get_tensor(f'encoder.layers.{i}.self_attn.relative_k_proj.weight').astype(np.float16).reshape(H, D // H, D)
    for half in (0, 1):
        src.setdefault(hashlib.sha1(np.ascontiguousarray(w[..., half::2]).tobytes()).digest(), ('rkp', i, half))

cands = {}
for i in range(L):
    w, b = bn(i); cands[('bn_w', i)] = w; cands[('bn_b', i)] = b
seg, consts, worst = [], bytearray(), {}
def lit(a, b):
    seg.append(['bytes', len(consts), b - a]); consts.extend(wb[a:b])
pos = 0
for off in sorted(blobs):
    bl = blobs[off]
    data = wb[bl['doff']:bl['doff'] + bl['size']]
    lit(pos, bl['doff']) if bl['doff'] > pos else None
    k = src.get(hashlib.sha1(data).digest())
    kind = None
    if k is not None:
        kind = list(k) if isinstance(k, tuple) else ['tensor', k]
    elif bl['dt'] == 'fp16' and len(bl['shape']) == 2 and D // 2 in bl['shape']:
        tr = int(bl['shape'][0] == D // 2); T = bl['shape'][tr]
        got = np.frombuffer(data, np.float16).reshape(bl['shape']).astype(np.float32)
        for fn in ('sin', 'cos'):
            want = sincos(fn, T).astype(np.float32); want = want.T if tr else want
            e = np.abs(want - got).max()
            if e <= 1e-3:
                kind = ['sincos', fn, T, tr]; worst['sincos'] = max(worst.get('sincos', 0), e); break
    elif bl['dt'] == 'fp16':
        got = np.frombuffer(data, np.float16)
        best = min(((np.abs(c.reshape(-1).astype(np.float32) - got.astype(np.float32)).max(), key)
                    for key, c in cands.items() if c.size == got.size), default=None)
        if best is not None and best[0] <= 2e-3 * max(np.abs(got).max(), 1e-9):
            kind = [best[1][0], best[1][1]]; worst[best[1][0]] = max(worst.get(best[1][0], 0), best[0])
    if kind is None:   # architecture-only constant: keep its bytes
        lit(bl['doff'], bl['doff'] + bl['size']); print('literal', bl['name'], bl['dt'], bl['shape'])
    else:
        seg.append(kind)
    pos = bl['doff'] + bl['size']
lit(pos, len(wb)) if len(wb) > pos else None
# merge neighbouring literal segments
out = []
for s in seg:
    if s[0] == 'bytes' and out and out[-1][0] == 'bytes' and out[-1][1] + out[-1][2] == s[1]: out[-1][2] += s[2]
    else: out.append(s)
from collections import Counter
print('segments', Counter(s[0] for s in out), 'literal bytes', len(consts), 'worst', worst)
os.makedirs(OUT, exist_ok=True)
for fn in ('model.mil', 'coremldata.bin', 'metadata.json'):
    shutil.copy(M + '/' + fn, OUT + '/' + fn)
os.makedirs(OUT + '/analytics', exist_ok=True); shutil.copy(M + '/analytics/coremldata.bin', OUT + '/analytics/coremldata.bin')
open(OUT + '/consts.bin', 'wb').write(consts)
json.dump({'weight_bin_size': len(wb), 'sha256': hashlib.sha256(wb).hexdigest(), 'segments': out}, open(OUT + '/recipe.json', 'w'))
print('written', OUT)
