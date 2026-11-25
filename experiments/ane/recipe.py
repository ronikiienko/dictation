"""Step 2 of making assets/parakeet-ultra-ane (step 1: export.py; NOTES.md "Neural Engine"): the three window
packages -> one multifunction model (functions w501/w1501/w3001 sharing the weights) -> compiled -> what the app ships
to rebuild it from moondream/parakeet-ultra's model.safetensors (src/asr/ane_build.rs): the compiled model folder
without weights/weight.bin, plus recipe.json for weight.bin: a list of segments, each one of
  ["bytes", start, len]              literal bytes from consts.bin (file/blob headers, architecture-only constants)
  ["tensor", name]                   a safetensors tensor as fp16 (F32 ones rounded), same element order
  ["bn_w", layer] / ["bn_b", layer]  depthwise conv with its batch norm folded in (weight / bias)
  ["pos", layer, P]                  relative_k_proj(position table of P positions), as [1, 8, 128, P]
Checks that numpy's version of every computed blob matches CoreML's (prints the worst difference).
usage: python recipe.py <hf dir> <dir with enc501/enc1501/enc3001.mlpackage> <out dir, e.g. assets/parakeet-ultra-ane>"""
import sys, os, re, json, struct, shutil, hashlib, subprocess, numpy as np
from safetensors import safe_open
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
ST, PK, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2], sys.argv[3]
d = MultiFunctionDescriptor()
for w in (501, 1501, 3001):
    d.add_function(f'{PK}/enc{w}.mlpackage', src_function_name='main', target_function_name=f'w{w}')
d.default_function_name = 'w3001'
save_multifunction(d, f'{PK}/enc.mlpackage')
subprocess.run(['xcrun', 'coremlcompiler', 'compile', f'{PK}/enc.mlpackage', PK], check=True, stdout=subprocess.DEVNULL)
M = f'{PK}/enc.mlmodelc'
cfg = json.load(open(ST + 'config.json'))
ecfg = cfg.get('encoder_config', cfg)
mil = open(M + '/model.mil').read()
wb = open(M + '/weights/weight.bin', 'rb').read()
pat = re.compile(r'tensor<(\w+), \[([\d, ]*)\]> (\w+) = const\(\)\[name = string\("([^"]+)"\), val = tensor<[^>]+>\(BLOBFILE\(path = string\("[^"]+"\), offset = uint64\((\d+)\)\)\)\]')
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

def pos_table(P):
    """transformers' ParakeetEncoderRelPositionalEncoding for P = 2T-1 positions (T-1 .. -(T-1))."""
    T = (P + 1) // 2
    pos = np.arange(T - 1, -T, -1, dtype=np.float32)
    inv = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float32) / D))
    ang = pos[:, None] * inv[None, :]
    pe = np.zeros((P, D), np.float32); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
    return pe

def posw(i, P):
    y = pos_table(P) @ g(f'encoder.layers.{i}.self_attn.relative_k_proj.weight').T   # [P, D]
    return y.reshape(P, H, D // H).transpose(1, 2, 0)[None].astype(np.float16)        # [1, H, D/H, P]

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
        kind = ['tensor', k]
    elif bl['dt'] == 'fp16' and len(bl['shape']) == 4 and bl['shape'][:3] == (1, H, D // H):
        P = bl['shape'][3]
        got = np.frombuffer(data, np.float16).reshape(bl['shape']).astype(np.float32)
        errs = [(np.abs(posw(i, P).astype(np.float32) - got).max(), i) for i in range(L)]
        e, i = min(errs); kind = ['pos', i, P]; worst['pos'] = max(worst.get('pos', 0), e / max(np.abs(got).max(), 1e-9))
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
