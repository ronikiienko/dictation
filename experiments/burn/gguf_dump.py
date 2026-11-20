"""List GGUF metadata and tensors. usage: python gguf_dump.py file.gguf [--meta-only]"""
import struct, sys
T = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
GT = {0:'F32',1:'F16',8:'Q8_0',2:'Q4_0',12:'Q4_K',14:'Q6_K',30:'BF16',24:'I8',26:'I32'}
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize(fmt)))[0]
def rstr(): return f.read(rd('Q')).decode('utf-8', 'replace')
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        et, n = rd('I'), rd('Q')
        v = [rval(et) for _ in range(n)]
        return v if n <= 8 else f'[{n} x type{et}] first: {v[:4]}'
    return rd(T[t])
assert f.read(4) == b'GGUF'; ver = rd('I'); nt = rd('Q'); nkv = rd('Q')
meta = {}
for _ in range(nkv):
    k = rstr(); meta[k] = rval(rd('I'))
for k, v in meta.items(): print(f'{k} = {v}')
if '--meta-only' in sys.argv: sys.exit()
for _ in range(nt):
    name = rstr(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); off = rd('Q')
    print(f'T {name} {GT.get(t,t)} {dims}')
