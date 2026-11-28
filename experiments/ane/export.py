"""Parakeet Ultra's encoder (moondream/parakeet-ultra, fp16 as published) -> CoreML for the Neural Engine, one fixed
window. Step 1 of making assets/parakeet-ultra-ane (step 2: recipe.py); NOTES.md "Neural Engine".
usage: python export.py <hf dir with config.json + model.safetensors> <mel frames: 501 | 1501 | 3001> <out dir>
Input `mel` [1, frames, 128] (frame-major) + `length` [1] (valid mel frames; the rest is masked), output
`encoder` [1, frames/8, 1024]. Python 3.12 + torch 2.7 + coremltools 9 + transformers 5.24 (see NOTES.md).
Attention's position term without the relative shift (pad + reshape, slow on the ANE): sin/cos of (i - j) split into i
and j, so qv . W pe(i-j) = A_i . cos(j w) + B_i . sin(j w), with relative_k_proj's sin and cos halves as two separate
weights. Same math (fp32: 3e-4 on outputs ~1000), encoder ~12% faster on the M1 (NOTES.md "Faster encoder?")."""
import sys, numpy as np, torch, torch.nn as nn
from transformers.models.parakeet import modeling_parakeet as mp
from transformers import ParakeetForTDT
HF, FR, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def sincos(t, C):
    # Computed here (f64, rounded to f32) and passed as finished constants: built from torch ops in the graph, CoreML
    # folded them itself with a poor sin for large angles (sin(374) = -0.113 instead of -0.150).
    ang = np.arange(t, dtype=np.float64)[:, None] / 10000 ** (np.arange(0, C, 2, dtype=np.float64) / C)
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

def attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
    # transformers' eager attention, but masked with -1e4 (its finfo(f32).min becomes -inf in fp16 -> NaN rows), and the
    # position term without the shift.
    b, t = hidden_states.shape[:2]; h, d = self.config.num_attention_heads, self.head_dim
    q = self.q_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    qu = q + self.bias_u.view(1, h, 1, d); qv = q + self.bias_v.view(1, h, 1, d)
    C = self.config.hidden_size
    SI, CI = map(torch.from_numpy, sincos(t, C))                       # [t, C/2]: sin(j w), cos(j w)
    Wh = self.relative_k_proj.weight.view(h, d, C)                     # relative_k_proj has no bias; pe = (sin, cos) interleaved
    Ws, Wc = Wh[..., 0::2].contiguous(), Wh[..., 1::2].contiguous()   # two weights: one interleaved output split on the ANE was slower
    us = qv @ Ws.unsqueeze(0); uc = qv @ Wc.unsqueeze(0)              # [b, h, t, C/2]
    A = us * SI + uc * CI; B = uc * SI - us * CI                       # rotated by i w
    bd = A @ CI.T + B @ SI.T                                           # [b, h, t, t] = qv_i . W pe(i - j)
    s = (qu @ k.transpose(-1, -2) + bd) * self.scaling
    if attention_mask is not None:
        s = s.masked_fill(attention_mask.logical_not(), -1e4)
    o = torch.softmax(s, dim=-1) @ v
    return self.o_proj(o.transpose(1, 2).reshape(b, t, h * d)), None
mp.ParakeetEncoderAttention.forward = attn_forward

enc = ParakeetForTDT.from_pretrained(HF, torch_dtype=torch.float32).encoder.eval()

class Wrap(nn.Module):
    def __init__(s, e): super().__init__(); s.e = e
    def forward(s, mel, length):
        mask = (torch.arange(mel.shape[1]).unsqueeze(0) < length.unsqueeze(1))
        return s.e(input_features=mel, attention_mask=mask).last_hidden_state

import coremltools as ct
ex = (torch.zeros(1, FR, 128), torch.tensor([FR], dtype=torch.int32))
tr = torch.jit.trace(Wrap(enc).eval(), ex, strict=False)
m = ct.convert(tr, inputs=[ct.TensorType(name='mel', shape=(1, FR, 128), dtype=np.float32),
                           ct.TensorType(name='length', shape=(1,), dtype=np.int32)],
               outputs=[ct.TensorType(name='encoder', dtype=np.float32)],
               compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14,
               compute_units=ct.ComputeUnit.CPU_AND_NE)
m.save(f'{OUT}/enc{FR}.mlpackage')
print('saved', f'{OUT}/enc{FR}.mlpackage')
