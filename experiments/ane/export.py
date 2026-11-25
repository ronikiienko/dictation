"""Parakeet Ultra's encoder (moondream/parakeet-ultra, fp16 as published) -> CoreML for the Neural Engine, one fixed
window. Step 1 of making assets/parakeet-ultra-ane (step 2: recipe.py); NOTES.md "Neural Engine".
usage: python export.py <hf dir with config.json + model.safetensors> <mel frames: 501 | 1501 | 3001> <out dir>
Input `mel` [1, frames, 128] (frame-major) + `length` [1] (valid mel frames; the rest is masked), output
`encoder` [1, frames/8, 1024]. Python 3.12 + torch 2.7 + coremltools 9 + transformers 5 (see NOTES.md). The committed version of this script
was not re-run after moving it here from .cache/tmp (only the reference check vs the GPU dump was removed)."""
import sys, numpy as np, torch, torch.nn as nn
from transformers.models.parakeet import modeling_parakeet as mp
from transformers import ParakeetForTDT
HF, FR, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def attn_forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
    # transformers' eager attention, but masked with -1e4: its finfo(f32).min becomes -inf in fp16 -> NaN rows.
    b, t = hidden_states.shape[:2]; h, d = self.config.num_attention_heads, self.head_dim
    q = self.q_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    qu = q + self.bias_u.view(1, h, 1, d); qv = q + self.bias_v.view(1, h, 1, d)
    p = self.relative_k_proj(position_embeddings).view(b, -1, h, d)
    bd = self._rel_shift(qv @ p.permute(0, 2, 3, 1))[..., :t]
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
