//! Parakeet (NeMo FastConformer) encoder in burn, weights from our converted file (weights.rs).
//! Structure follows transcribe.cpp `src/conformer/conformer.cpp` and `src/arch/parakeet/encoder.cpp`
//! (MIT, (c) 2026 The transcribe.cpp authors), including "kestrel" length masking in the subsampling stage.
//! GPU only (Linux, Windows); the sizes and helpers the CPU and Neural Engine paths share are in shapes.rs.
use super::shapes::{pos_emb_vec, stride2, valid_lengths, D, DK, H, LAYERS, PRE_CHUNK};
use super::weights::Weights;
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::module::{conv1d, conv2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};

const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

pub struct Ld<'a> {
    pub g: &'a Weights,
    pub dev: &'a Device,
    /// Working precision of the activations (F16 or F32); norms and softmax always run in f32 (ggml does too:
    /// the residual stream reaches ~±6800, and its square overflows f16).
    pub low: FloatDType,
    pub q8: bool,
}

impl Ld<'_> {
    pub fn up<const N: usize>(&self, v: Vec<f32>, shape: impl Into<burn::tensor::Shape>) -> Tensor<N> {
        let data = match self.low {
            FloatDType::F16 => TensorData::new(v.into_iter().map(half::f16::from_f32).collect::<Vec<_>>(), shape),
            _ => TensorData::new(v, shape),
        };
        let dt = if self.low == FloatDType::F16 { DType::F16 } else { DType::F32 };
        Tensor::from_data(data, (self.dev, dt))
    }
    pub fn t<const N: usize>(&self, name: &str) -> Tensor<N> {
        let (v, shape) = self.g.f32(name);
        assert_eq!(shape.len(), N, "{name}: shape {shape:?}");
        self.up(v, shape)
    }
    /// Vector shaped [1, n] for broadcasting over rows.
    pub fn row(&self, name: &str) -> Tensor<2> {
        let (v, _) = self.g.f32(name);
        let n = v.len();
        self.up(v, [1, n])
    }
    pub fn row_f32(&self, name: &str) -> Tensor<2> {
        let (v, _) = self.g.f32(name);
        let n = v.len();
        Tensor::from_data(TensorData::new(v, [1, n]), (self.dev, DType::F32))
    }
    /// Linear weight for y = x @ w ([in, out]). Q8_0 weights stay int8 on the GPU (Q8=1), f16 weights stay f16 with
    /// f16 activations: both uploaded straight from the weights file in their stored layout [out, in] and
    /// transposed as a view. Anything else is converted on the CPU.
    pub fn lin(&self, name: &str) -> Tensor<2> {
        let kind = self.g.kind(name);
        if (kind == "q8" && self.q8) || (kind == "f16" && self.low == FloatDType::F16) {
            let data = self.g.lin_data(name);
            let w = if kind == "q8" { Tensor::<2>::from_data(data, self.dev) } else { Tensor::<2>::from_data(data, (self.dev, DType::F16)) };
            return w.transpose();
        }
        let (v, shape) = self.g.f32(name);
        let (out, inp) = (shape[0], shape[1..].iter().product::<usize>());
        let mut tr = vec![0f32; v.len()];
        for o in 0..out {
            for i in 0..inp {
                tr[i * out + o] = v[o * inp + i];
            }
        }
        self.up(tr, [inp, out])
    }
}

pub struct Ln {
    w: Tensor<2>,
    b: Tensor<2>,
    low: FloatDType,
}
impl Ln {
    fn load(l: &Ld, p: &str) -> Ln {
        Ln { w: l.row_f32(&format!("{p}.weight")), b: l.row_f32(&format!("{p}.bias")), low: l.low }
    }
    /// LayerNorm in f32, result back in the working precision.
    fn fwd(&self, x: Tensor<2>) -> Tensor<2> {
        let x = x.cast(FloatDType::F32);
        let mean = x.clone().mean_dim(1);
        let xc = x - mean;
        let var = xc.clone().powf_scalar(2.0).mean_dim(1);
        (xc / var.add_scalar(LN_EPS).sqrt() * self.w.clone() + self.b.clone()).cast(self.low)
    }
}

pub struct Block {
    pub n_ff1: Ln,
    pub ff1: (Tensor<2>, Tensor<2>),
    pub n_att: Ln,
    q: Tensor<2>,
    k: Tensor<2>,
    v: Tensor<2>,
    o: Tensor<2>,
    pos: Tensor<2>,
    bias_u: Tensor<3>, // [H, 1, DK]
    bias_v: Tensor<3>,
    pub n_conv: Ln,
    pw1: Tensor<2>,
    dw: Tensor<3>, // [D, 1, 9]
    bn_scale: Tensor<3>, // [1, D, 1]
    bn_shift: Tensor<3>,
    pw2: Tensor<2>,
    n_ff2: Ln,
    ff2: (Tensor<2>, Tensor<2>),
    n_out: Ln,
    low: FloatDType,
}

impl Block {
    pub fn load(l: &Ld, i: usize) -> Block {
        let p = format!("enc.blocks.{i}");
        let bias = |n: &str| {
            let (v, _) = l.g.f32(&format!("{p}.attn.{n}"));
            l.up(v, [H, 1, DK])
        };
        let (w, _) = l.g.f32(&format!("{p}.conv.bn.weight"));
        let (b, _) = l.g.f32(&format!("{p}.conv.bn.bias"));
        let (m, _) = l.g.f32(&format!("{p}.conv.bn.running_mean"));
        let (var, _) = l.g.f32(&format!("{p}.conv.bn.running_var"));
        let scale: Vec<f32> = (0..D).map(|c| w[c] / (var[c] + BN_EPS).sqrt()).collect();
        let shift: Vec<f32> = (0..D).map(|c| b[c] - m[c] * scale[c]).collect();
        Block {
            n_ff1: Ln::load(l, &format!("{p}.norm_ff1")),
            ff1: (l.lin(&format!("{p}.ff1.linear1.weight")), l.lin(&format!("{p}.ff1.linear2.weight"))),
            n_att: Ln::load(l, &format!("{p}.norm_attn")),
            q: l.lin(&format!("{p}.attn.linear_q.weight")),
            k: l.lin(&format!("{p}.attn.linear_k.weight")),
            v: l.lin(&format!("{p}.attn.linear_v.weight")),
            o: l.lin(&format!("{p}.attn.linear_out.weight")),
            pos: l.lin(&format!("{p}.attn.linear_pos.weight")),
            bias_u: bias("pos_bias_u"),
            bias_v: bias("pos_bias_v"),
            n_conv: Ln::load(l, &format!("{p}.norm_conv")),
            pw1: l.lin(&format!("{p}.conv.pointwise1.weight")),
            dw: l.t::<3>(&format!("{p}.conv.depthwise.weight")),
            bn_scale: l.up(scale, [1, D, 1]),
            bn_shift: l.up(shift, [1, D, 1]),
            pw2: l.lin(&format!("{p}.conv.pointwise2.weight")),
            n_ff2: Ln::load(l, &format!("{p}.norm_ff2")),
            ff2: (l.lin(&format!("{p}.ff2.linear1.weight")), l.lin(&format!("{p}.ff2.linear2.weight"))),
            n_out: Ln::load(l, &format!("{p}.norm_out")),
            low: l.low,
        }
    }

    pub fn ff(x: Tensor<2>, w: &(Tensor<2>, Tensor<2>)) -> Tensor<2> {
        silu(x.matmul(w.0.clone())).matmul(w.1.clone())
    }

    /// Relative-position self-attention (Transformer-XL style, NeMo RelPositionMultiHeadAttention).
    /// x [T, D], pos [2T-1, D].
    pub fn attn(&self, x: Tensor<2>, pos: Tensor<2>) -> Tensor<2> {
        let t = x.dims()[0];
        let p_len = pos.dims()[0];
        let heads = |y: Tensor<2>, n: usize| y.reshape([n, H, DK]).swap_dims(0, 1); // [H, n, DK]
        let q = heads(x.clone().matmul(self.q.clone()), t);
        let k = heads(x.clone().matmul(self.k.clone()), t);
        let v = heads(x.matmul(self.v.clone()), t);
        let p = heads(pos.matmul(self.pos.clone()), p_len);
        let ac = (q.clone() + self.bias_u.clone()).matmul(k.swap_dims(1, 2)); // [H, T, T]
        let bd = (q + self.bias_v.clone()).matmul(p.swap_dims(1, 2)); // [H, T, 2T-1]
        // rel_shift: pad one zero column in front, view as [H, P+1, T], drop the first row, view back, keep T cols.
        let dev = bd.device();
        let bd = Tensor::cat(vec![Tensor::zeros([H, t, 1], (&dev, bd.dtype())), bd], 2)
            .reshape([H, p_len + 1, t])
            .narrow(1, 1, p_len)
            .reshape([H, t, p_len])
            .narrow(2, 0, t);
        let att = softmax((ac + bd).cast(FloatDType::F32).div_scalar((DK as f32).sqrt()), 2).cast(self.low);
        att.matmul(v).swap_dims(0, 1).reshape([t, D]).matmul(self.o.clone())
    }

    pub fn conv(&self, x: Tensor<2>) -> Tensor<2> {
        let t = x.dims()[0];
        let y = x.matmul(self.pw1.clone()); // [T, 2D]
        let y = y.clone().narrow(1, 0, D) * sigmoid(y.narrow(1, D, D)); // GLU
        let y = y.swap_dims(0, 1).reshape([1, D, t]);
        let y = conv1d(y, self.dw.clone(), None, ConvOptions::new([1], [4], [1], D));
        let y = silu(y * self.bn_scale.clone() + self.bn_shift.clone());
        y.reshape([D, t]).swap_dims(0, 1).matmul(self.pw2.clone())
    }

    pub fn fwd(&self, x: Tensor<2>, pos: Tensor<2>) -> Tensor<2> {
        let x = x.clone() + Self::ff(self.n_ff1.fwd(x), &self.ff1).mul_scalar(0.5);
        let x = x.clone() + self.attn(self.n_att.fwd(x), pos);
        let x = x.clone() + self.conv(self.n_conv.fwd(x));
        let x = x.clone() + Self::ff(self.n_ff2.fwd(x), &self.ff2).mul_scalar(0.5);
        self.n_out.fwd(x)
    }
}

pub struct PreEncode {
    c0: (Tensor<4>, Tensor<1>),
    c2: (Tensor<4>, Tensor<1>),
    c3: (Tensor<4>, Tensor<1>),
    c5: (Tensor<4>, Tensor<1>),
    c6: (Tensor<4>, Tensor<1>),
    out: Tensor<2>,
    out_b: Tensor<2>,
    low: FloatDType,
}

impl PreEncode {
    /// Convs in f32 (transcribe.cpp runs the kestrel subsampler's pointwise convs in f32 too); the output linear in
    /// the working precision.
    pub fn load(l: &Ld) -> PreEncode {
        let f = Ld { g: l.g, dev: l.dev, low: FloatDType::F32, q8: false };
        let c = |i: usize| (f.t::<4>(&format!("enc.pre_encode.conv.{i}.weight")), f.t::<1>(&format!("enc.pre_encode.conv.{i}.bias")));
        PreEncode { c0: c(0), c2: c(2), c3: c(3), c5: c(5), c6: c(6), out: l.lin("enc.pre_encode.out.weight"), out_b: l.row("enc.pre_encode.out.bias"), low: l.low }
    }
    /// mel [frames, 128] (f32) -> [frames/8, D]. Conv2d over (time, freq), dw_striding x3. Frames past the valid
    /// lengths `len` are zeroed after relu0, conv2 and conv5 (transcribe.cpp mask_after_stride).
    pub fn fwd(&self, mel: Tensor<2>, len: [usize; 4]) -> Tensor<2> {
        self.fwd_at(mel, len, 0)
    }

    /// Same result as `fwd`, computed `chunk` mel frames at a time (plus a margin on both sides) to bound the
    /// size of the intermediate buffers. Output frame j depends only on mel frames 8j-7 ..= 8j+7, so with chunk
    /// starts on multiples of 8 and a margin >= 8 each kept frame sees exactly its full-run inputs.
    pub fn fwd_chunked(&self, mel: Tensor<2>, len: [usize; 4], chunk: usize) -> Tensor<2> {
        const MARGIN: usize = 32; // mel frames, multiple of 8
        let t = mel.dims()[0];
        let n_out = stride2(stride2(stride2(t)));
        let step = (chunk / 8).max(1); // output frames per chunk
        if chunk == 0 || n_out <= step {
            return self.fwd(mel, len);
        }
        let mut parts = Vec::new();
        let mut j0 = 0;
        while j0 < n_out {
            let j1 = (j0 + step).min(n_out);
            let a = (8 * j0).saturating_sub(MARGIN);
            let b = (8 * j1 + MARGIN).min(t);
            let y = self.fwd_at(mel.clone().narrow(0, a, b - a), len, a);
            parts.push(y.narrow(0, j0 - a / 8, j1 - j0));
            j0 = j1;
        }
        Tensor::cat(parts, 0)
    }

    /// `mel` is the slice of the full mel starting at frame `off` (a multiple of 8); masks use global indices.
    fn fwd_at(&self, mel: Tensor<2>, len: [usize; 4], off: usize) -> Tensor<2> {
        let [t, f] = mel.dims();
        let s2 = |c: usize| ConvOptions::new([2, 2], [1, 1], [1, 1], c);
        let pw = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let dev = mel.device();
        let len = [len[0], len[1].saturating_sub(off / 2), len[2].saturating_sub(off / 4), len[3].saturating_sub(off / 8)];
        let mask = |x: Tensor<4>, valid: usize| {
            let tt = x.dims()[2];
            if valid >= tt {
                return x;
            }
            let m: Vec<f32> = (0..tt).map(|i| if i < valid { 1.0 } else { 0.0 }).collect();
            x * Tensor::<4>::from_data(TensorData::new(m, [1, 1, tt, 1]), (&dev, DType::F32))
        };
        let x = mel.cast(FloatDType::F32).reshape([1, 1, t, f]);
        let x = conv2d(x, self.c0.0.clone(), Some(self.c0.1.clone()), s2(1)).clamp_min(0.0);
        let x = mask(x, len[1]);
        let x = mask(conv2d(x, self.c2.0.clone(), Some(self.c2.1.clone()), s2(256)), len[2]);
        let x = conv2d(x, self.c3.0.clone(), Some(self.c3.1.clone()), pw.clone()).clamp_min(0.0);
        let x = mask(conv2d(x, self.c5.0.clone(), Some(self.c5.1.clone()), s2(256)), len[3]);
        let x = conv2d(x, self.c6.0.clone(), Some(self.c6.1.clone()), pw).clamp_min(0.0);
        let [_, c, t2, f2] = x.dims();
        let x = x.reshape([c, t2, f2]).swap_dims(0, 1).reshape([t2, c * f2]); // [T', C*F'] (channel-major)
        x.cast(self.low).matmul(self.out.clone()) + self.out_b.clone()
    }
}

/// `pos_emb_vec` as a tensor in the activations' dtype.
pub fn pos_emb(t: usize, dev: &Device, dt: DType) -> Tensor<2> {
    Tensor::from_data(TensorData::new(pos_emb_vec(t), [2 * t - 1, D]), (dev, dt))
}

pub struct Encoder {
    pub pre: PreEncode,
    pub blocks: Vec<Block>,
}

impl Encoder {
    pub fn load(l: &Ld) -> Encoder {
        Encoder { pre: PreEncode::load(l), blocks: (0..LAYERS).map(|i| Block::load(l, i)).collect() }
    }
    /// mel [frames, 128] of `n_samples` audio -> [T_valid, D]. Frames past the valid length are dropped after the
    /// subsampler, which equals transcribe.cpp's key-padding + conv masks (relative positions don't depend on T).
    pub fn fwd(&self, mel: Tensor<2>, n_samples: usize) -> Tensor<2> {
        let len = valid_lengths(n_samples, 160);
        let mut x = self.pre.fwd_chunked(mel, len, PRE_CHUNK);
        let t = x.dims()[0].min(len[3]);
        if t < x.dims()[0] {
            x = x.narrow(0, 0, t);
        }
        let pos = pos_emb(t, &x.device(), x.dtype());
        for b in &self.blocks {
            x = b.fwd(x, pos.clone());
        }
        x
    }
}

/// Speech/no-speech head on the subsampler output (transcribe.cpp `build_vad_graph`): k=1 conv d_model -> 128, silu,
/// k=5 conv 128 -> 128, silu, k=1 conv 128 -> 1, sigmoid. All f32.
pub struct Vad {
    proj: Tensor<2>, // [d, H]
    proj_b: Tensor<2>,
    ctx: Tensor<3>, // [H, H, K]
    ctx_b: Tensor<1>,
    out: Tensor<2>, // [H, 1]
    out_b: Tensor<2>,
}

impl Vad {
    pub fn load(l: &Ld) -> Vad {
        let f = Ld { g: l.g, dev: l.dev, low: FloatDType::F32, q8: false };
        let (out, _) = l.g.f32("vad.out.weight");
        let n = out.len();
        Vad {
            proj: f.lin("vad.proj.weight"),
            proj_b: f.row("vad.proj.bias"),
            ctx: f.t::<3>("vad.ctx.weight"),
            ctx_b: f.t::<1>("vad.ctx.bias"),
            out: f.up(out, [n, 1]),
            out_b: f.row("vad.out.bias"),
        }
    }
    /// Subsampler output [T, D] -> speech probability per frame [T].
    pub fn fwd(&self, x: Tensor<2>) -> Tensor<1> {
        let t = x.dims()[0];
        let k = self.ctx.dims()[2];
        let h = silu(x.cast(FloatDType::F32).matmul(self.proj.clone()) + self.proj_b.clone()); // [T, H]
        let hd = h.dims()[1];
        let c = conv1d(h.swap_dims(0, 1).reshape([1, hd, t]), self.ctx.clone(), Some(self.ctx_b.clone()), ConvOptions::new([1], [k / 2], [1], 1));
        let c = silu(c).reshape([hd, t]).swap_dims(0, 1); // [T, H]
        sigmoid(c.matmul(self.out.clone()) + self.out_b.clone()).reshape([t])
    }
}
