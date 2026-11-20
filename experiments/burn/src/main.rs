//! Parakeet Ultra encoder (NeMo FastConformer) in burn on Vulkan, weights from the transcribe.cpp GGUF.
//! Experiment: is burn slower than transcribe.cpp's ggml encoder, and does it give the same numbers?
//!
//! usage: burn-enc <model.gguf> <ref dir from ref_dump.py> [runs]
//! env:   F16=1  activations in half precision (default f32)
//!        Q8=1   keep Q8_0 weights as int8 on the GPU (default: dequantized)
mod gguf;

use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::module::{conv1d, conv2d};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType, Device, DeviceConfig, DeviceKind, FloatDType, Tensor, TensorData};
use gguf::Gguf;
use std::path::Path;
use std::time::Instant;

const D: usize = 1024;
const H: usize = 8;
const DK: usize = D / H;
const LAYERS: usize = 24;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

struct Ld<'a> {
    g: &'a Gguf,
    dev: &'a Device,
    /// Working precision of the activations (F16 or F32); norms and softmax always run in f32 (ggml does too:
    /// the residual stream reaches ~±6800, and its square overflows f16).
    low: FloatDType,
    q8: bool,
}

impl Ld<'_> {
    fn up<const N: usize>(&self, v: Vec<f32>, shape: impl Into<burn::tensor::Shape>) -> Tensor<N> {
        let data = match self.low {
            FloatDType::F16 => TensorData::new(v.into_iter().map(half::f16::from_f32).collect::<Vec<_>>(), shape),
            _ => TensorData::new(v, shape),
        };
        Tensor::from_data(data, self.dev)
    }
    fn t<const N: usize>(&self, name: &str) -> Tensor<N> {
        let (v, shape) = self.g.f32(name);
        assert_eq!(shape.len(), N, "{name}: shape {shape:?}");
        self.up(v, shape)
    }
    /// Vector shaped [1, n] for broadcasting over rows.
    fn row(&self, name: &str) -> Tensor<2> {
        let (v, _) = self.g.f32(name);
        let n = v.len();
        self.up(v, [1, n])
    }
    fn row_f32(&self, name: &str) -> Tensor<2> {
        let (v, _) = self.g.f32(name);
        let n = v.len();
        Tensor::from_data(TensorData::new(v, [1, n]), (self.dev, DType::F32))
    }
    /// Linear weight for y = x @ w ([in, out]). With Q8=1, Q8_0 weights stay int8 on the GPU: uploaded in the GGUF
    /// layout [out, in] (blocks of 32 along `in`, one f16 scale each = burn's Q8S per-block scheme) and transposed
    /// as a view.
    fn lin(&self, name: &str) -> Tensor<2> {
        if self.q8 {
            if let Some((q, d, shape)) = self.g.q8_0(name) {
                use burn::tensor::quantization::{QuantScheme, QuantValue, ScaleDtype};
                let scheme = QuantScheme::default().with_value(QuantValue::Q8S).per_block([32], ScaleDtype::F16);
                let data = TensorData::quantized(q, [shape[0], shape[1]], scheme, &d, None);
                return Tensor::<2>::from_data(data, self.dev).transpose();
            }
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

struct Ln {
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

struct Block {
    n_ff1: Ln,
    ff1: (Tensor<2>, Tensor<2>),
    n_att: Ln,
    q: Tensor<2>,
    k: Tensor<2>,
    v: Tensor<2>,
    o: Tensor<2>,
    pos: Tensor<2>,
    bias_u: Tensor<3>, // [H, 1, DK]
    bias_v: Tensor<3>,
    n_conv: Ln,
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
    fn load(l: &Ld, i: usize) -> Block {
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

    fn ff(x: Tensor<2>, w: &(Tensor<2>, Tensor<2>)) -> Tensor<2> {
        silu(x.matmul(w.0.clone())).matmul(w.1.clone())
    }

    /// Relative-position self-attention (Transformer-XL style, NeMo RelPositionMultiHeadAttention).
    /// x [T, D], pos [2T-1, D].
    fn attn(&self, x: Tensor<2>, pos: Tensor<2>) -> Tensor<2> {
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
        let bd = Tensor::cat(vec![Tensor::zeros([H, t, 1], &dev), bd], 2)
            .reshape([H, p_len + 1, t])
            .narrow(1, 1, p_len)
            .reshape([H, t, p_len])
            .narrow(2, 0, t);
        let att = softmax((ac + bd).cast(FloatDType::F32).div_scalar((DK as f32).sqrt()), 2).cast(self.low);
        att.matmul(v).swap_dims(0, 1).reshape([t, D]).matmul(self.o.clone())
    }

    fn conv(&self, x: Tensor<2>) -> Tensor<2> {
        let t = x.dims()[0];
        let y = x.matmul(self.pw1.clone()); // [T, 2D]
        let y = y.clone().narrow(1, 0, D) * sigmoid(y.narrow(1, D, D)); // GLU
        let y = y.swap_dims(0, 1).reshape([1, D, t]);
        let y = conv1d(y, self.dw.clone(), None, ConvOptions::new([1], [4], [1], D));
        let y = silu(y * self.bn_scale.clone() + self.bn_shift.clone());
        y.reshape([D, t]).swap_dims(0, 1).matmul(self.pw2.clone())
    }

    fn fwd(&self, x: Tensor<2>, pos: Tensor<2>, taps: &mut Vec<(String, Tensor<2>)>, i: usize) -> Tensor<2> {
        let x = x.clone() + Self::ff(self.n_ff1.fwd(x), &self.ff1).mul_scalar(0.5);
        if i == 0 {
            taps.push(("enc.block.0.ff1".into(), x.clone()));
        }
        let x = x.clone() + self.attn(self.n_att.fwd(x), pos);
        if i == 0 {
            taps.push(("enc.block.0.attn".into(), x.clone()));
        }
        let x = x.clone() + self.conv(self.n_conv.fwd(x));
        if i == 0 {
            taps.push(("enc.block.0.conv".into(), x.clone()));
        }
        let x = x.clone() + Self::ff(self.n_ff2.fwd(x), &self.ff2).mul_scalar(0.5);
        self.n_out.fwd(x)
    }
}

struct PreEncode {
    c0: (Tensor<4>, Tensor<1>),
    c2: (Tensor<4>, Tensor<1>),
    c3: (Tensor<4>, Tensor<1>),
    c5: (Tensor<4>, Tensor<1>),
    c6: (Tensor<4>, Tensor<1>),
    out: Tensor<2>,
    out_b: Tensor<2>,
}

impl PreEncode {
    fn load(l: &Ld) -> PreEncode {
        let c = |i: usize| (l.t::<4>(&format!("enc.pre_encode.conv.{i}.weight")), l.t::<1>(&format!("enc.pre_encode.conv.{i}.bias")));
        PreEncode { c0: c(0), c2: c(2), c3: c(3), c5: c(5), c6: c(6), out: l.lin("enc.pre_encode.out.weight"), out_b: l.row("enc.pre_encode.out.bias") }
    }
    /// mel [frames, 128] -> [frames/8, D]. Conv2d over (time, freq), dw_striding x3.
    fn fwd(&self, mel: Tensor<2>) -> Tensor<2> {
        let [t, f] = mel.dims();
        let s2 = |c: usize| ConvOptions::new([2, 2], [1, 1], [1, 1], c);
        let pw = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let x = mel.reshape([1, 1, t, f]);
        let x = conv2d(x, self.c0.0.clone(), Some(self.c0.1.clone()), s2(1)).clamp_min(0.0);
        let x = conv2d(x, self.c2.0.clone(), Some(self.c2.1.clone()), s2(256));
        let x = conv2d(x, self.c3.0.clone(), Some(self.c3.1.clone()), pw.clone()).clamp_min(0.0);
        let x = conv2d(x, self.c5.0.clone(), Some(self.c5.1.clone()), s2(256));
        let x = conv2d(x, self.c6.0.clone(), Some(self.c6.1.clone()), pw).clamp_min(0.0);
        let [_, c, t2, f2] = x.dims();
        let x = x.reshape([c, t2, f2]).swap_dims(0, 1).reshape([t2, c * f2]); // [T', C*F'] (channel-major)
        x.matmul(self.out.clone()) + self.out_b.clone()
    }
}

/// NeMo RelPositionalEncoding: positions T-1 .. -(T-1), sin on even, cos on odd features.
fn pos_emb(t: usize, dev: &Device) -> Tensor<2> {
    let n = 2 * t - 1;
    let mut v = vec![0f32; n * D];
    for r in 0..n {
        let pos = (t as f64 - 1.0) - r as f64;
        for i in (0..D).step_by(2) {
            let div = (-(i as f64) * (10000f64.ln() / D as f64)).exp();
            v[r * D + i] = (pos * div).sin() as f32;
            v[r * D + i + 1] = (pos * div).cos() as f32;
        }
    }
    Tensor::from_data(TensorData::new(v, [n, D]), dev)
}

struct Encoder {
    pre: PreEncode,
    blocks: Vec<Block>,
}

impl Encoder {
    fn fwd(&self, mel: Tensor<2>, taps: &mut Vec<(String, Tensor<2>)>) -> Tensor<2> {
        let mut x = self.pre.fwd(mel);
        taps.push(("enc.pre_encode.out".into(), x.clone()));
        let pos = pos_emb(x.dims()[0], &x.device());
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.fwd(x, pos.clone(), taps, i);
            if i == 0 || i == 12 {
                taps.push((format!("enc.block.{i}.out"), x.clone()));
            }
        }
        x
    }
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Relative L2 error and worst absolute difference against a transcribe.cpp dump.
fn compare(name: &str, t: Tensor<2>, refdir: &Path) {
    let got: Vec<f32> = t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
    let want = read_f32(&refdir.join(format!("{name}.f32")));
    assert_eq!(got.len(), want.len(), "{name}: size");
    let (mut num, mut den, mut worst) = (0f64, 0f64, 0f32);
    for (a, b) in got.iter().zip(&want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
        worst = worst.max((a - b).abs());
    }
    println!("  {name:22} rel.err {:.2e}  max|diff| {worst:.3e}", (num / den).sqrt());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, refdir) = (Path::new(&args[1]), Path::new(&args[2]));
    let runs: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(10);
    let f16 = std::env::var("F16").is_ok_and(|v| v == "1");
    let q8 = std::env::var("Q8").is_ok_and(|v| v == "1");

    let tp = Instant::now();
    let mut dev = Device::vulkan(DeviceKind::DiscreteGpu(0));
    if f16 {
        dev.configure(DeviceConfig::default().float_dtype(FloatDType::F16)).unwrap();
    }
    let memlog = std::env::var("MEMLOG").is_ok_and(|v| v == "1");
    let mem = |label: &str| {
        let u = dev.memory_pool_usage().unwrap();
        let smi = std::process::Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().find(|l| l.starts_with(&format!("{},", std::process::id()))).map(|l| l.split(',').nth(1).unwrap().trim().to_string()).unwrap_or_default())
            .unwrap_or_default();
        let tot = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        if memlog { println!("MEM {label:>8}  GPU total {tot} MiB  allocs {:5}  in_use {:6.1} MB  padding {:5.1} MB  reserved {:6.1} MB  driver {smi} MB",
            u.number_allocs, u.bytes_in_use as f64 / 1e6, u.bytes_padding as f64 / 1e6, u.bytes_reserved as f64 / 1e6); }
    };
    let _ = Tensor::<1>::zeros([1], &dev).into_data(); // create the device context
    mem("device");
    println!("device init {:.2}s", tp.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let g = Gguf::open(model).unwrap();
    let l = Ld { g: &g, dev: &dev, low: if f16 { FloatDType::F16 } else { FloatDType::F32 }, q8 };
    // PERSIST=1: weights as persistent allocations (own buffers, not carved out of the dynamic pool's pages).
    let enc = if std::env::var("PERSIST").is_ok_and(|v| v == "1") {
        dev.memory_persistent_allocations(&l, |l| Encoder { pre: PreEncode::load(l), blocks: (0..LAYERS).map(|i| Block::load(l, i)).collect() })
    } else {
        Encoder { pre: PreEncode::load(&l), blocks: (0..LAYERS).map(|i| Block::load(&l, i)).collect() }
    };
    drop(g);
    mem("weights");
    println!("loaded in {:.2}s ({}{})", t0.elapsed().as_secs_f64(), if f16 { "f16" } else { "f32" }, if q8 { ", Q8_0 weights int8" } else { "" });

    // Input: transcribe.cpp's own mel ([128, frames] row-major) -> [frames, 128].
    let mel = read_f32(&refdir.join("enc.mel.in.f32"));
    let frames = mel.len() / 128;
    let mel = Tensor::<2>::from_data(TensorData::new(mel, [128, frames]), &dev).swap_dims(0, 1);

    // SWEEP=1: pre-tune sweep. Every length from 8 mel frames up to 30 s (3000 frames) in steps of 8
    // (= 1 encoder frame, 80 ms); the 25 s reference mel is repeated to reach 30 s. Prints the slow
    // (tuning) lengths and the total, then exits. Run twice: the 2nd sweep should tune nothing.
    if std::env::var("SWEEP").is_ok_and(|v| v == "1") {
        let long = Tensor::cat(vec![mel.clone(), mel.clone()], 0).narrow(0, 0, 3000);
        let _ = long.clone().into_data();
        let ts = Instant::now();
        let (mut slow, mut slow_s, mut n_runs) = (0, 0.0, 0);
        for n in (8..=3000).step_by(8) {
            let t = Instant::now();
            let _ = enc.fwd(long.clone().narrow(0, 0, n), &mut Vec::new()).into_data();
            let s = t.elapsed().as_secs_f64();
            n_runs += 1;
            if s > 0.3 {
                slow += 1;
                slow_s += s;
                println!("  {n:5} mel frames ({:5.2} s audio): {s:6.2} s", n as f64 / 100.0);
            }
            if n % 400 == 0 {
                println!("  .. {n} frames, {:.0} s so far", ts.elapsed().as_secs_f64());
            }
        }
        println!("sweep: {n_runs} lengths in {:.1} s; {slow} slow (>0.3 s) totalling {slow_s:.1} s",
            ts.elapsed().as_secs_f64());
        mem("sweep");
        return;
    }
    if std::env::var("BISECT").is_ok_and(|v| v == "1") {
        // Which stage brings the one-time memory outside burn's pool?
        let x = enc.pre.fwd(mel.clone());
        let _ = x.clone().into_data();
        mem("pre-enc");
        let pos = pos_emb(x.dims()[0], &dev);
        let b = &enc.blocks[0];
        let y = Block::ff(b.n_ff1.fwd(x.clone()), &b.ff1);
        let _ = y.into_data();
        mem("ff");
        let y = b.attn(b.n_att.fwd(x.clone()), pos.clone());
        let _ = y.into_data();
        mem("attn");
        let y = b.conv(b.n_conv.fwd(x.clone()));
        let _ = y.into_data();
        mem("conv");
        let _ = b.fwd(x, pos, &mut Vec::new(), 1).into_data();
        mem("block");
    }
    let mut taps = Vec::new();
    let t1 = Instant::now();
    let out = enc.fwd(mel.clone(), &mut taps);
    let _ = out.clone().into_data();
    println!("first run (shader compile + autotune) {:.2}s", t1.elapsed().as_secs_f64());
    mem("1st run");
    println!("vs transcribe.cpp ({} mel frames -> {} encoder frames):", frames, out.dims()[0]);
    for (name, t) in taps {
        compare(&name, t, refdir);
    }
    compare("enc.final", out, refdir);

    mem("warm");
    let mut ms = Vec::new();
    for i in 0..runs {
        let t = Instant::now();
        let out = enc.fwd(mel.clone(), &mut Vec::new());
        let _ = out.into_data(); // wait for the GPU
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
        mem(&format!("run {i}"));
    }
    // LENS=1001,1500: other input lengths (mel frames, 100/s): first run (new kernels/tuning?) vs second.
    if let Ok(lens) = std::env::var("LENS") {
        for n in lens.split(',').map(|x| x.parse::<usize>().unwrap()) {
            let x = mel.clone().narrow(0, 0, n);
            let t = Instant::now();
            let _ = enc.fwd(x.clone(), &mut Vec::new()).into_data();
            let a = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let _ = enc.fwd(x, &mut Vec::new()).into_data();
            println!("len {n:5} frames: 1st {a:8.1} ms, 2nd {:6.1} ms", t.elapsed().as_secs_f64() * 1e3);
            mem(&format!("len {n}"));
        }
    }
    dev.memory_cleanup();
    mem("cleanup");
    ms.sort_by(f64::total_cmp);
    println!("encoder forward, {runs} warm runs: min {:.1} ms, median {:.1} ms", ms[0], ms[ms.len() / 2]);
}
