//! Kokoro-82M (hexgrad, Apache-2.0; StyleTTS2 + iSTFTNet) text-to-speech on our engine: phonemes in, 24 kHz wav out.
//! Port of the `kokoro` 0.9.4 Python package (model.py, modules.py, istftnet.py). Split: GPU (burn + own kernels) = ALBERT, convs,
//! style blocks, vocoder networks; CPU = the LSTMs (step by step, tiny), durations, the sine source and both small
//! STFTs (n_fft 20) of the vocoder.
//! Weights: `kokoro.safetensors` + `voices.safetensors` (f32, weight norm folded) written by the reference script
//! (.cache/tmp/kokoro/scripts/ref.py). Text -> phonemes: misaki-rs (Rust port of misaki, Kokoro v1.0's G2P, US English,
//! no espeak fallback: unknown words become ❓ and are dropped). PH=1: the input is already phonemes.
//! usage: cargo run --release --example kokoro -- <dir with kokoro.safetensors, voices.safetensors, config.json>
//!        <text> <out.wav> [voice=af_heart]
//!        or: <dir> --listen [voice] (macOS): select text anywhere, Ctrl+Option+R reads it aloud, again = stop.
//! Shelved 2026-11-22 (user decided against text-to-speech for now): not built by default. To build it again, add to
//! Cargo.toml (changes Cargo.lock, so the app's kernel cache key changes too: one re-tune, NOTES.md "Kokoro"):
//!   [dev-dependencies]
//!   burn = { version = "0.22.1", default-features = false, features = ["extension"] }
//!   burn-cubecl = { version = "0.22.1", default-features = false }
//!   misaki-rs = { version = "0.11", default-features = false }
//!   [[example]]
//!   name = "kokoro"
//!   path = "experiments/kokoro.rs"
//! Weights: .cache/tmp/kokoro/ref (kokoro.safetensors, voices.safetensors, config.json; made from hexgrad/Kokoro-82M).
//! Own CubeCL kernels (`mod kern`) for the vocoder's AdaIN + Snake + conv steps; OWN=0: burn's ops only.
//! KTEST=1: those kernels vs burn (error, speed). PROF=1: per-stage GPU times. BENCH=1: conv/elementwise benchmarks.
//! NOISE=0: no random phase/noise in the source (to compare with the reference). DUMP=<dir>: intermediates as raw f32.
//! HAR=<safetensors with "har">: use that source spectrum (test of the vocoder alone).
use burn::tensor::activation::{gelu_approximate, leaky_relu, softmax};
use burn::tensor::module::{conv1d, conv_transpose1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
// CubeCL's kernel cache asks for the build id our lib provides (lib.rs build_id__get): link it.
use dictation as _;

// ---------- weights (safetensors, f32) ----------
struct St {
    data: Vec<u8>,
    start: usize,
    header: serde_json::Map<String, serde_json::Value>,
}
impl St {
    fn open(p: &Path) -> St {
        let data = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header = serde_json::from_slice(&data[8..8 + n]).unwrap();
        St { data, start: 8 + n, header }
    }
    fn get(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let t = self.header.get(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(t["dtype"], "F32", "{name}");
        let o = &t["data_offsets"];
        let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        let v = self.data[self.start + a..self.start + b].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        (v, t["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
    }
    fn v(&self, name: &str) -> Vec<f32> {
        self.get(name).0
    }
}

// ---------- CPU parts ----------
fn sgemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, b_t: bool) -> Vec<f32> {
    // c[m][n] = a[m][k] * b[k][n]  (b_t: b stored as [n][k])
    let mut c = vec![0f32; m * n];
    let (rsb, csb) = if b_t { (1, k as isize) } else { (n as isize, 1) };
    unsafe { matrixmultiply::sgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), rsb, csb, 0.0, c.as_mut_ptr(), n as isize, 1) };
    c
}

/// Dot product in 8 lanes (a plain iterator sum stays sequential and doesn't vectorize).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Bidirectional 1-layer LSTM (PyTorch gate order i, f, g, o). x [t][i] -> [t][2h] (forward half, then backward).
struct Lstm {
    h: usize,
    dirs: [(Vec<f32>, Vec<f32>, Vec<f32>); 2], // w_ih [4h][i], w_hh [4h][h], bias_ih + bias_hh
}
impl Lstm {
    fn load(w: &St, p: &str) -> Lstm {
        let d = |s: &str| {
            let b: Vec<f32> = w.v(&format!("{p}.bias_ih_l0{s}")).iter().zip(w.v(&format!("{p}.bias_hh_l0{s}"))).map(|(a, b)| a + b).collect();
            (w.v(&format!("{p}.weight_ih_l0{s}")), w.v(&format!("{p}.weight_hh_l0{s}")), b)
        };
        let dirs = [d(""), d("_reverse")];
        Lstm { h: dirs[0].2.len() / 4, dirs }
    }
    fn fwd(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (h, i) = (self.h, x.len() / t);
        let run = |dir: usize| {
            let (wi, wh, b) = &self.dirs[dir];
            let mut pre = sgemm(x, wi, t, i, 4 * h, true);
            let mut out = vec![0f32; t * h];
            let (mut hs, mut cs) = (vec![0f32; h], vec![0f32; h]);
            for step in 0..t {
                let s = if dir == 0 { step } else { t - 1 - step };
                let g = &mut pre[s * 4 * h..(s + 1) * 4 * h];
                for (r, gv) in g.iter_mut().enumerate() {
                    let row = &wh[r * h..(r + 1) * h];
                    *gv += b[r] + dot(row, &hs);
                }
                for j in 0..h {
                    let (ig, fg, gg, og) = (sigm(g[j]), sigm(g[h + j]), g[2 * h + j].tanh(), sigm(g[3 * h + j]));
                    cs[j] = fg * cs[j] + ig * gg;
                    hs[j] = og * cs[j].tanh();
                }
                out[s * h..(s + 1) * h].copy_from_slice(&hs);
            }
            out
        };
        let (f, r) = rayon::join(|| run(0), || run(1));
        let mut y = vec![0f32; t * 2 * h];
        for s in 0..t {
            y[s * 2 * h..s * 2 * h + h].copy_from_slice(&f[s * h..(s + 1) * h]);
            y[s * 2 * h + h..(s + 1) * 2 * h].copy_from_slice(&r[s * h..(s + 1) * h]);
        }
        y
    }
}

/// y = W x + b for one vector (W [o][i]).
fn matvec(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let i = x.len();
    b.iter().enumerate().map(|(r, bv)| bv + dot(&w[r * i..(r + 1) * i], x)).collect()
}

fn layer_norm_rows(x: &mut [f32], c: usize, eps: f32) {
    for row in x.chunks_exact_mut(c) {
        let m = row.iter().sum::<f32>() / c as f32;
        let v = row.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / c as f32;
        let r = 1.0 / (v + eps).sqrt();
        row.iter_mut().for_each(|a| *a = (*a - m) * r);
    }
}

/// [rows][cols] -> [cols][rows]
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            y[c * rows + r] = x[r * cols + c];
        }
    }
    y
}

const N_FFT: usize = 20;
const HOP: usize = 5;
fn hann() -> Vec<f32> {
    (0..N_FFT).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()) as f32).collect()
}

/// torch.stft(center, reflect, hann, onesided) -> magnitude [11][frames] ++ phase [11][frames].
fn stft(x: &[f32]) -> (Vec<f32>, usize) {
    let w = hann();
    let pad = N_FFT / 2;
    let n = x.len();
    let at = |i: isize| -> f32 {
        let j = i - pad as isize;
        let j = if j < 0 { -j } else if j >= n as isize { 2 * (n as isize - 1) - j } else { j };
        x[j as usize]
    };
    let frames = 1 + n / HOP;
    let bins = N_FFT / 2 + 1;
    use rayon::prelude::*;
    let tw: Vec<(f32, f32)> =
        (0..N_FFT).map(|m| { let a = 2.0 * std::f64::consts::PI * m as f64 / N_FFT as f64; (a.cos() as f32, a.sin() as f32) }).collect();
    let per: Vec<[f32; 2 * (N_FFT / 2 + 1)]> = (0..frames)
        .into_par_iter()
        .map(|t| {
            let seg: Vec<f32> = (0..N_FFT).map(|i| at((t * HOP + i) as isize) * w[i]).collect();
            let mut o = [0f32; 2 * (N_FFT / 2 + 1)];
            for k in 0..bins {
                let (mut re, mut im) = (0f32, 0f32);
                for (i, s) in seg.iter().enumerate() {
                    let (c, si) = tw[(k * i) % N_FFT];
                    re += s * c;
                    im -= s * si;
                }
                o[k] = (re * re + im * im).sqrt();
                o[bins + k] = im.atan2(re);
            }
            o
        })
        .collect();
    let mut out = vec![0f32; 2 * bins * frames];
    for (t, o) in per.iter().enumerate() {
        for (r, v) in o.iter().enumerate() {
            out[r * frames + t] = *v;
        }
    }
    (out, frames)
}

/// torch.istft(center, hann, onesided) of spec * exp(i phase); spec, phase [11][frames].
fn istft(spec: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let w = hann();
    let bins = N_FFT / 2 + 1;
    let tw: Vec<(f32, f32)> =
        (0..N_FFT).map(|m| { let a = 2.0 * std::f64::consts::PI * m as f64 / N_FFT as f64; (a.cos() as f32, a.sin() as f32) }).collect();
    // irfft of each frame (X0 and X_{N/2}: real parts only, the others twice), windowed
    let seg: Vec<[f32; N_FFT]> = (0..frames)
        .into_par_iter()
        .map(|t| {
            let x: Vec<(f32, f32)> = (0..bins).map(|k| { let (m, p) = (spec[k * frames + t], phase[k * frames + t]); (m * p.cos(), m * p.sin()) }).collect();
            let mut o = [0f32; N_FFT];
            for (n, on) in o.iter_mut().enumerate() {
                let mut s = x[0].0 + x[N_FFT / 2].0 * if n % 2 == 0 { 1.0 } else { -1.0 };
                for (k, &(re, im)) in x.iter().enumerate().take(N_FFT / 2).skip(1) {
                    let (c, si) = tw[(k * n) % N_FFT];
                    s += 2.0 * (re * c - im * si);
                }
                *on = s / N_FFT as f32 * w[n];
            }
            o
        })
        .collect();
    let len = N_FFT + HOP * (frames - 1);
    let (mut y, mut env) = (vec![0f32; len], vec![0f32; len]);
    for (t, o) in seg.iter().enumerate() {
        for n in 0..N_FFT {
            y[t * HOP + n] += o[n];
            env[t * HOP + n] += w[n] * w[n];
        }
    }
    let pad = N_FFT / 2;
    (pad..len - pad).map(|i| if env[i] > 1e-11 { y[i] / env[i] } else { y[i] }).collect()
}

/// Small xorshift + Box-Muller (the source's random phase and noise).
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn n(&mut self) -> f32 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f32::consts::PI * self.u()).cos()
    }
}

// ---------- own GPU kernels (CubeCL, through burn's backend extension) ----------
// The vocoder's AdaINResBlock1 step = AdaIN -> Snake -> conv (+ residual). burn runs it as reductions, a few
// elementwise passes and its conv; here: one pass for the per-channel sums, then one conv kernel that applies
// AdaIN + Snake to its input while loading it and adds bias + residual when storing.
mod kern {
    use burn::backend::{Backend, backend_extension, tensor::FloatTensor};
    use burn::tensor::Shape;
    use burn_cubecl::cubecl;
    use burn_cubecl::cubecl::prelude::*;
    use burn_cubecl::kernel::into_contiguous;
    use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
    use burn_cubecl::{CubeBackend, tensor::CubeTensor};

    fn stats_shape(x: &Shape) -> Shape {
        Shape::from([x[1], 2])
    }
    fn conv_shape(x: &Shape, w: &Shape) -> Shape {
        Shape::from([1, w[1], x[2]])
    }

    #[backend_extension(Cube, Fusion)]
    pub trait KokoroOps: Backend {
        /// x [1, C, L] -> [C, 2]: sum and sum of squares over time per channel.
        #[fusion(dtype = x, shape = stats_shape(x))]
        fn chan_stats(x: FloatTensor<Self>) -> FloatTensor<Self>;
        /// conv1d(snake(adain(x)), w) + bias (+ res): stride 1, "same" padding, dilation dil; w per tap [k, cout, cin].
        /// stats from chan_stats(x); ab [2C] = AdaIN scale, shift (style and affine folded); alpha [C].
        #[fusion(dtype = x, shape = conv_shape(x, w))]
        #[allow(clippy::too_many_arguments)]
        fn conv_ada(
            x: FloatTensor<Self>,
            w: FloatTensor<Self>,
            bias: FloatTensor<Self>,
            stats: FloatTensor<Self>,
            ab: FloatTensor<Self>,
            alpha: FloatTensor<Self>,
            res: FloatTensor<Self>,
            has_res: bool,
            dil: usize,
        ) -> FloatTensor<Self>;
    }

    const ST: usize = 256; // threads per channel in chan_stats

    #[cube(launch)]
    fn stats_kernel<F: Float>(x: &Tensor<F>, out: &mut Tensor<F>, len: usize, #[define(F)] _d: ElemType) {
        let c = CUBE_POS_X as usize;
        let tid = UNIT_POS_X as usize;
        let base = c * len;
        let mut s = F::new(0.0_f32);
        let mut q = F::new(0.0_f32);
        let mut i = tid;
        while i < len {
            let v = x[base + i];
            s += v;
            q += v * v;
            i += ST;
        }
        let mut ss = Shared::<[F]>::new_slice(ST);
        let mut sq = Shared::<[F]>::new_slice(ST);
        ss[tid] = s;
        sq[tid] = q;
        sync_cube();
        #[unroll]
        for lv in 1..9 {
            let w = comptime![ST >> lv];
            if tid < w {
                let a = ss[tid] + ss[tid + w];
                ss[tid] = a;
                let b = sq[tid] + sq[tid + w];
                sq[tid] = b;
            }
            sync_cube();
        }
        if tid == 0 {
            out[c * 2] = ss[0];
            out[c * 2 + 1] = sq[0];
        }
    }

    // Conv as a matrix product on the GPU's matrix units (cmma 8x8x8 f32: Apple simdgroup matrices, NVIDIA
    // tensor cores where offered): Y[co][t] = sum over taps kk, channels ci of W_kk[co][ci] X[ci][t + kk dil - pad].
    // Cube = 128 output channels x 128 steps, 8 planes of 32 lanes, each 32 x 64 (4 x 8 fragments).
    // Weights per tap ([k][co][ci], `taps`) are read as fragments straight from memory; the input chunk
    // (kc channels, AdaIN + Snake applied, zero padded) sits in shared memory, a tap is an offset into it.
    // The tap loop stays a loop: fully unrolled, Metal ran k = 11 at ~550 instead of ~1400 GFLOPS (M1).
    const BM: usize = 128;
    const BN: usize = 128; // 2 planes x FN fragments x 8
    const FN: usize = 8;

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    fn conv_kernel<F: Float>(
        x: &Tensor<F>,
        wt: &Tensor<F>,
        bias: &Tensor<F>,
        stats: &Tensor<F>,
        ab: &Tensor<F>,
        alpha: &Tensor<F>,
        res: &Tensor<F>,
        y: &mut Tensor<F>,
        cin: usize,
        cout: usize,
        len: usize,
        #[comptime] k: usize,
        #[comptime] dil: usize,
        #[comptime] has_res: bool,
        #[comptime] kc: usize,
        #[define(F)] _d: ElemType,
    ) {
        let tid = UNIT_POS_X as usize;
        let sg = tid / 32;
        let lane = tid % 32;
        let sm = sg % 4;
        let sn = sg / 4;
        let t0 = CUBE_POS_X as usize * BN;
        let co0 = CUBE_POS_Y as usize * BM;
        let pad = comptime![(k - 1) * dil / 2];
        let span = comptime![BN + (k - 1) * dil];
        // row stride = 8 mod 16 floats: an 8 x 8 fragment load then hits each bank only twice
        let sp = comptime![(BN + (k - 1) * dil + 7) / 16 * 16 + 8];
        let mut xs = Shared::<[F]>::new_slice(comptime![kc * ((BN + (k - 1) * dil + 7) / 16 * 16 + 8)]);
        let mut scratch = Shared::<[F]>::new_slice(512usize);
        let mut accs = Sequence::<cmma::Matrix<F>>::new();
        #[unroll]
        for _ in 0..comptime![4 * FN] {
            let mut m = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined) };
            cmma::fill(&mut m, F::new(0.0_f32));
            accs.push(m);
        }
        let mut fa = Sequence::<cmma::Matrix<F>>::new();
        let mut fb = Sequence::<cmma::Matrix<F>>::new();
        #[unroll]
        for _ in 0..4 {
            fa.push(unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor) });
        }
        #[unroll]
        for _ in 0..FN {
            fb.push(unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor) });
        }
        let inv_len = F::new(1.0_f32) / F::cast_from(len as u32);
        let tap_stride = cout * cin;
        let mut ci0 = 0usize;
        while ci0 < cin {
            let mut idx = tid;
            while idx < comptime![kc * (BN + (k - 1) * dil)] {
                let c = idx / span;
                let p = idx % span;
                let ch = ci0 + c;
                let t = t0 + p;
                let mut v = F::new(0.0_f32);
                if t >= pad && t - pad < len {
                    let m = stats[ch * 2] * inv_len;
                    let var = stats[ch * 2 + 1] * inv_len - m * m;
                    let sc = ab[ch] / F::sqrt(var + F::new(1e-5_f32));
                    let a = alpha[ch];
                    let u = x[ch * len + t - pad] * sc + (ab[cin + ch] - sc * m);
                    let s = F::sin(a * u);
                    v = u + s * s / a;
                }
                xs[c * sp + p] = v;
                idx += 256;
            }
            sync_cube();
            for kk in 0..k {
                // not unrolled (see above)
                #[unroll]
                for c8 in 0..comptime![kc / 8] {
                    #[unroll]
                    for i in 0..4 {
                        let off = kk * tap_stride + (co0 + sm * 32 + i * 8) * cin + ci0 + c8 * 8;
                        cmma::load(fa.index_mut(i), &wt.slice(off, off + 7 * cin + 8), cin as u32);
                    }
                    #[unroll]
                    for j in 0..FN {
                        let off = c8 * 8 * sp + sn * FN * 8 + j * 8 + kk * dil;
                        cmma::load(fb.index_mut(j), &xs.slice(off, off + 7 * sp + 8), sp as u32);
                    }
                    #[unroll]
                    for i in 0..4 {
                        #[unroll]
                        for j in 0..FN {
                            cmma::execute(fa.index(i), fb.index(j), accs.index(i * FN + j), accs.index(i * FN + j));
                        }
                    }
                }
            }
            sync_cube();
            ci0 += kc;
        }
        // bias (+ residual) through a per-plane 8 x 8 scratch, bounds-checked at the end of the signal
        #[unroll]
        for i in 0..4 {
            #[unroll]
            for j in 0..FN {
                cmma::store(&mut scratch.slice_mut(sg * 64, sg * 64 + 64), accs.index(i * FN + j), 8u32, cmma::MatrixLayout::RowMajor);
                sync_cube();
                #[unroll]
                for h in 0..2usize {
                    let e = lane + 32 * h;
                    let co = co0 + sm * 32 + i * 8 + e / 8;
                    let t = t0 + sn * FN * 8 + j * 8 + e % 8;
                    if t < len {
                        let mut v = scratch[sg * 64 + e] + bias[co];
                        if comptime![has_res] {
                            v += res[co * len + t];
                        }
                        y[co * len + t] = v;
                    }
                }
                sync_cube();
            }
        }
    }

    impl KokoroOps for CubeBackend {
        fn chan_stats(x: CubeTensor) -> CubeTensor {
            let x = into_contiguous(x);
            let s = x.meta.shape().clone();
            let (c, len) = (s[1], s[2]);
            let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::from([c, 2]), x.dtype);
            let dt = burn::backend::cubecl::dtype_to_storage_type(x.dtype);
            stats_kernel::launch(
                &x.client,
                CubeCount::Static(c as u32, 1, 1),
                CubeDim::new_1d(ST as u32),
                x.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                len,
                dt,
            );
            out
        }
        fn conv_ada(
            x: CubeTensor,
            w: CubeTensor,
            bias: CubeTensor,
            stats: CubeTensor,
            ab: CubeTensor,
            alpha: CubeTensor,
            res: CubeTensor,
            has_res: bool,
            dil: usize,
        ) -> CubeTensor {
            let (x, w, res) = (into_contiguous(x), into_contiguous(w), into_contiguous(res));
            let (xs, ws) = (x.meta.shape().clone(), w.meta.shape().clone());
            let (cin, len, k, cout) = (xs[1], xs[2], ws[0], ws[1]);
            assert_eq!(ws[2], cin);
            let kc = if k <= 3 { 32 } else { 16 };
            assert!(cin % kc == 0 && cout % BM == 0, "conv_ada: channels {cin} -> {cout}");
            let y = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::from([1, cout, len]), x.dtype);
            let dt = burn::backend::cubecl::dtype_to_storage_type(x.dtype);
            conv_kernel::launch(
                &x.client,
                CubeCount::Static(len.div_ceil(BN) as u32, (cout / BM) as u32, 1),
                CubeDim::new_1d(256),
                x.clone().into_tensor_arg(),
                w.into_tensor_arg(),
                into_contiguous(bias).into_tensor_arg(),
                into_contiguous(stats).into_tensor_arg(),
                into_contiguous(ab).into_tensor_arg(),
                into_contiguous(alpha).into_tensor_arg(),
                res.into_tensor_arg(),
                y.clone().into_tensor_arg(),
                cin,
                cout,
                len,
                k,
                dil,
                has_res,
                kc,
                dt,
            );
            y
        }
    }
}

/// Our kernels from tensors (see `kern`).
fn chan_stats(x: Tensor<3>) -> Tensor<2> {
    use kern::KokoroOps;
    Tensor::from_dispatch(burn::backend::Dispatch::chan_stats(x.into_dispatch()))
}
#[allow(clippy::too_many_arguments)]
fn conv_ada(x: Tensor<3>, w: Tensor<3>, b: Tensor<1>, stats: Tensor<2>, ab: Tensor<1>, alpha: Tensor<1>, res: Option<Tensor<3>>, dil: usize) -> Tensor<3> {
    use kern::KokoroOps;
    let has = res.is_some();
    let res = res.unwrap_or_else(|| x.clone());
    Tensor::from_dispatch(burn::backend::Dispatch::conv_ada(
        x.into_dispatch(),
        w.into_dispatch(),
        b.into_dispatch(),
        stats.into_dispatch(),
        ab.into_dispatch(),
        alpha.into_dispatch(),
        res.into_dispatch(),
        has,
        dil,
    ))
}

// ---------- GPU parts ----------
struct Gpu<'a> {
    w: &'a St,
    dev: Device,
    /// Weights on the GPU, uploaded on first use (flat, with their shape).
    cache: std::cell::RefCell<HashMap<String, (Tensor<1>, Vec<usize>)>>,
    /// Own kernels for the vocoder blocks (OWN=0: burn's ops only).
    own: bool,
}
impl Gpu<'_> {
    fn up<const N: usize>(&self, v: Vec<f32>, shape: [usize; N]) -> Tensor<N> {
        Tensor::from_data(TensorData::new(v, shape), &self.dev)
    }
    fn t<const N: usize>(&self, name: &str) -> Tensor<N> {
        let mut c = self.cache.borrow_mut();
        let (t, s) = c.entry(name.to_string()).or_insert_with(|| {
            let (v, s) = self.w.get(name);
            let n = v.len();
            (self.up(v, [n]), s)
        });
        let shape: [usize; N] = s.clone().try_into().unwrap_or_else(|s| panic!("{name}: {s:?}"));
        t.clone().reshape(shape)
    }
    /// Conv weight [cout][cin][k] as [k][cout][cin] (our conv kernel), uploaded once.
    fn taps(&self, name: &str) -> Tensor<3> {
        let key = format!("{name}:taps");
        let mut c = self.cache.borrow_mut();
        let (t, s) = c.entry(key).or_insert_with(|| {
            let (v, s) = self.w.get(name);
            let (co, ci, k) = (s[0], s[1], s[2]);
            let mut r = vec![0f32; v.len()];
            for o in 0..co {
                for i in 0..ci {
                    for kk in 0..k {
                        r[(kk * co + o) * ci + i] = v[(o * ci + i) * k + kk];
                    }
                }
            }
            (self.up(r, [v.len()]), vec![k, co, ci])
        });
        t.clone().reshape([s[0], s[1], s[2]])
    }
    fn lin(&self, x: Tensor<2>, p: &str) -> Tensor<2> {
        let w: Tensor<2> = self.t(&format!("{p}.weight"));
        let b: Tensor<1> = self.t(&format!("{p}.bias"));
        x.matmul(w.transpose()) + b.unsqueeze()
    }
    fn conv(&self, x: Tensor<3>, p: &str, stride: usize, pad: usize, dil: usize) -> Tensor<3> {
        let b = self.w.header.contains_key(&format!("{p}.bias")).then(|| self.t::<1>(&format!("{p}.bias")));
        conv1d(x, self.t(&format!("{p}.weight")), b, ConvOptions::new([stride], [pad], [dil], 1))
    }
    fn conv_t(&self, x: Tensor<3>, p: &str, stride: usize, pad: usize, out_pad: usize, groups: usize) -> Tensor<3> {
        conv_transpose1d(x, self.t(&format!("{p}.weight")), Some(self.t(&format!("{p}.bias"))), ConvTransposeOptions::new([stride], [pad], [out_pad], [1], groups))
    }
    /// Instance norm over time (affine) + style: (1 + gamma) * norm(x) + beta, gamma/beta = fc(s).
    fn adain(&self, x: Tensor<3>, p: &str, s: &[f32]) -> Tensor<3> {
        let c = x.dims()[1];
        let h = matvec(&self.w.v(&format!("{p}.fc.weight")), &self.w.v(&format!("{p}.fc.bias")), s);
        let (nw, nb) = (self.w.v(&format!("{p}.norm.weight")), self.w.v(&format!("{p}.norm.bias")));
        let a: Vec<f32> = (0..c).map(|i| (1.0 + h[i]) * nw[i]).collect();
        let b: Vec<f32> = (0..c).map(|i| (1.0 + h[i]) * nb[i] + h[c + i]).collect();
        // one scale and shift per channel: y = x * a r + (b - a r m), r = 1 / sqrt(var + eps)
        let m = x.clone().mean_dim(2);
        let var = (x.clone() * x.clone()).mean_dim(2) - m.clone() * m.clone();
        let sc = self.up(a, [1, c, 1]) / (var + 1e-5).sqrt();
        let sh = self.up(b, [1, c, 1]) - sc.clone() * m;
        x * sc + sh
    }
    /// AdainResBlk1d (predictor F0/N blocks, decoder encode/decode).
    fn res_blk(&self, x: Tensor<3>, p: &str, s: &[f32], upsample: bool) -> Tensor<3> {
        let c = x.dims()[1];
        let mut r = leaky_relu(self.adain(x.clone(), &format!("{p}.norm1"), s), 0.2);
        if upsample {
            r = self.conv_t(r, &format!("{p}.pool"), 2, 1, 1, c);
        }
        r = self.conv(r, &format!("{p}.conv1"), 1, 1, 1);
        r = leaky_relu(self.adain(r, &format!("{p}.norm2"), s), 0.2);
        r = self.conv(r, &format!("{p}.conv2"), 1, 1, 1);
        let mut sc = x;
        if upsample {
            let [b, c, l] = sc.dims();
            sc = sc.unsqueeze_dim::<4>(3).repeat_dim(3, 2).reshape([b, c, 2 * l]);
        }
        if self.w.header.contains_key(&format!("{p}.conv1x1.weight")) {
            sc = self.conv(sc, &format!("{p}.conv1x1"), 1, 0, 1);
        }
        (r + sc) * std::f32::consts::FRAC_1_SQRT_2
    }
    fn snake(&self, x: Tensor<3>, alpha: &str) -> Tensor<3> {
        let a: Tensor<3> = self.t(alpha);
        let s = (x.clone() * a.clone()).sin();
        x + s.clone() * s / a
    }
    /// AdaIN's per-channel scale and shift before normalization, [2C]: a = (1 + gamma) w, b = (1 + gamma) bias + beta.
    fn ada_ab(&self, p: &str, s: &[f32]) -> Tensor<1> {
        let h = matvec(&self.w.v(&format!("{p}.fc.weight")), &self.w.v(&format!("{p}.fc.bias")), s);
        let (nw, nb) = (self.w.v(&format!("{p}.norm.weight")), self.w.v(&format!("{p}.norm.bias")));
        let c = nw.len();
        let ab: Vec<f32> = (0..c).map(|i| (1.0 + h[i]) * nw[i]).chain((0..c).map(|i| (1.0 + h[i]) * nb[i] + h[c + i])).collect();
        self.up(ab, [2 * c])
    }
    /// AdaINResBlock1 (vocoder), kernel k, dilations 1/3/5.
    fn res_blk1(&self, mut x: Tensor<3>, p: &str, s: &[f32], k: usize) -> Tensor<3> {
        if self.own {
            for (j, d) in [1usize, 3, 5].into_iter().enumerate() {
                let c1 = format!("{p}.convs1.{j}");
                let t = conv_ada(x.clone(), self.taps(&format!("{c1}.weight")), self.t(&format!("{c1}.bias")), chan_stats(x.clone()),
                    self.ada_ab(&format!("{p}.adain1.{j}"), s), self.t::<3>(&format!("{p}.alpha1.{j}")).flatten(0, 2), None, d);
                let c2 = format!("{p}.convs2.{j}");
                x = conv_ada(t.clone(), self.taps(&format!("{c2}.weight")), self.t(&format!("{c2}.bias")), chan_stats(t),
                    self.ada_ab(&format!("{p}.adain2.{j}"), s), self.t::<3>(&format!("{p}.alpha2.{j}")).flatten(0, 2), Some(x), 1);
            }
            return x;
        }
        for (j, d) in [1usize, 3, 5].into_iter().enumerate() {
            let mut t = self.adain(x.clone(), &format!("{p}.adain1.{j}"), s);
            t = self.snake(t, &format!("{p}.alpha1.{j}"));
            t = self.conv(t, &format!("{p}.convs1.{j}"), 1, (k * d - d) / 2, d);
            t = self.adain(t, &format!("{p}.adain2.{j}"), s);
            t = self.snake(t, &format!("{p}.alpha2.{j}"));
            t = self.conv(t, &format!("{p}.convs2.{j}"), 1, (k - 1) / 2, 1);
            x = t + x;
        }
        x
    }
    fn layer_norm_ch(&self, x: Tensor<3>, gamma: &str, beta: &str) -> Tensor<3> {
        let c = x.dims()[1];
        let m = x.clone().mean_dim(1);
        let xc = x - m;
        let v = (xc.clone() * xc.clone()).mean_dim(1);
        let g: Tensor<1> = self.t(gamma);
        let b: Tensor<1> = self.t(beta);
        xc / (v + 1e-5).sqrt() * g.reshape([1, c, 1]) + b.reshape([1, c, 1])
    }
    fn layer_norm_rows(&self, x: Tensor<2>, p: &str, eps: f32) -> Tensor<2> {
        let m = x.clone().mean_dim(1);
        let xc = x - m;
        let v = (xc.clone() * xc.clone()).mean_dim(1);
        let g: Tensor<1> = self.t(&format!("{p}.weight"));
        let b: Tensor<1> = self.t(&format!("{p}.bias"));
        xc / (v + eps).sqrt() * g.unsqueeze() + b.unsqueeze()
    }
}

/// PROF=1: wait for the GPU and print the time since the last mark.
fn mark(x: &Tensor<3>, what: &str, clock: &mut Instant) {
    if std::env::var("PROF").is_ok() {
        let _ = x.clone().sum().into_data();
        let _ = x.device().sync();
        eprintln!("  {what}: {:.0} ms", clock.elapsed().as_secs_f64() * 1e3);
        *clock = Instant::now();
    }
}

fn read(t: Tensor<3>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}
fn read2(t: Tensor<2>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}

struct Out {
    audio: Vec<f32>,
    dump: Vec<(&'static str, Vec<f32>)>,
    times: Vec<(&'static str, f64)>,
}

fn synth(g: &Gpu, ids: &[usize], style: &[f32], noise: bool) -> Out {
    let w = g.w;
    let t = ids.len();
    let mut times = Vec::new();
    let mut dump = Vec::new();
    let mut clock = Instant::now();
    let mut lap = |name: &'static str, times: &mut Vec<(&'static str, f64)>| {
        times.push((name, clock.elapsed().as_secs_f64() * 1e3));
        clock = Instant::now();
    };
    let (s_dec, s_pred) = (&style[..128], &style[128..]);
    let dumping = std::env::var("DUMP").is_ok();

    // ALBERT (GPU): embeddings on the CPU (gather + LayerNorm), 12 shared layers.
    let (we, pe, te) = (w.v("bert.embeddings.word_embeddings.weight"), w.v("bert.embeddings.position_embeddings.weight"), w.v("bert.embeddings.token_type_embeddings.weight"));
    let mut e = vec![0f32; t * 128];
    for (i, &id) in ids.iter().enumerate() {
        for j in 0..128 {
            e[i * 128 + j] = we[id * 128 + j] + pe[i * 128 + j] + te[j];
        }
    }
    let x = g.layer_norm_rows(g.up(e, [t, 128]), "bert.embeddings.LayerNorm", 1e-12);
    let mut x = g.lin(x, "bert.encoder.embedding_hidden_mapping_in");
    let l = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    for _ in 0..12 {
        let heads = |y: Tensor<2>| y.reshape([t, 12, 64]).swap_dims(0, 1);
        let q = heads(g.lin(x.clone(), &format!("{l}.attention.query")));
        let k = heads(g.lin(x.clone(), &format!("{l}.attention.key")));
        let v = heads(g.lin(x.clone(), &format!("{l}.attention.value")));
        let p = softmax(q.matmul(k.swap_dims(1, 2)) / 8.0, 2);
        let ctx = p.matmul(v).swap_dims(0, 1).reshape([t, 768]);
        let a = g.layer_norm_rows(x + g.lin(ctx, &format!("{l}.attention.dense")), &format!("{l}.attention.LayerNorm"), 1e-12);
        let f = g.lin(gelu_approximate(g.lin(a.clone(), &format!("{l}.ffn"))), &format!("{l}.ffn_output"));
        x = g.layer_norm_rows(f + a, &format!("{l}.full_layer_layer_norm"), 1e-12);
    }
    let d_en = read2(g.lin(x, "bert_encoder")); // [t][512]
    lap("bert", &mut times);

    // Duration encoder + durations (CPU).
    let cat_s = |x: &[f32], c: usize| -> Vec<f32> { x.chunks_exact(c).flat_map(|r| r.iter().chain(s_pred).copied()).collect() };
    let mut d = cat_s(&d_en, 512);
    for i in 0..3 {
        let y = Lstm::load(w, &format!("predictor.text_encoder.lstms.{}", 2 * i)).fwd(&d, t);
        let mut y = y;
        layer_norm_rows(&mut y, 512, 1e-5);
        let p = format!("predictor.text_encoder.lstms.{}", 2 * i + 1);
        let h = matvec(&w.v(&format!("{p}.fc.weight")), &w.v(&format!("{p}.fc.bias")), s_pred);
        for row in y.chunks_exact_mut(512) {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (1.0 + h[j]) * *v + h[512 + j];
            }
        }
        d = cat_s(&y, 512); // [t][640]
    }
    let x = Lstm::load(w, "predictor.lstm").fwd(&d, t);
    let (dw, db) = (w.v("predictor.duration_proj.linear_layer.weight"), w.v("predictor.duration_proj.linear_layer.bias"));
    let dur: Vec<usize> = x
        .chunks_exact(512)
        .map(|r| {
            let s: f32 = matvec(&dw, &db, r).into_iter().map(sigm).sum();
            (s.round_ties_even().max(1.0)) as usize
        })
        .collect();
    let frames: usize = dur.iter().sum();
    let src: Vec<usize> = dur.iter().enumerate().flat_map(|(i, &n)| std::iter::repeat_n(i, n)).collect();
    let en: Vec<f32> = src.iter().flat_map(|&i| d[i * 640..(i + 1) * 640].iter().copied()).collect(); // [frames][640]
    let shared = Lstm::load(w, "predictor.shared").fwd(&en, frames); // [frames][512]
    lap("durations+lstm", &mut times);

    // F0 / N (GPU).
    let sh = g.up(transpose(&shared, frames, 512), [1, 512, frames]);
    let curve = |p: &str| {
        let mut y = g.res_blk(sh.clone(), &format!("predictor.{p}.0"), s_pred, false);
        y = g.res_blk(y, &format!("predictor.{p}.1"), s_pred, true);
        y = g.res_blk(y, &format!("predictor.{p}.2"), s_pred, false);
        g.conv(y, &format!("predictor.{p}_proj"), 1, 0, 1) // [1, 1, 2 frames]
    };
    let (f0, nn) = (curve("F0"), curve("N"));

    // Text encoder: embedding (CPU) + CNN (GPU) + LSTM (CPU).
    let emb = w.v("text_encoder.embedding.weight");
    let xe: Vec<f32> = ids.iter().flat_map(|&id| emb[id * 512..(id + 1) * 512].iter().copied()).collect();
    let mut xt = g.up(transpose(&xe, t, 512), [1, 512, t]);
    for i in 0..3 {
        xt = g.conv(xt, &format!("text_encoder.cnn.{i}.0"), 1, 2, 1);
        xt = leaky_relu(g.layer_norm_ch(xt, &format!("text_encoder.cnn.{i}.1.gamma"), &format!("text_encoder.cnn.{i}.1.beta")), 0.2);
    }
    let xt = transpose(&read(xt), 512, t);
    let t_en = Lstm::load(w, "text_encoder.lstm").fwd(&xt, t); // [t][512]
    let asr: Vec<f32> = src.iter().flat_map(|&i| t_en[i * 512..(i + 1) * 512].iter().copied()).collect();
    let asr = g.up(transpose(&asr, frames, 512), [1, 512, frames]);
    let f0_curve = read(f0.clone());
    lap("f0n+text", &mut times);

    // Sine source (CPU): harmonics 1..9 of F0 per sample, phase summed per frame (as the reference), STFT.
    let hf = f0_curve.len(); // 80 Hz frames
    let up = 300;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (lw, lb) = (w.v("decoder.generator.m_source.l_linear.weight"), w.v("decoder.generator.m_source.l_linear.bias")[0]);
    let mut har = vec![0f32; hf * up];
    let mut phases = vec![0f32; 9 * hf];
    for h in 0..9 {
        let ini = if noise && h > 0 { rng.u() } else { 0.0 };
        let mut acc = 0f64; // torch's CPU cumsum adds in f64
        for i in 0..hf {
            let mut rad = (f0_curve[i] * (h + 1) as f32 / 24000.0).rem_euclid(1.0);
            if i == 0 {
                rad += ini;
            }
            acc += rad as f64;
            phases[h * hf + i] = acc as f32 * (2.0 * std::f32::consts::PI) * up as f32;
        }
    }
    {
        use rayon::prelude::*;
        har.par_chunks_mut(up * 8).enumerate().for_each(|(c, out)| {
            let mut rng = Rng(0x9E3779B97F4A7C15 ^ ((c as u64 + 1) * 0xD1B54A32D192ED03));
            for (o, v) in out.iter_mut().enumerate() {
                let j = c * up * 8 + o;
                let srcp = ((j as f32 + 0.5) / up as f32 - 0.5).max(0.0);
                let i0 = srcp.floor() as usize;
                let i1 = (i0 + 1).min(hf - 1);
                let lam = srcp - i0 as f32;
                let voiced = f0_curve[j / up] > 10.0;
                let mut s = lb;
                for h in 0..9 {
                    let p = &phases[h * hf..];
                    let ph = (1.0 - lam) * p[i0] + lam * p[i1];
                    let mut x = if voiced { ph.sin() * 0.1 } else { 0.0 };
                    if noise {
                        x += rng.n() * if voiced { 0.003 } else { 0.1 / 3.0 };
                    }
                    s += lw[h] * x;
                }
                *v = s.tanh();
            }
        });
    }
    let (mut spec, sf) = stft(&har);
    if let Ok(f) = std::env::var("HAR") {
        spec = St::open(Path::new(&f)).v("har"); // test: the reference's source spectrum
    }
    if dumping {
        dump.push(("src", har.clone()));
        dump.push(("har", spec.clone()));
    }
    let har = g.up(spec, [1, 22, sf]);
    lap("source", &mut times);

    // Decoder (GPU).
    let mut pc = Instant::now();
    mark(&har, "upload", &mut pc);
    let f0c = g.conv(f0, "decoder.F0_conv", 2, 1, 1);
    let nc = g.conv(nn, "decoder.N_conv", 2, 1, 1);
    let mut x = g.res_blk(Tensor::cat(vec![asr.clone(), f0c.clone(), nc.clone()], 1), "decoder.encode", s_dec, false);
    let asr_res = g.conv(asr, "decoder.asr_res.0", 1, 0, 1);
    for i in 0..4 {
        x = Tensor::cat(vec![x, asr_res.clone(), f0c.clone(), nc.clone()], 1);
        x = g.res_blk(x, &format!("decoder.decode.{i}"), s_dec, i == 3);
    }
    mark(&x, "decoder blocks", &mut pc);
    // Generator.
    if dumping {
        dump.push(("gen_in", read(x.clone())));
    }
    let p = "decoder.generator";
    for i in 0..2 {
        x = leaky_relu(x, 0.1);
        let xs = if i == 0 {
            conv1d(har.clone(), g.t(&format!("{p}.noise_convs.0.weight")), Some(g.t(&format!("{p}.noise_convs.0.bias"))), ConvOptions::new([6], [3], [1], 1))
        } else {
            g.conv(har.clone(), &format!("{p}.noise_convs.1"), 1, 0, 1)
        };
        let xs = g.res_blk1(xs, &format!("{p}.noise_res.{i}"), s_dec, if i == 0 { 7 } else { 11 });
        mark(&xs, "noise path", &mut pc);
        x = if i == 0 { g.conv_t(x, &format!("{p}.ups.0"), 10, 5, 0, 1) } else { g.conv_t(x, &format!("{p}.ups.1"), 6, 3, 0, 1) };
        if i == 1 {
            let l = x.dims()[2];
            x = Tensor::cat(vec![x.clone().narrow(2, 1, 1), x.narrow(2, 0, l)], 2); // reflection pad (1, 0)
        }
        x = x + xs;
        mark(&x, "ups", &mut pc);
        let mut acc: Option<Tensor<3>> = None;
        for (j, k) in [3usize, 7, 11].into_iter().enumerate() {
            let y = g.res_blk1(x.clone(), &format!("{p}.resblocks.{}", i * 3 + j), s_dec, k);
            acc = Some(match acc { None => y, Some(a) => a + y });
            mark(acc.as_ref().unwrap(), &format!("resblock k{k}"), &mut pc);
        }
        x = acc.unwrap() / 3.0;
    }
    x = leaky_relu(x, 0.01);
    mark(&x, "resblocks /3 + leaky", &mut pc);
    let post = g.conv(x, &format!("{p}.conv_post"), 1, 3, 1);
    mark(&post, "conv_post", &mut pc);
    let post = read(post); // [22][sf]
    mark(&g.up(vec![0.0], [1, 1, 1]), "read", &mut pc);
    if dumping {
        dump.push(("post", post.clone()));
    }
    lap("decoder (gpu)", &mut times);
    let bins = N_FFT / 2 + 1;
    let spec: Vec<f32> = post[..bins * sf].iter().map(|v| v.exp()).collect();
    let phase: Vec<f32> = post[bins * sf..].iter().map(|v| v.sin()).collect();
    let audio = istft(&spec, &phase, sf);
    lap("istft", &mut times);
    dump.push(("dur", dur.iter().map(|&v| v as f32).collect()));
    dump.push(("F0", f0_curve));
    dump.push(("t_en", transpose(&t_en, t, 512)));
    dump.push(("audio", audio.clone()));
    Out { audio, dump, times }
}

/// KTEST=1: our kernels vs burn's ops on random data (relative error) and their speed.
fn ktest(g: &Gpu) {
    let mut rng = Rng(7);
    let rel = |a: Vec<f32>, b: Vec<f32>| {
        let d: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        d / b.iter().map(|y| y * y).sum::<f32>().sqrt()
    };
    let all = [(128usize, 1000usize, 3usize, 1usize, false), (128, 55800, 3, 5, true), (256, 9300, 7, 3, false), (128, 55800, 11, 5, true), (128, 55800, 11, 1, true), (128, 55800, 11, 3, false), (256, 9300, 11, 1, false), (128, 55800, 11, 2, false), (128, 55800, 11, 4, false), (128, 55800, 3, 1, false), (128, 55800, 7, 1, false)];
    // KTEST=1: all; KTEST=3,4: those (by position)
    let pick: Vec<usize> = std::env::var("KTEST").unwrap().split(',').filter_map(|v| v.parse().ok()).filter(|&i: &usize| i > 1 || std::env::var("KTEST").unwrap() != "1").collect();
    let list: Vec<_> = if pick.is_empty() { all.to_vec() } else { pick.iter().map(|&i| all[i]).collect() };
    for (c, l, k, d, res) in list {
        let x: Tensor<3> = g.up((0..c * l).map(|_| rng.n() * 2.0 + 0.5).collect(), [1, c, l]);
        let st = chan_stats(x.clone());
        let want = Tensor::cat(vec![x.clone().sum_dim(2), (x.clone() * x.clone()).sum_dim(2)], 2).reshape([c, 2]);
        let e_st = rel(read2(st.clone()), read2(want));
        let w: Tensor<3> = g.up((0..c * c * k).map(|_| rng.n() * 0.05).collect(), [c, c, k]);
        let b: Tensor<1> = g.up((0..c).map(|_| rng.n()).collect(), [c]);
        let ab: Vec<f32> = (0..2 * c).map(|i| if i < c { 1.0 + 0.3 * rng.n() } else { 0.3 * rng.n() }).collect();
        let al: Vec<f32> = (0..c).map(|_| 0.5 + rng.u()).collect();
        let r = res.then(|| x.clone() * 0.5);
        let wt = w.clone().permute([2, 0, 1]);
        let got = conv_ada(x.clone(), wt.clone(), b.clone(), st, g.up(ab.clone(), [2 * c]), g.up(al.clone(), [c]), r.clone(), d);
        // burn: the same step from its ops
        let m = x.clone().mean_dim(2);
        let var = (x.clone() * x.clone()).mean_dim(2) - m.clone() * m.clone();
        let sc = g.up(ab[..c].to_vec(), [1, c, 1]) / (var + 1e-5).sqrt();
        let sh = g.up(ab[c..].to_vec(), [1, c, 1]) - sc.clone() * m;
        let u = x.clone() * sc + sh;
        let a = g.up(al, [1, c, 1]);
        let s = (u.clone() * a.clone()).sin();
        let mut want = conv1d(u + s.clone() * s / a, w.clone(), Some(b.clone()), ConvOptions::new([1], [(k - 1) * d / 2], [d], 1));
        if let Some(r) = r {
            want = want + r;
        }
        let e = rel(read(got), read(want));
        // speed of ours (stats + conv) vs burn's
        let time = |f: &dyn Fn() -> Tensor<3>| {
            for _ in 0..3 {
                let _ = f().narrow(1, 7, 1).into_data();
                let _ = g.dev.sync();
            }
            let t = Instant::now();
            for _ in 0..10 {
                let _ = f().narrow(1, 7, 1).into_data();
                let _ = g.dev.sync();
            }
            t.elapsed().as_secs_f64() * 1e2
        };
        let ab_t: Tensor<1> = g.up(ab.clone(), [2 * c]);
        let al_t: Tensor<1> = g.up((0..c).map(|i| 0.5 + i as f32 / c as f32).collect(), [c]);
        let ours = time(&|| conv_ada(x.clone(), wt.clone(), b.clone(), chan_stats(x.clone()), ab_t.clone(), al_t.clone(), None, d));
        let st0 = chan_stats(x.clone());
        let conv_only = time(&|| conv_ada(x.clone(), wt.clone(), b.clone(), st0.clone(), ab_t.clone(), al_t.clone(), None, d));
        let stats_only = time(&|| chan_stats(x.clone()).reshape([1, c, 2]));
        let burn_conv = time(&|| conv1d(x.clone(), w.clone(), Some(b.clone()), ConvOptions::new([1], [(k - 1) * d / 2], [d], 1)));
        eprintln!("  conv only {conv_only:.2} ms ({:.0} GFLOPS), stats only {stats_only:.2} ms, burn conv1d alone {burn_conv:.2} ms", 2.0 * (c * c * k * l) as f64 / (conv_only * 1e6));
        let a3: Tensor<3> = al_t.clone().reshape([1, c, 1]);
        let burn = time(&|| {
            let m = x.clone().mean_dim(2);
            let var = (x.clone() * x.clone()).mean_dim(2) - m.clone() * m.clone();
            let sc = a3.clone() / (var + 1e-5).sqrt();
            let u = x.clone() * sc.clone() + (a3.clone() - sc * m);
            let s = (u.clone() * a3.clone()).sin();
            conv1d(u + s.clone() * s / a3.clone(), w.clone(), Some(b.clone()), ConvOptions::new([1], [(k - 1) * d / 2], [d], 1))
        });
        let gf = 2.0 * (c * c * k * l) as f64 / (ours * 1e6);
        eprintln!("c{c} l{l} k{k} d{d} res {res}: stats rel {e_st:.2e}, conv rel {e:.2e} | ours {ours:.2} ms ({gf:.0} GFLOPS), burn {burn:.2} ms");
    }
}

/// BENCH=1: the vocoder's 4800 Hz conv (128 ch, 55800 steps) vs a matmul of the same work (im2col shape).
fn bench(g: &Gpu) {
    let l = 55800;
    let x: Tensor<3> = g.up((0..128 * l).map(|i| ((i % 97) as f32 - 48.0) / 50.0).collect(), [1, 128, l]);
    let time = |name: &str, f: &dyn Fn() -> Tensor<3>| {
        for _ in 0..3 {
            let _ = f().narrow(1, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        let t = Instant::now();
        for _ in 0..10 {
            let _ = f().narrow(1, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        eprintln!("{name}: {:.2} ms", t.elapsed().as_secs_f64() * 1e2);
    };
    let a: Tensor<3> = g.up(vec![0.7; 128], [1, 128, 1]);
    time("copy x*2", &|| x.clone() * 2.0);
    time("mean_dim", &|| x.clone().mean_dim(2));
    time("mean_dim(x*x)", &|| (x.clone() * x.clone()).mean_dim(2));
    time("scale+shift", &|| x.clone() * a.clone() + a.clone());
    time("snake", &|| { let s = (x.clone() * a.clone()).sin(); x.clone() + s.clone() * s / a.clone() });
    time("adain (2 reductions + apply)", &|| {
        let m = x.clone().mean_dim(2);
        let var = (x.clone() * x.clone()).mean_dim(2) - m.clone() * m.clone();
        let sc = a.clone() / (var + 1e-5).sqrt();
        let sh = a.clone() - sc.clone() * m;
        x.clone() * sc + sh
    });
    for (k, d) in [(3usize, 1usize), (3, 5), (11, 1), (11, 5)] {
        let wt: Tensor<3> = g.up((0..128 * 128 * k).map(|i| ((i % 89) as f32 - 44.0) / 4000.0).collect(), [128, 128, k]);
        let run = || conv1d(x.clone(), wt.clone(), None, ConvOptions::new([1], [(k * d - d) / 2], [d], 1));
        for _ in 0..2 {
            let _ = run().narrow(2, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        let t = Instant::now();
        for _ in 0..5 {
            let _ = run().narrow(2, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 5.0;
        let gf = 2.0 * 128.0 * 128.0 * k as f64 * l as f64 / (ms * 1e6);
        eprintln!("conv1d k{k} d{d}: {ms:.1} ms, {gf:.0} GFLOPS");
        let a: Tensor<2> = g.up((0..128 * 128 * k).map(|i| ((i % 89) as f32 - 44.0) / 4000.0).collect(), [128, 128 * k]);
        let b: Tensor<2> = g.up((0..128 * k * l).map(|i| ((i % 83) as f32 - 41.0) / 40.0).collect(), [128 * k, l]);
        let mm = || a.clone().matmul(b.clone());
        for _ in 0..2 {
            let _ = mm().narrow(1, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        let t = Instant::now();
        for _ in 0..5 {
            let _ = mm().narrow(1, 7, 1).into_data();
            let _ = g.dev.sync();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 5.0;
        eprintln!("   matmul [128x{}]x[{}x{l}]: {ms:.1} ms, {:.0} GFLOPS", 128 * k, 128 * k, gf * 0.0 + 2.0 * 128.0 * 128.0 * k as f64 * l as f64 / (ms * 1e6));
    }
}

// CubeCL's disk cache keys kernels by our build key (Cargo.lock + rustc, lib.rs), which doesn't see our own
// kernels: a cache folder per version of their source (`mod kern`), else an edited kernel loads the old binary.
fn set_cache() {
    let src = include_str!("kokoro.rs");
    let kern = &src[src.find("\nmod kern {").unwrap()..];
    let kern = &kern[..kern.find("\n}\n").unwrap()];
    let mut h = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(kern, &mut h);
    dictation::asr::set_kernel_cache(Path::new(&format!("target/kokoro-kernels/{:016x}", std::hash::Hasher::finish(&h))));
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&a[0]);
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let vocab: HashMap<String, usize> = cfg["vocab"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_u64().unwrap() as usize)).collect();
    if a[1] == "--listen" {
        set_cache();
        #[cfg(target_os = "macos")]
        listen::run(dir.to_path_buf(), a.get(2).cloned().unwrap_or("af_heart".into()), vocab);
        #[cfg(not(target_os = "macos"))]
        panic!("--listen: macOS only for now");
    }
    let t0 = Instant::now();
    let ph = &if std::env::var("PH").is_ok_and(|v| v == "1") {
        a[1].clone()
    } else {
        let g2p = misaki_rs::G2P::new(misaki_rs::Language::EnglishUS);
        let t1 = Instant::now();
        let (ph, _) = g2p.g2p(&a[1]).unwrap();
        eprintln!("g2p: load {:.0} ms, text {:.1} ms", (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
        println!("{ph}");
        ph
    };
    let mut ids = vec![0];
    ids.extend(ph.chars().filter_map(|c| vocab.get(&c.to_string()).copied()));
    ids.push(0);
    assert!(ids.len() <= 512, "too long for one pass: {} phonemes (max 510)", ids.len() - 2);
    let voice = a.get(3).map(String::as_str).unwrap_or("af_heart");
    let (v, _) = St::open(&dir.join("voices.safetensors")).get(voice);
    let row = ph.chars().count() - 1;
    let style = v[row * 256..(row + 1) * 256].to_vec();
    let noise = std::env::var("NOISE").map_or(true, |v| v != "0");

    set_cache();

    let t0 = Instant::now();
    let w = St::open(&dir.join("kokoro.safetensors"));
    #[cfg(target_os = "macos")]
    let dev = Device::metal(DeviceKind::DefaultDevice);
    #[cfg(not(target_os = "macos"))]
    let dev = Device::vulkan(DeviceKind::DefaultDevice);
    let own = std::env::var("OWN").map_or(true, |v| v != "0");
    let g = Gpu { w: &w, dev, cache: Default::default(), own };
    if std::env::var("BENCH").is_ok() {
        return bench(&g);
    }
    if std::env::var("KTEST").is_ok() {
        return ktest(&g);
    }
    eprintln!("loaded {:.2}s, {} phonemes -> {} ids", t0.elapsed().as_secs_f64(), ph.chars().count(), ids.len());
    let runs: usize = std::env::var("RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut out = None;
    for r in 0..runs {
        let t = Instant::now();
        let o = synth(&g, &ids, &style, noise);
        let s = t.elapsed().as_secs_f64();
        let secs = o.audio.len() as f64 / 24000.0;
        let parts: Vec<String> = o.times.iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
        eprintln!("run {r}: {s:.3} s for {secs:.2} s audio ({:.1}x real time) | ms: {}", secs / s, parts.join(", "));
        out = Some(o);
    }
    let o = out.unwrap();
    if let Ok(d) = std::env::var("DUMP") {
        std::fs::create_dir_all(&d).unwrap();
        for (n, v) in &o.dump {
            std::fs::write(Path::new(&d).join(format!("{n}.f32")), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        }
    }
    let spec = hound::WavSpec { channels: 1, sample_rate: 24000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut wr = hound::WavWriter::create(&a[2], spec).unwrap();
    for s in &o.audio {
        wr.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    wr.finalize().unwrap();
    eprintln!("written {}", a[2]);
}

/// `--listen` (macOS, a try-out of "read selection"): Ctrl+Option+R copies the selection (simulated Cmd+C; the
/// clipboard's text is put back after, other clipboard content is lost), reads it sentence by sentence (the next one is
/// made while the current one plays); pressing again stops. Needs Accessibility for the terminal (for Cmd+C).
#[cfg(target_os = "macos")]
mod listen {
    use super::*;
    use enigo::{Direction, Enigo, Key, Keyboard, Settings};
    use std::path::PathBuf;
    use std::sync::mpsc::{Receiver, channel};
    use std::time::Duration;

    pub fn run(dir: PathBuf, voice: String, vocab: HashMap<String, usize>) -> ! {
        use global_hotkey::hotkey::{Code, HotKey, Modifiers};
        use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
        use tao::event::{Event, StartCause};
        use tao::event_loop::{ControlFlow, EventLoopBuilder};
        use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
        let (tx, rx) = channel::<()>();
        std::thread::spawn(move || reader(&dir, &voice, &vocab, rx));
        GlobalHotKeyEvent::set_event_handler(Some(move |e: GlobalHotKeyEvent| {
            if e.state == HotKeyState::Pressed {
                let _ = tx.send(());
            }
        }));
        let mut el = EventLoopBuilder::new().build();
        el.set_activation_policy(ActivationPolicy::Accessory);
        let mut _manager = None; // kept alive: dropping it unregisters the shortcut
        el.run(move |ev, _, flow| {
            *flow = ControlFlow::Wait;
            if let Event::NewEvents(StartCause::Init) = ev {
                let m = GlobalHotKeyManager::new().unwrap(); // main thread only on macOS
                m.register(HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyR)).unwrap();
                _manager = Some(m);
            }
        })
    }

    fn reader(dir: &Path, voice: &str, vocab: &HashMap<String, usize>, rx: Receiver<()>) {
        let t0 = Instant::now();
        let w = St::open(&dir.join("kokoro.safetensors"));
        let (vs, _) = St::open(&dir.join("voices.safetensors")).get(voice);
        let own = std::env::var("OWN").map_or(true, |v| v != "0");
        let g = Gpu { w: &w, dev: Device::metal(DeviceKind::DefaultDevice), cache: Default::default(), own };
        let g2p = misaki_rs::G2P::new(misaki_rs::Language::EnglishUS);
        let mut kb = Enigo::new(&Settings { open_prompt_to_get_permissions: true, ..Default::default() }).unwrap();
        let mut cb = arboard::Clipboard::new().unwrap();
        eprintln!("loaded {:.1}s. Select text anywhere, press Ctrl+Option+R (again = stop). Ctrl+C here quits.", t0.elapsed().as_secs_f64());
        while rx.recv().is_ok() {
            while rx.try_recv().is_ok() {}
            let t = Instant::now();
            let Some(text) = selection(&mut kb, &mut cb) else {
                eprintln!("no selected text (or the terminal lacks Accessibility permission)");
                continue;
            };
            let copy_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut sink = rodio::DeviceSinkBuilder::open_default_sink()
                .or_else(|_| rodio::DeviceSinkBuilder::from_default_device()?.with_buffer_size(cpal::BufferSize::Default).open_stream())
                .unwrap();
            sink.log_on_drop(false);
            let player = rodio::Player::connect_new(sink.mixer());
            let mut stopped = false;
            let mut first = true;
            'all: for para in text.split('\n').map(str::trim).filter(|p| !p.is_empty()) {
                let Ok((ph, _)) = g2p.g2p(para) else { continue };
                for piece in pieces(&ph) {
                    if rx.try_recv().is_ok() {
                        stopped = true;
                        break 'all;
                    }
                    let mut ids = vec![0];
                    ids.extend(piece.chars().filter_map(|c| vocab.get(&c.to_string()).copied()));
                    ids.push(0);
                    if ids.len() <= 2 {
                        continue;
                    }
                    let row = (piece.chars().count() - 1).min(509);
                    let o = synth(&g, &ids, &vs[row * 256..(row + 1) * 256], true);
                    if first {
                        first = false;
                        eprintln!("copy {copy_ms:.0} ms, first audio after {:.2} s: {}", t.elapsed().as_secs_f64(), piece);
                    }
                    let one = std::num::NonZero::<u16>::MIN;
                    player.append(rodio::buffer::SamplesBuffer::new(one, std::num::NonZero::new(24000).unwrap(), o.audio));
                }
            }
            while !stopped && !player.empty() {
                stopped = rx.recv_timeout(Duration::from_millis(50)).is_ok();
            }
            player.stop();
            eprintln!("{}", if stopped { "stopped" } else { "done" });
        }
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGEventSourceFlagsState(state: i32) -> u64;
    }

    /// The selected text via Cmd+C. Waits until the shortcut's keys are let go (else the app gets Ctrl+Option+Cmd+C).
    fn selection(kb: &mut Enigo, cb: &mut arboard::Clipboard) -> Option<String> {
        let t = Instant::now();
        // shift, control, option, command (kCGEventFlagMask*); state 0 = combined session
        while unsafe { CGEventSourceFlagsState(0) } & 0x1e_0000 != 0 && t.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(10));
        }
        let old = cb.get_text().ok();
        let _ = cb.clear();
        // C as the raw key code (kVK_ANSI_C = 8), like the app's Cmd+V: no layout lookup off the main thread.
        kb.key(Key::Meta, Direction::Press).ok()?;
        kb.key(Key::Other(8), Direction::Press).ok()?;
        std::thread::sleep(Duration::from_millis(20));
        kb.key(Key::Other(8), Direction::Release).ok()?;
        kb.key(Key::Meta, Direction::Release).ok()?;
        let t = Instant::now();
        let mut text = None;
        while text.is_none() && t.elapsed() < Duration::from_millis(500) {
            std::thread::sleep(Duration::from_millis(20));
            text = cb.get_text().ok().filter(|s| !s.trim().is_empty());
        }
        if let Some(o) = old {
            let _ = cb.set_text(o);
        }
        text
    }

    /// Phonemes cut after each sentence end; a piece over Kokoro's 510 is cut at its last comma, else space, before.
    fn pieces(ph: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let cs: Vec<char> = ph.chars().collect();
        for (i, &c) in cs.iter().enumerate() {
            cur.push(c);
            if matches!(c, '.' | '!' | '?' | '…') && cs.get(i + 1).is_none_or(|n| *n == ' ') {
                out.push(std::mem::take(&mut cur));
            }
        }
        out.push(cur);
        let mut res = Vec::new();
        for mut p in out.into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()) {
            while p.chars().count() > 500 {
                let head: String = p.chars().take(500).collect();
                let cut = head.rfind(", ").map(|i| i + 1).or_else(|| head.rfind(' ')).unwrap_or(head.len());
                res.push(p[..cut].trim().to_string());
                p = p[cut..].trim().to_string();
            }
            res.push(p);
        }
        res
    }
}
