//! TDT greedy decoder (prediction network + joint), CPU: int8 (Q8_0) weights, f32 activations.
//! Ported from transcribe.cpp `src/arch/parakeet/decoder.cpp` `decode_tdt_greedy` (MIT, (c) 2026 The transcribe.cpp
//! authors), with the "kestrel" global symbol budget (`stt.parakeet.tdt.symbol_budget = global`).
use crate::weights::Weights;

pub struct Decoder {
    hidden: usize,
    vocab: usize, // token classes incl. blank (8193)
    blank: usize,
    durations: Vec<usize>,
    max_symbols: usize,
    global_budget: bool,
    embed: Vec<f32>, // [vocab][hidden]
    lstm: Vec<Lstm>,
    pub joint_h: usize,
    /// Joint encoder projection [joint_h][d_enc] + bias (applied by the caller, on the GPU, see `enc_proj`).
    pub enc_w: (Vec<f32>, Vec<usize>),
    pub enc_b: Vec<f32>,
    pred_w: Q8, // [joint_h][hidden]
    pred_b: Vec<f32>,
    out_w: Q8, // [vocab + n_dur][joint_h]
    out_b: Vec<f32>,
}

struct Lstm {
    wx: Q8, // [4H][H], gates i, f, g, o
    wh: Q8,
    b: Vec<f32>,
}

/// Q8_0 matrix kept as in the GGUF: int8 values + one scale per block of 32 along each row (4x smaller than f32, so
/// the decoder's ~13 MB of weights stay in the CPU cache).
pub struct Q8 {
    q: Vec<i8>,
    d: Vec<f32>,
    cols: usize,
}

impl Q8 {
    fn load(g: &Weights, name: &str) -> Q8 {
        let (q, d, shape) = g.q8_0(name).unwrap_or_else(|| panic!("{name}: expected Q8_0"));
        Q8 { q, d, cols: shape[1] }
    }
    /// y = W x (+ b).
    fn matvec(&self, x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports the features enabled on q8_avx2.
            return unsafe { q8_avx2(self, x, b, y) };
        }
        q8_body(self, x, b, y)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_avx2(w: &Q8, x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    use std::arch::x86_64::*;
    let nb = w.cols / 32;
    assert!(x.len() >= nb * 32 && w.q.len() >= y.len() * w.cols && w.d.len() >= y.len() * nb);
    for (r, out) in y.iter_mut().enumerate() {
        // SAFETY: bounds asserted above; unaligned loads.
        unsafe {
            let q = w.q.as_ptr().add(r * w.cols);
            let mut total = _mm256_setzero_ps();
            for blk in 0..nb {
                let mut acc = _mm256_setzero_ps();
                for j in 0..4 {
                    let o = blk * 32 + j * 8;
                    let qi = _mm_loadl_epi64(q.add(o) as *const __m128i);
                    let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qi));
                    acc = _mm256_fmadd_ps(qf, _mm256_loadu_ps(x.as_ptr().add(o)), acc);
                }
                total = _mm256_fmadd_ps(_mm256_set1_ps(*w.d.get_unchecked(r * nb + blk)), acc, total);
            }
            let h = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
            let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
            *out = _mm_cvtss_f32(h) + b.map_or(0.0, |b| b[r]);
        }
    }
}

#[inline(always)]
fn q8_body(w: &Q8, x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    let nb = w.cols / 32;
    for (r, out) in y.iter_mut().enumerate() {
        let q = &w.q[r * w.cols..(r + 1) * w.cols];
        let d = &w.d[r * nb..(r + 1) * nb];
        let mut s = 0f32;
        for (blk, (qc, xc)) in q.chunks_exact(32).zip(x.chunks_exact(32)).enumerate() {
            let mut acc = [0f32; 8];
            for j in 0..4 {
                for i in 0..8 {
                    acc[i] = (qc[j * 8 + i] as f32).mul_add(xc[j * 8 + i], acc[i]);
                }
            }
            s = d[blk].mul_add(acc.iter().sum::<f32>(), s);
        }
        *out = s + b.map_or(0.0, |b| b[r]);
    }
}

#[derive(Clone)]
struct State {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
}

/// y = W x + b, W row-major [rows][cols]. Runtime-dispatched to AVX2+FMA where available.
fn matvec(w: &[f32], x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports the features enabled on matvec_avx2.
        return unsafe { matvec_avx2(w, x, b, y) };
    }
    matvec_body(w, x, b, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_avx2(w: &[f32], x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    matvec_body(w, x, b, y)
}

#[inline(always)]
fn matvec_body(w: &[f32], x: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    let cols = x.len();
    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = [0f32; 16];
        let (rc, rt) = (row.chunks_exact(16), row.chunks_exact(16).remainder());
        for (wc, xc) in rc.zip(x.chunks_exact(16)) {
            for i in 0..16 {
                acc[i] = wc[i].mul_add(xc[i], acc[i]);
            }
        }
        let mut s: f32 = acc.iter().sum();
        let off = cols - rt.len();
        for (i, wv) in rt.iter().enumerate() {
            s += wv * x[off + i];
        }
        *out = s + b.map_or(0.0, |b| b[r]);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Decoder {
    pub fn load(g: &Weights) -> Decoder {
        let w = |n: &str| g.f32(n).0;
        let hidden = g.int("stt.parakeet.predictor.hidden") as usize;
        let layers = g.int("stt.parakeet.predictor.n_layers") as usize;
        let lstm = (0..layers)
            .map(|l| Lstm { wx: Q8::load(g, &format!("pred.lstm.{l}.Wx")), wh: Q8::load(g, &format!("pred.lstm.{l}.Wh")), b: w(&format!("pred.lstm.{l}.bias")) })
            .collect();
        Decoder {
            hidden,
            vocab: g.int("stt.parakeet.predictor.vocab") as usize,
            blank: g.int("tokenizer.ggml.blank_token_id") as usize,
            durations: g.ints("stt.parakeet.tdt.durations").into_iter().map(|d| d as usize).collect(),
            max_symbols: g.int("stt.parakeet.tdt.max_symbols") as usize,
            global_budget: matches!(g.meta.get("stt.parakeet.tdt.symbol_budget"), Some(crate::gguf::Value::Str(s)) if s == "global"),
            embed: w("pred.embed.weight"),
            lstm,
            joint_h: g.int("stt.parakeet.joint.hidden") as usize,
            enc_w: g.f32("joint.enc.weight"),
            enc_b: w("joint.enc.bias"),
            pred_w: Q8::load(g, "joint.pred.weight"),
            pred_b: w("joint.pred.bias"),
            out_w: Q8::load(g, "joint.out.weight"),
            out_b: w("joint.out.bias"),
        }
    }

    /// One LSTM step from `prev` with input token `tok` (None = start: zero embedding). Writes `next`.
    fn predict(&self, tok: Option<usize>, prev: &State, next: &mut State, gates: &mut Vec<f32>, tmp: &mut Vec<f32>) {
        let h = self.hidden;
        let mut x: Vec<f32> = match tok {
            Some(t) => self.embed[t * h..(t + 1) * h].to_vec(),
            None => vec![0.0; h],
        };
        gates.resize(4 * h, 0.0);
        tmp.resize(4 * h, 0.0);
        for (l, w) in self.lstm.iter().enumerate() {
            w.wx.matvec(&x, Some(&w.b), gates);
            w.wh.matvec(&prev.h[l], None, tmp);
            for k in 0..h {
                let i = sigmoid(gates[k] + tmp[k]);
                let f = sigmoid(gates[h + k] + tmp[h + k]);
                let gg = (gates[2 * h + k] + tmp[2 * h + k]).tanh();
                let o = sigmoid(gates[3 * h + k] + tmp[3 * h + k]);
                let c = f * prev.c[l][k] + i * gg;
                next.c[l][k] = c;
                next.h[l][k] = o * c.tanh();
            }
            x.copy_from_slice(&next.h[l]);
        }
    }

    /// Greedy TDT over `enc_proj` = [T][joint_h] (encoder frames already through joint.enc). Returns token ids.
    pub fn decode(&self, enc_proj: &[f32], t_enc: usize) -> Vec<usize> {
        self.decode_counted(enc_proj, t_enc).0
    }

    /// Also returns (iterations, predictor steps).
    pub fn decode_counted(&self, enc_proj: &[f32], t_enc: usize) -> (Vec<usize>, usize, usize) {
        let (h, jh) = (self.hidden, self.joint_h);
        let n_out = self.out_b.len();
        let zero = State { h: vec![vec![0.0; h]; self.lstm.len()], c: vec![vec![0.0; h]; self.lstm.len()] };
        let (mut state, mut next) = (zero.clone(), zero);
        let (mut gates, mut tmp) = (Vec::new(), Vec::new());
        let mut pred_proj = vec![0f32; jh];
        let mut act = vec![0f32; jh];
        let mut logits = vec![0f32; n_out];
        let mut out = Vec::new();
        let mut last: Option<usize> = None;
        let mut dirty = true;
        let (mut step, mut new_symbols) = (0usize, 0usize);
        let mut budget = self.max_symbols * t_enc;
        let max_iters = 16 * t_enc + 1024;
        let mut iter = 0;
        let mut preds = 0;
        while step < t_enc && iter < max_iters && (!self.global_budget || budget > 0) {
            iter += 1;
            if dirty {
                self.predict(last, &state, &mut next, &mut gates, &mut tmp);
                preds += 1;
                self.pred_w.matvec(&next.h[self.lstm.len() - 1], Some(&self.pred_b), &mut pred_proj);
                dirty = false;
            }
            let e = &enc_proj[step * jh..(step + 1) * jh];
            for k in 0..jh {
                act[k] = (e[k] + pred_proj[k]).max(0.0); // relu
            }
            self.out_w.matvec(&act, Some(&self.out_b), &mut logits);
            let argmax = |s: &[f32]| s.iter().enumerate().fold(0, |b, (i, &v)| if v > s[b] { i } else { b });
            let tok = argmax(&logits[..self.vocab]);
            let duration = self.durations[argmax(&logits[self.vocab..])];
            let blank = tok == self.blank;
            if !blank {
                out.push(tok);
                last = Some(tok);
                std::mem::swap(&mut state, &mut next);
                dirty = true;
            }
            if self.global_budget {
                step += if blank && duration == 0 { 1 } else { duration };
                budget -= 1;
                continue;
            }
            step += duration;
            new_symbols += 1;
            if duration != 0 {
                new_symbols = 0;
            } else if self.max_symbols > 0 && new_symbols >= self.max_symbols {
                step += 1;
                new_symbols = 0;
            }
        }
        (out, iter, preds)
    }

    /// First decode step's values (for comparing against transcribe.cpp's dec.* dumps):
    /// (lstm h/c per layer, log-softmaxed joint logits) at the start state and encoder frame 0.
    pub fn first_step(&self, enc_proj: &[f32]) -> (Vec<(Vec<f32>, Vec<f32>)>, Vec<f32>) {
        let h = self.hidden;
        let zero = State { h: vec![vec![0.0; h]; self.lstm.len()], c: vec![vec![0.0; h]; self.lstm.len()] };
        let mut next = zero.clone();
        self.predict(None, &zero, &mut next, &mut Vec::new(), &mut Vec::new());
        let mut pp = vec![0f32; self.joint_h];
        self.pred_w.matvec(&next.h[self.lstm.len() - 1], Some(&self.pred_b), &mut pp);
        let act: Vec<f32> = (0..self.joint_h).map(|k| (enc_proj[k] + pp[k]).max(0.0)).collect();
        let mut logits = vec![0f32; self.out_b.len()];
        self.out_w.matvec(&act, Some(&self.out_b), &mut logits);
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let lse = (logits.iter().map(|&v| ((v - mx) as f64).exp()).sum::<f64>()).ln() as f32 + mx;
        logits.iter_mut().for_each(|v| *v -= lse);
        (next.h.into_iter().zip(next.c).collect(), logits)
    }
}

/// CPU joint encoder projection (reference / fallback): [T][d] -> [T][joint_h].
pub fn enc_proj_cpu(d: &Decoder, enc: &[f32], t: usize) -> Vec<f32> {
    let din = enc.len() / t;
    let mut out = vec![0f32; t * d.joint_h];
    for i in 0..t {
        matvec(&d.enc_w.0, &enc[i * din..(i + 1) * din], Some(&d.enc_b), &mut out[i * d.joint_h..(i + 1) * d.joint_h]);
    }
    out
}
