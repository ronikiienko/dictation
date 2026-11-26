//! The encoder (subsampler, conformer blocks, pause head, joint's encoder projection) on the CPU, same maths as
//! encoder.rs on the GPU. Linears run as int8 x int8 dot products: weights as stored (Q8_0; f16 ones quantized at
//! load), activations quantized per row and block at each call (as ggml's CPU backend does for Q8_0). Blocks are
//! 32 values, or 128 when a matrix's scales come in groups of 128 (Parakeet Redux: ternary weights, one scale per
//! 128), so fewer float fix-ups. ARM: `sdot` (NEON dot product, checked at run time); elsewhere (x86 too, for now) a plain loop.
//! Attention and the subsampler's convs run in f32 (matrixmultiply). Threads: rayon.
use super::shapes::{pos_emb_vec, stride2, valid_lengths, D, DK, H, LAYERS, PRE_CHUNK};
use super::weights::Weights;
use rayon::prelude::*;

const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;

/// Output pointer shared by threads that write disjoint parts.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Int8 linear y = x W^T (+ b), W [n][k]. Packed for the kernel: per 8 outputs, per block, per 16 inputs, per 4
/// inputs: 8 outputs x 4 bytes (the layout `sdot` by lane wants); scales per 8 outputs, per block.
pub struct QLin {
    n: usize,
    k: usize,
    kb: usize,
    w: Packed,
    d: Vec<f32>,
    bias: Option<Vec<f32>>,
}

/// Weights in the kernel's order: int8, or (all codes -1/0/+1, Redux) 2 bits each: per 64 int8 in kernel order,
/// 16 bytes, byte i holding values i, 16+i, 32+i, 48+i (code + 1, two bits each, lowest first).
enum Packed {
    I8(Vec<i8>),
    T2(Vec<u8>),
}

impl Packed {
    /// One block of 8 outputs (8 x k values) as int8, unpacked into `buf` if needed.
    fn panel<'a>(&'a self, nt: usize, k: usize, buf: &'a mut Vec<i8>) -> &'a [i8] {
        match self {
            Packed::I8(w) => &w[nt * 8 * k..(nt + 1) * 8 * k],
            Packed::T2(p) => {
                buf.resize(8 * k, 0);
                unpack2(&p[nt * 2 * k..(nt + 1) * 2 * k], buf);
                buf
            }
        }
    }
}

fn unpack2(p: &[u8], out: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: 16 bytes in -> 64 out per step, lengths p.len() * 4 == out.len() (asserted).
    unsafe {
        use std::arch::aarch64::*;
        assert_eq!(p.len() * 4, out.len());
        let (three, one) = (vdupq_n_u8(3), vdupq_n_s8(1));
        for g in 0..p.len() / 16 {
            let b = vld1q_u8(p.as_ptr().add(g * 16));
            let o = out.as_mut_ptr().add(g * 64);
            vst1q_s8(o, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, three)), one));
            vst1q_s8(o.add(16), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(b), three)), one));
            vst1q_s8(o.add(32), vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(b), three)), one));
            vst1q_s8(o.add(48), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<6>(b)), one));
        }
    }
    // SSE2 is part of x86-64 itself: no run-time check. Shifting 16-bit lanes leaks the neighbour byte's bits into
    // the top, the mask with 3 drops them.
    #[cfg(target_arch = "x86_64")]
    // SAFETY: 16 bytes in -> 64 out per step, lengths p.len() * 4 == out.len() (asserted).
    unsafe {
        use std::arch::x86_64::*;
        assert_eq!(p.len() * 4, out.len());
        let (three, one) = (_mm_set1_epi8(3), _mm_set1_epi8(1));
        for g in 0..p.len() / 16 {
            let b = _mm_loadu_si128(p.as_ptr().add(g * 16) as *const __m128i);
            let o = out.as_mut_ptr().add(g * 64) as *mut __m128i;
            _mm_storeu_si128(o, _mm_sub_epi8(_mm_and_si128(b, three), one));
            _mm_storeu_si128(o.add(1), _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<2>(b), three), one));
            _mm_storeu_si128(o.add(2), _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(b), three), one));
            _mm_storeu_si128(o.add(3), _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<6>(b), three), one));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (g, b) in p.chunks_exact(16).enumerate() {
        for (i, &v) in b.iter().enumerate() {
            for j in 0..4 {
                out[g * 64 + j * 16 + i] = ((v >> (2 * j)) & 3) as i8 - 1;
            }
        }
    }
}

/// Rows of activations per kernel call and per thread task (columns: `TASK_COLS`).
const MR: usize = 4;
const TASK_ROWS: usize = 64;
const TASK_COLS: usize = 32;

impl QLin {
    fn load(g: &Weights, name: &str, bias: Option<&str>) -> QLin {
        let shape = g.shape(name);
        let (n, k) = (shape[0], shape[1..].iter().product::<usize>());
        let bias = bias.map(|b| g.f32(b).0);
        if let Some((q, d, _)) = g.ternary(name) {
            return QLin::pack(&q, &d, n, k, super::redux::GROUP, bias); // stays 2-bit (pack sees the codes)
        }
        if let Some((q, d, _)) = g.q8_0(name) {
            // Scales repeated over 4 blocks everywhere: one block of 128.
            let nb = k / 32;
            let wide = k % 128 == 0 && (0..n).all(|r| (0..nb).all(|b| d[r * nb + b] == d[r * nb + b / 4 * 4]));
            let kb = if wide { 128 } else { 32 };
            let d: Vec<f32> = d.iter().enumerate().filter(|(i, _)| (i % nb) % (kb / 32) == 0).map(|(_, &x)| x).collect();
            return QLin::pack(&q, &d, n, k, kb, bias);
        }
        let (v, _) = g.f32(name);
        // Blocks of 128 if that loses nothing (Redux: one scale per 128 already), else 32.
        for kb in [128, 32] {
            if k % kb != 0 {
                continue;
            }
            let (mut q, mut d) = quantize(&v, kb);
            // Ternary times a scale (Redux's pointwise convs, stored f16): codes -127/0/127 -> -1/0/1.
            if q.iter().all(|&c| c == 0 || c.abs() == 127) {
                q.iter_mut().for_each(|c| *c /= 127);
                d.iter_mut().for_each(|s| *s *= 127.0);
            }
            let exact = v.iter().enumerate().all(|(i, &x)| (q[i] as f32 * d[i / kb] - x).abs() <= 1e-6 * x.abs().max(1e-3));
            if exact || kb == 32 {
                return QLin::pack(&q, &d, n, k, kb, bias);
            }
        }
        unreachable!("{name}: k = {k} not a multiple of 32")
    }

    fn pack(q: &[i8], d: &[f32], n: usize, k: usize, kb: usize, bias: Option<Vec<f32>>) -> QLin {
        assert!(n % 8 == 0 && k % kb == 0 && kb % 16 == 0, "linear {n}x{k}");
        let nb = k / kb;
        let mut w = Vec::with_capacity(n * k);
        let mut dp = Vec::with_capacity(n * nb);
        for nt in 0..n / 8 {
            for b in 0..nb {
                for s in 0..kb / 16 {
                    for lane in 0..4 {
                        for j in 0..8 {
                            let r = nt * 8 + j;
                            let c = b * kb + s * 16 + lane * 4;
                            w.extend_from_slice(&q[r * k + c..r * k + c + 4]);
                        }
                    }
                }
                dp.extend((0..8).map(|j| d[(nt * 8 + j) * nb + b]));
            }
        }
        let w = if w.iter().all(|c| (-1..=1).contains(c)) {
            Packed::T2(w.chunks_exact(64).flat_map(|c| (0..16).map(move |i| (0..4).fold(0u8, |a, j| a | (((c[j * 16 + i] + 1) as u8) << (2 * j))))).collect())
        } else {
            Packed::I8(w)
        };
        QLin { n, k, kb, w, d: dp, bias }
    }

    /// x [m][k] -> [m][n].
    pub fn fwd(&self, x: &[f32], m: usize) -> Vec<f32> {
        let (n, k, kb) = (self.n, self.k, self.kb);
        let nb = k / kb;
        assert_eq!(x.len(), m * k);
        let mut xq = vec![0i8; m * k];
        let mut xd = vec![0f32; m * nb];
        xq.par_chunks_mut(k).zip(xd.par_chunks_mut(nb)).zip(x.par_chunks(k)).for_each(|((q, d), x)| quantize_row(x, kb, q, d));
        let mut y = vec![0f32; m * n];
        let out = Out(y.as_mut_ptr());
        let (row_tasks, col_tasks) = (m.div_ceil(TASK_ROWS), n.div_ceil(TASK_COLS));
        (0..row_tasks * col_tasks).into_par_iter().for_each(|task| {
            let out = out;
            let (r0, c0) = ((task / col_tasks) * TASK_ROWS, (task % col_tasks) * TASK_COLS);
            let (r1, c1) = ((r0 + TASK_ROWS).min(m), (c0 + TASK_COLS).min(n));
            let mut buf = Vec::new();
            for c in (c0..c1).step_by(8) {
                let nt = c / 8;
                let w = self.w.panel(nt, k, &mut buf);
                let wd = &self.d[nt * 8 * nb..(nt + 1) * 8 * nb];
                for r in (r0..r1).step_by(MR) {
                    let mr = MR.min(r1 - r);
                    let (x, d) = (&xq[r * k..(r + mr) * k], &xd[r * nb..(r + mr) * nb]);
                    // SAFETY: rows r..r+mr, columns c..c+8 of y belong to this task only; kernels stay in bounds.
                    unsafe { tile(mr, x, d, k, kb, w, wd, out.0.add(r * n + c), n) };
                }
            }
        });
        if let Some(b) = &self.bias {
            y.par_chunks_mut(n).for_each(|row| row.iter_mut().zip(b).for_each(|(v, b)| *v += b));
        }
        y
    }
}

/// One row of activations to int8 per block of `kb`: d = amax / 127, q = round(x / d) (half away from zero).
fn quantize_row(x: &[f32], kb: usize, q: &mut [i8], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if kb % 32 == 0 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU has AVX2; slices as in the plain loop below.
        return unsafe { quantize_row_avx2(x, kb, q, d) };
    }
    for (b, (blk, q)) in x.chunks_exact(kb).zip(q.chunks_exact_mut(kb)).enumerate() {
        let s = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        d[b] = s;
        for (o, v) in q.iter_mut().zip(blk) {
            *o = (v * inv).round() as i8;
        }
    }
}

/// As the plain loop of `quantize_row` (same results): without SSE4.1, `round` is a library call per value.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_row_avx2(x: &[f32], kb: usize, q: &mut [i8], d: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(x.len() == q.len() && x.len() == d.len() * kb && kb % 32 == 0);
    // SAFETY (whole body): every load/store is inside x, q (lengths checked above).
    unsafe {
        let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let sign = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        // round(v) = trunc(v + copysign(0.49999997, v)), as LLVM lowers `f32::round`.
        let half = _mm256_set1_ps(0.499_999_97);
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for b in 0..d.len() {
            let (xp, qp) = (x.as_ptr().add(b * kb), q.as_mut_ptr().add(b * kb));
            let mut m = _mm256_setzero_ps();
            for i in (0..kb).step_by(8) {
                m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(xp.add(i)), abs));
            }
            let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
            let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps::<1>(m2, m2)));
            let s = amax / 127.0;
            let inv = _mm256_set1_ps(if s > 0.0 { 1.0 / s } else { 0.0 });
            d[b] = s;
            for i in (0..kb).step_by(32) {
                let r = |j: usize| {
                    let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i + j)), inv);
                    _mm256_cvttps_epi32(_mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, sign))))
                };
                let p = _mm256_packs_epi16(_mm256_packs_epi32(r(0), r(8)), _mm256_packs_epi32(r(16), r(24)));
                _mm256_storeu_si256(qp.add(i) as *mut __m256i, _mm256_permutevar8x32_epi32(p, order));
            }
        }
    }
}

/// Q8_0-style: per block of `kb`, d = amax / 127, q = round(x / d).
fn quantize(v: &[f32], kb: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(v.len());
    let mut d = Vec::with_capacity(v.len() / kb);
    for b in v.chunks_exact(kb) {
        let s = b.iter().fold(0f32, |a, x| a.max(x.abs())) / 127.0;
        d.push(s);
        q.extend(b.iter().map(|x| if s == 0.0 { 0 } else { (x / s).round().clamp(-127.0, 127.0) as i8 }));
    }
    (q, d)
}

/// `mr` (1..=MR) rows x 8 columns of output.
unsafe fn tile(mr: usize, x: &[i8], xd: &[f32], k: usize, kb: usize, w: &[i8], wd: &[f32], y: *mut f32, ldy: usize) {
    // Every Apple M chip has `sdot`; some other ARM CPUs (e.g. Raspberry Pi 4) don't: plain loop there.
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: the CPU has the feature tile_neon is compiled for.
        return unsafe {
            match mr {
                4 => tile_neon::<4>(x, xd, k, kb, w, wd, y, ldy),
                3 => tile_neon::<3>(x, xd, k, kb, w, wd, y, ldy),
                2 => tile_neon::<2>(x, xd, k, kb, w, wd, y, ldy),
                _ => tile_neon::<1>(x, xd, k, kb, w, wd, y, ldy),
            }
        };
    }
    // x86: AVX2 is on nearly every x86-64 CPU since 2013-2015 (Haswell, Zen); older ones take the plain loop.
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU has the feature tile_avx2 is compiled for.
        return unsafe {
            match mr {
                4 => tile_avx2::<4>(x, xd, k, kb, w, wd, y, ldy),
                3 => tile_avx2::<3>(x, xd, k, kb, w, wd, y, ldy),
                2 => tile_avx2::<2>(x, xd, k, kb, w, wd, y, ldy),
                _ => tile_avx2::<1>(x, xd, k, kb, w, wd, y, ldy),
            }
        };
    }
    // SAFETY: as the caller's.
    unsafe { tile_plain(mr, x, xd, k, kb, w, wd, y, ldy) }
}

/// The packed panel holds, per 4 values of k, 8 outputs x 4 bytes = one 256-bit register: each 32-bit lane is one
/// output. x's 4 bytes are broadcast to all lanes; `maddubs` (unsigned x signed bytes, pairs summed to i16) + `madd`
/// with ones (pairs to i32) give the 4-term dot product per lane. maddubs needs one side unsigned: |w| and x with
/// w's sign. No overflow: |x| <= 127 (fwd's quantizing), |w| <= 128, a pair <= 2 * 128 * 127 < 32767.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2<const R: usize>(x: &[i8], xd: &[f32], k: usize, kb: usize, w: &[i8], wd: &[f32], y: *mut f32, ldy: usize) {
    use std::arch::x86_64::*;
    let nb = k / kb;
    debug_assert!(x.len() >= R * k && w.len() >= 8 * k && wd.len() >= 8 * nb && xd.len() >= R * nb);
    // SAFETY (whole body): pointers stay inside x, w, wd, xd (lengths above) and the caller's y tile.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut f = [_mm256_setzero_ps(); R];
        let mut wp = w.as_ptr() as *const __m256i;
        let xp = x.as_ptr();
        for b in 0..nb {
            let mut a = [_mm256_setzero_si256(); R];
            for g in 0..kb / 4 {
                let wv = _mm256_loadu_si256(wp);
                wp = wp.add(1);
                let wabs = _mm256_sign_epi8(wv, wv);
                for m in 0..R {
                    let xv = _mm256_set1_epi32((xp.add(m * k + b * kb + g * 4) as *const i32).read_unaligned());
                    let p = _mm256_maddubs_epi16(wabs, _mm256_sign_epi8(xv, wv));
                    a[m] = _mm256_add_epi32(a[m], _mm256_madd_epi16(p, ones));
                }
            }
            let dw = _mm256_loadu_ps(wd.as_ptr().add(b * 8));
            for m in 0..R {
                let dx = _mm256_set1_ps(*xd.get_unchecked(m * nb + b));
                f[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[m]), _mm256_mul_ps(dw, dx), f[m]);
            }
        }
        for m in 0..R {
            _mm256_storeu_ps(y.add(m * ldy), f[m]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn tile_neon<const R: usize>(x: &[i8], xd: &[f32], k: usize, kb: usize, w: &[i8], wd: &[f32], y: *mut f32, ldy: usize) {
    use std::arch::aarch64::*;
    let nb = k / kb;
    debug_assert!(x.len() >= R * k && w.len() >= 8 * k && wd.len() >= 8 * nb && xd.len() >= R * nb);
    // SAFETY (whole body): pointers stay inside x, w, wd, xd (lengths above) and the caller's y tile.
    unsafe {
        let mut f = [[vdupq_n_f32(0.0); 2]; R];
        let mut wp = w.as_ptr();
        let xp = x.as_ptr();
        for b in 0..nb {
            // Two accumulator sets (even / odd lanes): 4R independent sdot chains, enough to hide its latency.
            let mut a = [[vdupq_n_s32(0); 2]; R];
            let mut a2 = [[vdupq_n_s32(0); 2]; R];
            for s in 0..kb / 16 {
                let mut xv = [vdupq_n_s8(0); R];
                for m in 0..R {
                    xv[m] = vld1q_s8(xp.add(m * k + b * kb + s * 16));
                }
                macro_rules! lane {
                    ($acc:ident, $l:literal) => {
                        let w0 = vld1q_s8(wp);
                        let w1 = vld1q_s8(wp.add(16));
                        wp = wp.add(32);
                        for m in 0..R {
                            $acc[m][0] = vdotq_laneq_s32::<$l>($acc[m][0], w0, xv[m]);
                            $acc[m][1] = vdotq_laneq_s32::<$l>($acc[m][1], w1, xv[m]);
                        }
                    };
                }
                lane!(a, 0);
                lane!(a2, 1);
                lane!(a, 2);
                lane!(a2, 3);
            }
            for m in 0..R {
                a[m][0] = vaddq_s32(a[m][0], a2[m][0]);
                a[m][1] = vaddq_s32(a[m][1], a2[m][1]);
            }
            let d0 = vld1q_f32(wd.as_ptr().add(b * 8));
            let d1 = vld1q_f32(wd.as_ptr().add(b * 8 + 4));
            for m in 0..R {
                let dx = *xd.get_unchecked(m * nb + b);
                f[m][0] = vfmaq_f32(f[m][0], vcvtq_f32_s32(a[m][0]), vmulq_n_f32(d0, dx));
                f[m][1] = vfmaq_f32(f[m][1], vcvtq_f32_s32(a[m][1]), vmulq_n_f32(d1, dx));
            }
        }
        for m in 0..R {
            vst1q_f32(y.add(m * ldy), f[m][0]);
            vst1q_f32(y.add(m * ldy + 4), f[m][1]);
        }
    }
}

unsafe fn tile_plain(mr: usize, x: &[i8], xd: &[f32], k: usize, kb: usize, w: &[i8], wd: &[f32], y: *mut f32, ldy: usize) {
    let nb = k / kb;
    for m in 0..mr {
        for j in 0..8 {
            let mut f = 0f32;
            for b in 0..nb {
                let mut acc = 0i32;
                for s in 0..kb / 16 {
                    for lane in 0..4 {
                        let wo = ((b * (kb / 16) + s) * 4 + lane) * 32 + j * 4;
                        let xo = m * k + b * kb + s * 16 + lane * 4;
                        for e in 0..4 {
                            acc += w[wo + e] as i32 * x[xo + e] as i32;
                        }
                    }
                }
                f += acc as f32 * wd[b * 8 + j] * xd[m * nb + b];
            }
            // SAFETY: inside the caller's tile.
            unsafe { *y.add(m * ldy + j) = f };
        }
    }
}

/// c [m][n] = a [m][k] b [k][n] (+ c if `acc`), any strides (row, column), one thread.
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm(m: usize, k: usize, n: usize, a: *const f32, rsa: usize, csa: usize, b: *const f32, rsb: usize, csb: usize, c: *mut f32, rsc: usize, acc: bool) {
    // SAFETY: the caller's pointers and strides describe valid matrices.
    unsafe {
        matrixmultiply::sgemm(m, k, n, 1.0, a, rsa as isize, csa as isize, b, rsb as isize, csb as isize, if acc { 1.0 } else { 0.0 }, c, rsc as isize, 1)
    }
}

fn silu(v: &mut [f32]) {
    v.par_chunks_mut(4096).for_each(|c| c.iter_mut().for_each(|x| *x /= 1.0 + (-*x).exp()));
}

struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}
impl Ln {
    fn load(g: &Weights, p: &str) -> Ln {
        Ln { w: g.f32(&format!("{p}.weight")).0, b: g.f32(&format!("{p}.bias")).0 }
    }
    fn fwd(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        y.par_chunks_mut(D).for_each(|r| {
            let mean = r.iter().sum::<f32>() / D as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for ((v, w), b) in r.iter_mut().zip(&self.w).zip(&self.b) {
                *v = (*v - mean) * inv * w + b;
            }
        });
        y
    }
}

struct Block {
    n_ff1: Ln,
    ff1: (QLin, QLin),
    n_att: Ln,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    pos: QLin,
    bias_u: Vec<f32>, // [H][DK]
    bias_v: Vec<f32>,
    n_conv: Ln,
    pw1: QLin,
    dw: Vec<f32>, // [D][9]
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    pw2: QLin,
    n_ff2: Ln,
    ff2: (QLin, QLin),
    n_out: Ln,
}

impl Block {
    fn load(g: &Weights, i: usize) -> Block {
        let p = format!("enc.blocks.{i}");
        let lin = |n: &str| QLin::load(g, &format!("{p}.{n}.weight"), None);
        let f = |n: &str| g.f32(&format!("{p}.{n}")).0;
        let (w, b, m, var) = (f("conv.bn.weight"), f("conv.bn.bias"), f("conv.bn.running_mean"), f("conv.bn.running_var"));
        let bn_scale: Vec<f32> = (0..D).map(|c| w[c] / (var[c] + BN_EPS).sqrt()).collect();
        let bn_shift = (0..D).map(|c| b[c] - m[c] * bn_scale[c]).collect();
        let ln = |n: &str| Ln::load(g, &format!("{p}.{n}"));
        Block {
            n_ff1: ln("norm_ff1"),
            ff1: (lin("ff1.linear1"), lin("ff1.linear2")),
            n_att: ln("norm_attn"),
            q: lin("attn.linear_q"),
            k: lin("attn.linear_k"),
            v: lin("attn.linear_v"),
            o: lin("attn.linear_out"),
            pos: lin("attn.linear_pos"),
            bias_u: f("attn.pos_bias_u"),
            bias_v: f("attn.pos_bias_v"),
            n_conv: ln("norm_conv"),
            pw1: lin("conv.pointwise1"),
            dw: f("conv.depthwise.weight"),
            bn_scale,
            bn_shift,
            pw2: lin("conv.pointwise2"),
            n_ff2: ln("norm_ff2"),
            ff2: (lin("ff2.linear1"), lin("ff2.linear2")),
            n_out: ln("norm_out"),
        }
    }

    fn ff(x: &[f32], t: usize, w: &(QLin, QLin)) -> Vec<f32> {
        let mut h = w.0.fwd(x, t);
        silu(&mut h);
        w.1.fwd(&h, t)
    }

    /// Relative-position self-attention, as encoder.rs `Block::attn`. x [t][D], pos [2t-1][D] (already embedded).
    fn attn(&self, x: &[f32], t: usize, pos: &[f32]) -> Vec<f32> {
        let pl = 2 * t - 1;
        let (q, k, v) = (self.q.fwd(x, t), self.k.fwd(x, t), self.v.fwd(x, t));
        let p = self.pos.fwd(pos, pl);
        let mut out = vec![0f32; t * D];
        let o = Out(out.as_mut_ptr());
        let scale = 1.0 / (DK as f32).sqrt();
        // Tasks: head x block of query rows (many small tasks, so the slower E-cores don't hold up the end).
        const ROWS: usize = 64;
        let blocks = t.div_ceil(ROWS);
        (0..H * blocks).into_par_iter().for_each(|task| {
            let o = o;
            let (h, i0) = (task / blocks, (task % blocks) * ROWS);
            let (i1, col) = ((i0 + ROWS).min(t), h * DK);
            let rows = i1 - i0;
            let with = |bias: &[f32]| -> Vec<f32> {
                (i0..i1).flat_map(|i| (0..DK).map(move |d| (i, d))).map(|(i, d)| q[i * D + col + d] + bias[col + d]).collect()
            };
            let (qu, qv) = (with(&self.bias_u), with(&self.bias_v));
            // Query i reads relative position i - j for key j, i.e. pos row (t-1) - i + j: rows t-i1 .. 2t-1-i0.
            let (p0, pw) = (t - i1, rows + t - 1);
            let mut ac = vec![0f32; rows * t];
            let mut bd = vec![0f32; rows * pw];
            // SAFETY: shapes as given; k, p, v read as [*, D] with this head's columns; out's block is this task's.
            unsafe {
                sgemm(rows, DK, t, qu.as_ptr(), DK, 1, k.as_ptr().add(col), 1, D, ac.as_mut_ptr(), t, false);
                sgemm(rows, DK, pw, qv.as_ptr(), DK, 1, p.as_ptr().add(p0 * D + col), 1, D, bd.as_mut_ptr(), pw, false);
            }
            for r in 0..rows {
                let i = i0 + r;
                let row = &mut ac[r * t..(r + 1) * t];
                let b = &bd[r * pw + (t - 1 - i - p0)..];
                let mut mx = f32::MIN;
                for j in 0..t {
                    row[j] = (row[j] + b[j]) * scale;
                    mx = mx.max(row[j]);
                }
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                row.iter_mut().for_each(|v| *v /= sum);
            }
            unsafe { sgemm(rows, t, DK, ac.as_ptr(), t, 1, v.as_ptr().add(col), D, 1, o.0.add(i0 * D + col), D, false) };
        });
        self.o.fwd(&out, t)
    }

    fn conv(&self, x: &[f32], t: usize) -> Vec<f32> {
        let y = self.pw1.fwd(x, t); // [t][2D]
        let mut g = vec![0f32; t * D];
        g.par_chunks_mut(D).zip(y.par_chunks(2 * D)).for_each(|(g, y)| {
            for c in 0..D {
                g[c] = y[c] / (1.0 + (-y[D + c]).exp());
            }
        });
        let mut z = vec![0f32; t * D];
        z.par_chunks_mut(D).enumerate().for_each(|(i, z)| {
            for (j, row) in (i as isize - 4..=i as isize + 4).enumerate() {
                if row < 0 || row >= t as isize {
                    continue;
                }
                let gr = &g[row as usize * D..(row as usize + 1) * D];
                for c in 0..D {
                    z[c] += self.dw[c * 9 + j] * gr[c];
                }
            }
            for c in 0..D {
                let v = z[c] * self.bn_scale[c] + self.bn_shift[c];
                z[c] = v / (1.0 + (-v).exp());
            }
        });
        self.pw2.fwd(&z, t)
    }

    fn fwd(&self, x: &mut [f32], t: usize, pos: &[f32]) {
        let add = |x: &mut [f32], y: Vec<f32>, s: f32| x.par_chunks_mut(D).zip(y.par_chunks(D)).for_each(|(x, y)| x.iter_mut().zip(y).for_each(|(a, b)| *a += s * b));
        add(x, Self::ff(&self.n_ff1.fwd(x), t, &self.ff1), 0.5);
        add(x, self.attn(&self.n_att.fwd(x), t, pos), 1.0);
        add(x, self.conv(&self.n_conv.fwd(x), t), 1.0);
        add(x, Self::ff(&self.n_ff2.fwd(x), t, &self.ff2), 0.5);
        let y = self.n_out.fwd(x);
        x.copy_from_slice(&y);
    }
}

/// Subsampler: activations [C][time][freq].
struct Pre {
    c: [(Vec<f32>, Vec<f32>); 5], // convs 0, 2, 3, 5, 6: weights and biases
    out: QLin,
}

const C: usize = 256;

impl Pre {
    fn load(g: &Weights) -> Pre {
        let c = |i: usize| (g.f32(&format!("enc.pre_encode.conv.{i}.weight")).0, g.f32(&format!("enc.pre_encode.conv.{i}.bias")).0);
        Pre { c: [c(0), c(2), c(3), c(5), c(6)], out: QLin::load(g, "enc.pre_encode.out.weight", Some("enc.pre_encode.out.bias")) }
    }

    /// k=3 s=2 p=1 conv over (time, freq): `groups` = C (depthwise, input [C][t][f]) or 1 (input [t][f], one
    /// channel). Output [C][t2][f2].
    fn conv_s2(x: &[f32], t: usize, f: usize, w: &[f32], b: &[f32], depthwise: bool, relu: bool) -> (Vec<f32>, usize, usize) {
        let (t2, f2) = (stride2(t), stride2(f));
        // Input pixel of output (i, j) for tap (di, dj), or None in the zero padding.
        let at = |i: usize, j: usize, di: usize, dj: usize| {
            let (ti, fj) = ((2 * i + di) as isize - 1, (2 * j + dj) as isize - 1);
            (ti >= 0 && ti < t as isize && fj >= 0 && fj < f as isize).then(|| ti as usize * f + fj as usize)
        };
        let mut y = vec![0f32; C * t2 * f2];
        if !depthwise {
            // One input channel: the 9 taps of every output pixel as a [9][t2*f2] matrix, then W [C][9] times it.
            let n = t2 * f2;
            let mut cols = vec![0f32; 9 * n];
            cols.par_chunks_mut(n).enumerate().for_each(|(tap, row)| {
                for i in 0..t2 {
                    for j in 0..f2 {
                        row[i * f2 + j] = at(i, j, tap / 3, tap % 3).map_or(0.0, |p| x[p]);
                    }
                }
            });
            let o = Out(y.as_mut_ptr());
            let step = 1024;
            (0..n.div_ceil(step)).into_par_iter().for_each(|s| {
                let o = o;
                let (a, e) = (s * step, (s * step + step).min(n));
                // SAFETY: columns a..e of y are this task's.
                unsafe { sgemm(C, 9, e - a, w.as_ptr(), 9, 1, cols.as_ptr().add(a), n, 1, o.0.add(a), n, false) };
            });
            y.par_chunks_mut(n).enumerate().for_each(|(c, r)| r.iter_mut().for_each(|v| *v = if relu { (*v + b[c]).max(0.0) } else { *v + b[c] }));
            return (y, t2, f2);
        }
        y.par_chunks_mut(t2 * f2).enumerate().for_each(|(c, y)| {
            let src = &x[c * t * f..(c + 1) * t * f];
            let w = &w[c * 9..c * 9 + 9];
            for i in 0..t2 {
                for j in 0..f2 {
                    let mut s = b[c];
                    for tap in 0..9 {
                        if let Some(p) = at(i, j, tap / 3, tap % 3) {
                            s += w[tap] * src[p];
                        }
                    }
                    y[i * f2 + j] = if relu { s.max(0.0) } else { s };
                }
            }
        });
        (y, t2, f2)
    }

    /// Pointwise conv C -> C with relu, x [C][n].
    fn pointwise(x: &[f32], n: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; C * n];
        let o = Out(y.as_mut_ptr());
        let step = 256;
        (0..n.div_ceil(step)).into_par_iter().for_each(|s| {
            let o = o;
            let (a, e) = (s * step, (s * step + step).min(n));
            // SAFETY: columns a..e of y are this task's.
            unsafe { sgemm(C, C, e - a, w.as_ptr(), C, 1, x.as_ptr().add(a), n, 1, o.0.add(a), n, false) };
        });
        y.par_chunks_mut(n).enumerate().for_each(|(c, r)| r.iter_mut().for_each(|v| *v = (*v + b[c]).max(0.0)));
        y
    }

    /// Zero time rows >= valid of [C][t][f].
    fn mask(x: &mut [f32], t: usize, f: usize, valid: usize) {
        if valid < t {
            x.par_chunks_mut(t * f).for_each(|c| c[valid * f..].iter_mut().for_each(|v| *v = 0.0));
        }
    }

    /// mel [t][128] starting at global frame `off` -> [stride2^3(t)][D], as encoder.rs `PreEncode::fwd_at`.
    fn fwd_at(&self, mel: &[f32], t: usize, len: [usize; 4], off: usize) -> (Vec<f32>, usize) {
        let f = mel.len() / t;
        let len = [len[0], len[1].saturating_sub(off / 2), len[2].saturating_sub(off / 4), len[3].saturating_sub(off / 8)];
        let [c0, c2, c3, c5, c6] = &self.c;
        let (mut x, t1, f1) = Self::conv_s2(mel, t, f, &c0.0, &c0.1, false, true);
        Self::mask(&mut x, t1, f1, len[1]);
        let (mut x, t2, f2) = Self::conv_s2(&x, t1, f1, &c2.0, &c2.1, true, false);
        Self::mask(&mut x, t2, f2, len[2]);
        let x = Self::pointwise(&x, t2 * f2, &c3.0, &c3.1);
        let (mut x, t3, f3) = Self::conv_s2(&x, t2, f2, &c5.0, &c5.1, true, false);
        Self::mask(&mut x, t3, f3, len[3]);
        let x = Self::pointwise(&x, t3 * f3, &c6.0, &c6.1);
        let mut r = vec![0f32; t3 * C * f3]; // [t3][C*f3], channel-major
        r.par_chunks_mut(C * f3).enumerate().for_each(|(i, row)| {
            for c in 0..C {
                row[c * f3..(c + 1) * f3].copy_from_slice(&x[(c * t3 + i) * f3..(c * t3 + i + 1) * f3]);
            }
        });
        (self.out.fwd(&r, t3), t3)
    }

    /// As encoder.rs `PreEncode::fwd_chunked`: `PRE_CHUNK` mel frames at a time plus a margin, same result.
    fn fwd(&self, mel: &[f32], t: usize, len: [usize; 4]) -> (Vec<f32>, usize) {
        const MARGIN: usize = 32;
        let f = mel.len() / t;
        let n_out = stride2(stride2(stride2(t)));
        let step = PRE_CHUNK / 8;
        if n_out <= step {
            return self.fwd_at(mel, t, len, 0);
        }
        let mut y = Vec::with_capacity(n_out * D);
        let mut j0 = 0;
        while j0 < n_out {
            let j1 = (j0 + step).min(n_out);
            let a = (8 * j0).saturating_sub(MARGIN);
            let b = (8 * j1 + MARGIN).min(t);
            let (part, _) = self.fwd_at(&mel[a * f..b * f], b - a, len, a);
            y.extend_from_slice(&part[(j0 - a / 8) * D..(j1 - a / 8) * D]);
            j0 = j1;
        }
        (y, n_out)
    }
}

/// Pause head, as encoder.rs `Vad`.
struct Vad {
    proj: Vec<f32>, // [128][D]
    proj_b: Vec<f32>,
    ctx: Vec<Vec<f32>>, // per tap: [in][out]
    ctx_b: Vec<f32>,
    out: Vec<f32>, // [128]
    out_b: f32,
}

impl Vad {
    fn load(g: &Weights) -> Vad {
        let (ctx, s) = g.f32("vad.ctx.weight"); // [out][in][k]
        let (ho, hi, k) = (s[0], s[1], s[2]);
        let taps = (0..k).map(|j| (0..hi * ho).map(|x| ctx[(x % ho) * hi * k + (x / ho) * k + j]).collect()).collect();
        Vad {
            proj: g.f32("vad.proj.weight").0,
            proj_b: g.f32("vad.proj.bias").0,
            ctx: taps,
            ctx_b: g.f32("vad.ctx.bias").0,
            out: g.f32("vad.out.weight").0,
            out_b: g.f32("vad.out.bias").0[0],
        }
    }
    fn fwd(&self, x: &[f32], t: usize) -> Vec<f32> {
        let hd = self.proj_b.len();
        let k = self.ctx.len();
        let mut h = vec![0f32; t * hd];
        // SAFETY: shapes as allocated.
        unsafe { sgemm(t, D, hd, x.as_ptr(), D, 1, self.proj.as_ptr(), 1, D, h.as_mut_ptr(), hd, false) };
        h.chunks_mut(hd).for_each(|r| r.iter_mut().zip(&self.proj_b).for_each(|(v, b)| *v = (*v + b) / (1.0 + (-(*v + b)).exp())));
        let mut c: Vec<f32> = (0..t).flat_map(|_| self.ctx_b.iter().copied()).collect();
        for (j, w) in self.ctx.iter().enumerate() {
            // output row i reads input row i + j - k/2
            let shift = j as isize - (k / 2) as isize;
            let (i0, i1) = ((-shift).max(0) as usize, (t as isize - shift.max(0)).max(0) as usize);
            if i1 > i0 {
                let src = (i0 as isize + shift) as usize;
                unsafe { sgemm(i1 - i0, hd, hd, h.as_ptr().add(src * hd), hd, 1, w.as_ptr(), hd, 1, c.as_mut_ptr().add(i0 * hd), hd, true) };
            }
        }
        c.chunks(hd)
            .map(|r| {
                let s: f32 = r.iter().zip(&self.out).map(|(v, w)| v / (1.0 + (-v).exp()) * w).sum::<f32>() + self.out_b;
                1.0 / (1.0 + (-s).exp())
            })
            .collect()
    }
}

pub struct Encoder {
    pre: Pre,
    blocks: Vec<Block>,
    vad: Vad,
    joint: QLin,
}

impl Encoder {
    pub fn load(g: &Weights) -> Encoder {
        Encoder {
            pre: Pre::load(g),
            blocks: (0..LAYERS).map(|i| Block::load(g, i)).collect(),
            vad: Vad::load(g),
            joint: QLin::load(g, "joint.enc.weight", Some("joint.enc.bias")),
        }
    }

    /// Only what runs next to the Neural Engine's encoder (ane.rs): the subsampler + pause head (pause scan of long
    /// takes) and the joint's encoder projection; no conformer blocks (`encode` must not be called).
    #[cfg(target_os = "macos")]
    pub fn load_head(g: &Weights) -> Encoder {
        Encoder { pre: Pre::load(g), blocks: Vec::new(), vad: Vad::load(g), joint: QLin::load(g, "joint.enc.weight", Some("joint.enc.bias")) }
    }

    /// Encoder output [t][1024] -> joint's encoder projection [t][joint_h].
    #[cfg(target_os = "macos")]
    pub fn project(&self, x: &[f32], t: usize) -> Vec<f32> {
        self.joint.fwd(x, t)
    }

    /// mel [frames][128] of `n_samples` audio -> joint's encoder projection [T_valid][joint_h], and T_valid.
    pub fn encode(&self, mel: &[f32], frames: usize, n_samples: usize) -> (Vec<f32>, usize) {
        let len = valid_lengths(n_samples, 160);
        let (mut x, n) = self.pre.fwd(mel, frames, len);
        let t = n.min(len[3]);
        x.truncate(t * D);
        let pos = pos_emb_vec(t);
        for b in &self.blocks {
            b.fwd(&mut x, t, &pos);
        }
        (self.joint.fwd(&x, t), t)
    }

    /// Speech probability per encoder frame (subsampler + pause head), valid frames only.
    pub fn speech_probs(&self, mel: &[f32], frames: usize, n_samples: usize, hop: usize) -> Vec<f32> {
        let len = valid_lengths(n_samples, hop);
        let (x, n) = self.pre.fwd(mel, frames, len);
        let mut p = self.vad.fwd(&x, n);
        p.truncate(len[3]);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QLin (packing, int8 and 2-bit storage, every row count of the last tile) against a plain f32 product of the
    /// same quantized values; also the SIMD kernel in use (NEON / AVX2) against the plain loop.
    #[test]
    fn linear_matches_reference() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for (n, k, kb, ternary) in [(16, 256, 32, false), (24, 384, 128, true), (8, 128, 128, false)] {
            let q: Vec<i8> = (0..n * k).map(|_| if ternary { (rnd() * 2.99).round() as i8 } else { (rnd() * 254.0) as i8 }).collect();
            let d: Vec<f32> = (0..n * k / kb).map(|_| rnd().abs() + 0.01).collect();
            let l = QLin::pack(&q, &d, n, k, kb, None);
            assert_eq!(matches!(l.w, Packed::T2(_)), ternary);
            for m in 1..=7 {
                let x: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
                let y = l.fwd(&x, m);
                let (xq, xd) = quantize(&x, kb); // as fwd quantizes
                for r in 0..m {
                    for c in 0..n {
                        let want: f32 = (0..k).map(|i| xq[r * k + i] as f32 * xd[(r * k + i) / kb] * q[c * k + i] as f32 * d[(c * k + i) / kb]).sum();
                        assert!((y[r * n + c] - want).abs() <= 1e-3 * want.abs().max(1.0), "{n}x{k} m{m} [{r},{c}]: {} vs {want}", y[r * n + c]);
                    }
                }
                // The plain loop (CPUs without the SIMD kernel) gives the same as whatever `tile` picked here.
                let nb = k / kb;
                let mut buf = Vec::new();
                let w = l.w.panel(0, k, &mut buf);
                let mr = m.min(MR);
                let mut yp = vec![0f32; mr * 8];
                unsafe { tile_plain(mr, &xq[..mr * k], &xd[..mr * nb], k, kb, w, &l.d[..8 * nb], yp.as_mut_ptr(), 8) };
                for r in 0..mr {
                    for c in 0..8 {
                        assert!((yp[r * 8 + c] - y[r * n + c]).abs() <= 1e-3 * y[r * n + c].abs().max(1.0));
                    }
                }
            }
        }
    }

    /// The AVX2 activation quantizing gives the same bytes and scales as the plain loop, incl. halves (ties).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn quantize_avx2_matches_plain() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 11u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for kb in [32, 128] {
            let mut x: Vec<f32> = (0..kb * 64).map(|_| rnd() * 10.0).collect();
            // Blocks whose scale is exactly 1 (amax 127): values k + 0.5 are exact ties; and an all-zero block.
            for i in 0..kb {
                x[i] = if i == 0 { 127.0 } else { (i as f32 - kb as f32 / 2.0) + 0.5 };
            }
            x[kb..2 * kb].fill(0.0);
            let (mut q1, mut d1) = (vec![0i8; x.len()], vec![0f32; x.len() / kb]);
            let (mut q2, mut d2) = (q1.clone(), d1.clone());
            quantize_row(&x, kb, &mut q1, &mut d1);
            for (b, (blk, q)) in x.chunks_exact(kb).zip(q2.chunks_exact_mut(kb)).enumerate() {
                let s = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                d2[b] = s;
                q.iter_mut().zip(blk).for_each(|(o, v)| *o = (v * inv).round() as i8);
            }
            assert_eq!(d1, d2);
            assert_eq!(q1, q2);
        }
    }
}
