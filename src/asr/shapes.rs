//! Parakeet encoder sizes and helpers shared by the GPU (encoder.rs), CPU (cpu.rs) and Neural Engine (ane_build.rs)
//! paths. From transcribe.cpp (MIT, (c) 2026 The transcribe.cpp authors).

pub const D: usize = 1024;
pub const H: usize = 8;
pub const DK: usize = D / H;
pub const LAYERS: usize = 24;

/// The subsampler runs this many mel frames (5 s) at a time, everywhere (pieces and the pause scan). Same result as
/// all at once (each output frame sees only nearby mel frames); its largest buffer, the first conv's f32 output
/// (256 ch x 64 freq per 20 ms), is ~18 MB instead of ~100 MB for a 30 s piece. Measured 2026-11-20: peak VRAM over
/// 8 takes 1092 -> 951 MiB, texts identical, same speed on NVIDIA and the AMD iGPU.
pub const PRE_CHUNK: usize = 500;

/// Output length of a k=3, s=2, p=1 conv.
pub fn stride2(l: usize) -> usize {
    (l - 1) / 2 + 1
}

/// Kestrel valid lengths: [0] = n_samples / hop (one less than the mel frame count: NeMo's centre-padding frame),
/// [1..3] after each stride-2 conv.
pub fn valid_lengths(n_samples: usize, hop: usize) -> [usize; 4] {
    let l0 = n_samples / hop;
    let l1 = stride2(l0);
    let l2 = stride2(l1);
    [l0, l1, l2, stride2(l2)]
}

/// NeMo RelPositionalEncoding: positions T-1 .. -(T-1), sin on even, cos on odd features: [2T-1][D].
pub fn pos_emb_vec(t: usize) -> Vec<f32> {
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
    v
}
