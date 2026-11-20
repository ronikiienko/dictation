//! NeMo log-mel frontend (AudioToMelSpectrogramPreprocessor), CPU.
//! Ported from transcribe.cpp `src/transcribe-mel.cpp` (MIT, (c) 2026 The transcribe.cpp authors), Parakeet path:
//! pre-emphasis, zero pad n_fft/2 (pad_mode "constant", as NeMo's FilterbankFeatures), symmetric Hann (win_length,
//! centered in n_fft), |FFT|^2 in f64, Slaney mel filterbank, ln(x + 2^-24), per-feature normalize over the first
//! n/hop frames (unbiased std + 1e-5), the last frame (centre-padding artifact) set to 0. Dither never applied.
use realfft::RealFftPlanner;
use std::sync::Arc;

const LOG_EPS: f64 = 5.960_464_477_539_062_5e-8; // 2^-24
const NORM_EPS: f64 = 1e-5;

pub struct Mel {
    pub n_mels: usize,
    n_fft: usize,
    hop: usize,
    pre_emphasis: f64,
    window: Vec<f64>,
    /// [n_mels][n_freq] plus the nonzero range of each row.
    fb: Vec<Vec<f32>>,
    fb_range: Vec<(usize, usize)>,
    fft: Arc<dyn realfft::RealToComplex<f64>>,
}

fn hz_to_mel(hz: f64) -> f64 {
    let (fsp, min_log_hz) = (200.0 / 3.0, 1000.0);
    if hz < min_log_hz { hz / fsp } else { min_log_hz / fsp + (hz / min_log_hz).ln() / ((6.4f64).ln() / 27.0) }
}
fn mel_to_hz(mel: f64) -> f64 {
    let (fsp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / fsp;
    if mel < min_log_mel { mel * fsp } else { min_log_hz * ((6.4f64).ln() / 27.0 * (mel - min_log_mel)).exp() }
}

impl Mel {
    pub fn new(sr: usize, n_fft: usize, win: usize, hop: usize, n_mels: usize, fmin: f64, fmax: f64, pre_emphasis: f64) -> Mel {
        // Symmetric Hann (torch periodic=False) of length win, zero-padded to n_fft on both sides.
        let mut window = vec![0f64; n_fft];
        let pad = (n_fft - win) / 2;
        for k in 0..win {
            window[pad + k] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (win - 1) as f64).cos();
        }
        // librosa.filters.mel(norm="slaney").
        let n_freq = n_fft / 2 + 1;
        let fft_freqs: Vec<f64> = (0..n_freq).map(|k| sr as f64 * k as f64 / n_fft as f64).collect();
        let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
        let hz: Vec<f64> = (0..n_mels + 2).map(|m| mel_to_hz(mmin + (mmax - mmin) * m as f64 / (n_mels + 1) as f64)).collect();
        let mut fb = Vec::with_capacity(n_mels);
        let mut fb_range = Vec::with_capacity(n_mels);
        for m in 0..n_mels {
            let enorm = 2.0 / (hz[m + 2] - hz[m]);
            let row: Vec<f32> = (0..n_freq)
                .map(|k| {
                    let lower = (fft_freqs[k] - hz[m]) / (hz[m + 1] - hz[m]);
                    let upper = (hz[m + 2] - fft_freqs[k]) / (hz[m + 2] - hz[m + 1]);
                    (0f64.max(lower.min(upper)) * enorm) as f32
                })
                .collect();
            let lo = row.iter().position(|&w| w != 0.0).unwrap_or(0);
            let hi = row.iter().rposition(|&w| w != 0.0).map_or(0, |i| i + 1);
            fb.push(row);
            fb_range.push((lo, hi));
        }
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(n_fft);
        Mel { n_mels, n_fft, hop, pre_emphasis, window, fb, fb_range, fft }
    }

    /// Frames for n samples: floor(n / hop) + 1 (the last one is NeMo's centre-padding frame).
    pub fn n_frames(&self, n: usize) -> usize {
        n / self.hop + 1
    }

    /// Log-mel, normalized per feature. Returns frame-major [n_frames][n_mels] (flattened).
    /// Needs at least 2 * hop samples (two frames, so the statistics have n - 1 >= 1 frames).
    pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
        let n = pcm.len();
        let pad = self.n_fft / 2;
        assert!(n >= 2 * self.hop, "mel: need at least {} samples, got {n}", 2 * self.hop);
        let n_frames = self.n_frames(n);
        // Pre-emphasis, then zero padding on both sides.
        let mut emph = vec![0f64; n];
        emph[0] = pcm[0] as f64;
        for i in 1..n {
            emph[i] = pcm[i] as f64 - self.pre_emphasis * pcm[i - 1] as f64;
        }
        let mut padded = vec![0f64; n + 2 * pad];
        padded[pad..pad + n].copy_from_slice(&emph);

        let n_freq = self.n_fft / 2 + 1;
        let mut buf = self.fft.make_input_vec();
        let mut spec = self.fft.make_output_vec();
        let mut power = vec![0f32; n_freq];
        // log_mel is [n_mels][n_frames] while accumulating stats.
        let mut log_mel = vec![0f32; self.n_mels * n_frames];
        for t in 0..n_frames {
            let start = t * self.hop;
            for k in 0..self.n_fft {
                buf[k] = padded[start + k] * self.window[k];
            }
            self.fft.process(&mut buf, &mut spec).unwrap();
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = (c.re * c.re + c.im * c.im) as f32; // stored as f32, like transcribe.cpp
            }
            for m in 0..self.n_mels {
                let (lo, hi) = self.fb_range[m];
                let row = &self.fb[m];
                let mut sum = 0f64;
                for k in lo..hi {
                    sum += row[k] as f64 * power[k] as f64;
                }
                log_mel[m * n_frames + t] = (sum + LOG_EPS).ln() as f32;
            }
        }
        // Per-feature normalize over the valid frames (NeMo seq_len = n / hop = n_frames - 1), unbiased; the last
        // frame stays 0.
        let valid = n_frames - 1;
        let mut out = vec![0f32; n_frames * self.n_mels];
        for m in 0..self.n_mels {
            let row = &log_mel[m * n_frames..m * n_frames + valid];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / valid as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (valid as f64 - 1.0);
            let inv = 1.0 / (var.sqrt() + NORM_EPS);
            for t in 0..valid {
                out[t * self.n_mels + m] = ((row[t] as f64 - mean) * inv) as f32;
            }
        }
        out
    }
}
