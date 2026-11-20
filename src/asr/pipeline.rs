//! Whole pipeline: pcm (16 kHz mono f32) -> text. Mel and decoder on the CPU, encoder + pause head on the GPU.
//! Order of steps follows transcribe.cpp `src/arch/parakeet/model.cpp` (MIT, (c) 2026 The transcribe.cpp authors).
use super::decoder::Decoder;
use super::encoder::{valid_lengths, Encoder, Ld, Vad};
use super::weights::Weights;
use super::longform::{self, Params};
use super::mel::Mel;
use super::tokenizer::Tokenizer;
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};
use std::time::Instant;

/// Pause scan: the subsampler runs this many mel frames (30 s) at a time inside a 120 s scan block. Same result as
/// the whole block at once, ~600 MB less VRAM at the peak.
const SCAN_CHUNK: usize = 3000;

pub struct Model {
    dev: Device,
    enc: Encoder,
    vad: Vad,
    joint_enc: (Tensor<2>, Tensor<2>),
    dec: Decoder,
    tok: Tokenizer,
    mel: Mel,
    lf: Params,
    hop: usize,
}

/// Time spent per stage, milliseconds.
#[derive(Default, Debug, Clone, Copy)]
pub struct Timing {
    pub mel: f64,
    pub enc: f64,
    pub dec: f64,
    pub vad: f64,
}

/// GGUF floats like 0.1000000015 back to the decimal the converter meant (transcribe.cpp `as_decimal`).
fn dec(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

impl Model {
    /// Weights go to the GPU as f16 activations with int8 (Q8_0) matrices; the encoder's weights as persistent
    /// allocations (own buffers, not pages of the dynamic pool).
    pub fn load(g: &Weights, dev: Device) -> Model {
        let l = Ld { g, dev: &dev, low: FloatDType::F16, q8: true };
        let enc = dev.memory_persistent_allocations(&l, |l| Encoder::load(l));
        let vad = Vad::load(&l);
        let joint_enc = (l.lin("joint.enc.weight"), l.row("joint.enc.bias"));
        let sr = g.int("stt.frontend.sample_rate") as usize;
        let hop = g.int("stt.frontend.hop_length") as usize;
        let mel = Mel::new(
            sr,
            g.int("stt.frontend.n_fft") as usize,
            g.int("stt.frontend.win_length") as usize,
            hop,
            g.int("stt.frontend.num_mels") as usize,
            g.float("stt.frontend.f_min"),
            g.float("stt.frontend.f_max"),
            g.float("stt.frontend.pre_emphasis") as f32 as f64,
        );
        let sub = g.int("stt.parakeet.encoder.subsampling_factor") as usize;
        let lf = Params {
            sample_rate: sr,
            frame_seconds: hop as f64 / sr as f64 * sub as f64,
            speech_threshold: g.float("stt.parakeet.vad.speech_threshold") as f32,
            min_speech_seconds: dec(g.float("stt.parakeet.vad.min_speech_seconds")),
            min_gap_seconds: dec(g.float("stt.parakeet.vad.min_gap_seconds")),
            max_segment_seconds: dec(g.float("stt.parakeet.segmenter.max_segment_seconds")),
            min_segment_seconds: dec(g.float("stt.parakeet.segmenter.min_segment_seconds")),
            min_pause_seconds: dec(g.float("stt.parakeet.segmenter.min_pause_seconds")),
            scan_block_seconds: dec(g.float("stt.parakeet.segmenter.scan_block_seconds")),
            boundary_window_secs: 0.1,
            min_tail_seconds: 0.5,
            min_feature_samples: 320,
        };
        let dec = Decoder::load(g);
        let tok = Tokenizer::load(g);
        Model { enc, vad, joint_enc, dec, tok, mel, lf, hop, dev }
    }

    fn mel_tensor(&self, pcm: &[f32]) -> Tensor<2> {
        let m = self.mel.compute(pcm);
        let frames = m.len() / self.mel.n_mels;
        Tensor::from_data(TensorData::new(m, [frames, self.mel.n_mels]), (&self.dev, DType::F32))
    }

    /// Encoder output through the joint's encoder projection, on the GPU: [T_valid][joint_h] f32.
    fn encode(&self, pcm: &[f32], t: &mut Timing) -> (Vec<f32>, usize) {
        let t0 = Instant::now();
        let mel = self.mel_tensor(pcm);
        t.mel += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let x = self.enc.fwd(mel, pcm.len());
        let n = x.dims()[0];
        let p = (x.matmul(self.joint_enc.0.clone()) + self.joint_enc.1.clone()).cast(FloatDType::F32);
        let v = p.into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
        t.enc += t0.elapsed().as_secs_f64() * 1e3;
        (v, n)
    }

    /// One piece (<= 30 s).
    fn one_shot(&self, pcm: &[f32], t: &mut Timing) -> String {
        if pcm.len() < self.lf.min_feature_samples {
            return String::new();
        }
        let (proj, n) = self.encode(pcm, t);
        let t0 = Instant::now();
        let ids = self.dec.decode(&proj, n);
        let text = self.tok.decode(&ids);
        t.dec += t0.elapsed().as_secs_f64() * 1e3;
        text
    }

    /// Speech probability per encoder frame for one block (subsampler + pause head only).
    fn speech_probs(&self, pcm: &[f32]) -> Vec<f32> {
        let len = valid_lengths(pcm.len(), self.hop);
        let x = self.enc.pre.fwd_chunked(self.mel_tensor(pcm), len, SCAN_CHUNK);
        let p = self.vad.fwd(x).into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
        p[..p.len().min(len[3])].to_vec()
    }

    /// Pieces as (start sample, length); one piece for takes up to 30 s.
    fn segments(&self, pcm: &[f32]) -> Vec<(usize, usize)> {
        let rate = self.lf.sample_rate as f64;
        longform::pause_segments(pcm.len(), &self.lf, |start, len| {
            let probs = self.speech_probs(&pcm[start..start + len]);
            let duration = len as f64 / rate;
            longform::speech_regions(&probs, &self.lf).into_iter().map(|r| (r.0, r.1.min(duration))).collect()
        })
    }

    /// Takes over 30 s are cut at pauses (the model's own pause head) and the pieces' texts joined with spaces.
    /// Afterwards the GPU memory pool hands its free pages back.
    pub fn transcribe(&self, pcm: &[f32]) -> (String, Timing) {
        let mut t = Timing::default();
        let text = self.transcribe_inner(pcm, &mut t);
        self.dev.memory_cleanup();
        (text, t)
    }

    fn transcribe_inner(&self, pcm: &[f32], t: &mut Timing) -> String {
        if self.lf.fits_one_segment(pcm.len()) {
            return self.one_shot(pcm, t);
        }
        let t0 = Instant::now();
        let segs = self.segments(pcm);
        t.vad = t0.elapsed().as_secs_f64() * 1e3;
        let mut text = String::new();
        for (start, len) in segs {
            let s = self.one_shot(&pcm[start..start + len], t);
            if !s.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&s);
            }
        }
        text
    }
}
