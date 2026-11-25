//! Whole pipeline: pcm (16 kHz mono f32) -> text. Mel and decoder on the CPU, encoder + pause head on the GPU (or the
//! encoder on the Neural Engine, macOS, with the pause head on the CPU; or all on the CPU).
//! Order of steps follows transcribe.cpp `src/arch/parakeet/model.cpp` (MIT, (c) 2026 The transcribe.cpp authors).
use super::cpu;
use super::decoder::Decoder;
use super::encoder::{valid_lengths, Encoder, Ld, Vad, PRE_CHUNK};
use super::weights::Weights;
use super::longform::{self, Params};
use super::mel::Mel;
use super::tokenizer::Tokenizer;
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData};
use std::time::Instant;

/// Where the encoder runs: GPU (burn), CPU (cpu.rs), or Neural Engine (ane.rs; subsampler for the pause scan, pause
/// head and joint projection on the CPU: cpu.rs without its blocks).
enum Engine {
    Gpu { dev: Device, enc: Encoder, vad: Vad, joint_enc: (Tensor<2>, Tensor<2>) },
    Cpu(cpu::Encoder),
    #[cfg(target_os = "macos")]
    Ane { ane: super::ane::Encoder, head: cpu::Encoder },
}

pub struct Model {
    engine: Engine,
    dec: Decoder,
    tok: Tokenizer,
    mel: Mel,
    lf: Params,
    hop: usize,
}

/// A recognized word and when it was said (seconds from the start of the audio; `end` = start of its last piece +
/// one 80 ms frame).
#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
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
    /// `dev` None: the encoder on the CPU.
    pub fn load(g: &Weights, dev: Option<Device>) -> Model {
        let engine = match dev {
            Some(dev) => {
                // Test only: F32=1 f32 activations, DEQ=1 weights dequantized (no int8 kernels), to see what f16 +
                // int8 cost.
                let f32 = std::env::var("F32").is_ok();
                let l = Ld { g, dev: &dev, low: if f32 { FloatDType::F32 } else { FloatDType::F16 }, q8: std::env::var("DEQ").is_err() };
                let enc = dev.memory_persistent_allocations(&l, |l| Encoder::load(l));
                let vad = Vad::load(&l);
                let joint_enc = (l.lin("joint.enc.weight"), l.row("joint.enc.bias"));
                Engine::Gpu { dev, enc, vad, joint_enc }
            }
            None => Engine::Cpu(cpu::Encoder::load(g)),
        };
        Self::with_engine(g, engine)
    }

    /// Encoder on the Neural Engine (its models already loaded), the rest from `g` on the CPU.
    #[cfg(target_os = "macos")]
    pub fn load_ane(g: &Weights, ane: super::ane::Encoder) -> Model {
        Self::with_engine(g, Engine::Ane { ane, head: cpu::Encoder::load_head(g) })
    }

    fn with_engine(g: &Weights, engine: Engine) -> Model {
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
        Model { engine, dec, tok, mel, lf, hop }
    }

    /// Padded with zero frames to a whole number of encoder frames (8 mel frames = 80 ms, plus the centre-padding
    /// frame), so the GPU only ever sees lengths on the 80 ms grid: an off-grid length is a new kernel shape, which
    /// means compiling (and tuning) during a take. Same result: the subsampler masks everything past the valid
    /// length (`valid_lengths`, from the real sample count), and valid frames only read up to the last real one.
    /// The CPU encoder gets the same padding, so both see the same input.
    fn mel_padded(&self, pcm: &[f32]) -> (Vec<f32>, usize) {
        let mut m = self.mel.compute(pcm);
        let frames = (m.len() / self.mel.n_mels - 1).div_ceil(8) * 8 + 1;
        m.resize(frames * self.mel.n_mels, 0.0);
        (m, frames)
    }

    fn mel_tensor(&self, pcm: &[f32], dev: &Device) -> Tensor<2> {
        let (m, frames) = self.mel_padded(pcm);
        Tensor::from_data(TensorData::new(m, [frames, self.mel.n_mels]), (dev, DType::F32))
    }

    /// GPU: hand the memory pool's free pages back.
    fn cleanup(&self) {
        if let Engine::Gpu { dev, .. } = &self.engine {
            dev.memory_cleanup();
        }
    }

    /// Encoder output through the joint's encoder projection: [T_valid][joint_h] f32.
    fn encode(&self, pcm: &[f32], t: &mut Timing) -> (Vec<f32>, usize) {
        let (enc, joint_enc, dev) = match &self.engine {
            Engine::Gpu { dev, enc, joint_enc, .. } => (enc, joint_enc, dev),
            Engine::Cpu(c) => {
                let t0 = Instant::now();
                let (mel, frames) = self.mel_padded(pcm);
                t.mel += t0.elapsed().as_secs_f64() * 1e3;
                let t0 = Instant::now();
                let r = c.encode(&mel, frames, pcm.len());
                t.enc += t0.elapsed().as_secs_f64() * 1e3;
                return r;
            }
            #[cfg(target_os = "macos")]
            Engine::Ane { ane, head } => {
                // Unpadded: the model takes the valid frames and pads to its window itself.
                let t0 = Instant::now();
                let mel = self.mel.compute(pcm);
                t.mel += t0.elapsed().as_secs_f64() * 1e3;
                let t0 = Instant::now();
                let (x, n) = ane.encode(&mel, mel.len() / self.mel.n_mels).unwrap_or_else(|e| panic!("{e}"));
                let r = (head.project(&x, n), n);
                t.enc += t0.elapsed().as_secs_f64() * 1e3;
                return r;
            }
        };
        let t0 = Instant::now();
        let mel = self.mel_tensor(pcm, dev);
        t.mel += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let x = enc.fwd(mel, pcm.len());
        let n = x.dims()[0];
        let p = (x.matmul(joint_enc.0.clone()) + joint_enc.1.clone()).cast(FloatDType::F32);
        let v = p.into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
        t.enc += t0.elapsed().as_secs_f64() * 1e3;
        (v, n)
    }

    /// One piece (<= 30 s) -> tokens with their encoder frames.
    fn one_shot_tokens(&self, pcm: &[f32], t: &mut Timing) -> Vec<(usize, usize)> {
        if pcm.len() < self.lf.min_feature_samples {
            return Vec::new();
        }
        let (proj, n) = self.encode(pcm, t);
        let t0 = Instant::now();
        let toks = self.dec.decode(&proj, n);
        t.dec += t0.elapsed().as_secs_f64() * 1e3;
        toks
    }

    /// Tokens of each piece (start, length) of `pcm`. GPU: piece i is decoded on a CPU thread while the GPU encodes
    /// piece i + 1 (CPU encoder: one after the other, both want the same cores). `t.dec` then also counts decoder
    /// time hidden behind the encoder.
    fn pieces_tokens(&self, pcm: &[f32], segs: &[(usize, usize)], t: &mut Timing) -> Vec<Vec<(usize, usize)>> {
        if matches!(self.engine, Engine::Cpu(_)) || segs.len() < 2 {
            return segs.iter().map(|&(start, len)| self.one_shot_tokens(&pcm[start..start + len], t)).collect();
        }
        type Job<'s> = std::thread::ScopedJoinHandle<'s, (Vec<(usize, usize)>, f64)>;
        let dec = &self.dec;
        std::thread::scope(|s| {
            let mut out = Vec::new();
            let mut pending: Option<Job> = None;
            let join = |h: Job, out: &mut Vec<_>, t: &mut Timing| {
                let (toks, ms) = h.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
                t.dec += ms;
                out.push(toks);
            };
            for &(start, len) in segs {
                let piece = &pcm[start..start + len];
                let enc = (piece.len() >= self.lf.min_feature_samples).then(|| self.encode(piece, t));
                if let Some(h) = pending.take() {
                    join(h, &mut out, t);
                }
                pending = Some(s.spawn(move || {
                    let t0 = Instant::now();
                    let toks = enc.map_or_else(Vec::new, |(proj, n)| {
                        // Its own pool of a third of the threads: on all of them it slowed the GPU encoder, which
                        // needs a CPU thread to build and launch its work (Ryzen 5600H, 12 threads: 4 best; 8 and
                        // all 12 lost most of the gain, NOTES.md "Decoder on all cores").
                        static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
                        let pool = POOL.get_or_init(|| {
                            let n = std::thread::available_parallelism().map_or(1, |n| n.get() / 3).max(1);
                            rayon::ThreadPoolBuilder::new().num_threads(n).build().ok()
                        });
                        match pool {
                            Some(p) => p.install(|| dec.decode(&proj, n)),
                            None => dec.decode(&proj, n),
                        }
                    });
                    (toks, t0.elapsed().as_secs_f64() * 1e3)
                }));
            }
            if let Some(h) = pending.take() {
                join(h, &mut out, t);
            }
            out
        })
    }

    /// One piece (<= 30 s).
    fn one_shot(&self, pcm: &[f32], t: &mut Timing) -> String {
        let ids: Vec<usize> = self.one_shot_tokens(pcm, t).into_iter().map(|(id, _)| id).collect();
        self.tok.decode(&ids)
    }

    /// Like `transcribe`, but words with times (for subtitles). Same pieces, so the same text.
    pub fn transcribe_words(&self, pcm: &[f32]) -> (Vec<Word>, Timing) {
        let mut t = Timing::default();
        let segs = if self.lf.fits_one_segment(pcm.len()) {
            vec![(0, pcm.len())]
        } else {
            let t0 = Instant::now();
            let s = self.segments(pcm);
            t.vad = t0.elapsed().as_secs_f64() * 1e3;
            s
        };
        let (rate, frame) = (self.lf.sample_rate as f64, self.lf.frame_seconds);
        let mut words = Vec::new();
        for (&(start, _), toks) in segs.iter().zip(self.pieces_tokens(pcm, &segs, &mut t)) {
            let at = start as f64 / rate;
            words.extend(self.tok.words(&toks).into_iter().map(|(text, a, b)| Word {
                text,
                start: at + a as f64 * frame,
                end: at + (b + 1) as f64 * frame,
            }));
        }
        self.cleanup();
        (words, t)
    }

    /// Speech probability per encoder frame for one block (subsampler + pause head only).
    fn speech_probs(&self, pcm: &[f32]) -> Vec<f32> {
        let (enc, vad, dev) = match &self.engine {
            Engine::Gpu { dev, enc, vad, .. } => (enc, vad, dev),
            Engine::Cpu(c) => {
                let (mel, frames) = self.mel_padded(pcm);
                return c.speech_probs(&mel, frames, pcm.len(), self.hop);
            }
            #[cfg(target_os = "macos")]
            Engine::Ane { head, .. } => {
                let (mel, frames) = self.mel_padded(pcm);
                return head.speech_probs(&mel, frames, pcm.len(), self.hop);
            }
        };
        let len = valid_lengths(pcm.len(), self.hop);
        let x = enc.pre.fwd_chunked(self.mel_tensor(pcm, dev), len, PRE_CHUNK);
        let p = vad.fwd(x).into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
        p[..p.len().min(len[3])].to_vec()
    }

    /// Pause scan of one block (any length, normally only run on takes over 30 s); the sweep uses it to tune the
    /// pause head at short lengths (the last block of a take over 120 s).
    pub fn scan(&self, pcm: &[f32]) {
        let _ = self.speech_probs(pcm);
    }

    /// Audio just over 30 s: where its first piece ends (latest pause before 30 s, as for long takes). Runs the
    /// pause scan over all of `pcm` (the sweep uses it for that).
    pub fn first_cut(&self, pcm: &[f32]) -> usize {
        self.segments(pcm).first().map_or(pcm.len(), |&(s, l)| s + l)
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

    /// Neural Engine only: one encode of silence, so the ANE does not wind down while a take is being recorded (the
    /// first call after idle is slower: +10 ms after 3 s, +30 after 8, +100-140 after 16-25 s; NOTES.md "Neural
    /// Engine"). False (nothing done) on the GPU and CPU.
    pub fn keep_awake(&self) -> bool {
        match &self.engine {
            #[cfg(target_os = "macos")]
            Engine::Ane { ane, .. } => {
                let _ = ane.warm_up();
                true
            }
            _ => false,
        }
    }

    /// Takes over 30 s are cut at pauses (the model's own pause head) and the pieces' texts joined with spaces.
    /// Afterwards the GPU memory pool hands its free pages back.
    pub fn transcribe(&self, pcm: &[f32]) -> (String, Timing) {
        let mut t = Timing::default();
        let text = self.transcribe_inner(pcm, &mut t);
        self.cleanup();
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
        for toks in self.pieces_tokens(pcm, &segs, t) {
            let ids: Vec<usize> = toks.into_iter().map(|(id, _)| id).collect();
            let s = self.tok.decode(&ids);
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
