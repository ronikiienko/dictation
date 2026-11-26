//! Speech-to-text: Parakeet Ultra in Rust. Mel and the TDT decoder run on the CPU, the encoder and the pause head on
//! the GPU (burn on CubeCL, Vulkan; Linux and Windows), the CPU (cpu.rs) or the Neural Engine (macOS, ane.rs). Ported from transcribe.cpp (MIT, (c) 2026 The transcribe.cpp authors); the
//! research and every measurement behind it are in experiments/burn and NOTES.md ("burn pipeline").
//!
//! The model comes as transcribe.cpp's GGUF and is converted once into our weights file (weights.rs), which loads
//! with no CPU work: memory-mapped and uploaded as is.
#[cfg(target_os = "macos")]
mod ane;
mod ane_build;
mod cpu;
mod decoder;
#[cfg(gpu)]
mod encoder;
mod gguf;
#[cfg(gpu)]
mod gpu;
mod longform;
mod mel;
mod pipeline;
mod redux;
mod shapes;
mod tokenizer;
mod weights;

#[cfg(gpu)]
use burn::tensor::{Device, DeviceConfig, FloatDType, Tensor};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

pub use pipeline::{Timing, Word};
pub use gguf::{Gguf, Value};
pub use ane_build::{MODEL as ANE_MODEL, build as build_ane};
pub use redux::{convert as convert_redux, convert_ultra_head};
pub use weights::{Blob, Weights, convert, is_current, q8_bytes, write};
#[cfg(gpu)]
pub use gpu::{Gpu, choose_gpu, gpu, gpus, mark_swept, mem_report, set_kernel_cache, swept};

/// macOS: no GPU path (the Neural Engine or the CPU), so no GPUs to list or choose.
#[cfg(not(gpu))]
pub struct Gpu {
    pub name: String,
    pub kind: &'static str,
}
#[cfg(not(gpu))]
pub fn gpus() -> Vec<Gpu> {
    Vec::new()
}
#[cfg(not(gpu))]
pub fn gpu() -> &'static str {
    "none"
}
#[cfg(not(gpu))]
pub fn choose_gpu(_name: &str) {}

/// burn reports GPU failures (out of memory, lost device, ...) by panicking, the Neural Engine path its CoreML errors
/// too: turn them into errors.
fn guard<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(msg.unwrap_or_else(|| "speech model error".into()))
    })
}

/// What macOS's compiled copy of the Neural Engine models depends on, as far as we know: the app (its build and path;
/// an updated app is prepared again, quick if macOS still has them) and the model folder (its path).
fn ane_key() -> String {
    format!("{} {}", env!("DICTATION_BUILD_KEY"), std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default())
}

/// Whether `Asr::prepare_ane` ran for these Neural Engine models and this app.
pub fn ane_prepared(ane_dir: &Path) -> bool {
    std::fs::read_to_string(ane_dir.join("prepared")).is_ok_and(|s| s == ane_key())
}

/// A loaded model. 16 kHz mono f32 in, text out. Dropping it frees the GPU memory it used.
pub struct Asr {
    model: Option<pipeline::Model>,
    /// The GPU, if the encoder runs there.
    #[cfg(gpu)]
    dev: Option<Device>,
}

impl Asr {
    #[cfg(gpu)]
    pub fn load(weights_file: &Path) -> Result<Asr, String> {
        guard(|| {
            let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
            if w.has_ternary() {
                return Err("Parakeet Redux runs on the CPU only (GPU: not implemented). Choose Model → Parakeet Redux — CPU.".into());
            }
            // The best GPU (wgpu prefers "high power" ones). f16 activations.
            let mut dev = gpu::device();
            let _ = dev.configure(DeviceConfig::default().float_dtype(FloatDType::F16)); // fails if already set
            let model = pipeline::Model::load_gpu(&w, dev.clone());
            let _ = Tensor::<1>::zeros([1], &dev).into_data(); // wait for the uploads
            Ok(Asr { model: Some(model), dev: Some(dev) })
        })
    }

    /// Everything on the CPU (cpu.rs): no GPU, no kernels to compile or tune.
    pub fn load_cpu(weights_file: &Path) -> Result<Asr, String> {
        let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
        Ok(Asr { model: Some(pipeline::Model::load_cpu(&w)), #[cfg(gpu)] dev: None })
    }

    /// Parakeet Ultra's encoder on the Neural Engine (macOS, CoreML model ANE_MODEL in `ane_dir`, ane.rs), the rest
    /// on the CPU: no GPU, nothing to tune. Run once here (its first run is slow). `progress(done 0..1)` (only slow
    /// the first time: `prepare_ane`).
    pub fn load_ane(weights_file: &Path, ane_dir: &Path, progress: impl FnMut(f64)) -> Result<Asr, String> {
        #[cfg(target_os = "macos")]
        {
            let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
            if w.has_ternary() {
                return Err("the Neural Engine runs Parakeet Ultra only".into());
            }
            let ane = ane::Encoder::load(ane_dir, progress)?;
            ane.warm_up()?;
            Ok(Asr { model: Some(pipeline::Model::load_ane(&w, ane)) })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = progress;
            Err(format!("the Neural Engine is macOS only ({}, {})", weights_file.display(), ane_dir.display()))
        }
    }

    /// One-time: the first load of the Neural Engine model makes macOS compile it for it (~40 s, then cached by
    /// macOS); `load_ane` + `mark_ane_prepared`, like the GPU's sweep.
    pub fn prepare_ane(weights_file: &Path, ane_dir: &Path, progress: impl FnMut(f64)) -> Result<Asr, String> {
        let a = Asr::load_ane(weights_file, ane_dir, progress)?;
        std::fs::write(ane_dir.join("prepared"), ane_key()).map_err(|e| e.to_string())?;
        Ok(a)
    }

    /// Compiles (and tunes) ahead of time what real takes will need, so no take waits for it. Content doesn't
    /// matter, only shapes: synthetic noise (3 s loud, 3 s quiet, so long takes find pauses). Results go to the
    /// kernel cache, per build (build.rs) and GPU (`set_kernel_cache`). GPU only.
    ///
    /// Which lengths (measured 2026-11-20, NOTES "sweep"): new work only shows up right after certain sizes (encoder
    /// frames of 80 ms near 32, 48, 64, 96, 112, 128, 192, 256, 288, 320, and the last 1.2 s), so lengths <= 30 s
    /// run on a 0.4 s grid plus the full 80 ms grid in those places (only 80 ms steps exist: takes are padded to
    /// them, see `pipeline::Model::mel_tensor`). Takes over 30 s only add the pause scan over the whole take (their
    /// pieces are <= 30 s): pause scan alone at every second 31..120 s (a few lengths missed ~1 s of compiling on
    /// 41-52 s takes) and 240 s, and at the same short lengths as above (a take over
    /// 120 s is scanned in 120 s blocks, its last block can be short). `full`: every 80 ms up to 30 s (full model
    /// and pause scan) and full takes 31..120 s (the old sweep; to check that the short one misses nothing).
    /// `progress(done 0..1, seconds of audio, ms)` after every run; "done" is weighted by the expected cost.
    #[cfg(gpu)]
    pub fn sweep(&mut self, full: bool, mut progress: impl FnMut(f64, f64, f64)) -> Result<(), String> {
        let rate = 16_000;
        let mut seed = 1u32;
        let x: Vec<f32> = (0..240 * rate)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let on = (i / (rate * 3)) % 2 == 0; // 3 s "speech", 3 s quiet
                (seed >> 8) as f32 / (1u32 << 24) as f32 * if on { 0.2 } else { 0.002 } - if on { 0.1 } else { 0.001 }
            })
            .collect();
        let step = rate * 8 / 100; // 80 ms = one encoder frame
        let frames = 30 * rate / step; // 375
        let near = |f: usize| {
            [32, 48, 64, 96, 112, 128, 192, 256, 288, 320].iter().any(|&b| f + 1 >= b && f <= b + if b >= 128 { 16 } else { 8 })
        };
        let wanted = |f: usize| full || f <= 24 || f % 5 == 0 || near(f) || f + 16 > frames;
        #[derive(Clone, Copy, PartialEq)]
        enum Run {
            Full,  // the whole pipeline, as a take
            Take,  // pause scan over a take over 30 s, in 120 s blocks (no text)
            Block, // pause scan of one block
        }
        let mut runs: Vec<(usize, Run)> = vec![(30 * rate + step, Run::Take)]; // the shortest take that is cut at pauses
        runs.extend((1..=frames).filter(|&f| wanted(f)).map(|f| (f * step, Run::Full)));
        runs.extend((1..=frames).filter(|&f| wanted(f)).map(|f| (f * step, Run::Block)));
        if full {
            runs.extend((31..=120).map(|s| (s * rate, Run::Full)));
        } else {
            runs.extend((31..=120).chain([240]).map(|s| (s * rate, Run::Take)));
        }
        // Cost of one run, roughly: compiling/tuning (fixed) + running (grows with length; pause scan ~10x cheaper).
        let cost = |&(n, r): &(usize, Run)| {
            let secs = n as f64 / rate as f64;
            0.1 + if r == Run::Full { secs / 100.0 } else { secs / 1000.0 } + if n == 30 * rate + step { 10.0 } else { 0.0 }
        };
        let total: f64 = runs.iter().map(cost).sum();
        let mut done = 0.0;
        for r in &runs {
            let t = std::time::Instant::now();
            let x = &x[..r.0];
            match r.1 {
                Run::Full => drop(self.run(x)?),
                Run::Take => drop(self.first_cut(x)?),
                Run::Block => {
                    let m = self.model.as_ref().ok_or("model not loaded")?;
                    guard(|| Ok(m.scan(x)))?
                }
            }
            done += cost(r);
            progress(done / total, r.0 as f64 / rate as f64, t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Audio just over 30 s: where its first piece ends (latest pause before 30 s). The sweep uses it for the pause scan.
    #[cfg(gpu)]
    pub fn first_cut(&mut self, pcm: &[f32]) -> Result<usize, String> {
        let m = self.model.as_ref().ok_or("model not loaded")?;
        guard(|| Ok(m.first_cut(pcm)))
    }

    /// `run` with each word's time (subtitles).
    pub fn run_words(&mut self, pcm: &[f32]) -> Result<(Vec<Word>, Timing), String> {
        if pcm.is_empty() {
            return Ok((Vec::new(), Timing::default()));
        }
        let m = self.model.as_ref().ok_or("model not loaded")?;
        guard(|| Ok(m.transcribe_words(pcm)))
    }

    /// Neural Engine: one encode of silence to keep it from winding down during a take (pipeline::Model::keep_awake).
    /// False: nothing to do (GPU, CPU, no model).
    pub fn keep_awake(&mut self) -> bool {
        self.model.as_ref().is_some_and(|m| m.keep_awake())
    }

    pub fn run(&mut self, pcm: &[f32]) -> Result<(String, Timing), String> {
        if pcm.is_empty() {
            return Ok((String::new(), Timing::default()));
        }
        let m = self.model.as_ref().ok_or("model not loaded")?;
        guard(|| Ok(m.transcribe(pcm)))
    }
}

#[cfg(gpu)]
impl Drop for Asr {
    fn drop(&mut self) {
        self.model = None;
        let Some(dev) = &self.dev else { return };
        let _ = guard(|| {
            dev.memory_cleanup();
            Ok(())
        });
    }
}
