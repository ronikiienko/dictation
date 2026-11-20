//! Speech-to-text: Parakeet Ultra in Rust. Mel and the TDT decoder run on the CPU, the encoder and the pause head on
//! the GPU (burn on CubeCL, Vulkan). Ported from transcribe.cpp (MIT, (c) 2026 The transcribe.cpp authors); the
//! research and every measurement behind it are in experiments/burn and NOTES.md ("burn pipeline").
//!
//! The model comes as transcribe.cpp's GGUF and is converted once into our weights file (weights.rs), which loads
//! with no CPU work: memory-mapped and uploaded as is.
mod decoder;
mod encoder;
mod gguf;
mod longform;
mod mel;
mod pipeline;
mod tokenizer;
mod weights;

use burn::tensor::{Device, DeviceConfig, DeviceKind, FloatDType, Tensor};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

pub use pipeline::Timing;
pub use weights::{convert, is_current};

/// burn reports GPU failures (out of memory, lost device, ...) by panicking: turn them into errors.
fn guard<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(msg.unwrap_or_else(|| "GPU error".into()))
    })
}

/// Where CubeCL keeps its compiled-kernel cache (SPIR-V): a folder next to the model. Process-wide and only
/// settable before the first GPU use; later calls are ignored (the first folder stays).
pub fn set_kernel_cache(dir: &Path) {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let mut c = CubeClRuntimeConfig::default();
    c.compilation.cache = true;
    c.environment.path = CacheConfig::Directory(dir.to_path_buf());
    CubeClRuntimeConfig::try_set(c);
}

/// A loaded model. 16 kHz mono f32 in, text out. Dropping it frees the GPU memory it used.
pub struct Asr {
    model: Option<pipeline::Model>,
    dev: Device,
}

impl Asr {
    pub fn load(weights_file: &Path) -> Result<Asr, String> {
        guard(|| {
            let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
            // The best GPU Vulkan offers (wgpu prefers "high power" ones). f16 activations.
            let mut dev = Device::vulkan(DeviceKind::DefaultDevice);
            let _ = dev.configure(DeviceConfig::default().float_dtype(FloatDType::F16)); // fails if already set
            let model = pipeline::Model::load(&w, dev.clone());
            let _ = Tensor::<1>::zeros([1], &dev).into_data(); // wait for the uploads
            Ok(Asr { model: Some(model), dev })
        })
    }

    pub fn run(&mut self, pcm: &[f32]) -> Result<(String, Timing), String> {
        if pcm.is_empty() {
            return Ok((String::new(), Timing::default()));
        }
        let m = self.model.as_ref().ok_or("model not loaded")?;
        guard(|| Ok(m.transcribe(pcm)))
    }
}

impl Drop for Asr {
    fn drop(&mut self) {
        self.model = None;
        let _ = guard(|| {
            self.dev.memory_cleanup();
            Ok(())
        });
    }
}
