//! Speech-to-text: Parakeet Ultra in Rust. Mel and the TDT decoder run on the CPU, the encoder and the pause head on
//! the GPU (burn on CubeCL; Vulkan, macOS: Metal). Ported from transcribe.cpp (MIT, (c) 2026 The transcribe.cpp authors); the
//! research and every measurement behind it are in experiments/burn and NOTES.md ("burn pipeline").
//!
//! The model comes as transcribe.cpp's GGUF and is converted once into our weights file (weights.rs), which loads
//! with no CPU work: memory-mapped and uploaded as is.
mod cpu;
mod decoder;
mod encoder;
mod gguf;
mod live;
mod longform;
mod mel;
mod pipeline;
mod tokenizer;
mod weights;

use burn::tensor::{Device, DeviceConfig, DeviceKind, FloatDType, Tensor};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

pub use live::Live;
pub use pipeline::{Timing, Word};
pub use gguf::Value;
pub use weights::{Blob, Weights, convert, is_current, q8_bytes, write};

/// burn reports GPU failures (out of memory, lost device, ...) by panicking: turn them into errors.
fn guard<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(msg.unwrap_or_else(|| "GPU error".into()))
    })
}

/// Test only: what the GPU allocator holds (in use / reserved, per pool), in MB.
pub fn mem_report() -> String {
    let d = device();
    let mb = |b: u64| b as f64 / 1e6;
    let u = d.memory_pool_usage().unwrap_or_default();
    let pools: Vec<String> = d.memory_pool_report().unwrap_or_default().iter()
        .map(|p| format!("{}x{:.1}(peak {}, max alloc {:.1})", p.pages, mb(p.page_size), p.pages_peak, mb(p.largest_alloc))).collect();
    format!("in use {:.0} reserved {:.0} allocs {} | pools {}", mb(u.bytes_in_use), mb(u.bytes_reserved), u.number_allocs, pools.join(" "))
}

/// Where CubeCL keeps its kernel cache (compiled SPIR-V + autotune results): `kernels/<GPU name>/`, next to the
/// model. One folder per GPU because CubeCL files autotune results under "the default device", not under the GPU:
/// with one shared cache, a second GPU would reuse the first one's picks (an NVIDIA matmul pick crashed the AMD iGPU).
/// Process-wide and only settable before the first GPU use; later calls are ignored (the first folder stays).
pub fn set_kernel_cache(kernels: &Path) {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let mut c = CubeClRuntimeConfig::default();
    c.compilation.cache = true;
    let dir = gpu_dir(kernels);
    // Only the first call counts (see above), and it comes before CubeCL opens the database: our open/close of the
    // index file below must not happen while this process holds Turso's locks on it (closing any handle to a file
    // drops all of a process's fcntl locks on it).
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| repair_stale_wal_index(&dir));
    c.environment.path = CacheConfig::Directory(dir);
    CubeClRuntimeConfig::try_set(c);
}

/// CubeCL's cache is a Turso database in its experimental multi-process mode: `default.db`, the log `default.db-wal`
/// and the log's shared index `default.db-tshm`. Twice (during experiments) the log was emptied to 0 bytes while the
/// index still listed ~600 entries; every open then failed to read them, and CubeCL went on read-only, silently
/// saving nothing (tuning, kernels). Cause not reproduced (NOTES.md "Read-only cache again"). With the log empty the
/// index describes nothing that exists, so removing it loses nothing; Turso rebuilds it on open. Only when no other
/// process uses the database (Turso holds a lock on the index's first byte for as long as a process has it open).
fn repair_stale_wal_index(dir: &Path) {
    let (wal, index) = (dir.join("default.db-wal"), dir.join("default.db-tshm"));
    let wal_empty = std::fs::metadata(&wal).map_or(true, |m| m.len() == 0);
    if !wal_empty || !index.exists() || index_in_use(&index) {
        return;
    }
    match std::fs::remove_file(&index) {
        Ok(()) => eprintln!("kernel cache: removed a stale log index ({}), it would have made the cache read-only", index.display()),
        Err(e) => eprintln!("kernel cache: could not remove the stale log index {}: {e}", index.display()),
    }
}

/// Whether another process has the database open: it holds a lock on the index's byte 0. Unknown counts as in use.
#[cfg(unix)]
fn index_in_use(index: &Path) -> bool {
    use std::os::fd::AsRawFd;
    let Ok(f) = std::fs::OpenOptions::new().read(true).write(true).open(index) else { return true };
    // SAFETY: a plain fcntl query on an open file; `flock` is a C struct, all-zero is valid.
    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_WRLCK as _;
    lock.l_whence = libc::SEEK_SET as _;
    lock.l_start = 0;
    lock.l_len = 1;
    let r = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETLK, &mut lock) };
    r != 0 || lock.l_type as i32 != libc::F_UNLCK as i32
}

#[cfg(not(unix))]
fn index_in_use(_: &Path) -> bool {
    true // Windows: no check written, never touch it
}

fn gpu_dir(kernels: &Path) -> std::path::PathBuf {
    kernels.join(gpu().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect::<String>())
}

/// What a sweep's results depend on: the build (kernels, see build.rs), the tensor shapes our code produces (the
/// "shapes N" version: bump it when they change, e.g. a new chunk size, so the new shapes get compiled and tuned
/// once instead of during takes) and whether it tunes.
#[cfg(feature = "autotune")]
const SWEEP_KEY: &str = concat!(env!("DICTATION_BUILD_KEY"), " shapes ", "2", " tuned");
#[cfg(not(feature = "autotune"))]
const SWEEP_KEY: &str = concat!(env!("DICTATION_BUILD_KEY"), " shapes ", "2");

/// Whether this GPU's kernel cache already holds a full sweep for this build (`Asr::sweep` + `mark_swept`).
pub fn swept(kernels: &Path) -> bool {
    std::fs::read_to_string(gpu_dir(kernels).join("swept")).is_ok_and(|s| s.trim() == SWEEP_KEY)
}

pub fn mark_swept(kernels: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(gpu_dir(kernels))?;
    std::fs::write(gpu_dir(kernels).join("swept"), SWEEP_KEY)
}

/// A GPU adapter (Vulkan; macOS: Metal): its name and how to ask CubeCL for it.
pub struct Gpu {
    pub name: String,
    /// "discrete", "integrated", ...
    pub kind: &'static str,
    /// `CUBECL_WGPU_DEFAULT_DEVICE` value selecting it, e.g. `IntegratedGpu(0)`.
    selector: String,
}

/// The GPU API: Metal on macOS (it has no Vulkan), Vulkan elsewhere. Either way CubeCL writes the kernels in that
/// API's own language (MSL / SPIR-V), not WGSL.
const BACKENDS: burn::backend::wgpu::wgpu::Backends =
    if cfg!(target_os = "macos") { burn::backend::wgpu::wgpu::Backends::METAL } else { burn::backend::wgpu::wgpu::Backends::VULKAN };

/// The default GPU on that API (`gpu()` says which one).
fn device() -> Device {
    #[cfg(target_os = "macos")]
    return Device::metal(DeviceKind::DefaultDevice);
    #[cfg(not(target_os = "macos"))]
    return Device::vulkan(DeviceKind::DefaultDevice);
}

fn instance() -> burn::backend::wgpu::wgpu::Instance {
    use burn::backend::wgpu::wgpu;
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: BACKENDS,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// Every adapter (Vulkan; macOS: Metal), in the order CubeCL counts them (each type from 0).
pub fn gpus() -> Vec<Gpu> {
    use burn::backend::wgpu::wgpu::DeviceType;
    let mut count = [0usize; 5];
    block_on(instance().enumerate_adapters(BACKENDS))
        .into_iter()
        .map(|a| a.get_info())
        .map(|i| {
            let (t, kind, sel) = match i.device_type {
                DeviceType::DiscreteGpu => (0, "discrete", "DiscreteGpu"),
                DeviceType::IntegratedGpu => (1, "integrated", "IntegratedGpu"),
                DeviceType::VirtualGpu => (2, "virtual", "VirtualGpu"),
                DeviceType::Cpu => (3, "CPU, slow", "Cpu"),
                DeviceType::Other => (4, "other", "Other"),
            };
            let selector = if t == 3 { sel.to_string() } else { format!("{sel}({})", count[t]) };
            count[t] += 1;
            Gpu { name: i.name, kind, selector }
        })
        .collect()
}

/// Use the GPU with this name instead of the default one ("" or not found: the default). Call at startup, before
/// any other thread runs (it sets the process environment CubeCL reads), and before `gpu()`.
pub fn choose_gpu(name: &str) {
    if let Some(g) = gpus().into_iter().find(|g| !name.is_empty() && g.name == name) {
        // SAFETY: called first thing in main, while the process is single-threaded.
        unsafe { std::env::set_var("CUBECL_WGPU_DEFAULT_DEVICE", g.selector) };
    }
}

/// Name of the GPU `device()` uses. Same choice as CubeCL: the `CUBECL_WGPU_DEFAULT_DEVICE`
/// override (`DiscreteGpu(n)`, `IntegratedGpu(n)`, `VirtualGpu(n)`, `Cpu`) if set, else the adapter wgpu offers
/// for "high performance" on that API.
pub fn gpu() -> &'static str {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        use burn::backend::wgpu::wgpu;
        let name = match std::env::var("CUBECL_WGPU_DEFAULT_DEVICE") {
            Ok(sel) => gpus().into_iter().find(|g| g.selector == sel).map(|g| g.name),
            Err(_) => block_on(instance().request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .ok()
            .map(|a| a.get_info().name),
        };
        name.unwrap_or_else(|| "unknown GPU".into())
    })
}

/// wgpu's native adapter futures are ready at once; poll until done.
fn block_on<T>(f: impl Future<Output = T>) -> T {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// A loaded model. 16 kHz mono f32 in, text out. Dropping it frees the GPU memory it used.
pub struct Asr {
    model: Option<pipeline::Model>,
    /// None: encoder on the CPU.
    dev: Option<Device>,
}

impl Asr {
    pub fn load(weights_file: &Path) -> Result<Asr, String> {
        guard(|| {
            let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
            // The best GPU (wgpu prefers "high power" ones). f16 activations.
            let mut dev = device();
            let _ = dev.configure(DeviceConfig::default().float_dtype(FloatDType::F16)); // fails if already set
            let model = pipeline::Model::load(&w, Some(dev.clone()));
            let _ = Tensor::<1>::zeros([1], &dev).into_data(); // wait for the uploads
            Ok(Asr { model: Some(model), dev: Some(dev) })
        })
    }

    /// Everything on the CPU (cpu.rs): no GPU, no kernels to compile or tune.
    pub fn load_cpu(weights_file: &Path) -> Result<Asr, String> {
        let w = weights::Weights::open(weights_file).map_err(|e| e.to_string())?;
        Ok(Asr { model: Some(pipeline::Model::load(&w, None)), dev: None })
    }

    /// Compiles (and tunes) ahead of time what real takes will need, so no take waits for it. Content doesn't
    /// matter, only shapes: synthetic noise (3 s loud, 3 s quiet, so long takes find pauses). Results go to the
    /// kernel cache, per build (build.rs) and GPU (`set_kernel_cache`).
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
        let mut runs: Vec<(usize, Run)> = vec![(30 * rate + step, Run::Take)]; // live mode's window end
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

    /// Audio just over 30 s: where its first piece ends (latest pause before 30 s). Live mode's window end.
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
        let Some(dev) = &self.dev else { return };
        let _ = guard(|| {
            dev.memory_cleanup();
            Ok(())
        });
    }
}
