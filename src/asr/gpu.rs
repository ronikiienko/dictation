//! GPU only (Linux, Windows; macOS uses the Neural Engine or the CPU): which GPU, CubeCL's kernel cache, and the
//! one-time sweep's bookkeeping.
use burn::tensor::{Device, DeviceKind};
use std::path::Path;

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

/// A GPU adapter (Vulkan): its name and how to ask CubeCL for it.
pub struct Gpu {
    pub name: String,
    /// "discrete", "integrated", ...
    pub kind: &'static str,
    /// `CUBECL_WGPU_DEFAULT_DEVICE` value selecting it, e.g. `IntegratedGpu(0)`.
    selector: String,
}

/// The GPU API: Vulkan; CubeCL writes the kernels as SPIR-V, not WGSL.
const BACKENDS: burn::backend::wgpu::wgpu::Backends = burn::backend::wgpu::wgpu::Backends::VULKAN;

/// The default GPU on that API (`gpu()` says which one).
pub(super) fn device() -> Device {
    Device::vulkan(DeviceKind::DefaultDevice)
}

fn instance() -> burn::backend::wgpu::wgpu::Instance {
    use burn::backend::wgpu::wgpu;
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: BACKENDS,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// Every adapter (Vulkan), in the order CubeCL counts them (each type from 0).
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
