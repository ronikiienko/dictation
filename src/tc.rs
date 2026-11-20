//! Speech-to-text through the official `transcribe-cpp` crate (safe Rust bindings for transcribe.cpp).
//! transcribe.cpp itself is built from source by the crate (CMake, Vulkan) and linked statically into the binary.
//! This module only adapts it to what the app needs: load / run / drop, warnings and errors to stderr.
use std::path::Path;

pub use transcribe_cpp::Error;
use transcribe_cpp::{Model, RunOptions, Session, SessionOptions};

/// `log` messages (transcribe.cpp/ggml via init_logging, plus Rust crates) -> stderr, warnings and errors only
/// (transcribe.cpp is chatty at info level). Prefixed with their source.
struct StderrLog;
impl log::Log for StderrLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}: {}", r.target(), r.level(), r.args());
        }
    }
    fn flush(&self) {}
}

/// Call once before loading models.
pub fn init() -> Result<(), Error> {
    static LOG: StderrLog = StderrLog;
    if log::set_logger(&LOG).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    transcribe_cpp::init_logging();
    transcribe_cpp::init_backends_default()
}

/// Speech-to-text: a loaded model plus one session. 16 kHz mono f32 in, text out. Dropping it frees the model.
pub struct Asr {
    model: Model,
    session: Session,
}

impl Asr {
    pub fn load(path: &Path, threads: i32) -> Result<Self, Error> {
        let model = Model::load(path)?;
        let session = model.session_with(&SessionOptions { n_threads: threads, ..Default::default() })?;
        Ok(Asr { model, session })
    }

    pub fn backend(&self) -> String {
        self.model.backend()
    }

    pub fn run(&mut self, pcm: &[f32]) -> Result<String, Error> {
        if pcm.is_empty() {
            return Ok(String::new());
        }
        Ok(self.session.run(pcm, &RunOptions::default())?.text.trim().to_owned())
    }
}
