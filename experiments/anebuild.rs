//! Neural Engine model from moondream/parakeet-ultra's model.safetensors (src/asr/ane_build.rs), for checking it
//! against the export: `cargo run --release --example anebuild -- <model.safetensors> <out dir>` (the out dir is emptied first).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (st, out) = (std::path::Path::new(&a[0]), std::path::Path::new(&a[1]));
    std::fs::create_dir_all(out)?;
    let t = std::time::Instant::now();
    dictation::asr::build_ane(st, out, |_| {})?;
    eprintln!("built in {:.1} s", t.elapsed().as_secs_f64());
    Ok(())
}
