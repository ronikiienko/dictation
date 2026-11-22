//! Parakeet Redux -> our weights file (src/asr/redux.rs). Default: the ternary modules stored as Ultra's kinds (q8 /
//! f16, exact) so the GPU can run it too, to compare Redux with Ultra on the same runtime (same memory and speed as
//! Ultra there, only the accuracy is Redux's). `t2`: the app's CPU-only file (2-bit), as the app's download makes.
//! usage: cargo run --release --example redux -- <folder: model.safetensors, tokenizer.json, config.json> <out> [t2]
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let q8 = a.get(2).map(String::as_str) != Some("t2");
    dictation::asr::convert_redux(std::path::Path::new(&a[0]), std::path::Path::new(&a[1]), q8).unwrap();
    eprintln!("written {}", a[1]);
}
