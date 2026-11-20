//! Parakeet Ultra in Rust: mel + TDT decoder on the CPU, encoder (burn, Vulkan) on the GPU; weights converted once
//! from the transcribe.cpp GGUF (weights.rs). Experiment for replacing transcribe.cpp in the app.
//!
//! usage: burn-enc convert <model.gguf> <out weights file>             GGUF -> our weights file (once)
//!        burn-enc check <weights> <ref dir from ref_dump.py>   step-by-step comparison with transcribe.cpp
//!        burn-enc wav <weights> <file.wav>...                  transcribe (16 kHz mono wav), with timing
//!        burn-enc bench <weights> <ref dir> [runs]             encoder speed / VRAM (env MEMLOG, LENS, SWEEP)
//! env:   F16=0  f32 activations (default f16);  Q8=0  dequantized weights (default: Q8_0 stays int8 on the GPU)
//!        wav: SEGS=1 cut points, VADCMP=1 chunked vs whole-block pause scan, SCAN_CHUNK=<mel frames> (default 3000,
//!        0 = whole 120 s block), CLEANUP=0 keep the GPU pool after each take
mod decoder;
mod encoder;
mod gguf;
mod longform;
mod mel;
mod pipeline;
mod tokenizer;
mod weights;

use burn::tensor::{Device, DeviceConfig, DeviceKind, FloatDType, Tensor, TensorData};
use encoder::{Encoder, Ld};
use weights::Weights;
use std::path::Path;
use std::time::Instant;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Relative L2 error and worst absolute difference against a transcribe.cpp dump.
fn compare(name: &str, t: Tensor<2>, refdir: &Path) {
    let got: Vec<f32> = t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
    let want = read_f32(&refdir.join(format!("{name}.f32")));
    assert_eq!(got.len(), want.len(), "{name}: size");
    let (mut num, mut den, mut worst) = (0f64, 0f64, 0f32);
    for (a, b) in got.iter().zip(&want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
        worst = worst.max((a - b).abs());
    }
    println!("  {name:22} rel.err {:.2e}  max|diff| {worst:.3e}", (num / den).sqrt());
}

/// Same as `compare` for host vectors.
fn compare_vec(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: size");
    let (mut num, mut den, mut worst) = (0f64, 0f64, 0f32);
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
        worst = worst.max((a - b).abs());
    }
    println!("  {name:22} rel.err {:.2e}  max|diff| {worst:.3e}", (num / den).sqrt());
}

fn flag(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |v| v == "1")
}

fn device() -> (Device, FloatDType, bool) {
    let f16 = flag("F16", true);
    let mut dev = Device::vulkan(DeviceKind::DiscreteGpu(0));
    if f16 {
        dev.configure(DeviceConfig::default().float_dtype(FloatDType::F16)).unwrap();
    }
    let _ = Tensor::<1>::zeros([1], &dev).into_data(); // create the device context
    (dev, if f16 { FloatDType::F16 } else { FloatDType::F32 }, flag("Q8", true))
}

fn read_wav(p: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    assert!(spec.sample_rate == 16000 && spec.channels == 1, "{}: need 16 kHz mono", p.display());
    match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u32 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap() as f32 * scale).collect()
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("convert") => {
            let t = Instant::now();
            weights::convert(Path::new(&args[2]), Path::new(&args[3])).unwrap();
            println!("converted in {:.1}s", t.elapsed().as_secs_f64());
        }
        Some("check") => check(Path::new(&args[2]), Path::new(&args[3])),
        Some("wav") => wav(Path::new(&args[2]), &args[3..]),
        Some("bench") => bench(Path::new(&args[2]), Path::new(&args[3]), args.get(4).map_or(10, |s| s.parse().unwrap())),
        _ => eprintln!("usage: burn-enc convert|check|wav|bench ... (see src/main.rs)"),
    }
}

fn load(model: &Path) -> pipeline::Model {
    let t0 = Instant::now();
    let (dev, low, q8) = device();
    let t1 = Instant::now();
    let g = Weights::open(model).unwrap();
    let mut m = pipeline::Model::load(&g, dev, low, q8);
    if let Ok(v) = std::env::var("SCAN_CHUNK") {
        m.scan_chunk = v.parse().unwrap();
    }
    m.cleanup = flag("CLEANUP", true);
    println!("device init {:.2}s, model loaded in {:.2}s", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
    m
}

/// Each stage against transcribe.cpp's dumps (ref_dump.py), then the whole pipeline's text.
fn check(model: &Path, refdir: &Path) {
    let m = load(model);
    let pcm = read_f32(&refdir.join("pcm.f32"));
    let want_text = std::fs::read_to_string(refdir.join("text.txt")).unwrap().trim().to_string();

    println!("mel ({} samples):", pcm.len());
    let t = Instant::now();
    let mel = m.mel.compute(&pcm);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let frames = mel.len() / 128;
    let mut mel_t = vec![0f32; mel.len()]; // dump is [128][frames]
    for f in 0..frames {
        for k in 0..128 {
            mel_t[k * frames + f] = mel[f * 128 + k];
        }
    }
    compare_vec("enc.mel.in", &mel_t, &read_f32(&refdir.join("enc.mel.in.f32")));
    if let Ok(p) = std::env::var("DUMP_MEL") {
        std::fs::write(p, mel_t.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    println!("  ({ms:.1} ms)");

    println!("encoder on our mel:");
    let x = Tensor::<2>::from_data(TensorData::new(mel, [frames, 128]), (&m.dev, burn::tensor::DType::F32));
    let mut taps = Vec::new();
    let out = m.enc.fwd(x, pcm.len(), &mut taps);
    for (name, t) in taps {
        compare(&name, t, refdir);
    }
    compare("enc.final", out, refdir);

    println!("decoder on transcribe.cpp's encoder output:");
    let enc_out = read_f32(&refdir.join("dec.enc_out.f32"));
    let t_enc = enc_out.len() / 1024;
    let proj = decoder::enc_proj_cpu(&m.dec, &enc_out, t_enc);
    let (states, joint) = m.dec.first_step(&proj);
    for (l, (h, c)) in states.iter().enumerate() {
        compare_vec(&format!("dec.lstm.{l}.h.0"), h, &read_f32(&refdir.join(format!("dec.lstm.{l}.h.0.f32"))));
        compare_vec(&format!("dec.lstm.{l}.c.0"), c, &read_f32(&refdir.join(format!("dec.lstm.{l}.c.0.f32"))));
    }
    compare_vec("dec.joint.0", &joint, &read_f32(&refdir.join("dec.joint.0.f32")));
    let t = Instant::now();
    let (ids, iters, preds) = m.dec.decode_counted(&proj, t_enc);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  {iters} joint steps, {preds} predictor steps: {:.0} us per joint step", ms * 1e3 / iters as f64);
    let text = tokenizer_text(&m, &ids);
    println!("  text {} ({} tokens, {ms:.0} ms)", if text == want_text { "IDENTICAL" } else { "DIFFERS" }, ids.len());
    if text != want_text {
        println!("    ours: {text}\n    tc:   {want_text}");
    }

    println!("encode step only (mel upload + encoder + joint projection + download), back to back:");
    for _ in 0..5 {
        let mut t = pipeline::Timing::default();
        let _ = m.encode(&pcm, &mut t);
        print!(" {:.0}", t.enc);
    }
    println!(" ms");
    {
        let mel = m.mel.compute(&pcm);
        let frames = mel.len() / 128;
        let x = Tensor::<2>::from_data(TensorData::new(mel, [frames, 128]), (&m.dev, burn::tensor::DType::F32));
        print!("  encoder fwd + download of [T,1024] only:");
        for _ in 0..5 {
            let t = Instant::now();
            let _ = m.enc.fwd(x.clone(), pcm.len(), &mut Vec::new()).into_data();
            print!(" {:.0}", t.elapsed().as_secs_f64() * 1e3);
        }
        println!(" ms");
    }
    println!("whole pipeline:");
    for i in 0..3 {
        let (text, t) = m.transcribe(&pcm);
        println!("  run {i}: mel {:.0} ms, encoder {:.0} ms, decoder {:.0} ms -> text {}", t.mel, t.enc, t.dec,
            if text == want_text { "IDENTICAL" } else { "DIFFERS" });
        if i == 0 && text != want_text {
            println!("    ours: {text}\n    tc:   {want_text}");
        }
    }
}

fn tokenizer_text(m: &pipeline::Model, ids: &[usize]) -> String {
    m.decode_ids(ids)
}

fn wav(model: &Path, files: &[String]) {
    let mut m = load(model);
    for f in files {
        let pcm = read_wav(Path::new(f));
        if flag("SEGS", false) {
            let segs: Vec<String> = m.segments(&pcm).iter().map(|(s, n)| format!("{:.3}-{:.3}", *s as f64 / 16000.0, (s + n) as f64 / 16000.0)).collect();
            println!("segments: {}", segs.join(" "));
        }
        // VADCMP=1: speech probabilities of the first scan block, subsampler chunked vs whole block.
        if flag("VADCMP", false) {
            let n = pcm.len().min((m.lf.scan_block_seconds * 16000.0) as usize);
            let chunked = m.speech_probs(&pcm[..n]);
            let saved = m.scan_chunk;
            m.scan_chunk = 0;
            let whole = m.speech_probs(&pcm[..n]);
            m.scan_chunk = saved;
            let d = chunked.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("vadcmp: {} vs {} frames, max |diff| {d:.3e}", chunked.len(), whole.len());
        }
        let t0 = Instant::now();
        let (text, t) = m.transcribe(&pcm);
        println!("{f} ({:.1} s): total {:.0} ms (mel {:.0}, pauses {:.0}, encoder {:.0}, decoder {:.0})",
            pcm.len() as f64 / 16000.0, t0.elapsed().as_secs_f64() * 1e3, t.mel, t.vad, t.enc, t.dec);
        println!("  {text}");
    }
}

/// Encoder only, on transcribe.cpp's mel: accuracy per stage, speed, VRAM. Env MEMLOG=1, LENS=1001,1500 (mel
/// frames: 1st/2nd run at other lengths), SWEEP=1 (every length up to 30 s: pre-tune sweep, only meaningful with
/// the autotune feature).
fn bench(model: &Path, refdir: &Path, runs: usize) {
    let tp = Instant::now();
    let (dev, low, q8) = device();
    let memlog = std::env::var("MEMLOG").is_ok_and(|v| v == "1");
    let mem = |label: &str| {
        let u = dev.memory_pool_usage().unwrap();
        let smi = std::process::Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().find(|l| l.starts_with(&format!("{},", std::process::id()))).map(|l| l.split(',').nth(1).unwrap().trim().to_string()).unwrap_or_default())
            .unwrap_or_default();
        let tot = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        if memlog { println!("MEM {label:>8}  GPU total {tot} MiB  allocs {:5}  in_use {:6.1} MB  padding {:5.1} MB  reserved {:6.1} MB  driver {smi} MB",
            u.number_allocs, u.bytes_in_use as f64 / 1e6, u.bytes_padding as f64 / 1e6, u.bytes_reserved as f64 / 1e6); }
    };
    mem("device");
    println!("device init {:.2}s", tp.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let g = Weights::open(model).unwrap();
    let l = Ld { g: &g, dev: &dev, low, q8 };
    // Weights as persistent allocations (own buffers, not carved out of the dynamic pool's pages).
    let enc = dev.memory_persistent_allocations(&l, |l| Encoder::load(l));
    drop(g);
    mem("weights");
    println!("loaded in {:.2}s ({}{})", t0.elapsed().as_secs_f64(), if low == FloatDType::F16 { "f16" } else { "f32" }, if q8 { ", Q8_0 weights int8" } else { "" });

    // Input: transcribe.cpp's own mel ([128, frames] row-major) -> [frames, 128].
    let mel = read_f32(&refdir.join("enc.mel.in.f32"));
    let frames = mel.len() / 128;
    let mel = Tensor::<2>::from_data(TensorData::new(mel, [128, frames]), &dev).swap_dims(0, 1);

    // SWEEP=1: pre-tune sweep. Every length from 8 mel frames up to 30 s (3000 frames) in steps of 8
    // (= 1 encoder frame, 80 ms); the 25 s reference mel is repeated to reach 30 s. Prints the slow
    // (tuning) lengths and the total, then exits. Run twice: the 2nd sweep should tune nothing.
    if std::env::var("SWEEP").is_ok_and(|v| v == "1") {
        let long = Tensor::cat(vec![mel.clone(), mel.clone()], 0).narrow(0, 0, 3000);
        let _ = long.clone().into_data();
        let ts = Instant::now();
        let (mut slow, mut slow_s, mut n_runs) = (0, 0.0, 0);
        for n in (8..=3000).step_by(8) {
            let t = Instant::now();
            let _ = enc.fwd(long.clone().narrow(0, 0, n), (n - 1) * 160, &mut Vec::new()).into_data();
            let s = t.elapsed().as_secs_f64();
            n_runs += 1;
            if s > 0.3 {
                slow += 1;
                slow_s += s;
                println!("  {n:5} mel frames ({:5.2} s audio): {s:6.2} s", n as f64 / 100.0);
            }
            if n % 400 == 0 {
                println!("  .. {n} frames, {:.0} s so far", ts.elapsed().as_secs_f64());
            }
        }
        println!("sweep: {n_runs} lengths in {:.1} s; {slow} slow (>0.3 s) totalling {slow_s:.1} s",
            ts.elapsed().as_secs_f64());
        mem("sweep");
        return;
    }
    let mut taps = Vec::new();
    let t1 = Instant::now();
    let out = enc.fwd(mel.clone(), (frames - 1) * 160, &mut taps);
    let _ = out.clone().into_data();
    println!("first run (shader compile + autotune) {:.2}s", t1.elapsed().as_secs_f64());
    mem("1st run");
    println!("vs transcribe.cpp ({} mel frames -> {} encoder frames):", frames, out.dims()[0]);
    for (name, t) in taps {
        compare(&name, t, refdir);
    }
    compare("enc.final", out, refdir);

    mem("warm");
    let mut ms = Vec::new();
    for i in 0..runs {
        let t = Instant::now();
        let out = enc.fwd(mel.clone(), (frames - 1) * 160, &mut Vec::new());
        let _ = out.into_data(); // wait for the GPU
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
        mem(&format!("run {i}"));
    }
    // LENS=1001,1500: other input lengths (mel frames, 100/s): first run (new kernels/tuning?) vs second.
    if let Ok(lens) = std::env::var("LENS") {
        for n in lens.split(',').map(|x| x.parse::<usize>().unwrap()) {
            let x = mel.clone().narrow(0, 0, n);
            let t = Instant::now();
            let _ = enc.fwd(x.clone(), (n - 1) * 160, &mut Vec::new()).into_data();
            let a = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let _ = enc.fwd(x, (n - 1) * 160, &mut Vec::new()).into_data();
            println!("len {n:5} frames: 1st {a:8.1} ms, 2nd {:6.1} ms", t.elapsed().as_secs_f64() * 1e3);
            mem(&format!("len {n}"));
        }
    }
    dev.memory_cleanup();
    mem("cleanup");
    ms.sort_by(f64::total_cmp);
    println!("encoder forward, {runs} warm runs: min {:.1} ms, median {:.1} ms", ms[0], ms[ms.len() / 2]);
}
