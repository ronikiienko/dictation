//! Record from a source the way the app does, write a 16 kHz mono wav (test tool).
//! `cargo run --release --example rec -- list` | `… -- <source id: "" = default mic, "system-audio", or a device id> <seconds> <out.wav>`
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "list" {
        for (id, name) in dictation::audio::input_devices() {
            println!("{id:?}  {name}");
        }
        return Ok(());
    }
    let (src, secs, out) = (&a[1], a[2].parse::<f64>()?, &a[3]);
    let r = dictation::audio::Recorder::start(src, || {})?;
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    let x = r.stop()?;
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let first = x.iter().position(|v| v.abs() > 0.01).map_or(-1.0, |i| i as f64 / 16000.0);
    println!("first sound at {first:.2} s");
    println!("{} samples ({:.2} s at 16 kHz), peak {peak:.3}, rms {rms:.4}", x.len(), x.len() as f64 / 16000.0);
    let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(out, spec)?;
    for v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
