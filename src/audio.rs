//! Microphone capture (cpal, PulseAudio protocol -> PipeWire) and short sounds (rodio).
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex, mpsc};

use crate::RATE;

/// Frames per block requested from the sound server (50 ms at 16 kHz).
const BLOCK: u32 = 800;
/// Keep recording this long after stop is pressed, so the block in flight is delivered.
const TAIL: std::time::Duration = std::time::Duration::from_millis(150);

fn host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    if let Ok(h) = cpal::host_from_id(cpal::HostId::PulseAudio) {
        return h;
    }
    cpal::default_host()
}

/// (id, human name) of every input device; id "" = system default.
pub fn input_devices() -> Vec<(String, String)> {
    let mut out = vec![(String::new(), "System default".to_string())];
    if let Ok(devs) = host().input_devices() {
        for d in devs {
            let (Ok(id), Ok(desc)) = (d.id(), d.description()) else { continue };
            out.push((id.to_string(), desc.name().to_string()));
        }
    }
    out
}

/// A running recording. Audio is captured on its own thread (cpal streams are not Send); stop() returns 16 kHz mono.
pub struct Recorder {
    stop: mpsc::Sender<()>,
    thread: std::thread::JoinHandle<Result<Vec<f32>, String>>,
}

impl Recorder {
    pub fn start(device: &str) -> Result<Recorder, String> {
        let device = device.to_string();
        let (stop_tx, stop_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || -> Result<Vec<f32>, String> {
            let buf = Arc::new(Mutex::new(Vec::<f32>::with_capacity(RATE * 60)));
            let (stream, rate, channels) = match open(&device, buf.clone()) {
                Ok(s) => {
                    let _ = ready_tx.send(Ok(()));
                    s
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e.clone()));
                    return Err(e);
                }
            };
            let _ = stop_rx.recv();
            std::thread::sleep(TAIL); // let the last block arrive before closing
            drop(stream);
            let raw = std::mem::take(&mut *buf.lock().unwrap());
            Ok(to_16k_mono(&raw, rate, channels))
        });
        ready_rx.recv().map_err(|_| "audio thread died".to_string())??;
        Ok(Recorder { stop: stop_tx, thread })
    }

    pub fn stop(self) -> Result<Vec<f32>, String> {
        let _ = self.stop.send(());
        self.thread.join().map_err(|_| "audio thread panicked".to_string())?
    }
}

fn open(device: &str, buf: Arc<Mutex<Vec<f32>>>) -> Result<(cpal::Stream, u32, u16), String> {
    let host = host();
    let dev = if device.is_empty() {
        host.default_input_device()
    } else {
        device.parse().ok().and_then(|id| host.device_by_id(&id)).or_else(|| host.default_input_device())
    }
    .ok_or("no microphone found")?;
    let err = |e: cpal::Error| eprintln!("audio: {e}");
    // Ask for exactly what the model wants; the sound server resamples. Fall back to the device's own format.
    // Small fixed blocks (50 ms): with the default the server hands over audio in ~1 s blocks, and the block still
    // filling when the stream closes was lost -> up to 1 s cut off the end of every take.
    let want = cpal::StreamConfig { channels: 1, sample_rate: RATE as u32, buffer_size: cpal::BufferSize::Fixed(BLOCK) };
    let b = buf.clone();
    if let Ok(s) = dev.build_input_stream(want, move |d: &[f32], _: &_| b.lock().unwrap().extend_from_slice(d), err, None)
    {
        s.play().map_err(|e| e.to_string())?;
        return Ok((s, RATE as u32, 1));
    }
    let def = dev.default_input_config().map_err(|e| e.to_string())?;
    let mut cfg: cpal::StreamConfig = def.config();
    cfg.buffer_size = cpal::BufferSize::Fixed(cfg.sample_rate / 20);
    let (rate, ch) = (cfg.sample_rate, cfg.channels);
    let s = match def.sample_format() {
        cpal::SampleFormat::F32 => {
            dev.build_input_stream(cfg.clone(), move |d: &[f32], _: &_| buf.lock().unwrap().extend_from_slice(d), err, None)
        }
        cpal::SampleFormat::I16 => dev.build_input_stream(
            cfg.clone(),
            move |d: &[i16], _: &_| buf.lock().unwrap().extend(d.iter().map(|&v| v as f32 / 32768.0)),
            err,
            None,
        ),
        f => return Err(format!("unsupported sample format {f:?}")),
    }
    .map_err(|e| e.to_string())?;
    s.play().map_err(|e| e.to_string())?;
    Ok((s, rate, ch))
}

/// Downmix + linear resample (only used when the device can't give 16 kHz mono itself).
fn to_16k_mono(x: &[f32], rate: u32, channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    let mono: Vec<f32> =
        if ch == 1 { x.to_vec() } else { x.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect() };
    if rate as usize == RATE || mono.is_empty() {
        return mono;
    }
    let step = rate as f64 / RATE as f64;
    let n = (mono.len() as f64 / step) as usize;
    (0..n)
        .map(|i| {
            let p = i as f64 * step;
            let (j, f) = (p as usize, (p.fract()) as f32);
            let a = mono[j];
            let b = *mono.get(j + 1).unwrap_or(&a);
            a + (b - a) * f
        })
        .collect()
}

/// Start / stop cue. Generated in code, so the app needs no sound files.
#[derive(Clone, Copy, Debug)]
pub enum Cue {
    /// Two rising notes.
    Start,
    /// Two falling notes.
    Stop,
}

const CUE_RATE: u32 = 48000;

/// Two short sine notes with soft edges (no clicks).
fn cue_samples(cue: Cue) -> Vec<f32> {
    let (a, b) = match cue {
        Cue::Start => (660.0, 990.0),
        Cue::Stop => (990.0, 660.0),
    };
    let note = |freq: f32, secs: f32| {
        let n = (CUE_RATE as f32 * secs) as usize;
        let fade = (CUE_RATE as f32 * 0.008) as usize;
        (0..n).map(move |i| {
            let env = (i.min(n - 1 - i) as f32 / fade as f32).min(1.0);
            0.25 * env * (std::f32::consts::TAU * freq * i as f32 / CUE_RATE as f32).sin()
        })
    };
    note(a, 0.07).chain(std::iter::repeat_n(0.0, CUE_RATE as usize / 50)).chain(note(b, 0.09)).collect()
}

/// Play a cue without blocking (the output device is opened only while it plays).
pub fn play(cue: Cue) {
    std::thread::spawn(move || {
        let run = || -> Result<(), Box<dyn std::error::Error>> {
            let mut sink = rodio::DeviceSinkBuilder::open_default_sink()?;
            sink.log_on_drop(false);
            let player = rodio::Player::connect_new(sink.mixer());
            let rate = std::num::NonZero::new(CUE_RATE).unwrap();
            player.append(rodio::buffer::SamplesBuffer::new(std::num::NonZero::<u16>::MIN, rate, cue_samples(cue)));
            player.sleep_until_end();
            Ok(())
        };
        if let Err(e) = run() {
            eprintln!("sound {cue:?}: {e}");
        }
    });
}
