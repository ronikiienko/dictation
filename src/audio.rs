//! Recording (microphone or system audio; cpal, PulseAudio protocol -> PipeWire on Linux) and short sounds (rodio).
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::RATE;

/// Keep recording this long after stop is pressed, so the block in flight is delivered.
const TAIL: Duration = Duration::from_millis(150);

/// A microphone that still gives only exact zeros after this long is taken as live anyway (muted, no permission):
/// the take then goes on as before instead of hanging in "starting".
const WAKE_LIMIT: Duration = Duration::from_millis(2500);

fn host() -> cpal::Host {
    #[cfg(target_os = "linux")]
    if let Ok(h) = cpal::host_from_id(cpal::HostId::PulseAudio) {
        return h;
    }
    cpal::default_host()
}

/// Device id meaning "record what the computer plays" (the default output device) instead of a microphone.
pub const SYSTEM_AUDIO: &str = "system-audio";

/// (id, human name) of every input device; id "" = system default. On macOS and Windows also SYSTEM_AUDIO (on
/// Linux the sound server already lists "Monitor of …" sources for that).
pub fn input_devices() -> Vec<(String, String)> {
    let mut out = vec![(String::new(), "System default".to_string())];
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    out.push((SYSTEM_AUDIO.to_string(), "System audio (what's playing)".to_string()));
    if let Ok(devs) = host().input_devices() {
        for d in devs {
            let (Ok(id), Ok(desc)) = (d.id(), d.description()) else { continue };
            out.push((id.to_string(), desc.name().to_string()));
        }
    }
    out
}

/// A running recording. Audio is captured on its own thread (cpal streams are not Send); stop() returns the take.
pub struct Recorder {
    stop: mpsc::Sender<()>,
    thread: std::thread::JoinHandle<Result<Captured, String>>,
}

/// A stopped take.
pub struct Captured {
    /// 16 kHz mono: what asr gets.
    pub audio: Vec<f32>,
    /// Mono at the device's own rate (`rate`), for the full-quality copy of the recording.
    pub full: Vec<f32>,
    pub rate: u32,
}

impl Recorder {
    /// `on_live` is called once, from the recording thread, when the microphone delivers real sound. Until then a
    /// Bluetooth headset is still switching to its headset mode (0.4-1.6 s on Bluetooth headphones) and macOS hands over exact
    /// zeros: those are dropped, and the app shows "recording" only from `on_live` on, so speech isn't lost in them.
    /// System audio is live at once (silence there is real zeros); after WAKE_LIMIT a mic counts as live anyway.
    pub fn start(device: &str, on_live: impl FnOnce() + Send + 'static) -> Result<Recorder, String> {
        let device = device.to_string();
        let (stop_tx, stop_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let buf = Arc::new(Mutex::new(Vec::<f32>::with_capacity(RATE * 60)));
        let live = Arc::new(AtomicBool::new(device == SYSTEM_AUDIO));
        let (b, l) = (buf, live);
        let thread = std::thread::spawn(move || -> Result<Captured, String> {
            let (stream, rate, channels) = match open(&device, b.clone(), l.clone()) {
                Ok(s) => {
                    let _ = ready_tx.send(Ok(()));
                    s
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e.clone()));
                    return Err(e);
                }
            };
            let t = Instant::now();
            let mut on_live = Some(on_live);
            loop {
                match stop_rx.recv_timeout(Duration::from_millis(10)) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    _ => break,
                }
                if on_live.is_some() && (l.load(Ordering::Relaxed) || t.elapsed() >= WAKE_LIMIT) {
                    l.store(true, Ordering::Relaxed);
                    println!("microphone live after {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
                    on_live.take().unwrap()();
                }
            }
            std::thread::sleep(TAIL); // let the last block arrive before closing
            drop(stream);
            let raw = std::mem::take(&mut *b.lock().unwrap());
            let full = mono(&raw, channels);
            Ok(Captured { audio: to_16k(&full, rate), full, rate })
        });
        ready_rx.recv().map_err(|_| "audio thread died".to_string())??;
        Ok(Recorder { stop: stop_tx, thread })
    }

    pub fn stop(self) -> Result<Captured, String> {
        let _ = self.stop.send(());
        self.thread.join().map_err(|_| "audio thread panicked".to_string())?
    }
}

fn open(device: &str, buf: Arc<Mutex<Vec<f32>>>, live: Arc<AtomicBool>) -> Result<(cpal::Stream, u32, u16), String> {
    let host = host();
    // System audio: an input stream on the output device. cpal records it with a Core Audio process tap on macOS
    // (14.6+, asks for "System Audio Recording" permission; denied = silence, not an error), WASAPI loopback on Windows.
    let system = device == SYSTEM_AUDIO;
    let dev = if system {
        host.default_output_device()
    } else if device.is_empty() {
        host.default_input_device()
    } else {
        device.parse().ok().and_then(|id| host.device_by_id(&id)).or_else(|| host.default_input_device())
    }
    .ok_or(if system { "no output device found" } else { "no microphone found" })?;
    let err = |e: cpal::Error| eprintln!("audio: {e}");
    // Record in the format the device already runs at and convert in the app (to_16k). Never ask for another
    // format: on macOS cpal applies the requested rate to the device itself, for every app, and leaves it there
    // (16 kHz on Bluetooth headphones = headset/call quality, even after the app quits).
    let def = if system { dev.default_output_config() } else { dev.default_input_config() }.map_err(|e| e.to_string())?;
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut cfg: cpal::StreamConfig = def.config();
    // Linux (PulseAudio protocol): the block size is a property of our stream only. Without it the server hands over
    // ~1 s blocks and the block still filling when the stream closes was lost (up to 1 s cut off every take); ask
    // for 50 ms. Elsewhere it can be a device setting (macOS) -> keep the device's own.
    #[cfg(target_os = "linux")]
    {
        let mut frames = cfg.sample_rate / 20;
        if let cpal::SupportedBufferSize::Range { min, max } = *def.buffer_size() {
            frames = frames.clamp(min, max);
        }
        cfg.buffer_size = cpal::BufferSize::Fixed(frames);
    }
    let (rate, ch) = (cfg.sample_rate, cfg.channels);
    // Appends a block; before the mic is live, from its first non-zero value on (whole frames), dropping the rest.
    let keep = move |d: &mut dyn Iterator<Item = f32>| {
        let mut b = buf.lock().unwrap();
        if live.load(Ordering::Relaxed) {
            b.extend(d);
            return;
        }
        let x: Vec<f32> = d.collect();
        if let Some(i) = x.iter().position(|&v| v != 0.0) {
            live.store(true, Ordering::Relaxed);
            b.extend_from_slice(&x[i - i % ch as usize..]);
        }
    };
    // Any format the device runs at, converted to f32 (cpal's sample conversion). E.g. laptop mics on PipeWire
    // run at s32.
    fn stream<T>(
        dev: &cpal::Device,
        cfg: cpal::StreamConfig,
        keep: impl Fn(&mut dyn Iterator<Item = f32>) + Send + 'static,
        err: impl FnMut(cpal::Error) + Send + 'static,
    ) -> Result<cpal::Stream, cpal::Error>
    where
        T: cpal::SizedSample,
        f32: cpal::FromSample<T>,
    {
        dev.build_input_stream(cfg, move |d: &[T], _: &_| keep(&mut d.iter().map(|v| v.to_sample::<f32>())), err, None)
    }
    use cpal::{I24, SampleFormat as F, U24};
    let s = match def.sample_format() {
        F::F32 => stream::<f32>(&dev, cfg.clone(), keep, err),
        F::F64 => stream::<f64>(&dev, cfg.clone(), keep, err),
        F::I8 => stream::<i8>(&dev, cfg.clone(), keep, err),
        F::I16 => stream::<i16>(&dev, cfg.clone(), keep, err),
        F::I24 => stream::<I24>(&dev, cfg.clone(), keep, err),
        F::I32 => stream::<i32>(&dev, cfg.clone(), keep, err),
        F::I64 => stream::<i64>(&dev, cfg.clone(), keep, err),
        F::U8 => stream::<u8>(&dev, cfg.clone(), keep, err),
        F::U16 => stream::<u16>(&dev, cfg.clone(), keep, err),
        F::U24 => stream::<U24>(&dev, cfg.clone(), keep, err),
        F::U32 => stream::<u32>(&dev, cfg.clone(), keep, err),
        F::U64 => stream::<u64>(&dev, cfg.clone(), keep, err),
        f => return Err(format!("unsupported sample format {f:?}")),
    }
    .map_err(|e| e.to_string())?;
    s.play().map_err(|e| e.to_string())?;
    Ok((s, rate, ch))
}

/// Downmix (average of channels) + resample to 16 kHz (rubato, band-limited, so nothing above 8 kHz folds back
/// into the speech band).
fn mono(x: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 { x.to_vec() } else { x.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect() }
}

fn to_16k(mono: &[f32], rate: u32) -> Vec<f32> {
    if rate as usize == RATE || mono.is_empty() {
        return mono.to_vec();
    }
    crate::media::mono_to_16k(mono, rate as usize).unwrap_or_else(|e| {
        eprintln!("audio: {e}");
        Vec::new()
    })
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

/// Play a cue without blocking (the output device is opened only while it plays). If rodio's default (a fixed buffer
/// of ~50 ms, 2048 frames at 44.1 kHz) is refused, the device picks its own buffer size: Bluetooth headphones in
/// headset mode (switched to it as soon as their mic records, so at every start cue) allow 14..=1024 frames.
pub fn play(cue: Cue) {
    std::thread::spawn(move || {
        let open = || {
            rodio::DeviceSinkBuilder::open_default_sink().or_else(|_| {
                rodio::DeviceSinkBuilder::from_default_device()?.with_buffer_size(cpal::BufferSize::Default).open_stream()
            })
        };
        let run = || -> Result<(), Box<dyn std::error::Error>> {
            let mut sink = open()?;
            sink.log_on_drop(false);
            let player = rodio::Player::connect_new(sink.mixer());
            let rate = std::num::NonZero::new(CUE_RATE).unwrap();
            player.append(rodio::buffer::SamplesBuffer::new(std::num::NonZero::<u16>::MIN, rate, cue_samples(cue)));
            player.sleep_until_end();
            Ok(())
        };
        if let Err(e) = run() {
            let mut msg = e.to_string();
            let mut src = e.source();
            while let Some(s) = src {
                msg += &format!(": {s}");
                src = s.source();
            }
            eprintln!("sound {cue:?}: {msg}");
        }
    });
}
