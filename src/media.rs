//! Audio and video files -> 16 kHz mono f32 for asr (`dictation transcribe`). symphonia (pure Rust) reads the file's
//! audio track: wav, mp3, flac, ogg/vorbis, opus, m4a/aac, alac, aiff, caf, and the sound of mp4/mov/mkv/webm videos
//! (the picture is skipped). Opus (most .webm, voice messages) via symphonia-opus (pure Rust; symphonia has none
//! yet). Not covered: AC-3, .avi -> an error naming the problem.
//! rubato's FFT resampler converts to 16 kHz while decoding, so a long 48 kHz stereo file never sits in memory at its
//! own rate (1 h at 16 kHz mono = 230 MB).
use crate::RATE;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::common::Tier;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

/// symphonia's codecs plus Opus. At the fallback tier, so a future symphonia Opus decoder takes over by itself.
static CODECS: LazyLock<CodecRegistry> = LazyLock::new(|| {
    let mut r = CodecRegistry::new();
    symphonia::default::register_enabled_codecs(&mut r);
    r.register_audio_decoder_at_tier::<symphonia_opus::OpusDecoder>(Tier::Fallback);
    r
});

/// Mono resampler (rubato FFT), any rate -> `to` (16 kHz for asr; 48 kHz for Opus), chunk by chunk.
pub(crate) struct To16k {
    rs: Option<Fft<f32>>, // None: already at `to`
    pending: Vec<f32>,
    inbuf: Vec<f32>,
    outbuf: Vec<f32>,
    out: Vec<f32>,
    frames_in: usize,
    rate: usize,
    to: usize,
}

impl To16k {
    pub(crate) fn new(rate: usize) -> Result<To16k, String> {
        Self::to(rate, RATE)
    }

    /// The output so far (taken: the next call returns only what came after). The resampler's start-up delay stays
    /// in (a few ms of silence at the very start; `finish` cuts it, streams don't need to).
    pub(crate) fn take(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.out)
    }

    fn to(rate: usize, to: usize) -> Result<To16k, String> {
        let rs = if rate == to {
            None
        } else {
            Some(Fft::<f32>::new(rate, to, 1024, 1, FixedSync::Both).map_err(|e| format!("resampler: {e}"))?)
        };
        let (inbuf, outbuf) = rs.as_ref().map_or((vec![], vec![]), |r| (vec![0.0; r.input_frames_max()], vec![0.0; r.output_frames_max()]));
        Ok(To16k { rs, pending: Vec::new(), inbuf, outbuf, out: Vec::new(), frames_in: 0, rate, to })
    }

    pub(crate) fn push(&mut self, x: &[f32]) -> Result<(), String> {
        self.frames_in += x.len();
        let Some(rs) = self.rs.as_mut() else {
            self.out.extend_from_slice(x);
            return Ok(());
        };
        self.pending.extend_from_slice(x);
        let n = rs.input_frames_next();
        let mut used = 0;
        while self.pending.len() - used >= n {
            self.inbuf[..n].copy_from_slice(&self.pending[used..used + n]);
            Self::chunk(rs, &self.inbuf, &mut self.outbuf, &mut self.out, None)?;
            used += n;
        }
        self.pending.drain(..used);
        Ok(())
    }

    fn chunk(rs: &mut Fft<f32>, inbuf: &[f32], outbuf: &mut [f32], out: &mut Vec<f32>, partial: Option<usize>) -> Result<(), String> {
        let (ni, no) = (rs.input_frames_next(), rs.output_frames_next());
        let input = InterleavedSlice::new(&inbuf[..ni], 1, ni).map_err(|e| e.to_string())?;
        let mut output = InterleavedSlice::new_mut(&mut outbuf[..no], 1, no).map_err(|e| e.to_string())?;
        let idx = Indexing { partial_len: partial, ..Default::default() };
        let (_, produced) = rs.process_into_buffer(&input, &mut output, Some(&idx)).map_err(|e| format!("resample: {e}"))?;
        out.extend_from_slice(&outbuf[..produced]);
        Ok(())
    }

    /// The rest of the input, then silence until the resampler's delay is out; that delay is cut off the start.
    fn finish(mut self) -> Result<Vec<f32>, String> {
        let Some(rs) = self.rs.as_mut() else { return Ok(self.out) };
        let want = (self.frames_in as u64 * self.to as u64).div_ceil(self.rate as u64) as usize;
        let delay = rs.output_delay();
        let mut partial = Some(self.pending.len());
        self.inbuf.fill(0.0);
        self.inbuf[..self.pending.len()].copy_from_slice(&self.pending);
        while self.out.len() < want + delay {
            Self::chunk(rs, &self.inbuf, &mut self.outbuf, &mut self.out, partial)?;
            partial = Some(0);
        }
        self.out.drain(..delay);
        self.out.truncate(want);
        Ok(self.out)
    }
}

/// Mono samples at `rate` -> 16 kHz (also used for recordings).
pub(crate) fn mono_to_16k(x: &[f32], rate: usize) -> Result<Vec<f32>, String> {
    let mut c = To16k::new(rate)?;
    c.push(x)?;
    c.finish()
}

/// Mono samples at `rate` -> an Ogg Opus file (`opus-pure`, pure Rust), fullband ~48 kbit/s: the recordings'
/// full-quality copy (the 16 kHz wav next to it is what asr heard). Opus takes 8/12/16/24/48 kHz; any other rate
/// (44.1 kHz USB mics) is resampled to 48 kHz first.
pub(crate) fn write_opus(path: &Path, x: &[f32], rate: u32) -> Result<(), String> {
    use opus_pure::{Application, MAX_PACKET_BYTES, OggOpusWriter, OpusEncoder, OpusHead};
    let e = |e: opus_pure::Error| e.to_string();
    let resampled;
    let (x, rate) = if [8000, 12000, 16000, 24000, 48000].contains(&rate) {
        (x, rate)
    } else {
        let mut c = To16k::to(rate as usize, 48000)?;
        c.push(x)?;
        resampled = c.finish()?;
        (&resampled[..], 48000)
    };
    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).map_err(e)?;
    enc.bitrate_bps = 48_000;
    let head = OpusHead::for_encoder(&enc, rate);
    // As opus-pure's example: encode the encoder's lookahead too (zeros), then end the last page exactly at the
    // audio's end (granule = pre-skip + length, in 48 kHz ticks).
    let (frame, ticks) = (rate as usize / 50, 48000 / rate as usize);
    let frames = (x.len() + (head.pre_skip as usize).div_ceil(ticks)).div_ceil(frame);
    let end = u64::from(head.pre_skip) + (x.len() * ticks) as u64;
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut w = OggOpusWriter::new(std::io::BufWriter::new(file), head).map_err(e)?;
    let (mut packet, mut block) = (vec![0u8; MAX_PACKET_BYTES], vec![0.0f32; frame]);
    for i in 0..frames {
        let s = (i * frame).min(x.len());
        let t = (s + frame).min(x.len());
        block[..t - s].copy_from_slice(&x[s..t]);
        block[t - s..].fill(0.0);
        let n = enc.encode(&block, frame, &mut packet).map_err(e)?;
        if i + 1 == frames {
            w.write_packet_with_duration(&packet[..n], (end - w.granule() as u64) as u32).map_err(e)?;
        } else {
            w.write_packet(&packet[..n]).map_err(e)?;
        }
    }
    w.finish().map_err(e)?;
    Ok(())
}

/// A file that tells how far into it reading has got (bytes): `read_progress`.
struct Tracked {
    file: std::fs::File,
    pos: Arc<AtomicU64>,
}

impl Read for Tracked {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(buf)?;
        self.pos.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

impl Seek for Tracked {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let at = self.file.seek(to)?;
        self.pos.store(at, Ordering::Relaxed);
        Ok(at)
    }
}

impl symphonia::core::io::MediaSource for Tracked {
    fn is_seekable(&self) -> bool {
        self.file.is_seekable()
    }
    fn byte_len(&self) -> Option<u64> {
        self.file.byte_len()
    }
}

/// The file's first audio track as 16 kHz mono (channels averaged).
pub fn read(path: &Path) -> Result<Vec<f32>, String> {
    read_progress(path, |_| {})
}

/// `read` with `progress(done 0..1)` after every packet: the share of the file's bytes read (audio and video
/// tracks are interleaved through the file, so it runs with the audio).
pub fn read_progress(path: &Path, mut progress: impl FnMut(f64)) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = file.metadata().map_or(0, |m| m.len()).max(1) as f64;
    let pos = Arc::new(AtomicU64::new(0));
    let mss = MediaSourceStream::new(Box::new(Tracked { file, pos: pos.clone() }), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let unsupported = |what: &str| format!("{}: {what} (supported: wav, mp3, flac, ogg, m4a/aac, opus, mp4, mov, mkv, webm)", path.display());
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|_| unsupported("not a format this app reads"))?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| unsupported("no audio track"))?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or_else(|| unsupported("no audio track"))?;
    let track_id = track.id;
    let mut decoder = CODECS
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|_| unsupported("its sound codec is not supported (AC-3, ...)"))?;
    let mut conv: Option<To16k> = None;
    let (mut inter, mut mono) = (Vec::<f32>::new(), Vec::<f32>::new());
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // Chained streams (a new track list): the first one is the recording.
            Err(Error::ResetRequired) => break,
            // Some files end without a clean end marker.
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        progress((pos.load(Ordering::Relaxed) as f64 / len).min(1.0));
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(Error::DecodeError(_)) => continue, // one damaged packet: skip it, as players do
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let ch = buf.spec().channels().count().max(1);
        let rate = buf.spec().rate() as usize;
        let c = match conv.as_mut() {
            Some(c) => c,
            None => conv.insert(To16k::new(rate)?),
        };
        buf.copy_to_vec_interleaved(&mut inter);
        mono.clear();
        mono.extend(inter.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        c.push(&mono)?;
    }
    match conv {
        Some(c) => c.finish(),
        None => Err(format!("{}: no audio in the file", path.display())),
    }
}
