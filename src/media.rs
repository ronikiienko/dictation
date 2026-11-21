//! Audio and video files -> 16 kHz mono f32 for asr (`dictation transcribe`). symphonia (pure Rust) reads the file's
//! audio track: wav, mp3, flac, ogg/vorbis, m4a/aac, alac, aiff, caf, and the sound of mp4/mov/mkv/webm videos
//! (the picture is skipped). Not covered: Opus (most .webm, voice messages), AC-3, .avi -> an error naming the codec.
//! rubato's FFT resampler converts to 16 kHz while decoding, so a long 48 kHz stereo file never sits in memory at its
//! own rate (1 h at 16 kHz mono = 230 MB).
use crate::RATE;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use std::path::Path;

/// Mono samples at the file's rate in, 16 kHz out, chunk by chunk.
struct To16k {
    rs: Option<Fft<f32>>, // None: the file is 16 kHz already
    pending: Vec<f32>,
    inbuf: Vec<f32>,
    outbuf: Vec<f32>,
    out: Vec<f32>,
    frames_in: usize,
    rate: usize,
}

impl To16k {
    fn new(rate: usize) -> Result<To16k, String> {
        let rs = if rate == RATE {
            None
        } else {
            Some(Fft::<f32>::new(rate, RATE, 1024, 1, FixedSync::Both).map_err(|e| format!("resampler: {e}"))?)
        };
        let (inbuf, outbuf) = rs.as_ref().map_or((vec![], vec![]), |r| (vec![0.0; r.input_frames_max()], vec![0.0; r.output_frames_max()]));
        Ok(To16k { rs, pending: Vec::new(), inbuf, outbuf, out: Vec::new(), frames_in: 0, rate })
    }

    fn push(&mut self, x: &[f32]) -> Result<(), String> {
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
        let want = (self.frames_in as u64 * RATE as u64).div_ceil(self.rate as u64) as usize;
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

/// The file's first audio track as 16 kHz mono (channels averaged).
pub fn read(path: &Path) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let unsupported = |what: &str| format!("{}: {what} (supported: wav, mp3, flac, ogg, m4a/aac, mp4, mov, mkv, webm without Opus)", path.display());
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|_| unsupported("not a format this app reads"))?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| unsupported("no audio track"))?;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or_else(|| unsupported("no audio track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|_| unsupported("its sound codec is not supported (Opus, AC-3, ...)"))?;
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
