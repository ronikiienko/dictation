//! Long takes: cut at pauses into pieces of at most 30 s.
//! Ported from transcribe.cpp `src/arch/parakeet/longform.cpp` + `run_longform` in `model.cpp` (MIT, (c) 2026 The
//! transcribe.cpp authors). Speech probabilities come from the model's own head on the subsampler output, computed
//! per scan block (120 s).

/// [start, end) in seconds.
pub type Regions = Vec<(f64, f64)>;

pub struct Params {
    pub sample_rate: usize,
    pub frame_seconds: f64,
    pub speech_threshold: f32,
    pub min_speech_seconds: f64,
    pub min_gap_seconds: f64,
    pub max_segment_seconds: f64,
    pub min_segment_seconds: f64,
    pub min_pause_seconds: f64,
    pub scan_block_seconds: f64,
    pub boundary_window_secs: f64,
    pub min_tail_seconds: f64,
    pub min_feature_samples: usize,
}

const PAUSE_EPS: f64 = 1e-6;

fn round_half_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

impl Params {
    fn samples_for(&self, seconds: f64) -> i64 {
        round_half_even(seconds * self.sample_rate as f64)
    }
    pub fn fits_one_segment(&self, n: usize) -> bool {
        n as i64 <= self.samples_for(self.max_segment_seconds)
    }
}

pub fn speech_regions(probs: &[f32], p: &Params) -> Regions {
    let mut regions: Regions = Vec::new();
    let mut run_start: Option<usize> = None;
    for i in 0..=probs.len() {
        let speaking = i < probs.len() && probs[i] >= p.speech_threshold;
        match (speaking, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(s)) => {
                let (start, end) = (s as f64 * p.frame_seconds, i as f64 * p.frame_seconds);
                match regions.last_mut() {
                    Some(last) if start - last.1 < p.min_gap_seconds => last.1 = end,
                    _ => regions.push((start, end)),
                }
                run_start = None;
            }
            _ => {}
        }
    }
    regions.retain(|r| r.1 - r.0 >= p.min_speech_seconds);
    regions
}

fn pauses_from_speech(mut speech: Regions, duration: f64, p: &Params) -> Regions {
    let least = p.min_pause_seconds - PAUSE_EPS;
    speech.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut pauses = Vec::new();
    let mut previous = 0.0f64;
    for r in &speech {
        if r.0 - previous >= least {
            pauses.push((previous, r.0));
        }
        previous = previous.max(r.1);
    }
    if duration - previous >= least {
        pauses.push((previous, duration));
    }
    pauses
}

/// Middle of the latest pause lying within [min_segment, max_segment]; else the latest whose middle does; else a hard
/// cut at max_segment.
fn next_cut(pauses: &Regions, p: &Params) -> f64 {
    let inside = pauses.iter().filter(|r| r.0 >= p.min_segment_seconds && r.1 <= p.max_segment_seconds).last();
    let mid = |r: &(f64, f64)| (r.0 + r.1) / 2.0;
    let pick = inside.or_else(|| pauses.iter().filter(|r| (p.min_segment_seconds..=p.max_segment_seconds).contains(&mid(r))).last());
    pick.map_or(p.max_segment_seconds, mid)
}

fn has_speech(regions: &Regions, seconds: f64) -> bool {
    regions.iter().any(|r| r.1 > 0.0 && r.0 < seconds)
}

fn scan_block_sizes(n: i64, p: &Params) -> Vec<i64> {
    let max = p.samples_for(p.scan_block_seconds);
    let window = 4.max(p.samples_for(p.boundary_window_secs));
    let min_tail = window.max(p.samples_for(p.min_tail_seconds));
    let mut sizes = Vec::new();
    let mut emitted = 0;
    while emitted < n {
        let remaining = n - emitted;
        let b = if remaining <= max { remaining } else { max.min(remaining - min_tail) };
        sizes.push(b);
        emitted += b;
    }
    sizes
}

/// Pieces (start sample, length). `speech(start, len)` returns the speech regions of that block (seconds, relative
/// to the block start).
pub fn pause_segments(n: usize, p: &Params, mut speech: impl FnMut(usize, usize) -> Regions) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    if p.fits_one_segment(n) {
        out.push((0, n));
        return out;
    }
    let rate = p.sample_rate as f64;
    let cap = p.samples_for(p.max_segment_seconds);
    let mut regions: Regions = Vec::new();
    let (mut consumed, mut carry, mut block_start, mut cuts) = (0i64, 0i64, 0i64, 0);
    for block_len in scan_block_sizes(n as i64, p) {
        let offset = carry as f64 / rate;
        let block = if (block_len as usize) < p.min_feature_samples {
            vec![(0.0, block_len as f64 / rate)]
        } else {
            speech(block_start as usize, block_len as usize)
        };
        regions.extend(block.iter().map(|r| (r.0 + offset, r.1 + offset)));
        carry += block_len;
        block_start += block_len;
        while carry > cap {
            let cut = next_cut(&pauses_from_speech(regions.clone(), carry as f64 / rate, p), p);
            let index = round_half_even(cut * rate);
            let seconds = index as f64 / rate;
            if has_speech(&regions, seconds) {
                out.push((consumed as usize, index as usize));
            }
            consumed += index;
            carry -= index;
            cuts += 1;
            regions = regions.iter().filter(|r| r.1 > seconds).map(|r| ((r.0 - seconds).max(0.0), r.1 - seconds)).collect();
        }
    }
    if cuts > 0 && ((carry as usize) < p.min_feature_samples || !has_speech(&regions, carry as f64 / rate)) {
        return out;
    }
    out.push((consumed as usize, carry as usize));
    out
}
