// What the overlay shows and how it's drawn: the same on every platform (included by overlay/mod.rs; also by the
// Linux type-check crate, so no `crate::` paths here). Pixels: tiny-skia, premultiplied RGBA.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

/// What the overlay shows.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum View {
    Hidden,
    /// A take started, the microphone gives no sound yet (flat, dim bars).
    Starting,
    /// Bars following the microphone's loudness.
    Recording,
    /// The take is being transcribed: pulsing dots, or a progress line (percent) for takes over 30 s.
    Transcribing(Option<u32>),
    /// The take is done (its number, so it shows once): a mark for [`DONE_FOR`], then the overlay hides itself.
    Done(Outcome, u64),
}

/// How a take ended, as the overlay shows it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outcome {
    /// Text pasted, or put on the clipboard when that's the chosen output: a green check.
    Text,
    /// Meant to be pasted, but the paste keys couldn't be pressed (macOS: no Accessibility permission): the text is
    /// on the clipboard. An amber check.
    Copied,
    /// Nothing recognized: a dim dash.
    Empty,
    /// Copying or transcribing failed: a red cross.
    Failed,
}

/// How long the done mark shows: drawn in over the first `MARK_IN`, the pill fades out over the last `FADE`.
pub const DONE_FOR: Duration = Duration::from_millis(650);
const MARK_IN: Duration = Duration::from_millis(160);
const FADE: Duration = Duration::from_millis(180);

/// Size in logical pixels (points).
pub const WIDTH: f32 = 136.0;
pub const HEIGHT: f32 = 40.0;
/// Distance from the bottom of the screen (above the Dock / panel).
pub const BOTTOM: f32 = 96.0;
/// Redraw interval while shown.
pub const FRAME: Duration = Duration::from_millis(16);
/// A new bar every STEP: BARS * STEP = how much time the bars show (0.95 s).
const BARS: usize = 19;
const STEP: Duration = Duration::from_millis(50);

/// The loudest sample (absolute value, f32 bits) since the overlay last took it. Positive f32s order like their
/// bits, so fetch_max works on them.
static PEAK: AtomicU32 = AtomicU32::new(0);

/// From the microphone callback: the loudest sample of a block.
pub fn level(peak: f32) {
    PEAK.fetch_max(peak.abs().to_bits(), Ordering::Relaxed);
}

/// Peak -> bar height 0..1: -50 dBFS (room noise) .. -10 dBFS (loud speech).
fn norm(peak: f32) -> f32 {
    let db = 20.0 * peak.max(1e-6).log10();
    ((db + 50.0) / 40.0).clamp(0.0, 1.0)
}

/// Keeps the bars and draws frames.
pub struct Painter {
    view: View,
    bars: [f32; BARS],
    last_step: Instant,
    since: Instant,
    /// The last take whose done mark was shown (each shows once).
    done: Option<u64>,
}

impl Default for Painter {
    fn default() -> Self {
        Painter { view: View::Hidden, bars: [0.0; BARS], last_step: Instant::now(), since: Instant::now(), done: None }
    }
}

impl Painter {
    /// Show this; true if it changed. A new take starts with flat bars.
    pub fn set(&mut self, mut v: View) -> bool {
        if let View::Done(_, n) = v {
            // Once per take, and only right after the pill showed that take (not out of nowhere).
            if self.done == Some(n) || !matches!(self.view, View::Transcribing(_)) {
                v = if matches!(self.view, View::Done(..)) { self.view } else { View::Hidden };
            } else {
                self.done = Some(n);
            }
        }
        if v == self.view {
            return false;
        }
        if matches!(v, View::Starting) || (matches!(v, View::Recording) && !matches!(self.view, View::Starting)) {
            self.bars = [0.0; BARS];
            PEAK.store(0, Ordering::Relaxed);
        }
        if !matches!((self.view, v), (View::Transcribing(_), View::Transcribing(_))) {
            self.since = Instant::now();
        }
        self.view = v;
        true
    }

    pub fn visible(&self) -> bool {
        self.view != View::Hidden
    }

    /// Hides the done mark once its time is up; true if it just did (the window should hide too).
    pub fn expire(&mut self) -> bool {
        if matches!(self.view, View::Done(..)) && self.since.elapsed() >= DONE_FOR {
            self.view = View::Hidden;
            return true;
        }
        false
    }

    /// The next frame, `scale` pixels per point.
    pub fn frame(&mut self, scale: f32) -> Option<Pixmap> {
        let now = Instant::now();
        // Steps on a fixed 50 ms grid (not "now"): frames rarely land exactly on a step, and restarting the clock at
        // the frame showed one frame without motion every step or so (the scroll stuttered). After a pause (hidden),
        // catch up at once.
        if now.duration_since(self.last_step) >= STEP * BARS as u32 {
            self.last_step = now - STEP;
        }
        while now.duration_since(self.last_step) >= STEP {
            self.last_step += STEP;
            let peak = f32::from_bits(PEAK.swap(0, Ordering::Relaxed));
            self.bars.rotate_left(1);
            self.bars[BARS - 1] = if self.view == View::Recording { norm(peak) } else { 0.0 };
        }
        let (w, h) = (WIDTH * scale, HEIGHT * scale);
        let mut px = Pixmap::new(w.ceil() as u32, h.ceil() as u32)?;
        let t = Transform::from_scale(scale, scale);

        // Fading out at the end of the done mark: everything, pill included.
        let shown = now.duration_since(self.since);
        let fade = match self.view {
            View::Done(..) => (DONE_FOR.saturating_sub(shown).as_secs_f32() / FADE.as_secs_f32()).min(1.0),
            _ => 1.0,
        };
        let rgba = |r: u8, g: u8, b: u8, a: u8| paint(Color::from_rgba8(r, g, b, (a as f32 * fade) as u8));

        // The pill: dark, slightly see-through, a faint edge.
        let pill = rounded(0.5, 0.5, WIDTH - 1.0, HEIGHT - 1.0, (HEIGHT - 1.0) / 2.0)?;
        px.fill_path(&pill, &rgba(18, 18, 20, 220), FillRule::Winding, t, None);
        let edge = Stroke { width: 1.0, ..Default::default() };
        px.stroke_path(&pill, &rgba(255, 255, 255, 40), &edge, t, None);

        let white = |a: u8| rgba(255, 255, 255, a);
        match self.view {
            View::Hidden => {}
            View::Starting | View::Recording => {
                let (bw, gap) = (3.5, 3.0);
                let total = BARS as f32 * bw + (BARS - 1) as f32 * gap;
                let x0 = (WIDTH - total) / 2.0;
                let alpha = if self.view == View::Starting { 90 } else { 255 };
                // The bars reach into the round ends; there they're cut off by the curve (7 pt inside it).
                let r = HEIGHT / 2.0;
                // Smooth scrolling: the row slides left between steps, the incoming bar (right end) follows the
                // loudest sample so far in this step; it becomes the newest bar at the next step.
                let frac = (now.duration_since(self.last_step).as_secs_f32() / STEP.as_secs_f32()).min(1.0);
                let live = if self.view == View::Recording { norm(f32::from_bits(PEAK.load(Ordering::Relaxed))) } else { 0.0 };
                for (i, &v) in self.bars.iter().chain(std::iter::once(&live)).enumerate() {
                    let x = x0 + (i as f32 - frac) * (bw + gap);
                    let cx = x + bw / 2.0;
                    let dx = (r - cx).max(cx - (WIDTH - r)).max(0.0);
                    let room = 2.0 * ((r * r - dx * dx).max(0.0).sqrt() - 7.0);
                    let bh = (3.0 + v * (HEIGHT - 14.0)).min(room.max(3.0));
                    // Fade in the round ends only: full on the straight part, about a third at the outermost bars.
                    let k = (dx / (r - x0 - bw / 2.0)).clamp(0.0, 1.0);
                    let fade = 1.0 - 0.65 * k * k * (3.0 - 2.0 * k);
                    let a = (alpha as f32 * fade) as u8;
                    if let Some(p) = rounded(x, (HEIGHT - bh) / 2.0, bw, bh, bw / 2.0) {
                        px.fill_path(&p, &white(a), FillRule::Winding, t, None);
                    }
                }
            }
            View::Transcribing(Some(pct)) => {
                let (lw, lh) = (WIDTH - 40.0, 4.0);
                let (x, y) = ((WIDTH - lw) / 2.0, (HEIGHT - lh) / 2.0);
                if let Some(p) = rounded(x, y, lw, lh, lh / 2.0) {
                    px.fill_path(&p, &white(60), FillRule::Winding, t, None);
                }
                let done = lw * pct.min(100) as f32 / 100.0;
                if let Some(p) = rounded(x, y, done.max(lh), lh, lh / 2.0) {
                    px.fill_path(&p, &white(255), FillRule::Winding, t, None);
                }
            }
            View::Done(o, _) => {
                // Drawn in like a pen stroke over MARK_IN.
                let k = (shown.as_secs_f32() / MARK_IN.as_secs_f32()).min(1.0);
                let (cx, cy) = (WIDTH / 2.0, HEIGHT / 2.0);
                let (strokes, color): (&[&[(f32, f32)]], Paint) = match o {
                    Outcome::Text => (&[&[(-7.5, 0.5), (-2.5, 5.5), (8.0, -6.0)]], rgba(48, 209, 88, 255)),
                    Outcome::Copied => (&[&[(-7.5, 0.5), (-2.5, 5.5), (8.0, -6.0)]], rgba(255, 179, 64, 255)),
                    Outcome::Empty => (&[&[(-7.0, 0.0), (7.0, 0.0)]], white(150)),
                    Outcome::Failed => (&[&[(-5.5, -5.5), (5.5, 5.5)], &[(5.5, -5.5), (-5.5, 5.5)]], rgba(255, 69, 58, 255)),
                };
                let pen = Stroke { width: 2.6, line_cap: tiny_skia::LineCap::Round, line_join: tiny_skia::LineJoin::Round, ..Default::default() };
                for pts in strokes {
                    if let Some(p) = partial(pts, k, cx, cy) {
                        px.stroke_path(&p, &color, &pen, t, None);
                    }
                }
            }
            View::Transcribing(None) => {
                // Three dots, a wave running through them.
                let s = now.duration_since(self.since).as_secs_f32();
                for i in 0..3 {
                    let phase = s * 6.0 - i as f32 * 0.9;
                    let k = 0.5 + 0.5 * phase.sin();
                    let r = 3.0 + 1.2 * k;
                    let x = WIDTH / 2.0 + (i as f32 - 1.0) * 14.0;
                    let mut pb = PathBuilder::new();
                    pb.push_circle(x, HEIGHT / 2.0, r);
                    if let Some(p) = pb.finish() {
                        px.fill_path(&p, &white((110.0 + 145.0 * k) as u8), FillRule::Winding, t, None);
                    }
                }
            }
        }
        Some(px)
    }
}

fn paint(c: Color) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color(c);
    p.anti_alias = true;
    p
}

/// The first `k` (0..1) of the polyline `pts` (offsets from `cx`, `cy`), by length.
fn partial(pts: &[(f32, f32)], k: f32, cx: f32, cy: f32) -> Option<tiny_skia::Path> {
    let seg = |i: usize| ((pts[i + 1].0 - pts[i].0).hypot(pts[i + 1].1 - pts[i].1), pts[i], pts[i + 1]);
    let total: f32 = (0..pts.len() - 1).map(|i| seg(i).0).sum();
    let mut left = total * k.max(0.01);
    let mut pb = PathBuilder::new();
    pb.move_to(cx + pts[0].0, cy + pts[0].1);
    for i in 0..pts.len() - 1 {
        let (len, a, b) = seg(i);
        let f = (left / len).min(1.0);
        pb.line_to(cx + a.0 + (b.0 - a.0) * f, cy + a.1 + (b.1 - a.1) * f);
        left -= len;
        if left <= 0.0 {
            break;
        }
    }
    pb.finish()
}

/// A rectangle with round corners (radius `r`), as one path.
fn rounded(x: f32, y: f32, w: f32, h: f32, r: f32) -> Option<tiny_skia::Path> {
    let r = r.min(w / 2.0).min(h / 2.0);
    let k = r * 0.552_284_8; // cubic Bézier handle for a quarter circle
    let (x1, y1) = (x + w, y + h);
    let mut pb = PathBuilder::new();
    pb.move_to(x + r, y);
    pb.line_to(x1 - r, y);
    pb.cubic_to(x1 - r + k, y, x1, y + r - k, x1, y + r);
    pb.line_to(x1, y1 - r);
    pb.cubic_to(x1, y1 - r + k, x1 - r + k, y1, x1 - r, y1);
    pb.line_to(x + r, y1);
    pb.cubic_to(x + r - k, y1, x, y1 - r + k, x, y1 - r);
    pb.line_to(x, y + r);
    pb.cubic_to(x, y + r - k, x + r - k, y, x + r, y);
    pb.close();
    pb.finish()
}
