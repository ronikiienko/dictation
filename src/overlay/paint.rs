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
}

/// Size in logical pixels (points).
pub const WIDTH: f32 = 136.0;
pub const HEIGHT: f32 = 40.0;
/// Distance from the bottom of the screen (above the Dock / panel).
pub const BOTTOM: f32 = 96.0;
/// Redraw interval while shown.
pub const FRAME: Duration = Duration::from_millis(33);
/// A new bar every STEP: BARS * STEP = how much time the bars show (0.75 s).
const BARS: usize = 15;
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
}

impl Default for Painter {
    fn default() -> Self {
        Painter { view: View::Hidden, bars: [0.0; BARS], last_step: Instant::now(), since: Instant::now() }
    }
}

impl Painter {
    /// Show this; true if it changed. A new take starts with flat bars.
    pub fn set(&mut self, v: View) -> bool {
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

    /// The next frame, `scale` pixels per point.
    pub fn frame(&mut self, scale: f32) -> Option<Pixmap> {
        let now = Instant::now();
        if now.duration_since(self.last_step) >= STEP {
            self.last_step = now;
            let peak = f32::from_bits(PEAK.swap(0, Ordering::Relaxed));
            self.bars.rotate_left(1);
            self.bars[BARS - 1] = if self.view == View::Recording { norm(peak) } else { 0.0 };
        }
        let (w, h) = (WIDTH * scale, HEIGHT * scale);
        let mut px = Pixmap::new(w.ceil() as u32, h.ceil() as u32)?;
        let t = Transform::from_scale(scale, scale);

        // The pill: dark, slightly see-through, a faint edge.
        let pill = rounded(0.5, 0.5, WIDTH - 1.0, HEIGHT - 1.0, (HEIGHT - 1.0) / 2.0)?;
        px.fill_path(&pill, &paint(Color::from_rgba8(18, 18, 20, 220)), FillRule::Winding, t, None);
        let edge = Stroke { width: 1.0, ..Default::default() };
        px.stroke_path(&pill, &paint(Color::from_rgba8(255, 255, 255, 40)), &edge, t, None);

        let white = |a: u8| paint(Color::from_rgba8(255, 255, 255, a));
        match self.view {
            View::Hidden => {}
            View::Starting | View::Recording => {
                let (bw, gap) = (3.5, 3.0);
                let total = BARS as f32 * bw + (BARS - 1) as f32 * gap;
                let x0 = (WIDTH - total) / 2.0;
                let alpha = if self.view == View::Starting { 90 } else { 255 };
                for (i, &v) in self.bars.iter().enumerate() {
                    let bh = 3.0 + v * (HEIGHT - 14.0);
                    let x = x0 + i as f32 * (bw + gap);
                    if let Some(p) = rounded(x, (HEIGHT - bh) / 2.0, bw, bh, bw / 2.0) {
                        px.fill_path(&p, &white(alpha), FillRule::Winding, t, None);
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
