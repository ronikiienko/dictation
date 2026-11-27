// What the overlay shows and how it's drawn: the same on every platform (included by overlay/mod.rs; also by the
// Linux type-check crate, so no `crate::` paths here). Pixels: tiny-skia, premultiplied RGBA.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

/// What the overlay shows.
#[derive(Clone, PartialEq, Debug)]
pub enum View {
    Hidden,
    /// A take started, the microphone gives no sound yet (flat, dim bars).
    Starting,
    /// Bars following the microphone's loudness.
    Recording,
    /// The take is being transcribed: pulsing dots, or a progress line (percent) for takes over 30 s.
    Transcribing(Option<u32>),
    /// Live subtitles: the text so far (its last two lines are shown).
    Subtitles(String),
}

/// Size in logical pixels (points).
pub const WIDTH: f32 = 136.0;
pub const HEIGHT: f32 = 40.0;
/// Size of the subtitles box, its text size and line height (points).
pub const SUB_WIDTH: f32 = 640.0;
pub const SUB_HEIGHT: f32 = 76.0;
const TEXT_SIZE: f32 = 20.0;
const LINE: f32 = 26.0;
/// Fonts tried for subtitles (proof of concept: the system's own, nothing bundled). Cyrillic needed.
const FONTS: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
];
/// Distance from the bottom of the screen (above the Dock / panel).
pub const BOTTOM: f32 = 96.0;
/// Redraw interval while shown.
pub const FRAME: Duration = Duration::from_millis(33);
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
    /// Loaded on the first subtitles (None inside: no font found).
    text: Option<Option<Text>>,
}

impl Default for Painter {
    fn default() -> Self {
        Painter { view: View::Hidden, bars: [0.0; BARS], last_step: Instant::now(), since: Instant::now(), text: None }
    }
}

/// A font and its rasterized glyphs at one size.
struct Text {
    font: fontdue::Font,
    px: f32,
    glyphs: HashMap<char, (fontdue::Metrics, Vec<u8>)>,
}

impl Text {
    fn load() -> Option<Text> {
        let font = FONTS.iter().find_map(|p| {
            let bytes = std::fs::read(p).ok()?;
            fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()).ok()
        });
        if font.is_none() {
            eprintln!("subtitles: no font found (tried {FONTS:?})");
        }
        Some(Text { font: font?, px: 0.0, glyphs: HashMap::new() })
    }

    fn advance(&self, c: char) -> f32 {
        self.font.metrics(c, self.px).advance_width
    }

    fn width(&self, s: &str) -> f32 {
        s.chars().map(|c| self.advance(c)).sum()
    }

    /// `s` wrapped at word boundaries to `max` pixels wide.
    fn wrap(&self, s: &str, max: f32) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut line = String::new();
        for w in s.split_whitespace() {
            let candidate = if line.is_empty() { w.to_string() } else { format!("{line} {w}") };
            if !line.is_empty() && self.width(&candidate) > max {
                lines.push(std::mem::replace(&mut line, w.to_string()));
            } else {
                line = candidate;
            }
        }
        if !line.is_empty() {
            lines.push(line);
        }
        lines
    }

    /// White text, its baseline at (x, y) pixels, `alpha` 0..255.
    fn draw(&mut self, px: &mut Pixmap, s: &str, mut x: f32, y: f32, alpha: u8) {
        let (w, h) = (px.width() as i32, px.height() as i32);
        for c in s.chars() {
            let size = self.px;
            let (m, bitmap) = self.glyphs.entry(c).or_insert_with(|| self.font.rasterize(c, size));
            let (gx, gy) = (x.round() as i32 + m.xmin, (y.round() as i32) - m.height as i32 - m.ymin);
            let data = px.data_mut();
            for row in 0..m.height as i32 {
                for col in 0..m.width as i32 {
                    let (px_x, px_y) = (gx + col, gy + row);
                    if px_x < 0 || px_y < 0 || px_x >= w || px_y >= h {
                        continue;
                    }
                    let cov = bitmap[(row * m.width as i32 + col) as usize] as u32 * alpha as u32 / 255;
                    let i = ((px_y * w + px_x) * 4) as usize;
                    // White over premultiplied RGBA.
                    for k in 0..4 {
                        data[i + k] = (cov + data[i + k] as u32 * (255 - cov) / 255) as u8;
                    }
                }
            }
            x += m.advance_width;
        }
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
        if !(matches!(self.view, View::Transcribing(_)) && matches!(v, View::Transcribing(_))) {
            self.since = Instant::now();
        }
        self.view = v;
        true
    }

    pub fn visible(&self) -> bool {
        self.view != View::Hidden
    }

    /// Size of what's shown, points.
    pub fn size(&self) -> (f32, f32) {
        if matches!(self.view, View::Subtitles(_)) { (SUB_WIDTH, SUB_HEIGHT) } else { (WIDTH, HEIGHT) }
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
        let (ww, hh) = self.size();
        let (w, h) = (ww * scale, hh * scale);
        let mut px = Pixmap::new(w.ceil() as u32, h.ceil() as u32)?;
        let t = Transform::from_scale(scale, scale);

        // The pill: dark, slightly see-through, a faint edge. Subtitles: a box with round corners.
        let radius = if ww == WIDTH { (HEIGHT - 1.0) / 2.0 } else { 14.0 };
        let pill = rounded(0.5, 0.5, ww - 1.0, hh - 1.0, radius)?;
        px.fill_path(&pill, &paint(Color::from_rgba8(18, 18, 20, 220)), FillRule::Winding, t, None);
        let edge = Stroke { width: 1.0, ..Default::default() };
        px.stroke_path(&pill, &paint(Color::from_rgba8(255, 255, 255, 40)), &edge, t, None);

        let white = |a: u8| paint(Color::from_rgba8(255, 255, 255, a));
        match &self.view {
            View::Hidden => {}
            View::Subtitles(s) => {
                let text = self.text.get_or_insert_with(Text::load);
                if let Some(text) = text {
                    if text.px != TEXT_SIZE * scale {
                        text.px = TEXT_SIZE * scale;
                        text.glyphs.clear();
                    }
                    let pad = 16.0 * scale;
                    let lines = text.wrap(s, w - 2.0 * pad);
                    let ascent = text.font.horizontal_line_metrics(text.px).map_or(text.px * 0.8, |m| m.ascent);
                    let top = (h - 2.0 * LINE * scale) / 2.0 + (LINE * scale - text.px) / 2.0 + ascent;
                    if lines.is_empty() {
                        text.draw(&mut px, "Listening…", pad, top, 110);
                    }
                    for (i, line) in lines.iter().rev().take(2).rev().enumerate() {
                        text.draw(&mut px, line, pad, top + i as f32 * LINE * scale, 255);
                    }
                }
            }
            View::Starting | View::Recording => {
                let (bw, gap) = (3.5, 3.0);
                let total = BARS as f32 * bw + (BARS - 1) as f32 * gap;
                let x0 = (WIDTH - total) / 2.0;
                let alpha = if matches!(self.view, View::Starting) { 90 } else { 255 };
                // The bars reach into the round ends; there they're cut off by the curve (7 pt inside it).
                let r = HEIGHT / 2.0;
                for (i, &v) in self.bars.iter().enumerate() {
                    let x = x0 + i as f32 * (bw + gap);
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
            &View::Transcribing(Some(pct)) => {
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
