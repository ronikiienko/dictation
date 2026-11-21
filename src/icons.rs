//! The tray icons, drawn in code (no image files): a coloured disc with a white symbol, readable on light and dark
//! panels (Linux, Windows); macOS: the symbol alone as a template image (`glyph_rgba`), like other menu bar icons.
//! Rendered at any size with 4x4 supersampling.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    /// Ready: microphone on grey.
    Idle,
    /// Microphone on red.
    Recording,
    /// Three dots on blue.
    Transcribing,
    /// Down arrow on teal.
    Downloading,
    /// Open ring on amber (one-time GPU tuning).
    Optimizing,
    /// "!" on orange (no model yet).
    Warning,
}

impl Icon {
    pub const ALL: [Icon; 6] = [Icon::Idle, Icon::Recording, Icon::Transcribing, Icon::Downloading, Icon::Optimizing, Icon::Warning];

    fn background(self) -> [u8; 3] {
        match self {
            Icon::Idle => [0x4d, 0x53, 0x5c],
            Icon::Recording => [0xdf, 0x35, 0x3b],
            Icon::Transcribing => [0x2f, 0x7c, 0xe0],
            Icon::Downloading => [0x16, 0x9c, 0x8f],
            Icon::Optimizing => [0xd9, 0x8c, 0x12],
            Icon::Warning => [0xe0, 0x62, 0x1a],
        }
    }

    /// Whether the white symbol covers point (x, y) of the unit square (y down).
    fn symbol(self, x: f32, y: f32) -> bool {
        match self {
            Icon::Idle | Icon::Recording => microphone(x, y),
            Icon::Transcribing => [0.29, 0.5, 0.71].iter().any(|&cx| dist(x, y, cx, 0.5) <= 0.075),
            Icon::Downloading => {
                segment(x, y, (0.5, 0.24), (0.5, 0.52), 0.055)
                    || triangle(x, y, (0.29, 0.47), (0.71, 0.47), (0.5, 0.7))
                    || segment(x, y, (0.3, 0.78), (0.7, 0.78), 0.04)
            }
            Icon::Optimizing => {
                // A ring with a gap at the top right, like a progress circle.
                let r = dist(x, y, 0.5, 0.5);
                let a = (y - 0.5).atan2(x - 0.5).to_degrees(); // -180..180, 0 = right, -90 = up
                (r - 0.22).abs() <= 0.055 && !(-90.0..=-10.0).contains(&a)
            }
            Icon::Warning => segment(x, y, (0.5, 0.24), (0.5, 0.56), 0.065) || dist(x, y, 0.5, 0.73) <= 0.07,
        }
    }

    /// `size` x `size` pixels, RGBA (straight alpha), rows top to bottom.
    pub fn rgba(self, size: u32) -> Vec<u8> {
        const SS: u32 = 4;
        let bg = self.background();
        let mut out = Vec::with_capacity((size * size * 4) as usize);
        for py in 0..size {
            for px in 0..size {
                let (mut disc, mut white) = (0u32, 0u32);
                for sy in 0..SS {
                    for sx in 0..SS {
                        let x = (px as f32 + (sx as f32 + 0.5) / SS as f32) / size as f32;
                        let y = (py as f32 + (sy as f32 + 0.5) / SS as f32) / size as f32;
                        if dist(x, y, 0.5, 0.5) <= 0.48 {
                            disc += 1;
                            if self.symbol(x, y) {
                                white += 1;
                            }
                        }
                    }
                }
                if disc == 0 {
                    out.extend_from_slice(&[0, 0, 0, 0]);
                    continue;
                }
                // Colour = mix of background and white over the disc; alpha = disc coverage.
                let w = white as f32 / disc as f32;
                let mix = |c: u8| (c as f32 + (255.0 - c as f32) * w).round() as u8;
                out.extend_from_slice(&[mix(bg[0]), mix(bg[1]), mix(bg[2]), (disc * 255 / (SS * SS)) as u8]);
            }
        }
        out
    }
}

impl Icon {
    /// macOS menu bar look: the symbol alone (no disc), scaled up to fill the bar's height. Black + alpha, to be shown
    /// as a template image (macOS tints it for light/dark menu bars), except Recording: a red symbol, not a template
    /// (macOS apps show recording in red). Returns (RGBA, template).
    pub fn glyph_rgba(self, size: u32) -> (Vec<u8>, bool) {
        const SS: u32 = 4;
        // The symbols are drawn inside the disc (about 0.18..0.82 of the square): zoom in so they span ~0.03..0.97.
        const ZOOM: f32 = 0.66;
        let template = self != Icon::Recording;
        let rgb = if template { [0, 0, 0] } else { self.background() };
        let mut out = Vec::with_capacity((size * size * 4) as usize);
        for py in 0..size {
            for px in 0..size {
                let mut hit = 0u32;
                for sy in 0..SS {
                    for sx in 0..SS {
                        let x = (px as f32 + (sx as f32 + 0.5) / SS as f32) / size as f32;
                        let y = (py as f32 + (sy as f32 + 0.5) / SS as f32) / size as f32;
                        if self.symbol(0.5 + (x - 0.5) * ZOOM, 0.5 + (y - 0.5) * ZOOM) {
                            hit += 1;
                        }
                    }
                }
                out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], (hit * 255 / (SS * SS)) as u8]);
            }
        }
        (out, template)
    }
}

fn dist(x: f32, y: f32, cx: f32, cy: f32) -> f32 {
    ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()
}

/// Within `r` of the segment a-b (a capsule).
fn segment(x: f32, y: f32, a: (f32, f32), b: (f32, f32), r: f32) -> bool {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let t = (((x - a.0) * dx + (y - a.1) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
    dist(x, y, a.0 + t * dx, a.1 + t * dy) <= r
}

fn triangle(x: f32, y: f32, a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let side = |p: (f32, f32), q: (f32, f32)| (q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0);
    let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
    (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0)
}

fn microphone(x: f32, y: f32) -> bool {
    let body = segment(x, y, (0.5, 0.29), (0.5, 0.47), 0.105);
    // The holder: lower half of a ring around the body.
    let r = dist(x, y, 0.5, 0.47);
    let holder = y >= 0.47 && (r - 0.19).abs() <= 0.035;
    let stem = segment(x, y, (0.5, 0.66), (0.5, 0.77), 0.035);
    let base = segment(x, y, (0.38, 0.78), (0.62, 0.78), 0.035);
    body || holder || stem || base
}
