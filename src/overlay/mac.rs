//! macOS: the overlay as a tao window (the tray's event loop runs it, main thread): no frame, transparent, always on
//! top, never focused (`with_focusable(false)`: macOS won't make it the key window, so typing and pasting stay in the
//! app in front), clicks pass through. Frames go onto the window's layer as a CGImage (premultiplied RGBA, as
//! tiny-skia draws them).
use super::{BOTTOM, FRAME, HEIGHT, Painter, View, WIDTH};
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::ffi::c_void;
use std::time::Instant;
use tao::dpi::{LogicalPosition, LogicalSize};
use tao::event_loop::EventLoopWindowTarget;
use tao::platform::macos::{WindowBuilderExtMacOS, WindowExtMacOS};
use tao::window::{Window as TaoWindow, WindowBuilder};

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;
    fn CGColorSpaceRelease(space: *mut c_void);
    fn CGDataProviderCreateWithData(
        info: *mut c_void,
        data: *const c_void,
        size: usize,
        release: Option<unsafe extern "C" fn(*mut c_void, *const c_void, usize)>,
    ) -> *mut c_void;
    fn CGDataProviderRelease(provider: *mut c_void);
    fn CGImageCreate(
        width: usize,
        height: usize,
        bits_per_component: usize,
        bits_per_pixel: usize,
        bytes_per_row: usize,
        space: *mut c_void,
        bitmap_info: u32,
        provider: *mut c_void,
        decode: *const f64,
        should_interpolate: bool,
        intent: i32,
    ) -> *mut c_void;
    fn CGImageRelease(image: *mut c_void);
}

/// kCGImageAlphaPremultipliedLast, default byte order: R, G, B, A in memory.
const RGBA_PREMULTIPLIED: u32 = 1;

/// Frees the pixels once CoreGraphics is done with them.
unsafe extern "C" fn free_pixels(info: *mut c_void, _data: *const c_void, _size: usize) {
    drop(unsafe { Box::from_raw(info as *mut Vec<u8>) });
}

pub struct Window {
    win: TaoWindow,
    painter: Painter,
    next: Instant,
    /// Current size, points (the pill or the subtitles box).
    size: (f32, f32),
}

impl Window {
    pub fn new<T: 'static>(target: &EventLoopWindowTarget<T>) -> Result<Window, String> {
        let win = WindowBuilder::new()
            .with_title("Dictation")
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false)
            .with_always_on_top(true)
            .with_focused(false)
            .with_focusable(false)
            .with_visible(false)
            .with_visible_on_all_workspaces(true)
            .with_has_shadow(false)
            .with_inner_size(LogicalSize::new(WIDTH as f64, HEIGHT as f64))
            .build(target)
            .map_err(|e| e.to_string())?;
        let _ = win.set_ignore_cursor_events(true);
        Ok(Window { win, painter: Painter::default(), next: Instant::now(), size: (WIDTH, HEIGHT) })
    }

    /// Show `v` (Hidden = hide). Placed at the bottom middle of the screen the mouse is on, when it appears.
    pub fn set(&mut self, v: View) {
        let was = self.painter.visible();
        if !self.painter.set(v) {
            return;
        }
        if !self.painter.visible() {
            self.win.set_visible(false);
            return;
        }
        let size = self.painter.size();
        if size != self.size {
            self.size = size;
            self.win.set_inner_size(LogicalSize::new(size.0 as f64, size.1 as f64));
            if was {
                self.place();
                self.draw();
            }
        }
        if !was {
            self.place();
            self.draw();
            self.win.set_visible(true);
        }
    }

    fn place(&self) {
        let mon = self
            .win
            .cursor_position()
            .ok()
            .and_then(|p| self.win.monitor_from_point(p.x, p.y))
            .or_else(|| self.win.primary_monitor());
        let Some(m) = mon else { return };
        let scale = m.scale_factor();
        let (pos, size) = (m.position().to_logical::<f64>(scale), m.size().to_logical::<f64>(scale));
        let (w, h) = self.size;
        let x = pos.x + (size.width - w as f64) / 2.0;
        let y = pos.y + size.height - h as f64 - BOTTOM as f64;
        self.win.set_outer_position(LogicalPosition::new(x, y));
    }

    /// When the next frame is due; None while hidden.
    pub fn next_frame(&self) -> Option<Instant> {
        self.painter.visible().then_some(self.next)
    }

    /// Draw a frame if one is due.
    pub fn tick(&mut self) {
        if self.painter.visible() && Instant::now() >= self.next {
            self.draw();
        }
    }

    fn draw(&mut self) {
        self.next = Instant::now() + FRAME;
        let scale = self.win.scale_factor() as f32;
        let Some(px) = self.painter.frame(scale) else { return };
        let (w, h) = (px.width() as usize, px.height() as usize);
        let pixels = Box::new(px.take());
        unsafe {
            let data = pixels.as_ptr() as *const c_void;
            let len = pixels.len();
            let provider = CGDataProviderCreateWithData(Box::into_raw(pixels) as *mut c_void, data, len, Some(free_pixels));
            let space = CGColorSpaceCreateDeviceRGB();
            let image = CGImageCreate(w, h, 8, 32, w * 4, space, RGBA_PREMULTIPLIED, provider, std::ptr::null(), false, 0);
            CGColorSpaceRelease(space);
            CGDataProviderRelease(provider);
            if image.is_null() {
                return;
            }
            let ns_window = self.win.ns_window() as *mut AnyObject;
            let view: *mut AnyObject = msg_send![ns_window, contentView];
            if !view.is_null() {
                let _: () = msg_send![view, setWantsLayer: true];
                let layer: *mut AnyObject = msg_send![view, layer];
                if !layer.is_null() {
                    let _: () = msg_send![layer, setContents: image as *mut AnyObject];
                }
            }
            CGImageRelease(image); // the layer keeps its own reference
        }
    }
}
