//! macOS: the overlay as a tao window (the tray's event loop runs it, main thread): no frame, transparent, always on
//! top, never focused (`with_focusable(false)`: macOS won't make it the key window, so typing and pasting stay in the
//! app in front), clicks pass through. Frames go onto the window's layer as a CGImage (premultiplied RGBA, as
//! tiny-skia draws them).
use super::{BOTTOM, FRAME, HEIGHT, Painter, View, WIDTH};
use objc2::msg_send;
use objc2::runtime::AnyObject;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tao::dpi::{LogicalPosition, LogicalSize};
use tao::event_loop::EventLoopWindowTarget;
use tao::platform::macos::{MonitorHandleExtMacOS, WindowBuilderExtMacOS, WindowExtMacOS};
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

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVDisplayLinkCreateWithActiveCGDisplays(link: *mut *mut c_void) -> i32;
    fn CVDisplayLinkSetOutputCallback(link: *mut c_void, cb: LinkCallback, ctx: *mut c_void) -> i32;
    fn CVDisplayLinkStart(link: *mut c_void) -> i32;
    fn CVDisplayLinkSetCurrentCGDisplay(link: *mut c_void, display: u32) -> i32;
    fn CVDisplayLinkStop(link: *mut c_void) -> i32;
    fn CVDisplayLinkRelease(link: *mut c_void);
}

type LinkCallback = unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, u64, *mut u64, *mut c_void) -> i32;

/// What the display link's thread calls: the wake-up, once until the event loop took it.
struct Wake {
    wake: Box<dyn Fn() + Send + Sync>,
    pending: Arc<AtomicBool>,
}

unsafe extern "C" fn on_refresh(_: *mut c_void, _: *const c_void, _: *const c_void, _: u64, _: *mut u64, ctx: *mut c_void) -> i32 {
    let w = unsafe { &*(ctx as *const Wake) };
    if !w.pending.swap(true, Ordering::AcqRel) {
        (w.wake)();
    }
    0
}

/// A CoreVideo display link: a callback right after each screen refresh, so every refresh gets one new frame
/// (a timer drifts against the screen's 60 Hz: now and then a frame is skipped or shown twice, a hitch in motion).
struct DisplayLink {
    link: *mut c_void,
    ctx: *mut Wake,
    pending: Arc<AtomicBool>,
}

impl DisplayLink {
    fn new(wake: Box<dyn Fn() + Send + Sync>) -> Option<DisplayLink> {
        let pending = Arc::new(AtomicBool::new(false));
        let ctx = Box::into_raw(Box::new(Wake { wake, pending: pending.clone() }));
        let mut link = std::ptr::null_mut();
        // SAFETY: plain CoreVideo calls; ctx lives until Drop (after the link is stopped and released).
        unsafe {
            if CVDisplayLinkCreateWithActiveCGDisplays(&mut link) != 0 || link.is_null() {
                drop(Box::from_raw(ctx));
                return None;
            }
            CVDisplayLinkSetOutputCallback(link, on_refresh, ctx as *mut c_void);
        }
        Some(DisplayLink { link, ctx, pending })
    }
    fn set_display(&self, display: u32) {
        unsafe { CVDisplayLinkSetCurrentCGDisplay(self.link, display) };
    }
    fn start(&self) {
        self.pending.store(false, Ordering::Release);
        unsafe { CVDisplayLinkStart(self.link) };
    }
    fn stop(&self) {
        unsafe { CVDisplayLinkStop(self.link) };
    }
}

impl Drop for DisplayLink {
    fn drop(&mut self) {
        // SAFETY: stopping waits for a running callback; then nothing uses ctx any more.
        unsafe {
            CVDisplayLinkStop(self.link);
            CVDisplayLinkRelease(self.link);
            drop(Box::from_raw(self.ctx));
        }
    }
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
    /// Wakes the event loop once per screen refresh while shown (None: couldn't make one, a timer drives it).
    link: Option<DisplayLink>,
}

impl Window {
    /// `wake` is called from the display link's thread once per screen refresh while shown (at most one pending):
    /// it should make the event loop call [`Window::refresh`].
    pub fn new<T: 'static>(target: &EventLoopWindowTarget<T>, wake: impl Fn() + Send + Sync + 'static) -> Result<Window, String> {
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
        let link = DisplayLink::new(Box::new(wake));
        if link.is_none() {
            log::warn!("recording window: no display link, drawing on a timer");
        }
        Ok(Window { win, painter: Painter::default(), next: Instant::now(), link })
    }

    /// Show `v` (Hidden = hide). Placed at the bottom middle of the screen the mouse is on, when it appears.
    pub fn set(&mut self, v: View) {
        let was = self.painter.visible();
        if !self.painter.set(v) {
            return;
        }
        if !self.painter.visible() {
            self.win.set_visible(false);
            if let Some(l) = &self.link {
                l.stop();
            }
            return;
        }
        if !was {
            self.place();
            self.draw();
            self.win.set_visible(true);
            if let Some(l) = &self.link {
                l.start();
            }
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
        let x = pos.x + (size.width - WIDTH as f64) / 2.0;
        let y = pos.y + size.height - HEIGHT as f64 - BOTTOM as f64;
        self.win.set_outer_position(LogicalPosition::new(x, y));
        // Time frames by this screen's refreshes (screens can differ: 120 Hz built-in, 60 Hz external).
        if let Some(l) = &self.link {
            l.set_display(m.native_id());
        }
    }

    /// When the next frame is due; None while hidden or when the display link drives the frames.
    pub fn next_frame(&self) -> Option<Instant> {
        (self.painter.visible() && self.link.is_none()).then_some(self.next)
    }

    /// Draw a frame if one is due.
    pub fn tick(&mut self) {
        if self.link.is_some() {
            return; // frames come from `refresh`
        }
        if self.painter.expire() {
            self.win.set_visible(false);
            return;
        }
        if self.painter.visible() && Instant::now() >= self.next {
            self.draw();
        }
    }

    /// The screen refreshed (the display link's wake-up): draw the next frame.
    pub fn refresh(&mut self) {
        let Some(l) = &self.link else { return };
        l.pending.store(false, Ordering::Release);
        if self.painter.expire() {
            self.win.set_visible(false);
            l.stop();
            return;
        }
        if self.painter.visible() {
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
