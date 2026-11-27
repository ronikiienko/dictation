//! Linux (Wayland): the overlay as a layer-shell surface (wlr-layer-shell, KWin has it): the desktop's own kind of
//! surface for panels and popups. Anchored to the bottom middle, above windows, never gets keyboard focus, clicks
//! pass through. A normal window can't do that on Wayland (no positioning, no "always on top", and a new window
//! takes focus, so the paste would land in it). Own thread with its own connection; frames are drawn into shared
//! memory (wl_shm, premultiplied ARGB). No layer shell (GNOME) or no Wayland (X11): no overlay, said once in the log.
use super::{BOTTOM, FRAME, HEIGHT, Painter, View, WIDTH};
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState, Region};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface, LayerSurfaceConfigure,
};
use smithay_client_toolkit::shm::slot::SlotPool;
use smithay_client_toolkit::shm::{Shm, ShmHandler};
use smithay_client_toolkit::{delegate_dispatch2, delegate_registry, registry_handlers};
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{wl_output, wl_shm, wl_surface};
use wayland_client::{Connection, QueueHandle};

/// What to show, set from the app; the overlay thread picks it up. None until the thread is started.
static WANT: Mutex<Option<View>> = Mutex::new(None);

/// Show `v` (Hidden = hide). The thread starts on the first call.
pub fn show(v: View) {
    let mut want = WANT.lock().unwrap();
    if want.is_none() && v != View::Hidden {
        let started = std::thread::Builder::new().name("overlay".into()).spawn(|| {
            if let Err(e) = run() {
                eprintln!("overlay: {e}");
            }
        });
        if started.is_err() {
            return;
        }
    }
    if want.is_some() || v != View::Hidden {
        *want = Some(v);
    }
}

fn wanted() -> View {
    WANT.lock().unwrap().clone().unwrap_or(View::Hidden)
}

struct State {
    registry: RegistryState,
    outputs: OutputState,
    compositor: CompositorState,
    layer_shell: LayerShell,
    shm: Shm,
    pool: SlotPool,
    /// While shown.
    layer: Option<LayerSurface>,
    /// The compositor sent the first configure: frames may be attached.
    configured: bool,
    /// Pixels per point of the screen it's on.
    scale: u32,
    /// Size of the open surface, points.
    size: (f32, f32),
    painter: Painter,
    next: Instant,
}

fn run() -> Result<(), String> {
    let conn = Connection::connect_to_env().map_err(|e| format!("no Wayland ({e}); no recording window"))?;
    let (globals, mut queue) = registry_queue_init::<State>(&conn).map_err(|e| e.to_string())?;
    let qh = queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).map_err(|e| e.to_string())?;
    let layer_shell =
        LayerShell::bind(&globals, &qh).map_err(|_| "the desktop has no layer shell (GNOME?); no recording window".to_string())?;
    let shm = Shm::bind(&globals, &qh).map_err(|e| e.to_string())?;
    let pool = SlotPool::new((WIDTH * HEIGHT * 4.0 * 4.0) as usize, &shm).map_err(|e| e.to_string())?;
    let mut s = State {
        registry: RegistryState::new(&globals),
        outputs: OutputState::new(&globals, &qh),
        compositor,
        layer_shell,
        shm,
        pool,
        layer: None,
        configured: false,
        scale: 1,
        size: (WIDTH, HEIGHT),
        painter: Painter::default(),
        next: Instant::now(),
    };
    loop {
        let v = wanted();
        let hidden = v == View::Hidden;
        if s.painter.set(v) {
            if hidden {
                s.layer = None; // destroys the surface
                s.configured = false;
            } else if s.layer.is_none() || s.painter.size() != s.size {
                s.open(&qh); // a new size (pill <-> subtitles): a new surface, configured for it
            }
        }
        if s.configured && s.painter.visible() && Instant::now() >= s.next {
            s.draw();
        }
        // Wait for Wayland events until the next frame (or a short while, to notice a new view).
        queue.flush().map_err(|e| e.to_string())?;
        if let Some(guard) = queue.prepare_read() {
            let wait = if s.painter.visible() { s.next.saturating_duration_since(Instant::now()) } else { Duration::from_millis(50) };
            let mut fd = libc::pollfd { fd: guard.connection_fd().as_raw_fd(), events: libc::POLLIN, revents: 0 };
            let n = unsafe { libc::poll(&mut fd, 1, wait.as_millis().min(1000) as i32) };
            if n > 0 {
                guard.read().map_err(|e| e.to_string())?;
            }
        }
        queue.dispatch_pending(&mut s).map_err(|e| e.to_string())?;
    }
}

impl State {
    fn open(&mut self, qh: &QueueHandle<State>) {
        let surface = self.compositor.create_surface(qh);
        let layer = self.layer_shell.create_layer_surface(qh, surface, Layer::Overlay, Some("dictation"), None);
        self.size = self.painter.size();
        layer.set_anchor(Anchor::BOTTOM);
        layer.set_margin(0, 0, BOTTOM as i32, 0);
        layer.set_size(self.size.0 as u32, self.size.1 as u32);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        if let Ok(r) = Region::new(&self.compositor) {
            layer.set_input_region(Some(r.wl_region())); // empty: clicks go to what's below
        }
        let _ = layer.set_buffer_scale(self.scale);
        layer.commit(); // no buffer yet: asks for the first configure
        self.layer = Some(layer);
        self.configured = false;
    }

    fn draw(&mut self) {
        self.next = Instant::now() + FRAME;
        let Some(layer) = &self.layer else { return };
        let Some(px) = self.painter.frame(self.scale as f32) else { return };
        let (w, h) = (px.width() as i32, px.height() as i32);
        let Ok((buffer, canvas)) = self.pool.create_buffer(w, h, w * 4, wl_shm::Format::Argb8888) else { return };
        // RGBA (tiny-skia) -> ARGB8888 little-endian = B, G, R, A in memory; both premultiplied.
        for (d, s) in canvas.chunks_exact_mut(4).zip(px.data().chunks_exact(4)) {
            d.copy_from_slice(&[s[2], s[1], s[0], s[3]]);
        }
        let surface = layer.wl_surface();
        let _ = layer.set_buffer_scale(self.scale);
        surface.damage_buffer(0, 0, w, h);
        if buffer.attach_to(surface).is_ok() {
            layer.commit();
        }
    }
}

impl CompositorHandler for State {
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, factor: i32) {
        self.scale = factor.max(1) as u32;
        self.next = Instant::now(); // redraw at the new scale
    }
    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: wl_output::Transform) {}
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
}

impl OutputHandler for State {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.outputs
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl LayerShellHandler for State {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        // The compositor removed it (e.g. its screen went away): open a new one on the next show.
        self.layer = None;
        self.configured = false;
        self.painter.set(View::Hidden);
    }
    fn configure(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface, _: LayerSurfaceConfigure, _: u32) {
        self.configured = true;
        self.next = Instant::now(); // draw now
    }
}

impl ShmHandler for State {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for State {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry
    }
    registry_handlers![OutputState];
}

delegate_registry!(State);
delegate_dispatch2!(State);
