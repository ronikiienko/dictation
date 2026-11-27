//! Tray icon + menu: described once here, shown by a per-platform backend. Linux: ksni (StatusNotifierItem over
//! D-Bus: the desktop draws the menu itself; no GTK, works on X11 and Wayland). macOS/Windows: tray-icon + muda (its
//! menus) on a tao event loop; built into those OSes, nothing to install (on Linux tray-icon would need
//! libayatana-appindicator, hence ksni there). The macOS/Windows global shortcut (global-hotkey) lives here too: it
//! needs the same main-thread event loop.
//!
//! The tray owns the main thread on every platform (`run`) and the app runs on another one: macOS only allows
//! on-screen things (the menu-bar icon included) on the main thread, Windows wants them on the thread that made them.
use std::sync::Arc;

/// What a click does (sends an event to the app).
pub type Click = Arc<dyn Fn() + Send + Sync>;

pub fn click(f: impl Fn() + Send + Sync + 'static) -> Click {
    Arc::new(f)
}

pub enum Item {
    /// A greyed-out line of text.
    Label(String),
    Separator,
    Action { label: String, enabled: bool, click: Click },
    Check { label: String, checked: bool, enabled: bool, click: Click },
    /// One of several (each option: label, what choosing it does).
    Radio { options: Vec<(String, Click)>, selected: usize },
    Sub { label: String, items: Vec<Item> },
}

/// The app's side: what to show, built from its state whenever the backend needs it.
pub trait Model: Send + 'static {
    /// The tray icon (drawn in code, `icons`).
    fn icon(&self) -> crate::icons::Icon;
    /// Ask for attention (recording).
    fn attention(&self) -> bool;
    /// (title, description).
    fn tooltip(&self) -> (String, String);
    fn menu(&self) -> Vec<Item>;
    /// Left click on the icon.
    fn activate(&self);
    /// The menu is about to open (refresh what may have changed, e.g. microphones).
    fn menu_opening(&mut self) {}
    /// macOS/Windows: the global shortcut to register, e.g. "Ctrl+Alt+Space" ("" = none). Not used on Linux: there
    /// the desktop's portal owns shortcuts.
    fn hotkey(&self) -> String {
        String::new()
    }
    fn hotkey_pressed(&self) {}
    fn hotkey_released(&self) {}
    /// What the recording window (crate::overlay) shows.
    fn overlay(&self) -> crate::overlay::View {
        crate::overlay::View::Hidden
    }
    /// Registering `hotkey()` worked, or why not (taken by another app, not a valid key).
    fn hotkey_result(&self, _key: &str, _result: Result<(), String>) {}
}

/// Show the tray on this (the main) thread and run `app` on a new thread with a handle to update it. Returns when
/// `app` does, with its result; the tray is gone by then.
pub fn run<M: Model, R: Send + 'static>(
    model: M,
    app: impl FnOnce(Handle<M>) -> R + Send + 'static,
) -> Result<R, Box<dyn std::error::Error>> {
    imp::run(model, app)
}

pub use imp::Handle;

#[cfg(target_os = "linux")]
mod imp {
    use super::{Item, Model};

    /// Updates the tray from any thread.
    pub struct Handle<M: Model>(ksni::Handle<Ksni<M>>);

    impl<M: Model> Clone for Handle<M> {
        fn clone(&self) -> Self {
            Handle(self.0.clone())
        }
    }

    impl<M: Model> Handle<M> {
        /// Change the model; the tray redraws. None once the tray is gone.
        pub async fn update<R: Send + 'static>(&self, f: impl FnOnce(&mut M) -> R + Send + 'static) -> Option<R> {
            self.0
                .update(|k| {
                    let r = f(&mut k.0);
                    crate::overlay::show(k.0.overlay());
                    r
                })
                .await
        }
    }

    pub fn run<M: Model, R: Send + 'static>(
        model: M,
        app: impl FnOnce(Handle<M>) -> R + Send + 'static,
    ) -> Result<R, Box<dyn std::error::Error>> {
        // ksni serves D-Bus as a task of the current tokio runtime: a small one on this thread.
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        rt.block_on(async {
            use ksni::TrayMethods;
            let tray = Ksni(model).spawn().await?;
            let h = Handle(tray.clone());
            let (done_tx, done_rx) = tokio::sync::oneshot::channel();
            std::thread::Builder::new().name("app".into()).spawn(move || {
                let _ = done_tx.send(app(h));
            })?;
            let r = done_rx.await.map_err(|_| "app thread panicked")?;
            tray.shutdown().await;
            Ok(r)
        })
    }

    pub struct Ksni<M>(M);

    fn convert<M: Model>(items: Vec<Item>) -> Vec<ksni::MenuItem<Ksni<M>>> {
        use ksni::menu::*;
        items
            .into_iter()
            .map(|it| match it {
                Item::Label(label) => StandardItem { label, enabled: false, ..Default::default() }.into(),
                Item::Separator => MenuItem::Separator,
                Item::Action { label, enabled, click } => StandardItem {
                    label,
                    enabled,
                    activate: Box::new(move |_: &mut Ksni<M>| click()),
                    ..Default::default()
                }
                .into(),
                Item::Check { label, checked, enabled, click } => {
                    CheckmarkItem { label, checked, enabled, activate: Box::new(move |_: &mut Ksni<M>| click()), ..Default::default() }
                        .into()
                }
                Item::Radio { options, selected } => {
                    let labels = options.iter().map(|(l, _)| RadioItem { label: l.clone(), ..Default::default() }).collect();
                    let clicks: Vec<_> = options.into_iter().map(|(_, c)| c).collect();
                    RadioGroup {
                        selected,
                        select: Box::new(move |_: &mut Ksni<M>, i| {
                            if let Some(c) = clicks.get(i) {
                                c();
                            }
                        }),
                        options: labels,
                        ..Default::default()
                    }
                    .into()
                }
                Item::Sub { label, items } => {
                    SubMenu { label, submenu: convert(items), ..Default::default() }.into()
                }
            })
            .collect()
    }

    impl<M: Model> ksni::Tray for Ksni<M> {
        fn id(&self) -> String {
            "dictation".into()
        }
        fn title(&self) -> String {
            "Dictation".into()
        }
        fn icon_pixmap(&self) -> Vec<ksni::Icon> {
            let icon = self.0.icon();
            [22, 32, 48, 64]
                .into_iter()
                .map(|size| {
                    // RGBA -> ARGB (network byte order), as the StatusNotifierItem spec wants.
                    let data = icon.rgba(size as u32).chunks_exact(4).flat_map(|p| [p[3], p[0], p[1], p[2]]).collect();
                    ksni::Icon { width: size, height: size, data }
                })
                .collect()
        }
        fn status(&self) -> ksni::Status {
            if self.0.attention() { ksni::Status::NeedsAttention } else { ksni::Status::Active }
        }
        fn tool_tip(&self) -> ksni::ToolTip {
            let (title, description) = self.0.tooltip();
            ksni::ToolTip { title, description, ..Default::default() }
        }
        fn activate(&mut self, _x: i32, _y: i32) {
            self.0.activate();
        }
        fn menu_about_to_show(&mut self) {
            self.0.menu_opening();
        }
        fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
            convert(self.0.menu())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::{Click, Item, Model};
    use crate::icons::Icon;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tao::event::{Event, StartCause};
    use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
    use tao::platform::run_return::EventLoopExtRunReturn;
    use tray_icon::menu::{CheckMenuItem, IsMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu};
    use tray_icon::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
    use global_hotkey::hotkey::HotKey;
    use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

    /// Tray icon size in pixels (the OS scales it; macOS menu bar is 22 pt = 44 px on Retina).
    const SIZE: u32 = if cfg!(target_os = "macos") { 44 } else { 32 };

    /// What reaches the main thread's event loop.
    pub enum Ev<M> {
        Update(Box<dyn FnOnce(&mut M) + Send>),
        Tray(TrayIconEvent),
        Menu(MenuEvent),
        HotKey(GlobalHotKeyEvent),
        /// The app thread is done: stop the loop.
        Done,
    }

    /// Updates the tray from any thread (by sending the change to the main thread).
    pub struct Handle<M: Model>(Arc<Mutex<EventLoopProxy<Ev<M>>>>);

    impl<M: Model> Clone for Handle<M> {
        fn clone(&self) -> Self {
            Handle(self.0.clone())
        }
    }

    impl<M: Model> Handle<M> {
        /// Change the model; the tray redraws. None once the tray is gone.
        pub async fn update<R: Send + 'static>(&self, f: impl FnOnce(&mut M) -> R + Send + 'static) -> Option<R> {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if !self.send(Ev::Update(Box::new(move |m| {
                let _ = tx.send(f(m));
            }))) {
                return None;
            }
            rx.await.ok()
        }

        fn send(&self, ev: Ev<M>) -> bool {
            self.0.lock().is_ok_and(|p| p.send_event(ev).is_ok())
        }
    }

    pub fn run<M: Model, R: Send + 'static>(
        mut model: M,
        app: impl FnOnce(Handle<M>) -> R + Send + 'static,
    ) -> Result<R, Box<dyn std::error::Error>> {
        let mut el = EventLoopBuilder::<Ev<M>>::with_user_event().build();
        #[cfg(target_os = "macos")]
        {
            use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
            el.set_activation_policy(ActivationPolicy::Accessory); // menu-bar icon only, no Dock icon
        }
        let h = Handle(Arc::new(Mutex::new(el.create_proxy())));
        let ht = h.clone();
        TrayIconEvent::set_event_handler(Some(move |e| {
            ht.send(Ev::Tray(e));
        }));
        let hm = h.clone();
        MenuEvent::set_event_handler(Some(move |e| {
            hm.send(Ev::Menu(e));
        }));
        let hk = h.clone();
        GlobalHotKeyEvent::set_event_handler(Some(move |e| {
            hk.send(Ev::HotKey(e));
        }));
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new().name("app".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app(h.clone())));
            let _ = done_tx.send(r);
            h.send(Ev::Done);
        })?;

        let mut tray: Option<TrayIcon> = None;
        let mut keys = Keys { manager: None, registered: None };
        let mut shown = Shown::default();
        let mut clicks: HashMap<MenuId, Click> = HashMap::new();
        #[cfg(target_os = "macos")]
        let mut overlay: Option<crate::overlay::Window> = None;
        el.run_return(|event, _target, flow| {
            // The recording window animates while shown: wake up for its next frame.
            #[cfg(target_os = "macos")]
            if let Some(o) = overlay.as_mut() {
                o.tick();
            }
            *flow = ControlFlow::Wait;
            #[cfg(target_os = "macos")]
            if let Some(t) = overlay.as_ref().and_then(|o| o.next_frame()) {
                *flow = ControlFlow::WaitUntil(t);
            }
            match event {
                // macOS: the tray can only be made once the loop runs.
                Event::NewEvents(StartCause::Init) => {
                    // Same as Linux everywhere: left click = activate (record), right click = menu.
                    match TrayIconBuilder::new().with_menu_on_left_click(false).build() {
                        Ok(t) => tray = Some(t),
                        Err(e) => log::error!("tray icon: {e}"),
                    }
                    // Also main-thread only on macOS.
                    keys.manager = Some(GlobalHotKeyManager::new().map_err(|e| e.to_string()));
                    #[cfg(target_os = "macos")]
                    match crate::overlay::Window::new(_target) {
                        Ok(w) => overlay = Some(w),
                        Err(e) => log::error!("recording window: {e}"),
                    }
                }
                Event::UserEvent(Ev::HotKey(e)) => {
                    if keys.registered.is_some_and(|k| k.id() == e.id) {
                        match e.state {
                            HotKeyState::Pressed => model.hotkey_pressed(),
                            HotKeyState::Released => model.hotkey_released(),
                        }
                    }
                    return;
                }
                Event::UserEvent(Ev::Update(f)) => f(&mut model),
                Event::UserEvent(Ev::Menu(e)) => {
                    if let Some(c) = clicks.get(&e.id) {
                        c();
                    }
                    shown.menu.clear(); // the OS may have toggled a checkmark itself: redraw from the model
                }
                Event::UserEvent(Ev::Tray(TrayIconEvent::Click { button, button_state, .. })) => {
                    match button_state {
                        MouseButtonState::Down => model.menu_opening(),
                        MouseButtonState::Up if button == MouseButton::Left => model.activate(),
                        _ => {}
                    }
                }
                Event::UserEvent(Ev::Done) => {
                    *flow = ControlFlow::Exit;
                    return;
                }
                _ => return,
            }
            keys.update(&model, &mut shown);
            #[cfg(target_os = "macos")]
            if let Some(o) = overlay.as_mut() {
                o.set(model.overlay());
                if let Some(t) = o.next_frame() {
                    *flow = ControlFlow::WaitUntil(t);
                }
            }
            if let Some(t) = &tray {
                refresh(t, &model, &mut shown, &mut clicks);
            }
        });
        drop(tray);
        match done_rx.recv() {
            Ok(Ok(r)) => Ok(r),
            _ => Err("app thread panicked".into()),
        }
    }

    /// The global shortcut.
    struct Keys {
        /// Made once the loop runs; Err = the OS refused (no shortcuts then).
        manager: Option<Result<GlobalHotKeyManager, String>>,
        registered: Option<HotKey>,
    }

    impl Keys {
        /// Register the model's key if it changed (and report how that went).
        fn update<M: Model>(&mut self, m: &M, shown: &mut Shown) {
            let Some(manager) = &self.manager else { return };
            let want = m.hotkey();
            if shown.hotkey.as_ref() == Some(&want) {
                return;
            }
            shown.hotkey = Some(want.clone());
            let manager = match manager {
                Ok(manager) => manager,
                Err(e) => return m.hotkey_result(&want, Err(format!("global shortcuts unavailable: {e}"))),
            };
            if let Some(old) = self.registered.take() {
                let _ = manager.unregister(old);
            }
            if want.is_empty() {
                return m.hotkey_result(&want, Ok(()));
            }
            let r = want
                .parse::<HotKey>()
                .map_err(|e| format!("\"{want}\" is not a valid shortcut ({e})"))
                .and_then(|k| {
                    manager.register(k).map_err(|e| format!("{want} is taken by another app ({e})"))?;
                    self.registered = Some(k);
                    Ok(())
                });
            m.hotkey_result(&want, r);
        }
    }

    /// What the tray shows now, to only redraw what changed.
    #[derive(Default)]
    struct Shown {
        hotkey: Option<String>,
        icon: Option<Icon>,
        tooltip: String,
        menu: String,
        /// The menu's shape (kinds and nesting, no texts or states) and its items in the order `add` made them.
        shape: String,
        items: Vec<Kept>,
    }

    /// A menu item we keep, to change it in place.
    enum Kept {
        Item(MenuItem),
        Check(CheckMenuItem),
        Sub(Submenu),
        Separator,
    }

    fn refresh<M: Model>(t: &TrayIcon, m: &M, shown: &mut Shown, clicks: &mut HashMap<MenuId, Click>) {
        let icon = m.icon();
        if shown.icon != Some(icon) {
            // macOS: the symbol alone, as a template image (except Recording, red); elsewhere the coloured disc.
            #[cfg(target_os = "macos")]
            let (rgba, template) = icon.glyph_rgba(SIZE);
            #[cfg(not(target_os = "macos"))]
            let rgba = icon.rgba(SIZE);
            match tray_icon::Icon::from_rgba(rgba, SIZE, SIZE) {
                #[cfg(target_os = "macos")]
                Ok(i) if template => {
                    let _ = t.set_icon_templated(Some(i));
                }
                Ok(i) => {
                    let _ = t.set_icon(Some(i));
                }
                Err(e) => log::error!("tray icon: {e}"),
            }
            shown.icon = Some(icon);
        }
        let (title, description) = m.tooltip();
        let tooltip = if description.is_empty() { title } else { format!("{title}\n{description}") };
        if tooltip != shown.tooltip {
            let _ = t.set_tooltip(Some(&tooltip));
            shown.tooltip = tooltip;
        }
        let items = m.menu();
        let mut key = String::new();
        fingerprint(&items, &mut key);
        if key != shown.menu {
            // Same shape: change texts and states in place. Replacing the menu closes it if it is open (macOS), and
            // the status line changes while it is (download / optimizing progress).
            let mut shape = String::new();
            shape_of(&items, &mut shape);
            clicks.clear();
            if shape == shown.shape {
                update(items, &mut shown.items.iter(), clicks);
            } else {
                let menu = Menu::new();
                shown.items.clear();
                add(items, &|i| {
                    let _ = menu.append(i);
                }, clicks, &mut shown.items);
                t.set_menu(Some(Box::new(menu)));
                shown.shape = shape;
            }
            shown.menu = key;
        }
    }

    /// The menu's shape: what `add` makes, without texts or states.
    fn shape_of(items: &[Item], out: &mut String) {
        for it in items {
            match it {
                Item::Label(_) | Item::Action { .. } => *out += "A",
                Item::Separator => *out += "-",
                Item::Check { .. } => *out += "C",
                Item::Radio { options, .. } => *out += &"C".repeat(options.len()),
                Item::Sub { items, .. } => {
                    *out += "S[";
                    shape_of(items, out);
                    *out += "]";
                }
            }
        }
    }

    /// Give the items `add` made (same shape) the new texts, states and clicks.
    fn update<'a>(items: Vec<Item>, h: &mut impl Iterator<Item = &'a Kept>, clicks: &mut HashMap<MenuId, Click>) {
        for it in items {
            match it {
                Item::Label(label) => {
                    if let Some(Kept::Item(i)) = h.next() {
                        i.set_text(label);
                    }
                }
                Item::Separator => {
                    h.next();
                }
                Item::Action { label, enabled, click } => {
                    if let Some(Kept::Item(i)) = h.next() {
                        i.set_text(label);
                        i.set_enabled(enabled);
                        clicks.insert(i.id().clone(), click);
                    }
                }
                Item::Check { label, checked, enabled, click } => {
                    if let Some(Kept::Check(i)) = h.next() {
                        i.set_text(label);
                        i.set_enabled(enabled);
                        i.set_checked(checked);
                        clicks.insert(i.id().clone(), click);
                    }
                }
                Item::Radio { options, selected } => {
                    for (n, (label, click)) in options.into_iter().enumerate() {
                        if let Some(Kept::Check(i)) = h.next() {
                            i.set_text(label);
                            i.set_checked(n == selected);
                            clicks.insert(i.id().clone(), click);
                        }
                    }
                }
                Item::Sub { label, items } => {
                    if let Some(Kept::Sub(s)) = h.next() {
                        s.set_text(label);
                    }
                    update(items, h, clicks);
                }
            }
        }
    }

    /// The menu as text (labels and states), to tell whether it changed.
    fn fingerprint(items: &[Item], out: &mut String) {
        for it in items {
            match it {
                Item::Label(l) => *out += &format!("L{l}\n"),
                Item::Separator => *out += "-\n",
                Item::Action { label, enabled, .. } => *out += &format!("A{label}{enabled}\n"),
                Item::Check { label, checked, enabled, .. } => *out += &format!("C{label}{checked}{enabled}\n"),
                Item::Radio { options, selected } => {
                    *out += &format!("R{selected}\n");
                    options.iter().for_each(|(l, _)| *out += &format!("{l}\n"));
                }
                Item::Sub { label, items, .. } => {
                    *out += &format!("S{label}[\n");
                    fingerprint(items, out);
                    *out += "]\n";
                }
            }
        }
    }

    /// Menu items through `push`; each one also goes to `kept` (in order, a submenu before its items).
    fn add(items: Vec<Item>, push: &dyn Fn(&dyn IsMenuItem), clicks: &mut HashMap<MenuId, Click>, kept: &mut Vec<Kept>) {
        for it in items {
            match it {
                Item::Label(label) => {
                    let i = MenuItem::new(label, false, None);
                    push(&i);
                    kept.push(Kept::Item(i));
                }
                Item::Separator => {
                    push(&PredefinedMenuItem::separator());
                    kept.push(Kept::Separator);
                }
                Item::Action { label, enabled, click } => {
                    let i = MenuItem::new(label, enabled, None);
                    clicks.insert(i.id().clone(), click);
                    push(&i);
                    kept.push(Kept::Item(i));
                }
                Item::Check { label, checked, enabled, click } => {
                    let i = CheckMenuItem::new(label, enabled, checked, None);
                    clicks.insert(i.id().clone(), click);
                    push(&i);
                    kept.push(Kept::Check(i));
                }
                // No radio items in muda: checkmarks, one checked.
                Item::Radio { options, selected } => {
                    for (n, (label, click)) in options.into_iter().enumerate() {
                        let i = CheckMenuItem::new(label, true, n == selected, None);
                        clicks.insert(i.id().clone(), click);
                        push(&i);
                        kept.push(Kept::Check(i));
                    }
                }
                Item::Sub { label, items } => {
                    let s = Submenu::new(label, true);
                    let at = kept.len();
                    kept.push(Kept::Separator); // placeholder: the submenu goes before its items
                    add(items, &|i| {
                        let _ = s.append(i);
                    }, clicks, kept);
                    push(&s);
                    kept[at] = Kept::Sub(s);
                }
            }
        }
    }
}
