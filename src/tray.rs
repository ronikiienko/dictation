//! Tray icon + menu: described once here, shown by a per-platform backend. Linux: ksni (StatusNotifierItem over
//! D-Bus: the desktop draws the menu itself; no GTK, works on X11 and Wayland). macOS/Windows: tray-icon + muda (its
//! menus) on a tao event loop; built into those OSes, nothing to install (on Linux tray-icon would need
//! libayatana-appindicator, hence ksni there).
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
            self.0.update(|k| f(&mut k.0)).await
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

    /// Tray icon size in pixels (the OS scales it; macOS menu bar is 22 pt = 44 px on Retina).
    const SIZE: u32 = if cfg!(target_os = "macos") { 44 } else { 32 };

    /// What reaches the main thread's event loop.
    pub enum Ev<M> {
        Update(Box<dyn FnOnce(&mut M) + Send>),
        Tray(TrayIconEvent),
        Menu(MenuEvent),
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
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new().name("app".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| app(h.clone())));
            let _ = done_tx.send(r);
            h.send(Ev::Done);
        })?;

        let mut tray: Option<TrayIcon> = None;
        let mut shown = Shown::default();
        let mut clicks: HashMap<MenuId, Click> = HashMap::new();
        el.run_return(|event, _, flow| {
            *flow = ControlFlow::Wait;
            match event {
                // macOS: the tray can only be made once the loop runs.
                Event::NewEvents(StartCause::Init) => {
                    // Same as Linux everywhere: left click = activate (record), right click = menu.
                    match TrayIconBuilder::new().with_menu_on_left_click(false).build() {
                        Ok(t) => tray = Some(t),
                        Err(e) => log::error!("tray icon: {e}"),
                    }
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

    /// What the tray shows now, to only redraw what changed.
    #[derive(Default)]
    struct Shown {
        icon: Option<Icon>,
        tooltip: String,
        menu: String,
    }

    fn refresh<M: Model>(t: &TrayIcon, m: &M, shown: &mut Shown, clicks: &mut HashMap<MenuId, Click>) {
        let icon = m.icon();
        if shown.icon != Some(icon) {
            match tray_icon::Icon::from_rgba(icon.rgba(SIZE), SIZE, SIZE) {
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
            let menu = Menu::new();
            clicks.clear();
            add(items, &|i| {
                let _ = menu.append(i);
            }, clicks);
            t.set_menu(Some(Box::new(menu)));
            shown.menu = key;
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

    /// Menu items through `push`.
    fn add(items: Vec<Item>, push: &dyn Fn(&dyn IsMenuItem), clicks: &mut HashMap<MenuId, Click>) {
        for it in items {
            match it {
                Item::Label(label) => push(&MenuItem::new(label, false, None)),
                Item::Separator => push(&PredefinedMenuItem::separator()),
                Item::Action { label, enabled, click } => {
                    let i = MenuItem::new(label, enabled, None);
                    clicks.insert(i.id().clone(), click);
                    push(&i);
                }
                Item::Check { label, checked, enabled, click } => {
                    let i = CheckMenuItem::new(label, enabled, checked, None);
                    clicks.insert(i.id().clone(), click);
                    push(&i);
                }
                // No radio items in muda: checkmarks, one checked.
                Item::Radio { options, selected } => {
                    for (n, (label, click)) in options.into_iter().enumerate() {
                        let i = CheckMenuItem::new(label, true, n == selected, None);
                        clicks.insert(i.id().clone(), click);
                        push(&i);
                    }
                }
                Item::Sub { label, items } => {
                    let s = Submenu::new(label, true);
                    add(items, &|i| {
                        let _ = s.append(i);
                    }, clicks);
                    push(&s);
                }
            }
        }
    }
}
