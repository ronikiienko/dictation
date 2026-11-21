//! Tray icon + menu: described once here, shown by a per-platform backend. Linux: ksni (StatusNotifierItem over
//! D-Bus: the desktop draws the menu itself; no GTK, works on X11 and Wayland). macOS/Windows later: tray-icon + muda.
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
    Action { label: String, icon: &'static str, enabled: bool, click: Click },
    Check { label: String, checked: bool, enabled: bool, click: Click },
    /// One of several (each option: label, what choosing it does).
    Radio { options: Vec<(String, Click)>, selected: usize },
    Sub { label: String, icon: &'static str, items: Vec<Item> },
}

/// The app's side: what to show, built from its state whenever the backend needs it.
pub trait Model: Send + 'static {
    /// freedesktop icon name.
    fn icon(&self) -> &'static str;
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
        pub async fn update<R>(&self, f: impl FnOnce(&mut M) -> R) -> Option<R> {
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
                Item::Action { label, icon, enabled, click } => StandardItem {
                    label,
                    icon_name: icon.into(),
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
                Item::Sub { label, icon, items } => {
                    SubMenu { label, icon_name: icon.into(), submenu: convert(items), ..Default::default() }.into()
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
        fn icon_name(&self) -> String {
            self.0.icon().into()
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
