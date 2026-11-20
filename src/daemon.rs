//! The tray app: model loaded once, global shortcut (XDG portal), record -> transcribe (transcribe.cpp splits long takes at pauses) -> paste.
//! All settings live in the tray menu (saved to config.toml on every change).
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use ksni::TrayMethods;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::config::{Config, KeepLoaded, Output, PasteKeys};
use crate::output::{Keyboard, Notice};
use crate::{RATE, audio, models, tc};

pub const APP_ID: &str = "local.dictation.Dictation";

pub fn socket_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("XDG_RUNTIME_DIR").unwrap_or("/tmp".into())).join("dictation.sock")
}

pub enum Ev {
    Toggle,
    /// New settings chosen in the tray menu.
    Set(Config),
    /// config.toml edited by hand (`dictation reload`).
    Reload,
    Download(&'static str),
    Downloaded(&'static str, Result<(), String>),
    ConfigureShortcut,
    Quit,
    /// Periodic check for the idle unload.
    Tick,
    UnloadNow,
    Shortcut(String),
    EngineLoaded(Result<String, String>),
    Done(Result<Take, String>),
}

pub struct Take {
    audio: Vec<f32>,
    text: String,
    ms: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    NoModel,
    Idle,
    Recording,
    Transcribing,
}

/// Whether the speech model is in (GPU) memory. Independent of State: a take can start while it loads.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mem {
    Unloaded,
    Loading,
    Loaded,
}

// ---------------------------------------------------------------- engine: owns the model, runs on its own thread

enum Job {
    Load { model: std::path::PathBuf, threads: i32 },
    Unload,
    Transcribe(Vec<f32>),
}

fn engine(jobs: std_mpsc::Receiver<Job>, ev: mpsc::UnboundedSender<Ev>) {
    let mut asr: Option<tc::Asr> = None;
    if let Err(e) = tc::init() {
        let _ = ev.send(Ev::EngineLoaded(Err(e.to_string())));
        return;
    }
    for job in jobs {
        match job {
            Job::Load { model, threads } => {
                asr = None; // free the old model first (VRAM)
                let t = Instant::now();
                let r = (|| -> Result<String, tc::Error> {
                    let mut a = tc::Asr::load(&model, threads)?;
                    a.run(&vec![0.0; RATE])?; // warm-up (first run of a new binary also compiles GPU shaders)
                    let backend = a.backend();
                    asr = Some(a);
                    Ok(format!("{backend}, loaded in {:.1} s", t.elapsed().as_secs_f64()))
                })();
                let _ = ev.send(Ev::EngineLoaded(r.map_err(|e| e.to_string())));
            }
            Job::Unload => {
                asr = None;
                unsafe { libc::malloc_trim(0) }; // hand freed heap back to the OS
            }
            Job::Transcribe(x) => {
                let t = Instant::now();
                let r = match asr.as_mut() {
                    Some(a) => a.run(&x).map_err(|e| e.to_string()),
                    None => Err("model not loaded".into()),
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let _ = ev.send(Ev::Done(r.map(|text| Take { audio: x, text, ms })));
            }
        }
    }
}

// ---------------------------------------------------------------- tray (status + every setting)

struct Tray {
    state: State,
    mem: Mem,
    shortcut: String,
    detail: String,
    cfg: Config,
    mics: Vec<(String, String)>,
    /// (model id, percent, checking the SHA-256) while downloading.
    download: Option<(&'static str, u32, bool)>,
    tx: mpsc::UnboundedSender<Ev>,
}

impl Tray {
    fn status_text(&self) -> String {
        let key = if self.shortcut.is_empty() { String::new() } else { format!(" — {}", self.shortcut) };
        match self.download {
            Some((id, _, true)) => return format!("Checking {}…", models::find(id).name),
            Some((id, pct, false)) => return format!("Downloading {}… {pct}%", models::find(id).name),
            None => {}
        }
        match self.state {
            State::NoModel => "No model yet — click to download".into(),
            State::Idle => format!("Ready{key}"),
            State::Recording => format!("● Recording{key}"),
            State::Transcribing => "Transcribing…".into(),
        }
    }

    fn mem_text(&self) -> String {
        match self.mem {
            Mem::Unloaded => "Model not loaded (loads when you start, ~1 s)".into(),
            Mem::Loading => "Loading model…".into(),
            Mem::Loaded => format!("Model loaded: {}", self.detail),
        }
    }

    /// Send a changed copy of the settings to the main loop.
    fn set(&self, f: impl FnOnce(&mut Config)) {
        let mut c = self.cfg.clone();
        f(&mut c);
        let _ = self.tx.send(Ev::Set(c));
    }
}

/// "Keep model loaded" choices.
const KEEP: [(KeepLoaded, &str); 4] = [
    (KeepLoaded::AfterTake, "Unload right after each take"),
    (KeepLoaded::OneMinute, "1 minute after the last take"),
    (KeepLoaded::TenMinutes, "10 minutes after the last take"),
    (KeepLoaded::Always, "Always"),
];

fn output_label(o: Output, k: PasteKeys) -> &'static str {
    match (o, k) {
        (Output::Clipboard, _) => "Clipboard only",
        (Output::Paste, PasteKeys::CtrlV) => "Paste with Ctrl+V",
        (Output::Paste, PasteKeys::CtrlShiftV) => "Paste with Ctrl+Shift+V (terminals)",
        (Output::Paste, PasteKeys::ShiftInsert) => "Paste with Shift+Insert (most apps and terminals)",
    }
}

impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "dictation".into()
    }
    fn title(&self) -> String {
        "Dictation".into()
    }
    fn icon_name(&self) -> String {
        match self.state {
            State::Recording => "media-record",
            _ if self.download.is_some() => "download",
            State::Transcribing => "view-refresh",
            State::NoModel => "dialog-warning",
            State::Idle => "audio-input-microphone",
        }
        .into()
    }
    fn status(&self) -> ksni::Status {
        if self.state == State::Recording { ksni::Status::NeedsAttention } else { ksni::Status::Active }
    }
    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip { title: format!("Dictation: {}", self.status_text()), description: self.mem_text(), ..Default::default() }
    }
    fn activate(&mut self, _x: i32, _y: i32) {
        if self.state == State::NoModel && self.download.is_none() {
            let _ = self.tx.send(Ev::Download(models::find(&self.cfg.model).id));
        } else {
            let _ = self.tx.send(Ev::Toggle);
        }
    }
    fn menu_about_to_show(&mut self) {
        self.mics = audio::input_devices(); // pick up newly plugged microphones
    }
    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        let recording = self.state == State::Recording;

        let model_items = models::CATALOG
            .iter()
            .map(|m| {
                let installed = m.installed();
                let label = match self.download {
                    Some((id, _, true)) if id == m.id => format!("{} — checking…", m.name),
                    Some((id, pct, false)) if id == m.id => format!("{} — downloading {pct}%", m.name),
                    _ if installed => format!("{} — {}", m.name, m.note),
                    _ => format!("{} — download {:.0} MB", m.name, m.size as f64 / 1e6),
                };
                CheckmarkItem {
                    label,
                    checked: self.cfg.model == m.id,
                    enabled: self.download.is_none() || installed,
                    activate: Box::new(move |t: &mut Self| {
                        if m.installed() {
                            t.set(|c| c.model = m.id.into());
                        } else if t.download.is_none() {
                            let _ = t.tx.send(Ev::Download(m.id));
                        }
                    }),
                    ..Default::default()
                }
                .into()
            })
            .chain([
                MenuItem::Separator,
                StandardItem {
                    label: "Open models folder…".into(),
                    icon_name: "folder-open".into(),
                    activate: Box::new(|_: &mut Self| {
                        if let Err(e) = models::open_dir() {
                            eprintln!("{e}");
                        }
                    }),
                    ..Default::default()
                }
                .into(),
            ])
            .collect();

        let mic_selected = self.mics.iter().position(|(id, _)| *id == self.cfg.microphone).unwrap_or(0);
        let mic_ids: Vec<String> = self.mics.iter().map(|m| m.0.clone()).collect();
        let outputs = [
            (Output::Paste, PasteKeys::CtrlV),
            (Output::Paste, PasteKeys::CtrlShiftV),
            (Output::Paste, PasteKeys::ShiftInsert),
            (Output::Clipboard, self.cfg.paste_keys),
        ];
        let out_selected = if self.cfg.output == Output::Clipboard {
            3
        } else {
            outputs.iter().position(|&(_, k)| k == self.cfg.paste_keys).unwrap_or(0)
        };

        vec![
            StandardItem { label: self.status_text(), enabled: false, ..Default::default() }.into(),
            StandardItem { label: self.mem_text(), enabled: false, ..Default::default() }.into(),
            MenuItem::Separator,
            StandardItem {
                label: if recording { "Stop dictation" } else { "Start dictation" }.into(),
                icon_name: if recording { "media-playback-stop" } else { "media-record" }.into(),
                enabled: matches!(self.state, State::Idle | State::Recording | State::Transcribing),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(Ev::Toggle);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            SubMenu { label: "Model".into(), icon_name: "run-build".into(), submenu: model_items, ..Default::default() }.into(),
            SubMenu {
                label: "Keep model loaded".into(),
                icon_name: "memory".into(),
                submenu: vec![
                    RadioGroup {
                        selected: KEEP.iter().position(|&(k, _)| k == self.cfg.keep_loaded).unwrap_or(1),
                        select: Box::new(|t: &mut Self, i| t.set(|c| c.keep_loaded = KEEP[i].0)),
                        options: KEEP.iter().map(|&(_, l)| RadioItem { label: l.into(), ..Default::default() }).collect(),
                        ..Default::default()
                    }
                    .into(),
                    MenuItem::Separator,
                    StandardItem {
                        label: "Unload now".into(),
                        enabled: self.mem == Mem::Loaded && self.state == State::Idle,
                        activate: Box::new(|t: &mut Self| {
                            let _ = t.tx.send(Ev::UnloadNow);
                        }),
                        ..Default::default()
                    }
                    .into(),
                ],
                ..Default::default()
            }
            .into(),
            SubMenu {
                label: "Microphone".into(),
                icon_name: "audio-input-microphone".into(),
                submenu: vec![
                    RadioGroup {
                        selected: mic_selected,
                        select: Box::new(move |t: &mut Self, i| {
                            if let Some(id) = mic_ids.get(i).cloned() {
                                t.set(|c| c.microphone = id);
                            }
                        }),
                        options: self.mics.iter().map(|(_, name)| RadioItem { label: name.clone(), ..Default::default() }).collect(),
                        ..Default::default()
                    }
                    .into(),
                ],
                ..Default::default()
            }
            .into(),
            SubMenu {
                label: "Output".into(),
                icon_name: "edit-paste".into(),
                submenu: vec![
                    RadioGroup {
                        selected: out_selected,
                        select: Box::new(move |t: &mut Self, i| {
                            let (o, k) = outputs[i];
                            t.set(|c| {
                                c.output = o;
                                c.paste_keys = k;
                            });
                        }),
                        options: outputs.iter().map(|&(o, k)| RadioItem { label: output_label(o, k).into(), ..Default::default() }).collect(),
                        ..Default::default()
                    }
                    .into(),
                ],
                ..Default::default()
            }
            .into(),
            CheckmarkItem {
                label: "Sounds".into(),
                checked: self.cfg.sounds,
                activate: Box::new(|t: &mut Self| t.set(|c| c.sounds = !c.sounds)),
                ..Default::default()
            }
            .into(),
            CheckmarkItem {
                label: "Notifications".into(),
                checked: self.cfg.notifications,
                activate: Box::new(|t: &mut Self| t.set(|c| c.notifications = !c.notifications)),
                ..Default::default()
            }
            .into(),
            CheckmarkItem {
                label: "Keep recordings".into(),
                checked: self.cfg.save_recordings,
                activate: Box::new(|t: &mut Self| t.set(|c| c.save_recordings = !c.save_recordings)),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: if self.shortcut.is_empty() { "Set shortcut…".into() } else { format!("Change shortcut ({})…", self.shortcut) },
                icon_name: "configure-shortcuts".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(Ev::ConfigureShortcut);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = t.tx.send(Ev::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

// ---------------------------------------------------------------- global shortcut (XDG portal)

async fn shortcut_task(tx: mpsc::UnboundedSender<Ev>, mut configure: mpsc::UnboundedReceiver<()>) -> ashpd::Result<()> {
    use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut, Shortcut};
    ashpd::register_host_app(APP_ID.try_into()?).await?;
    let gs = GlobalShortcuts::new().await?;
    let session = gs.create_session(Default::default()).await?;
    let wanted = [NewShortcut::new("toggle", "Start / stop dictation").preferred_trigger("Meta+Y")];
    let bound = gs.bind_shortcuts(&session, &wanted, None, Default::default()).await?.response()?;
    let describe = |s: &[Shortcut]| {
        s.iter().find(|s| s.id() == "toggle").map(|s| s.trigger_description().to_string()).unwrap_or_default()
    };
    let _ = tx.send(Ev::Shortcut(describe(bound.shortcuts())));
    let mut activated = gs.receive_activated().await?;
    let mut changed = gs.receive_shortcuts_changed().await?;
    loop {
        tokio::select! {
            Some(a) = activated.next() => if a.shortcut_id() == "toggle" { let _ = tx.send(Ev::Toggle); },
            Some(c) = changed.next() => { let _ = tx.send(Ev::Shortcut(describe(c.shortcuts()))); },
            Some(()) = configure.recv() => {
                if gs.version() >= 2 {
                    if let Err(e) = gs.configure_shortcuts(&session, None, Default::default()).await {
                        eprintln!("configure shortcuts: {e}");
                    }
                } else {
                    eprintln!("portal too old to open the shortcut dialog; use System Settings -> Shortcuts");
                }
            },
            else => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- control socket (`dictation toggle|reload|status`)

async fn socket_task(tx: mpsc::UnboundedSender<Ev>, status: tokio::sync::watch::Receiver<String>) {
    let path = socket_path();
    let _ = std::fs::remove_file(&path);
    let listener = match tokio::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => return eprintln!("socket {}: {e}", path.display()),
    };
    while let Ok((conn, _)) = listener.accept().await {
        let (tx, status) = (tx.clone(), status.clone());
        tokio::spawn(async move {
            let (r, mut w) = conn.into_split();
            let mut line = String::new();
            if BufReader::new(r).read_line(&mut line).await.is_err() {
                return;
            }
            let reply = match line.trim() {
                "toggle" => tx.send(Ev::Toggle).map(|_| "ok".to_string()),
                "reload" => tx.send(Ev::Reload).map(|_| "ok".to_string()),
                "unload" => tx.send(Ev::UnloadNow).map(|_| "ok".to_string()),
                "status" => Ok(status.borrow().clone()),
                _ => Ok("unknown command".to_string()),
            }
            .unwrap_or_else(|_| "shutting down".into());
            let _ = w.write_all(format!("{reply}\n").as_bytes()).await;
        });
    }
}

/// Send one command to a running app. -> reply line.
pub fn send(cmd: &str) -> std::io::Result<String> {
    use std::io::{BufRead, Write};
    let mut s = std::os::unix::net::UnixStream::connect(socket_path())?;
    s.set_read_timeout(Some(Duration::from_secs(2)))?;
    s.write_all(format!("{cmd}\n").as_bytes())?;
    let mut line = String::new();
    std::io::BufReader::new(s).read_line(&mut line)?;
    Ok(line.trim().to_string())
}

// ---------------------------------------------------------------- recordings (same format as gauge / dictate.py)

fn save_take(t: &Take) -> std::io::Result<()> {
    let dir = crate::root().join("recordings");
    std::fs::create_dir_all(&dir)?;
    let name = time_name();
    let spec = hound::WavSpec { channels: 1, sample_rate: RATE as u32, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(dir.join(format!("{name}.wav")), spec).map_err(std::io::Error::other)?;
    for &v in &t.audio {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(std::io::Error::other)?;
    }
    w.finalize().map_err(std::io::Error::other)?;
    use std::io::Write;
    let line = serde_json::json!({"wav": format!("{name}.wav"), "seconds": (t.audio.len() as f64 / RATE as f64 * 100.0).round() / 100.0,
        "parakeet": t.text, "final_model": "ultra-vk", "parakeet_ms": t.ms.round(),
        "segmenting": "transcribe.cpp", "app": "dictation-rs"});
    writeln!(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("log.jsonl"))?, "{line}")
}

/// Local time as YYYYMMDD-HHMMSS (same names as gauge), via libc to avoid a date crate.
fn time_name() -> String {
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:04}{:02}{:02}-{:02}{:02}{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

// ---------------------------------------------------------------- main loop

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    if send("status").is_ok() {
        return Err("Dictation is already running (see the tray).".into());
    }
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(main_loop())
}

fn start_download(id: &'static str, tx: &mpsc::UnboundedSender<Ev>, tray: &ksni::Handle<Tray>) {
    let p = Arc::new(models::Progress::default());
    {
        let (p, tx) = (p.clone(), tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Downloaded(id, models::download(models::find(id), &p)));
        });
    }
    let tray = tray.clone();
    tokio::spawn(async move {
        loop {
            let (done, total) = (p.done.load(Ordering::Relaxed), p.total.load(Ordering::Relaxed));
            let pct = if total > 0 { (done * 100 / total) as u32 } else { 0 };
            let verifying = p.verifying.load(Ordering::Relaxed);
            let alive = tray.update(move |t: &mut Tray| {
                if t.download.is_some() {
                    t.download = Some((id, pct, verifying));
                }
                t.download.is_some()
            });
            if !alive.await.unwrap_or(false) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
}

async fn main_loop() -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = Config::load();
    let (tx, mut rx) = mpsc::unbounded_channel::<Ev>();
    let (jobs, jobs_rx) = std_mpsc::channel::<Job>();
    {
        let tx = tx.clone();
        std::thread::Builder::new().name("engine".into()).spawn(move || engine(jobs_rx, tx))?;
    }
    let installed = |cfg: &Config| models::find(&cfg.model).installed();
    let mut state = if installed(&cfg) { State::Idle } else { State::NoModel };
    let tray = Tray {
        state,
        mem: Mem::Unloaded,
        shortcut: String::new(),
        detail: String::new(),
        cfg: cfg.clone(),
        mics: audio::input_devices(),
        download: None,
        tx: tx.clone(),
    }
    .spawn()
    .await?;
    let (status_tx, status_rx) = tokio::sync::watch::channel(String::new());
    tokio::spawn(socket_task(tx.clone(), status_rx));
    let (configure_tx, configure_rx) = mpsc::unbounded_channel();
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            if let Err(e) = shortcut_task(tx, configure_rx).await {
                eprintln!("global shortcut unavailable ({e}); bind `dictation toggle` in System Settings -> Shortcuts");
            }
        });
    }
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            let _ = tx.send(Ev::Quit);
        });
    }
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                if tx.send(Ev::Tick).is_err() {
                    break;
                }
            }
        });
    }
    let mut clipboard = crate::output::Clipboard::default();
    let mut kb = Keyboard::new().map_err(|e| format!("virtual keyboard (/dev/uinput): {e}"))?;
    let mut notice = Notice { enabled: cfg.notifications, ..Default::default() };
    let mut rec: Option<audio::Recorder> = None;
    let mut shortcut = String::new();
    let mut busy = 0usize; // takes queued in the engine
    let mut downloading = false;
    let mut pending_reload = false;
    let mut mem = Mem::Unloaded;
    let mut last_used = Instant::now();
    let mut load_failed = false; // stops "Always" from retrying a broken load every tick; a take or settings change retries

    // Queue a (re)load of the configured model on the engine thread. Takes queued after it wait for it.
    let load = |cfg: &Config, mem: &mut Mem| {
        let m = models::find(&cfg.model);
        let _ = jobs.send(Job::Load { model: m.path(), threads: cfg.threads });
        *mem = Mem::Loading;
    };
    if state == State::NoModel {
        notice.show("Dictation needs a model", "Click the tray icon to download Parakeet Ultra (740 MB).", "dialog-information", false).await;
    } else if cfg.keep_loaded == KeepLoaded::Always {
        load(&cfg, &mut mem);
    }

    let _ = status_tx.send(serde_json::json!({"state": format!("{state:?}"), "model_memory": format!("{mem:?}"), "shortcut": "", "model": cfg.model}).to_string());
    loop {
        let Some(ev) = rx.recv().await else { break };
        match ev {
            Ev::Toggle => match (state, rec.take()) {
                (State::Idle | State::Transcribing, None) => match audio::Recorder::start(&cfg.microphone) {
                    Ok(r) => {
                        rec = Some(r);
                        state = State::Recording;
                        last_used = Instant::now();
                        if mem == Mem::Unloaded {
                            load_failed = false;
                            load(&cfg, &mut mem); // loads while you speak
                        }
                        if cfg.sounds {
                            audio::play(audio::Cue::Start);
                        }
                        notice.show("● Recording…", "Press the shortcut again to stop", "audio-input-microphone", true).await;
                    }
                    Err(e) => notice.show("Microphone error", &e, "dialog-error", false).await,
                },
                (_, Some(r)) => {
                    if cfg.sounds {
                        audio::play(audio::Cue::Stop);
                    }
                    state = if busy > 0 { State::Transcribing } else { State::Idle };
                    match r.stop() {
                        Ok(x) if x.len() >= RATE / 4 => {
                            let wait = if mem == Mem::Loading { " (model still loading)" } else { "" };
                            notice.show("Transcribing…", &format!("{:.1} s of audio{wait}", x.len() as f64 / RATE as f64), "view-refresh", true).await;
                            busy += 1;
                            let _ = jobs.send(Job::Transcribe(x));
                            state = State::Transcribing;
                        }
                        Ok(_) => notice.show("Too short, skipped", "", "dialog-information", false).await,
                        Err(e) => notice.show("Microphone error", &e, "dialog-error", false).await,
                    }
                    last_used = Instant::now();
                }
                (State::NoModel, None) if !downloading => {
                    downloading = true;
                    let id = models::find(&cfg.model).id;
                    start_download(id, &tx, &tray);
                    tray.update(move |t: &mut Tray| t.download = Some((id, 0, false))).await;
                }
                _ => {}
            },
            Ev::Done(r) => {
                busy = busy.saturating_sub(1);
                last_used = Instant::now();
                if state == State::Transcribing && busy == 0 {
                    state = State::Idle;
                }
                match r {
                    Ok(t) => {
                        if !t.text.is_empty() {
                            if let Err(e) = clipboard.set(&t.text) {
                                eprintln!("clipboard: {e}");
                            } else if cfg.output == Output::Paste {
                                if let Err(e) = kb.paste(cfg.paste_keys) {
                                    eprintln!("paste: {e}");
                                }
                            }
                        }
                        let body = if t.text.chars().count() > 200 { t.text.chars().take(200).collect::<String>() + "…" } else { t.text.clone() };
                        let title = match (t.text.is_empty(), cfg.output) {
                            (true, _) => "Nothing recognized",
                            (false, Output::Paste) => "Pasted",
                            (false, Output::Clipboard) => "Copied to clipboard",
                        };
                        if state != State::Recording {
                            notice.show(title, &body, if t.text.is_empty() { "dialog-information" } else { "dialog-ok" }, false).await;
                        }
                        println!("{:.1}s audio, {:.0} ms: {}", t.audio.len() as f64 / RATE as f64, t.ms, t.text);
                        if cfg.save_recordings {
                            if let Err(e) = save_take(&t) {
                                eprintln!("save recording: {e}");
                            }
                        }
                    }
                    Err(e) => notice.show("Transcription failed", &e, "dialog-error", false).await,
                }
            }
            Ev::EngineLoaded(r) => match r {
                Ok(info) => {
                    println!("model loaded ({info})");
                    mem = Mem::Loaded;
                    let detail = format!("{}, {info}", models::find(&cfg.model).name);
                    tray.update(move |t: &mut Tray| t.detail = detail).await;
                }
                Err(e) => {
                    eprintln!("model load failed: {e}");
                    mem = Mem::Unloaded; // the next take tries again
                    load_failed = true;
                    notice.show("Model failed to load", &e, "dialog-error", false).await;
                }
            },
            Ev::Tick => {}
            Ev::UnloadNow => {
                if rec.is_none() && busy == 0 && mem == Mem::Loaded {
                    let _ = jobs.send(Job::Unload);
                    mem = Mem::Unloaded;
                    println!("model unloaded");
                }
            }
            Ev::Download(id) => {
                if !downloading {
                    downloading = true;
                    start_download(id, &tx, &tray);
                    tray.update(move |t: &mut Tray| t.download = Some((id, 0, false))).await;
                }
            }
            Ev::Downloaded(id, r) => {
                downloading = false;
                tray.update(|t: &mut Tray| t.download = None).await;
                match r {
                    Ok(()) => {
                        notice.show("Model downloaded", models::find(id).name, "dialog-ok", false).await;
                        let mut c = cfg.clone();
                        c.model = id.into();
                        let _ = tx.send(Ev::Set(c));
                    }
                    Err(e) => notice.show("Download failed", &e, "dialog-error", false).await,
                }
            }
            Ev::Set(new) => {
                if let Err(e) = new.save() {
                    eprintln!("save config: {e}");
                }
                pending_reload |= apply(&mut cfg, new);
                load_failed = false;
            }
            Ev::Reload => {
                pending_reload |= apply(&mut cfg, Config::load());
                load_failed = false;
            }
            Ev::Shortcut(s) => {
                println!("shortcut: {s}");
                shortcut = s.clone();
                tray.update(move |t: &mut Tray| t.shortcut = s).await;
            }
            Ev::ConfigureShortcut => {
                let _ = configure_tx.send(());
            }
            Ev::Quit => break,
        }
        notice.enabled = cfg.notifications;
        if state == State::NoModel && installed(&cfg) {
            state = State::Idle;
        } else if state == State::Idle && busy == 0 && !installed(&cfg) {
            // Deleted from the models folder (or never there for a new choice): offer the download again.
            state = State::NoModel;
            if mem != Mem::Unloaded {
                let _ = jobs.send(Job::Unload);
                mem = Mem::Unloaded;
            }
            println!("model file missing: {}", models::find(&cfg.model).path().display());
        }
        // Settings changes apply once the current take is pasted: a loaded model is swapped for the new choice,
        // an unloaded one just loads the new choice next time. "Always" loads right away.
        if rec.is_none() && busy == 0 && state != State::NoModel {
            if pending_reload && mem != Mem::Unloaded {
                load(&cfg, &mut mem);
            } else if cfg.keep_loaded == KeepLoaded::Always && mem == Mem::Unloaded && !load_failed {
                load(&cfg, &mut mem);
            }
            pending_reload = false;
            // Idle unload: checked after every event (a pasted take, a 10 s tick, a settings change).
            if let Some(limit) = cfg.keep_loaded.idle() {
                if mem == Mem::Loaded && last_used.elapsed() >= limit {
                    let _ = jobs.send(Job::Unload);
                    mem = Mem::Unloaded;
                    println!("model unloaded ({:?})", cfg.keep_loaded);
                }
            }
        }
        let (s, m, c) = (state, mem, cfg.clone());
        tray.update(move |t: &mut Tray| {
            t.state = s;
            t.mem = m;
            t.cfg = c;
        })
        .await;
        let _ = status_tx.send(
            serde_json::json!({"state": format!("{state:?}"), "model_memory": format!("{mem:?}"), "shortcut": shortcut, "model": cfg.model})
                .to_string(),
        );
    }
    if let Some(r) = rec {
        let _ = r.stop();
    }
    let _ = std::fs::remove_file(socket_path());
    tray.shutdown().await;
    Ok(())
}

/// Take new settings. -> whether a loaded model has to be swapped (model or thread count changed).
fn apply(cfg: &mut Config, new: Config) -> bool {
    let reload = new.model != cfg.model || new.threads != cfg.threads;
    *cfg = new;
    reload
}
