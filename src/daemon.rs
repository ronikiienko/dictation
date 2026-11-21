//! The tray app: global shortcut (XDG portal), record -> transcribe (asr: long takes are cut at pauses) -> paste.
//! All settings live in the tray menu (saved to config.toml on every change).
use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::config::{Config, KeepLoaded, Output, PasteKeys};
use crate::output::{Keyboard, Notice};
use crate::icons::Icon;
use crate::{RATE, asr, audio, models, tray};

pub const APP_ID: &str = "local.dictation.Dictation";

/// Linux/macOS: the control socket's file, in the user's own runtime/temp folder.
pub fn socket_path() -> std::path::PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, Into::into).join("dictation.sock")
}

/// Where the running app listens (interprocess "local socket"): Linux/macOS the socket file above, Windows a named
/// pipe (pipe names are machine-wide: per user).
fn socket_name() -> std::io::Result<interprocess::local_socket::Name<'static>> {
    use interprocess::local_socket::{GenericFilePath, GenericNamespaced, ToFsName, ToNsName};
    if cfg!(windows) {
        format!("dictation-{}", std::env::var("USERNAME").unwrap_or_default()).to_ns_name::<GenericNamespaced>()
    } else {
        socket_path().to_fs_name::<GenericFilePath>()
    }
}

pub enum Ev {
    Toggle,
    /// New settings chosen in the tray menu.
    Set(Config),
    /// config.toml edited by hand (`dictation reload`).
    Download,
    Downloaded(Result<(), String>),
    ConfigureShortcut,
    Quit,
    /// Periodic check for the idle unload.
    Tick,
    UnloadNow,
    /// The global shortcut now in effect ("" = none).
    Shortcut(String),
    /// macOS/Windows: the chosen shortcut could not be registered (why).
    ShortcutFailed(String),
    EngineLoaded(Result<String, String>),
    Done(Result<Take, String>),
    /// Every 300 ms: in live mode, re-transcribe if enough new audio is in.
    LiveTick,
    /// Live mode: text to paste now (None: nothing new yet).
    Live(Option<String>),
    /// The one-time GPU optimization (sweep): share done, 0..1.
    SweepProgress(f64),
    /// The sweep finished; the model stays loaded. Ok = load/sweep info.
    Swept(Result<String, String>),
}

pub struct Take {
    audio: Vec<f32>,
    /// The whole take's text.
    text: String,
    /// What is left to paste (live mode pasted the rest while recording).
    paste: String,
    ms: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    NoModel,
    Idle,
    Recording,
    Transcribing,
    /// One-time per GPU: compiling (and tuning) kernels for every take length (Job::Sweep). No takes meanwhile.
    Optimizing,
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
    Load { weights: std::path::PathBuf, kernels: std::path::PathBuf },
    /// Load + sweep (asr::Asr::sweep) + mark the GPU's kernel cache as swept. Progress -> Ev::SweepProgress.
    Sweep { weights: std::path::PathBuf, kernels: std::path::PathBuf },
    Unload,
    Transcribe(Vec<f32>),
    /// Live mode (paste while speaking): a take starts.
    LiveStart,
    /// Live mode: everything recorded so far -> Ev::Live with newly agreed words.
    LiveCheck(Vec<f32>),
    /// Live mode: the take stopped (whole audio); the current window's final pass -> Ev::Done.
    LiveFinish(Vec<f32>),
}

fn timing_line(what: &str, s: &asr::Timing) {
    println!("  {what}mel {:.0} ms, pauses {:.0} ms, encoder {:.0} ms, decoder {:.0} ms", s.mel, s.vad, s.enc, s.dec);
}

fn engine(jobs: std_mpsc::Receiver<Job>, ev: mpsc::UnboundedSender<Ev>) {
    let mut model: Option<asr::Asr> = None;
    let mut live: Option<asr::Live> = None;
    for job in jobs {
        match job {
            Job::Load { weights, kernels } => {
                model = None; // free the old model first (VRAM)
                asr::set_kernel_cache(&kernels);
                let t = Instant::now();
                let r = (|| -> Result<String, String> {
                    let mut a = asr::Asr::load(&weights)?;
                    a.run(&vec![0.0; RATE])?; // warm-up (the first run of new kernels compiles them, then cached)
                    model = Some(a);
                    Ok(format!("loaded in {:.1} s", t.elapsed().as_secs_f64()))
                })();
                let _ = ev.send(Ev::EngineLoaded(r));
            }
            Job::Sweep { weights, kernels } => {
                model = None;
                asr::set_kernel_cache(&kernels);
                let t = Instant::now();
                let r = (|| -> Result<String, String> {
                    let mut a = asr::Asr::load(&weights)?;
                    let mut sent = 0.0;
                    a.sweep(false, |done, _, _| {
                        if done - sent >= 0.01 || done >= 1.0 {
                            sent = done;
                            let _ = ev.send(Ev::SweepProgress(done));
                        }
                    })?;
                    asr::mark_swept(&kernels).map_err(|e| e.to_string())?;
                    model = Some(a);
                    Ok(format!("optimized in {:.0} s", t.elapsed().as_secs_f64()))
                })();
                let _ = ev.send(Ev::Swept(r));
            }
            Job::Unload => {
                model = None;
                #[cfg(all(target_os = "linux", target_env = "gnu"))]
                unsafe {
                    libc::malloc_trim(0); // hand freed heap back to the OS (glibc keeps it otherwise)
                }
            }
            Job::Transcribe(x) => {
                let t = Instant::now();
                let r = match model.as_mut() {
                    Some(a) => a.run(&x),
                    None => Err("model not loaded".into()),
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if let Ok((_, s)) = &r {
                    timing_line("", s);
                }
                let _ = ev.send(Ev::Done(r.map(|(text, _)| Take { audio: x, paste: text.clone(), text, ms })));
            }
            Job::LiveStart => live = Some(asr::Live::default()),
            Job::LiveCheck(x) => {
                let mut piece = None;
                if let (Some(a), Some(l)) = (model.as_mut(), live.as_mut()) {
                    piece = l.check(a, &x);
                }
                let _ = ev.send(Ev::Live(piece));
            }
            Job::LiveFinish(x) => {
                let mut l = live.take().unwrap_or_default();
                let t = Instant::now();
                let r = match model.as_mut() {
                    Some(a) => l.finish(a, &x),
                    None => Err("model not loaded".into()),
                };
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let _ = ev.send(Ev::Done(r.map(|paste| Take { audio: x, text: l.text, paste, ms })));
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
    /// (percent, checking the SHA-256) while downloading the model.
    download: Option<(u32, bool)>,
    /// The GPU in use (asr::gpu) and every one there is (name, kind).
    gpu: String,
    gpus: Vec<(String, &'static str)>,
    /// While optimizing: percent done.
    sweep: u32,
    tx: mpsc::UnboundedSender<Ev>,
}

impl Tray {
    fn status_text(&self) -> String {
        let key = if self.shortcut.is_empty() { String::new() } else { format!(" — {}", self.shortcut) };
        match self.download {
            Some((_, true)) => return format!("Checking and converting {}…", models::MODEL.name),
            Some((pct, false)) => return format!("Downloading {}… {pct}%", models::MODEL.name),
            None => {}
        }
        match self.state {
            State::NoModel => "No model yet — click to download".into(),
            State::Idle => format!("Ready{key}"),
            State::Recording => format!("● Recording{key}"),
            State::Transcribing => "Transcribing…".into(),
            State::Optimizing => {
                format!("Speeding up dictation for {}… {}%", self.gpu, self.sweep)
            }
        }
    }

    fn mem_text(&self) -> String {
        if self.state == State::Optimizing {
            return "Makes dictation faster on this GPU".into();
        }
        match self.mem {
            Mem::Unloaded => "Model not loaded (loads when you start, ~1 s)".into(),
            Mem::Loading => "Loading model…".into(),
            Mem::Loaded => format!("Model loaded: {}", self.detail),
        }
    }
}

/// macOS/Windows: the shortcut choices in the menu (any other: `shortcut` in config.toml).
const SHORTCUTS: [&str; 4] = ["Ctrl+Alt+Space", "Ctrl+Shift+Space", "Ctrl+Alt+D", "F9"];

impl Tray {
    /// Linux: the desktop's shortcut settings (portal). macOS/Windows: pick from a list (no system place for it).
    fn shortcut_item(&self) -> tray::Item {
        if cfg!(target_os = "linux") {
            return tray::Item::Action {
                label: if self.shortcut.is_empty() { "Set shortcut…".into() } else { format!("Change shortcut ({})…", self.shortcut) },
                enabled: true,
                click: self.send(|| Ev::ConfigureShortcut),
            };
        }
        let mut keys: Vec<String> = SHORTCUTS.iter().map(|k| k.to_string()).collect();
        keys.push(String::new());
        if !keys.contains(&self.cfg.shortcut) {
            keys.insert(0, self.cfg.shortcut.clone()); // one typed into config.toml
        }
        let selected = keys.iter().position(|k| *k == self.cfg.shortcut).unwrap_or(0);
        let options = keys
            .into_iter()
            .map(|k| {
                let label = if k.is_empty() { "None".to_string() } else { k.clone() };
                (label, self.setting(move |c| c.shortcut = k.clone()))
            })
            .collect();
        tray::Item::Sub {
            label: format!("Shortcut ({})", if self.shortcut.is_empty() { "none" } else { &self.shortcut }),
            items: vec![
                tray::Item::Radio { options, selected },
                tray::Item::Label("Other keys: \"shortcut\" in config.toml".into()),
            ],
        }
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

impl Tray {
    /// A menu click that sends this event.
    fn send(&self, ev: impl Fn() -> Ev + Send + Sync + 'static) -> tray::Click {
        let tx = self.tx.clone();
        tray::click(move || {
            let _ = tx.send(ev());
        })
    }

    /// A menu click that changes a setting.
    fn setting(&self, f: impl Fn(&mut Config) + Send + Sync + 'static) -> tray::Click {
        let (tx, cfg) = (self.tx.clone(), self.cfg.clone());
        tray::click(move || {
            let mut c = cfg.clone();
            f(&mut c);
            let _ = tx.send(Ev::Set(c));
        })
    }
}

impl tray::Model for Tray {
    fn icon(&self) -> Icon {
        match self.state {
            State::Recording => Icon::Recording,
            _ if self.download.is_some() => Icon::Downloading,
            State::Transcribing => Icon::Transcribing,
            State::Optimizing => Icon::Optimizing,
            State::NoModel => Icon::Warning,
            State::Idle => Icon::Idle,
        }
    }
    fn attention(&self) -> bool {
        self.state == State::Recording
    }
    fn tooltip(&self) -> (String, String) {
        (format!("Dictation: {}", self.status_text()), self.mem_text())
    }
    fn activate(&self) {
        if self.state == State::NoModel && self.download.is_none() {
            let _ = self.tx.send(Ev::Download);
        } else {
            let _ = self.tx.send(Ev::Toggle);
        }
    }
    fn menu_opening(&mut self) {
        self.mics = audio::input_devices(); // pick up newly plugged microphones
    }
    fn hotkey(&self) -> String {
        self.cfg.shortcut.clone()
    }
    fn hotkey_pressed(&self) {
        let _ = self.tx.send(Ev::Toggle);
    }
    fn hotkey_result(&self, key: &str, result: Result<(), String>) {
        match result {
            Ok(()) => {
                let _ = self.tx.send(Ev::Shortcut(key.to_string()));
            }
            Err(e) => {
                let _ = self.tx.send(Ev::Shortcut(String::new()));
                let _ = self.tx.send(Ev::ShortcutFailed(e));
            }
        }
    }
    fn menu(&self) -> Vec<tray::Item> {
        use tray::Item::*;
        let recording = self.state == State::Recording;

        let mut gpu_options = vec![("Automatic (fastest)".to_string(), self.setting(|c| c.gpu = String::new()))];
        for (name, kind) in &self.gpus {
            let n = name.clone();
            gpu_options.push((format!("{name} ({kind})"), self.setting(move |c| c.gpu = n.clone())));
        }
        let gpu_selected = self.gpus.iter().position(|g| g.0 == self.cfg.gpu).map_or(0, |i| i + 1);

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

        let mut items = vec![Label(self.status_text()), Label(self.mem_text())];
        if self.state == State::Optimizing {
            // Own line: one long line made the whole menu too wide.
            items.push(Label("Runs once; the computer may feel slow until it's done".into()));
        }
        items.extend([
            Separator,
            Action {
                label: if recording { "Stop dictation" } else { "Start dictation" }.into(),
                enabled: matches!(self.state, State::Idle | State::Recording | State::Transcribing),
                click: self.send(|| Ev::Toggle),
            },
            Separator,
            Sub {
                label: "GPU".into(),
                items: vec![
                    Radio { options: gpu_options, selected: gpu_selected },
                    Separator,
                    Label(format!("In use: {}", self.gpu)),
                    Label("Switching restarts the app".into()),
                ],
            },
            Sub {
                label: "Keep model loaded".into(),
                items: vec![
                    Radio {
                        options: KEEP.iter().map(|&(k, l)| (l.to_string(), self.setting(move |c| c.keep_loaded = k))).collect(),
                        selected: KEEP.iter().position(|&(k, _)| k == self.cfg.keep_loaded).unwrap_or(1),
                    },
                    Separator,
                    Action {
                        label: "Unload now".into(),
                        enabled: self.mem == Mem::Loaded && self.state == State::Idle,
                        click: self.send(|| Ev::UnloadNow),
                    },
                ],
            },
            Sub {
                label: "Microphone".into(),
                items: vec![Radio {
                    options: self
                        .mics
                        .iter()
                        .map(|(id, name)| {
                            let id = id.clone();
                            (name.clone(), self.setting(move |c| c.microphone = id.clone()))
                        })
                        .collect(),
                    selected: self.mics.iter().position(|(id, _)| *id == self.cfg.microphone).unwrap_or(0),
                }],
            },
            Sub {
                label: "Output".into(),
                items: vec![
                    Radio {
                        options: outputs
                            .iter()
                            .map(|&(o, k)| {
                                let set = self.setting(move |c| {
                                    c.output = o;
                                    c.paste_keys = k;
                                });
                                (output_label(o, k).to_string(), set)
                            })
                            .collect(),
                        selected: out_selected,
                    },
                    Separator,
                    Check {
                        label: "Paste while speaking".into(),
                        checked: self.cfg.live,
                        enabled: self.cfg.output == Output::Paste,
                        click: self.setting(|c| c.live = !c.live),
                    },
                    // Live words are pasted once two passes agree, with only ~1-2 s of audio after them; pasted text
                    // is never corrected. Without live mode the whole take is transcribed at once (full context).
                    Label("Slightly less accurate: words are pasted".into()),
                    Label("before the model hears what follows".into()),
                ],
            },
            Check { label: "Sounds".into(), checked: self.cfg.sounds, enabled: true, click: self.setting(|c| c.sounds = !c.sounds) },
            Check {
                label: "Notifications".into(),
                checked: self.cfg.notifications,
                enabled: true,
                click: self.setting(|c| c.notifications = !c.notifications),
            },
            Check {
                label: "Keep recordings".into(),
                checked: self.cfg.save_recordings,
                enabled: true,
                click: self.setting(|c| c.save_recordings = !c.save_recordings),
            },
            Action {
                label: "Open app folder…".into(),
                enabled: true,
                click: tray::click(|| {
                    if let Err(e) = crate::open_root() {
                        eprintln!("{e}");
                    }
                }),
            },
            self.shortcut_item(),
            Separator,
            Action { label: "Quit".into(), enabled: true, click: self.send(|| Ev::Quit) },
        ]);
        items
    }
}

// ---------------------------------------------------------------- global shortcut (XDG portal)

#[cfg(target_os = "linux")]
async fn shortcut_task(tx: mpsc::UnboundedSender<Ev>, mut configure: mpsc::UnboundedReceiver<()>) -> ashpd::Result<()> {
    use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut, Shortcut};
    // The portal identifies the app by its .desktop file: make sure it exists and points here.
    let written = match ensure_desktop_file() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("desktop file: {e}");
            false
        }
    };
    // A just-written file may take the desktop a moment to notice: retry for a few seconds then.
    let mut tries = if written { 10 } else { 1 };
    loop {
        match ashpd::register_host_app(APP_ID.try_into()?).await {
            Ok(()) => break,
            Err(e) if tries > 1 => {
                tries -= 1;
                eprintln!("register app id: {e}; retrying");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) => return Err(e),
        }
    }
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

/// The app's `.desktop` file (`~/.local/share/applications/<APP_ID>.desktop`; the desktop's app list and what the
/// shortcut portal identifies us by), pointing at this program. Written when missing or different (e.g. the program
/// moved). -> whether it was written.
#[cfg(target_os = "linux")]
fn ensure_desktop_file() -> Result<bool, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    // Exec: quoted, with the characters the spec wants escaped inside quotes.
    let mut quoted = String::from("\"");
    for c in exe.to_string_lossy().chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    quoted.push('"');
    let text = format!(
        "[Desktop Entry]\nType=Application\nName=Dictation\nComment=Offline speech-to-text\nExec={quoted}\n\
         Icon=audio-input-microphone\nTerminal=false\nCategories=Utility;\n"
    );
    let dir = directories::BaseDirs::new().ok_or("no home directory")?.data_dir().join("applications");
    let path = dir.join(format!("{APP_ID}.desktop"));
    if std::fs::read_to_string(&path).is_ok_and(|t| t == text) {
        return Ok(false);
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(true)
}

// ---------------------------------------------------------------- control socket (`dictation toggle`; `status` = single-instance check)

async fn socket_task(tx: mpsc::UnboundedSender<Ev>, status: tokio::sync::watch::Receiver<String>) {
    use interprocess::local_socket::ListenerOptions;
    use interprocess::local_socket::tokio::prelude::*;
    // A leftover socket file from a crash is replaced (run() already checked no other copy answers).
    let listener = match socket_name().and_then(|n| ListenerOptions::new().name(n).try_overwrite(true).create_tokio()) {
        Ok(l) => l,
        Err(e) => return eprintln!("control socket: {e}"),
    };
    while let Ok(conn) = listener.accept().await {
        let (tx, status) = (tx.clone(), status.clone());
        tokio::spawn(async move {
            let mut line = String::new();
            if BufReader::new(&conn).read_line(&mut line).await.is_err() {
                return;
            }
            let reply = match line.trim() {
                "toggle" => tx.send(Ev::Toggle).map(|_| "ok".to_string()),
                "status" => Ok(status.borrow().clone()),
                _ => Ok("unknown command".to_string()),
            }
            .unwrap_or_else(|_| "shutting down".into());
            let _ = (&conn).write_all(format!("{reply}\n").as_bytes()).await;
        });
    }
}

/// Send one command to a running app. -> reply line.
pub fn send(cmd: &str) -> std::io::Result<String> {
    use interprocess::local_socket::{Stream, prelude::*};
    use std::io::{BufRead, Write};
    let s = Stream::connect(socket_name()?)?;
    s.set_recv_timeout(Some(Duration::from_secs(2)))?;
    let mut s = std::io::BufReader::new(s);
    s.get_mut().write_all(format!("{cmd}\n").as_bytes())?;
    let mut line = String::new();
    s.read_line(&mut line)?;
    Ok(line.trim().to_string())
}

/// Press the paste keys, making the virtual keyboard first if it isn't there yet (e.g. permission granted since).
fn press_paste(kb: &mut Option<Keyboard>, keys: PasteKeys) -> Result<(), String> {
    if kb.is_none() {
        *kb = Some(Keyboard::new(false).map_err(|e| e.to_string())?);
    }
    let r = kb.as_mut().map_or(Ok(()), |k| k.paste(keys)).map_err(|e| e.to_string());
    if r.is_err() {
        *kb = None;
    }
    r
}

// ---------------------------------------------------------------- recordings (wav + one JSON line per take)

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
        "text": t.text, "ms": t.ms.round()});
    writeln!(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("log.jsonl"))?, "{line}")
}

/// Local time as YYYYMMDD-HHMMSS.
fn time_name() -> String {
    jiff::ZonedDateTime::now().strftime("%Y%m%d-%H%M%S").to_string()
}

// ---------------------------------------------------------------- main loop

/// `log` messages (wgpu and other crates) -> stderr, warnings and errors only, prefixed with their source.
struct StderrLog;
impl log::Log for StderrLog {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}: {}", r.target(), r.level(), r.args());
        }
    }
    fn flush(&self) {}
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    static LOG: StderrLog = StderrLog;
    if log::set_logger(&LOG).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    if send("status").is_ok() {
        return Err("Dictation is already running (see the tray).".into());
    }
    let cfg = Config::load();
    let (tx, rx) = mpsc::unbounded_channel::<Ev>();
    let state = if models::MODEL.installed() { State::Idle } else { State::NoModel };
    let model = Tray {
        state,
        mem: Mem::Unloaded,
        shortcut: String::new(),
        detail: String::new(),
        cfg: cfg.clone(),
        mics: audio::input_devices(),
        download: None,
        gpu: asr::gpu().to_string(),
        gpus: asr::gpus().into_iter().map(|g| (g.name, g.kind)).collect(),
        sweep: 0,
        tx: tx.clone(),
    };
    // The tray on this (main) thread, everything else on the app thread's runtime.
    let restart = tray::run(model, move |tray| -> Result<bool, String> {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().map_err(|e| e.to_string())?;
        let r = rt.block_on(main_loop(cfg, state, tx, rx, tray)).map_err(|e| e.to_string());
        rt.shutdown_background();
        r
    })??;
    if restart {
        // Another GPU: CubeCL picks the GPU and its kernel cache once per process, so start over.
        let exe = std::env::current_exe().ok().filter(|p| p.exists()).unwrap_or_else(|| std::env::args().next().unwrap_or_default().into());
        let mut cmd = std::process::Command::new(exe);
        // The GPU we chose (asr::choose_gpu) is in our environment; without this, "Automatic" after a named GPU
        // would inherit it and stay on that GPU. The new process picks again from config.toml.
        cmd.env_remove("CUBECL_WGPU_DEFAULT_DEVICE");
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            return Err(format!("restart: {}", cmd.exec()).into());
        }
        #[cfg(not(unix))]
        cmd.spawn()?;
    }
    Ok(())
}

fn start_download(tx: &mpsc::UnboundedSender<Ev>, tray: &tray::Handle<Tray>) {
    let p = Arc::new(models::Progress::default());
    {
        let (p, tx) = (p.clone(), tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(Ev::Downloaded(models::download(&models::MODEL, &p)));
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
                    t.download = Some((pct, verifying));
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

/// -> whether to restart (another GPU was chosen).
async fn main_loop(
    mut cfg: Config,
    mut state: State,
    tx: mpsc::UnboundedSender<Ev>,
    mut rx: mpsc::UnboundedReceiver<Ev>,
    tray: tray::Handle<Tray>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (jobs, jobs_rx) = std_mpsc::channel::<Job>();
    {
        let tx = tx.clone();
        std::thread::Builder::new().name("engine".into()).spawn(move || engine(jobs_rx, tx))?;
    }
    let installed = || models::MODEL.installed();
    let (status_tx, status_rx) = tokio::sync::watch::channel(String::new());
    tokio::spawn(socket_task(tx.clone(), status_rx));
    let (configure_tx, configure_rx) = mpsc::unbounded_channel();
    // Linux: the portal. macOS/Windows: the tray backend registers `cfg.shortcut` (tray::Model::hotkey).
    #[cfg(not(target_os = "linux"))]
    drop(configure_rx);
    #[cfg(target_os = "linux")]
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
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(300)).await;
                if tx.send(Ev::LiveTick).is_err() {
                    break;
                }
            }
        });
    }
    let mut clipboard = crate::output::Clipboard::default();
    // Without it the app still works, the text just stays on the clipboard (press_paste retries, notices say why).
    let mut kb = Keyboard::new(true).map_err(|e| eprintln!("virtual keyboard: {e}")).ok();
    let mut notice = Notice { enabled: cfg.notifications, ..Default::default() };
    let mut rec: Option<audio::Recorder> = None;
    let mut shortcut = String::new();
    let mut busy = 0usize; // takes queued in the engine
    let mut downloading = false;
    let mut mem = Mem::Unloaded;
    let mut last_used = Instant::now();
    let mut live_on = false; // the current take pastes piece by piece
    let mut live_busy = false; // a LiveCheck is queued in the engine
    let mut load_failed = false; // stops "Always" from retrying a broken load every tick; a take or settings change retries
    let mut sweep_tried = false; // once per run: a failed sweep is not retried (takes still work, first ones slower)
    let mut restart = false; // another GPU chosen: restart once no take is running

    // Queue a (re)load of the configured model on the engine thread. Takes queued after it wait for it.
    let load = |mem: &mut Mem| {
        let m = &models::MODEL;
        let _ = jobs.send(Job::Load { weights: m.weights(), kernels: m.kernel_cache() });
        *mem = Mem::Loading;
    };
    if state == State::NoModel && models::MODEL.download_started() {
        // A download to finish, or a GGUF from an older version of the app to check and convert.
        downloading = true;
        start_download(&tx, &tray);
        tray.update(|t: &mut Tray| t.download = Some((0, false))).await;
    } else if state == State::NoModel {
        notice.show("Dictation needs a model", "Click the tray icon to download Parakeet Ultra (740 MB).", "dialog-information").await;
    }
    let _ = tx.send(Ev::Tick); // first pass of the checks below: optimizing for the GPU, "Always" loading

    let _ = status_tx.send(serde_json::json!({"state": format!("{state:?}"), "model_memory": format!("{mem:?}"), "shortcut": ""}).to_string());
    loop {
        let Some(ev) = rx.recv().await else { break };
        // Live mode events change no state shown in the tray: handled here, without the bookkeeping below.
        let ev = match ev {
            Ev::LiveTick => {
                if let Some(r) = rec.as_ref().filter(|_| live_on && !live_busy && mem == Mem::Loaded) {
                    let _ = jobs.send(Job::LiveCheck(r.snapshot()));
                    live_busy = true;
                }
                continue;
            }
            Ev::SweepProgress(done) => {
                tray.update(move |t: &mut Tray| t.sweep = (done * 100.0) as u32).await;
                continue;
            }
            Ev::Live(piece) => {
                live_busy = false;
                if let Some(p) = piece {
                    if let Err(e) = clipboard.set(&p) {
                        eprintln!("clipboard: {e}");
                    } else if let Err(e) = press_paste(&mut kb, cfg.paste_keys) {
                        eprintln!("paste: {e}");
                    }
                }
                continue;
            }
            ev => ev,
        };
        match ev {
            Ev::Toggle => match (state, rec.take()) {
                (State::Idle | State::Transcribing, None) => match audio::Recorder::start(&cfg.microphone) {
                    Ok(r) => {
                        rec = Some(r);
                        state = State::Recording;
                        last_used = Instant::now();
                        if mem == Mem::Unloaded {
                            load_failed = false;
                            load(&mut mem); // loads while you speak
                        }
                        live_on = cfg.live && cfg.output == Output::Paste;
                        if live_on {
                            let _ = jobs.send(Job::LiveStart);
                        }
                        if cfg.sounds {
                            audio::play(audio::Cue::Start);
                        }
                    }
                    Err(e) => notice.show("Microphone error", &e, "dialog-error").await,
                },
                (_, Some(r)) => {
                    if cfg.sounds {
                        audio::play(audio::Cue::Stop);
                    }
                    state = if busy > 0 { State::Transcribing } else { State::Idle };
                    match r.stop() {
                        Ok(x) if live_on => {
                            busy += 1;
                            let _ = jobs.send(Job::LiveFinish(x));
                            state = State::Transcribing;
                        }
                        Ok(x) if x.len() >= RATE / 4 => {
                            busy += 1;
                            let _ = jobs.send(Job::Transcribe(x));
                            state = State::Transcribing;
                        }
                        Ok(_) => notice.show("Too short, skipped", "", "dialog-information").await,
                        Err(e) => notice.show("Microphone error", &e, "dialog-error").await,
                    }
                    live_on = false;
                    last_used = Instant::now();
                }
                (State::Optimizing, None) => {
                    notice
                        .show("Still speeding up dictation for your GPU", "One-time tuning. Dictation is available when it's done; progress is in the tray menu.", "dialog-information")
                        .await
                }
                (State::NoModel, None) if !downloading => {
                    downloading = true;
                    start_download(&tx, &tray);
                    tray.update(|t: &mut Tray| t.download = Some((0, false))).await;
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
                        // (title, body) of a notice about what went wrong.
                        let mut problem: Option<(&str, String)> = None;
                        if !t.paste.is_empty() {
                            if let Err(e) = clipboard.set(&t.paste) {
                                eprintln!("clipboard: {e}");
                                problem = Some(("Couldn't copy the text", e.to_string()));
                            } else if cfg.output == Output::Paste {
                                if let Err(e) = press_paste(&mut kb, cfg.paste_keys) {
                                    eprintln!("paste: {e}");
                                    problem = Some(("Not pasted: the text is on the clipboard", crate::output::PASTE_HELP.into()));
                                }
                            }
                        }
                        let body = if t.text.chars().count() > 200 { t.text.chars().take(200).collect::<String>() + "…" } else { t.text.clone() };
                        let title = match (t.text.is_empty(), cfg.output) {
                            (true, _) => "Nothing recognized",
                            (false, Output::Paste) => "Pasted",
                            (false, Output::Clipboard) => "Copied to clipboard",
                        };
                        if let Some((title, body)) = problem {
                            notice.show(title, &body, "dialog-error").await;
                        } else if state != State::Recording {
                            notice.show(title, &body, if t.text.is_empty() { "dialog-information" } else { "dialog-ok" }).await;
                        }
                        println!("{:.1}s audio, {:.0} ms: {}", t.audio.len() as f64 / RATE as f64, t.ms, t.text);
                        if cfg.save_recordings {
                            if let Err(e) = save_take(&t) {
                                eprintln!("save recording: {e}");
                            }
                        }
                    }
                    Err(e) => notice.show("Transcription failed", &e, "dialog-error").await,
                }
            }
            Ev::EngineLoaded(r) => match r {
                Ok(info) => {
                    println!("model loaded ({info})");
                    mem = Mem::Loaded;
                    let detail = format!("{}, {info}", models::MODEL.name);
                    tray.update(move |t: &mut Tray| t.detail = detail).await;
                }
                Err(e) => {
                    eprintln!("model load failed: {e}");
                    mem = Mem::Unloaded; // the next take tries again
                    load_failed = true;
                    notice.show("Model failed to load", &e, "dialog-error").await;
                }
            },
            Ev::Swept(r) => {
                state = State::Idle;
                last_used = Instant::now();
                let gpu = asr::gpu();
                match r {
                    Ok(info) => {
                        println!("GPU optimized: {gpu}, {info}");
                        mem = Mem::Loaded;
                        let detail = format!("{}, {info}", models::MODEL.name);
                        tray.update(move |t: &mut Tray| t.detail = detail).await;
                        notice.show("Dictation is ready", &format!("Tuned for {gpu}. This won't be needed again."), "dialog-ok").await;
                    }
                    Err(e) => {
                        eprintln!("GPU optimization failed: {e}");
                        mem = Mem::Unloaded;
                        notice.show("GPU optimization failed", &format!("{e}\nDictation still works; the first takes may be slow."), "dialog-error").await;
                    }
                }
            }
            Ev::Tick => {}
            Ev::UnloadNow => {
                if rec.is_none() && busy == 0 && mem == Mem::Loaded {
                    let _ = jobs.send(Job::Unload);
                    mem = Mem::Unloaded;
                    println!("model unloaded");
                }
            }
            Ev::Download => {
                if !downloading {
                    downloading = true;
                    start_download(&tx, &tray);
                    tray.update(|t: &mut Tray| t.download = Some((0, false))).await;
                }
            }
            Ev::Downloaded(r) => {
                downloading = false;
                tray.update(|t: &mut Tray| t.download = None).await;
                match r {
                    Ok(()) => notice.show("Model downloaded", models::MODEL.name, "dialog-ok").await,
                    Err(e) => notice.show("Download failed", &e, "dialog-error").await,
                }
            }
            Ev::Set(new) => {
                if let Err(e) = new.save() {
                    eprintln!("save config: {e}");
                }
                restart |= new.gpu != cfg.gpu;
                cfg = new;
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
            Ev::ShortcutFailed(e) => notice.show("Shortcut not set", &format!("{e}. Pick another in the tray menu."), "dialog-error").await,
            Ev::Quit => break,
            Ev::LiveTick | Ev::Live(_) | Ev::SweepProgress(_) => unreachable!("handled above"),
        }
        notice.enabled = cfg.notifications;
        if state == State::NoModel && installed() {
            state = State::Idle;
        } else if state == State::Idle && busy == 0 && !installed() {
            // Deleted from the models folder: offer the download again.
            state = State::NoModel;
            if mem != Mem::Unloaded {
                let _ = jobs.send(Job::Unload);
                mem = Mem::Unloaded;
            }
            println!("model file missing: {}", models::MODEL.weights().display());
        }
        // A GPU change applies once the current take is pasted.
        if restart && rec.is_none() && busy == 0 {
            println!("GPU changed to {:?}: restarting", if cfg.gpu.is_empty() { "automatic" } else { &cfg.gpu });
            break;
        }
        // Once per GPU (and build): compile/tune kernels for every take length before the first take, so no take
        // waits for that. Takes are blocked meanwhile; the tray shows progress.
        if state == State::Idle && rec.is_none() && busy == 0 && !sweep_tried && !asr::swept(&models::MODEL.kernel_cache()) {
            sweep_tried = true;
            let m = &models::MODEL;
            let _ = jobs.send(Job::Sweep { weights: m.weights(), kernels: m.kernel_cache() });
            state = State::Optimizing;
            mem = Mem::Loading;
            tray.update(|t: &mut Tray| t.sweep = 0).await;
            println!("optimizing for {}", asr::gpu());
            let body = format!(
                "Tuning for {} so every dictation runs faster. This runs only once and can take a few minutes, \
                 longer on integrated graphics. It works the GPU much harder than dictation does, so the computer \
                 may feel slow until it's done. Progress is in the tray menu.",
                asr::gpu()
            );
            notice.show("Speeding up dictation for your GPU", &body, "dialog-information").await;
        }
        if rec.is_none() && busy == 0 && !matches!(state, State::NoModel | State::Optimizing) {
            if cfg.keep_loaded == KeepLoaded::Always && mem == Mem::Unloaded && !load_failed {
                load(&mut mem);
            }
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
            serde_json::json!({"state": format!("{state:?}"), "model_memory": format!("{mem:?}"), "shortcut": shortcut})
                .to_string(),
        );
    }
    if let Some(r) = rec {
        let _ = r.stop();
    }
    #[cfg(unix)]
    let _ = std::fs::remove_file(socket_path());
    Ok(restart)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(unix)] // compares with `date`
    fn time_name_is_local_yyyymmdd_hhmmss() {
        let n = super::time_name();
        let date = std::process::Command::new("date").arg("+%Y%m%d-%H%M").output().unwrap();
        assert!(n.starts_with(String::from_utf8_lossy(&date.stdout).trim()), "{n}");
        assert_eq!(n.len(), 15);
    }
}
