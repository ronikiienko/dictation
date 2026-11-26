# Dictation

Offline speech-to-text in the tray, using the Parakeet speech model. The main goal of the project is speed: most of
the work went into getting the text as soon as possible after you stop talking. The other goal is running the same
app on every system.

## Features

- Press a shortcut, speak, press it again: the text is pasted where your cursor is.
- Runs entirely on your machine. The only network use is the one-time model download.
- 25 European languages, no need to pick one. Tested daily in English and Ukrainian.
- Transcribes audio and video files to text or subtitles (`.srt`) from the tray menu.
- Records from the microphone or from system audio.
- Runs on the GPU (Vulkan) on Linux and Windows, on the Neural Engine on macOS, or on the CPU. On a MacBook Air M1
  a 30 s take is text in about 0.2 s.
- Can keep your recordings (off by default).

## Supported systems

| System | State |
|---|---|
| Linux, KDE Plasma (Wayland) | used daily |
| macOS (Apple Silicon) | used daily |
| Windows | not tested: the code is written, but the app hasn't been built or run there |
| Other Linux desktops | untested |

## Install

macOS: download `Dictation-macos-arm64.zip` from Releases, unzip it and move `Dictation.app` to Applications. The
app isn't signed by Apple, so the first open is blocked: then System Settings → Privacy & Security → Open Anyway.

## First start

The app downloads the model (~740 MB; ~1.26 GB on macOS) and prepares it for your machine, which
takes 1–2 minutes. Progress is shown in the tray menu. Later starts take about a second.

## Use

- Shortcut: Ctrl+Alt+Space on macOS, Meta+Y suggested on Linux (the desktop asks you to confirm it). Change it in
  the tray menu.
- All settings are in the tray menu.
- The model takes ~900 MB of memory while loaded (~1.2 GB on the Neural Engine). By default it's unloaded 1 minute
  after the last take, and the next take loads it again in about a second. Tray menu → Keep model loaded.
- Settings, models and saved recordings are in one folder: `~/.local/share/dictation/` on Linux,
  `~/Library/Application Support/dictation/` on macOS. Delete it to remove everything.

## Which model to pick

Tray menu → Model. Switching restarts the app.

- **Parakeet Ultra on the GPU** (default on Linux and Windows): the most accurate.
- **Parakeet Ultra — Neural Engine** (default on macOS): same accuracy.
- **Parakeet Ultra — CPU**: same accuracy, slower. For machines without a usable GPU.
- **Parakeet Redux — CPU**: a compressed version of Ultra. Uses about 380 MB of memory instead of ~900 MB, but
  makes more mistakes (on our English test takes, 10% of words wrong instead of 6.5%; worse on Ukrainian and
  technical terms). Pick it only when memory is tight.

## Notes per system

**Linux**

- To paste, the app needs access to `/dev/uinput`. Run once, then log out and back in:

  ```sh
  echo 'KERNEL=="uinput", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/70-uinput.rules
  ```

- The tray icon needs a desktop with StatusNotifierItem support, and the shortcut needs the desktop's global
  shortcuts portal. KDE Plasma has both.

**macOS**

- On the first paste macOS asks to allow the app under Accessibility.
- Neural Engine: the memory it uses doesn't show in Activity Monitor. When free disk space gets low (under ~15 GB),
  macOS deletes the prepared model, and the next take waits ~40 s while it's prepared again.

## Credits

- Models (CC-BY-4.0, downloaded by the app):
  [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) and
  [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) by moondream, based on
  [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3); 8-bit GGUF by
  [handy-computer](https://huggingface.co/handy-computer/parakeet-ultra-gguf).
  `assets/parakeet-ultra-ane/` is derived from Parakeet Ultra.
- The Rust pipeline is a port of [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT).
- The GPU path is built on [burn](https://github.com/tracel-ai/burn) and CubeCL.

## License

MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)), at your option.
