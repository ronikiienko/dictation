#!/usr/bin/env python3
"""Proof of concept: dictate into whatever window is focused, anywhere in KDE (Wayland).

  .venv/bin/python experiments/dictate.py            run the app (keeps the model loaded; Ctrl+C quits)
  .venv/bin/python experiments/dictate.py toggle     start / stop a take in the running app (bind this to a KDE shortcut)

Shortcut (once): System Settings -> Keyboard -> Shortcuts -> Add New -> Command or Script:
  /home/rostyslav/Documents/WhateverProjects/dictation/.venv/bin/python
      /home/rostyslav/Documents/WhateverProjects/dictation/dictate.py toggle
and give it a key (e.g. Meta+D).

Flow: first press -> sound, recording (Silero VAD follows along, as in gauge); second press -> sound, the take is cut
at pauses and transcribed (Parakeet Ultra Q8_0 via transcribe.cpp, gauge's --final ultra-vk), the text is put on the
clipboard (Klipper over D-Bus) and pasted with Ctrl+V from a virtual keyboard (/dev/uinput; KDE gives the logged-in
user write access). Clipboard + paste keeps Ukrainian/English/punctuation exact whatever keyboard layout is active.
Terminals paste with Ctrl+Shift+V, so there you get nothing (POC). Takes are saved to recordings/ like gauge's.
"""
import fcntl
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "dictation.sock")
SOUND_START = "/usr/share/sounds/ocean/stereo/audio-volume-change.oga"
SOUND_STOP = "/usr/share/sounds/ocean/stereo/completion-success.oga"


class Keyboard:
    """Minimal virtual keyboard on /dev/uinput (only the keys we press)."""
    EV_SYN, EV_KEY, SYN_REPORT = 0, 1, 0
    KEY_LEFTCTRL, KEY_V = 29, 47
    UI_SET_EVBIT, UI_SET_KEYBIT, UI_DEV_SETUP, UI_DEV_CREATE, UI_DEV_DESTROY = 0x40045564, 0x40045565, 0x405C5503, 0x5501, 0x5502

    def __init__(self):
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, self.UI_SET_EVBIT, self.EV_KEY)
        for k in (self.KEY_LEFTCTRL, self.KEY_V):
            fcntl.ioctl(self.fd, self.UI_SET_KEYBIT, k)
        # struct uinput_setup: input_id {bustype=BUS_VIRTUAL, vendor, product, version}, name[80], ff_effects_max
        fcntl.ioctl(self.fd, self.UI_DEV_SETUP, struct.pack("HHHH80sI", 0x06, 0x1, 0x1, 1, b"dictation-keyboard", 0))
        fcntl.ioctl(self.fd, self.UI_DEV_CREATE)
        time.sleep(0.5)  # give KWin a moment to pick up the new device before the first keys

    def _ev(self, typ, code, value):
        os.write(self.fd, struct.pack("llHHi", 0, 0, typ, code, value))

    def _key(self, code, down):
        self._ev(self.EV_KEY, code, 1 if down else 0)
        self._ev(self.EV_SYN, self.SYN_REPORT, 0)

    def paste(self):
        self._key(self.KEY_LEFTCTRL, True)
        self._key(self.KEY_V, True)
        time.sleep(0.02)
        self._key(self.KEY_V, False)
        self._key(self.KEY_LEFTCTRL, False)

    def close(self):
        fcntl.ioctl(self.fd, self.UI_DEV_DESTROY)
        os.close(self.fd)


def set_clipboard(text):
    subprocess.run(["qdbus6", "org.kde.klipper", "/klipper", "setClipboardContents", text], check=True,
                   stdout=subprocess.DEVNULL)


def sound(path):
    subprocess.Popen(["paplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Notice:
    """One desktop notification, updated in place: recording (stays until stop) -> transcribing -> result."""

    def __init__(self):
        self.id = None

    def show(self, title, body="", stay=True, icon="audio-input-microphone"):
        cmd = ["notify-send", "-p", "-a", "Dictation", "-i", icon, title, body]
        cmd += ["-u", "critical"] if stay else ["-u", "normal", "-t", "4000", "-e"]  # critical: no timeout in KDE
        if self.id:
            cmd += ["-r", self.id]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        self.id = out or self.id


class Recorder:
    """parecord + incremental Silero VAD in a thread, as gauge's live loop."""

    def __init__(self, gauge):
        self.g = gauge
        self.vad = gauge.Vad()
        self.chunks = []
        self.proc = subprocess.Popen(["parecord", "--raw", "--format=s16le", f"--rate={gauge.RATE}", "--channels=1",
                                      "--latency-msec=50"], stdout=subprocess.PIPE)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while data := self.proc.stdout.read(self.g.CHUNK * 2):
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
            self.chunks.append(x)
            self.vad.accept(x)

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.thread.join()
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)
        return audio, self.vad.gaps()


def _quit(*_):
    raise KeyboardInterrupt


def serve():
    signal.signal(signal.SIGINT, _quit)   # also when started from a script (background jobs inherit SIGINT ignored)
    signal.signal(signal.SIGTERM, _quit)  # e.g. later as a systemd user service
    import gauge
    t = time.perf_counter()
    final = gauge.load_final("ultra-vk", 4)
    gauge.FINAL = "ultra-vk"
    kb = Keyboard()
    print(f"model loaded in {time.perf_counter() - t:.1f}s; waiting for `dictate.py toggle` (Ctrl+C quits)", flush=True)

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    rec = None
    notice = Notice()
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                cmd = conn.recv(64).decode().strip()
                if cmd != "toggle":
                    conn.sendall(b"unknown command\n")
                    continue
                if rec is None:
                    rec = Recorder(gauge)
                    sound(SOUND_START)
                    notice.show("● Recording…", "Press the shortcut again to stop")
                    conn.sendall(b"recording\n")
                    print("● recording", flush=True)
                    continue
                audio, gaps = rec.stop()
                rec = None
                sound(SOUND_STOP)
                conn.sendall(b"stopped\n")
            if len(audio) < gauge.RATE // 4:
                notice.show("Too short, skipped", stay=False, icon="dialog-information")
                print("  (too short, skipped)", flush=True)
                continue
            notice.show("Transcribing…", f"{len(audio) / gauge.RATE:.1f} s of audio", icon="view-refresh")
            t = time.perf_counter()
            text, pieces = gauge.transcribe(final, audio, gaps)
            ms = (time.perf_counter() - t) * 1000
            if text:
                set_clipboard(text)
                kb.paste()
            notice.show("Pasted" if text else "Nothing recognized",
                        (text[:200] + "…" if len(text) > 200 else text), stay=False,
                        icon="dialog-ok" if text else "dialog-information")
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
            gauge.RECORDINGS.mkdir(exist_ok=True)
            gauge.write_wav(gauge.RECORDINGS / f"{name}.wav", audio)
            with open(gauge.RECORDINGS / "log.jsonl", "a") as f:
                f.write(json.dumps({"wav": f"{name}.wav", "seconds": round(len(audio) / gauge.RATE, 2),
                                    "parakeet": text, "final_model": "ultra-vk", "parakeet_ms": round(ms),
                                    "pieces_s": [round(p, 2) for p in pieces], "app": "dictate"},
                                   ensure_ascii=False) + "\n")
            print(f"  {len(audio) / gauge.RATE:.1f}s audio, {ms:.0f} ms: {text}", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        if rec is not None:
            rec.stop()
            notice.show("Dictation stopped", stay=False, icon="dialog-information")
        srv.close()
        os.unlink(SOCK)
        kb.close()


def toggle():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCK)
    except (FileNotFoundError, ConnectionRefusedError):
        subprocess.run(["notify-send", "-a", "Dictation", "Dictation is not running",
                        "Start it: .venv/bin/python experiments/dictate.py"])
        sys.exit(1)
    s.sendall(b"toggle\n")
    print(s.recv(64).decode().strip())


if __name__ == "__main__":
    if sys.argv[1:] == ["toggle"]:
        toggle()
    elif not sys.argv[1:]:
        serve()
    else:
        sys.exit(__doc__)
