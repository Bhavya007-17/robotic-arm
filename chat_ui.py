"""Tkinter chat window for the robot arm agent.

Main thread runs the chat UI. A worker thread owns the SimEnv (with the MuJoCo
viewer) and the agent: it idle-steps the simulation when no instruction is
pending and runs agent instructions from a queue. The worker posts chat lines
back to the UI through a second queue.

Voice input is push-to-talk: hold the Mic button to record, release to
transcribe locally with faster-whisper (see voice.py). The transcript is queued
exactly as a typed instruction would be. Transcription runs on its own thread so
neither the Tk main loop nor the sim worker blocks while Whisper runs.

Usage:
    python chat_ui.py [--test "pick up the blue box and put it on the shelf"]
"""

import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import mujoco
import PIL.Image
import PIL.ImageTk

import agent as agent_mod
import voice
from agent import build_agent, run_instruction
from sim_env import SimEnv

FEED_W, FEED_H = 320, 240
FEED_INTERVAL_MS = 150

instructions = queue.Queue()  # UI -> worker
outgoing = queue.Queue()  # worker -> UI
statuses = queue.Queue()  # voice threads -> UI status label
shutdown = threading.Event()


def emit(line):
    outgoing.put(line)
    print(line, flush=True)


def worker():
    agent_mod.ENV = SimEnv()
    agent_mod.ENV.step_settle(200)
    agent_mod.ENV.randomize_boxes()
    agent = build_agent()
    emit("system: Robot ready. Boxes on the table: red, green, blue, yellow.")
    while not shutdown.is_set():
        try:
            text, source = instructions.get(timeout=0.0)
        except queue.Empty:
            agent_mod.ENV.step_settle(10)  # keep physics/viewer alive while idle
            continue
        emit(f"you ({source}): {text}" if source else f"you: {text}")
        try:
            run_instruction(agent, text, emit=emit)
        except Exception as e:
            emit(f"system: error while executing instruction: {e}")
    agent_mod.ENV.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="auto-send this instruction on startup")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Robot Arm Chat")
    root.geometry("680x640")

    # Live camera feeds (external + wrist), rendered in the main thread.
    feeds = tk.Frame(root)
    feeds.pack(fill=tk.X, padx=8, pady=(8, 0))
    feed_labels = {}
    for col, (cam, title) in enumerate(
        [("front_34", "External camera"), ("wrist_cam", "Wrist camera")]
    ):
        cell = tk.Frame(feeds)
        cell.grid(row=0, column=col, padx=4)
        tk.Label(cell, text=title, font=("Segoe UI", 9, "bold")).pack()
        placeholder = PIL.ImageTk.PhotoImage(
            PIL.Image.new("RGB", (FEED_W, FEED_H), (20, 20, 20))
        )
        lbl = tk.Label(cell, image=placeholder, bg="black")
        lbl.image = placeholder
        lbl.pack()
        feed_labels[cam] = lbl

    renderer_holder = {"r": None}

    def update_feeds():
        env = agent_mod.ENV
        if env is not None:
            if renderer_holder["r"] is None:
                renderer_holder["r"] = mujoco.Renderer(env.model, FEED_H, FEED_W)
            r = renderer_holder["r"]
            for cam, lbl in feed_labels.items():
                try:
                    r.update_scene(env.data, camera=cam)
                    photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(r.render()))
                    lbl.configure(image=photo)
                    lbl.image = photo  # keep a reference
                except Exception:
                    pass  # tolerate transient render glitches while sim steps
        root.after(FEED_INTERVAL_MS, update_feeds)

    history = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 10)
    )
    history.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

    entry_row = tk.Frame(root)
    entry_row.pack(fill=tk.X, padx=8, pady=(0, 8))
    entry = tk.Entry(entry_row, font=("Segoe UI", 10))
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    entry.focus_set()

    def append(line):
        history.configure(state=tk.NORMAL)
        history.insert(tk.END, line + "\n")
        history.see(tk.END)
        history.configure(state=tk.DISABLED)

    def send(event=None):
        text = entry.get().strip()
        if text:
            entry.delete(0, tk.END)
            instructions.put((text, None))

    tk.Button(entry_row, text="Send", command=send).pack(side=tk.LEFT, padx=(6, 0))
    entry.bind("<Return>", send)

    # --- Voice: push-to-talk ------------------------------------------------

    mic = tk.Button(entry_row, text="🎤 Hold to talk", state=tk.DISABLED)
    mic.pack(side=tk.LEFT, padx=(6, 0))
    status = tk.Label(root, text="voice: starting…", anchor=tk.W, fg="#666")
    status.pack(fill=tk.X, padx=8, pady=(0, 6))

    transcriber = voice.Transcriber()
    recorder = voice.Recorder()
    take = {"started_at": None}

    def set_status(text):
        statuses.put(text)

    def load_model():
        try:
            transcriber.load()
        except Exception as e:
            set_status(f"voice: unavailable ({e})")
            return
        set_status(f"voice: ready ({transcriber.model_name}) — hold to talk")
        mic.after(0, lambda: mic.configure(state=tk.NORMAL))

    def on_mic_press(event=None):
        if not transcriber.ready:
            set_status("voice: model still loading…")
            return
        try:
            recorder.start()
        except Exception as e:
            set_status(f"voice: microphone error ({e})")
            return
        take["started_at"] = time.time()
        set_status("voice: listening… release to send")

    def on_mic_release(event=None):
        if take["started_at"] is None:
            return
        held = time.time() - take["started_at"]
        take["started_at"] = None
        try:
            audio = recorder.stop()
        except Exception as e:
            set_status(f"voice: microphone error ({e})")
            return
        set_status(f"voice: transcribing {held:.1f}s…")
        threading.Thread(target=transcribe, args=(audio,), daemon=True).start()

    def transcribe(audio):
        started = time.time()
        try:
            text = transcriber.transcribe(audio)
        except Exception as e:
            set_status(f"voice: transcription failed ({e})")
            return
        elapsed = time.time() - started
        if not text:
            set_status("voice: no speech detected — nothing sent")
            return
        set_status(f"voice: heard in {elapsed:.1f}s — hold to talk")
        instructions.put((text, "voice"))

    mic.bind("<ButtonPress-1>", on_mic_press)
    mic.bind("<ButtonRelease-1>", on_mic_release)

    def poll():
        try:
            while True:
                append(outgoing.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                status.configure(text=statuses.get_nowait())
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close():
        shutdown.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=load_model, daemon=True).start()
    if args.test:
        instructions.put((args.test, None))
    root.after(100, poll)
    root.after(500, update_feeds)
    root.mainloop()


if __name__ == "__main__":
    main()
