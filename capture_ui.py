"""Screenshot the chat UI with both camera feeds live and a command executing.

capture_demo.py runs headless and saves raw camera frames; it cannot show the
interface. This launches the real chat window, sends one instruction, waits for
the arm to be mid-task, and grabs the actual window off the screen.

Usage:
    python capture_ui.py [--out docs/images/06_chat_ui.png] [--delay 14]
"""

import argparse
import threading
import time

import PIL.ImageGrab

import chat_ui

DEFAULT_OUT = "docs/images/06_chat_ui.png"
DEFAULT_DELAY = 14.0  # long enough for the arm to be visibly mid-transfer


def _grab_after(root, out, delay):
    time.sleep(delay)
    # The MuJoCo passive viewer opens its own window over the same screen area,
    # so the chat window has to be raised or the grab captures the viewer.
    root.after(0, lambda: (root.lift(), root.attributes("-topmost", True)))
    time.sleep(1.5)
    root.update_idletasks()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    box = (x, y, x + root.winfo_width(), y + root.winfo_height())
    PIL.ImageGrab.grab(bbox=box).save(out)
    print(f"saved {out} from window box {box}", flush=True)
    root.after(0, lambda: (root.attributes("-topmost", False), root.destroy()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--command", default="put the red box on the shelf")
    args = parser.parse_args()

    original_main = chat_ui.main

    def patched_main():
        # chat_ui.main() builds the window and blocks in mainloop, so the grab
        # has to be armed from a thread that waits for the window to exist.
        threading.Thread(target=_arm_grab, args=(args,), daemon=True).start()
        original_main()

    import sys

    sys.argv = ["chat_ui.py", "--test", args.command]
    patched_main()


def _arm_grab(args):
    # Wait for the Tk root to come into existence, then schedule the grab.
    for _ in range(200):
        root = getattr(chat_ui.tk, "_default_root", None)
        if root is not None:
            _grab_after(root, args.out, args.delay)
            return
        time.sleep(0.1)
    print("timed out waiting for the chat window", flush=True)


if __name__ == "__main__":
    main()
