# record_demo.py
# Records your gameplay: frames + key inputs.
# Run this, switch to the game, play the track.
# Recording starts when you first press W.
# Recording stops when the finish screen is detected.
#
# Usage: .venv/bin/python record_demo.py --track summer_1 --run 1

import argparse
import os
import time
import cv2
import numpy as np
import pickle
from mss import mss
from Quartz import (
    CGEventSourceCreate,
    CGEventSourceKeyState,
    kCGEventSourceStateHIDSystemState,
)

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="summer_1")
parser.add_argument("--run", type=int, default=1)
args = parser.parse_args()

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

# macOS virtual key codes
KEY_W = 13
KEY_A = 0
KEY_D = 2
KEY_S = 1
KEY_ESC = 53
KEY_UP = 126
KEY_DOWN = 125
KEY_LEFT = 123
KEY_RIGHT = 124

# Load finish template for auto-stop
base_dir = os.path.dirname(__file__)
finish_template = cv2.imread(os.path.join(base_dir, "finish_template.png"), cv2.IMREAD_GRAYSCALE)
FINISH_THRESHOLD = 0.8

def is_key_pressed(keycode):
    """Check if a key is currently pressed using macOS Quartz."""
    return CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, keycode)

def keys_to_action():
    """Convert currently held keys to our action space."""
    w = is_key_pressed(KEY_W) or is_key_pressed(KEY_UP)
    a = is_key_pressed(KEY_A) or is_key_pressed(KEY_LEFT)
    d = is_key_pressed(KEY_D) or is_key_pressed(KEY_RIGHT)

    if w and a:
        return 1  # forward + left
    elif w and d:
        return 2  # forward + right
    elif w:
        return 0  # forward only
    elif a:
        return 4  # coast + left
    elif d:
        return 5  # coast + right
    else:
        return 3  # coast

def check_finish(gray_frame):
    """Check if finish screen is visible."""
    if finish_template is None:
        return False
    result = cv2.matchTemplate(gray_frame, finish_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= FINISH_THRESHOLD

os.makedirs("demos", exist_ok=True)
demo_file = f"demos/{args.track}_run{args.run}.pkl"

print(f"Recording demo for track: {args.track}, run: {args.run}")
print("Switch to the game. Recording starts when you press W.")
print("Recording stops at the finish line (or press ESC).")

sct = mss()

# Wait for first W press
while not (is_key_pressed(KEY_W) or is_key_pressed(KEY_UP)):
    time.sleep(0.01)

print("Recording!")

frames = []
actions = []
finish_streak = 0

while True:
    # Check ESC
    if is_key_pressed(KEY_ESC):
        print("ESC pressed — stopping.")
        break

    # Grab frame
    raw = np.array(sct.grab(REGION))[:, :, :3]
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # Check finish
    if check_finish(gray):
        finish_streak += 1
        if finish_streak >= 3:
            print("Finish line detected — stopping.")
            break
    else:
        finish_streak = 0

    action = keys_to_action()
    frames.append(small)
    actions.append(action)

    time.sleep(0.05)  # ~20 FPS

print(f"\nRecorded {len(frames)} frames ({len(frames)/20:.1f} seconds)")

# Show action distribution
ACTION_NAMES = {0: "forward", 1: "fwd+left", 2: "fwd+right", 3: "coast", 4: "coast+left", 5: "coast+right"}
unique, counts = np.unique(actions, return_counts=True)
for a, c in zip(unique, counts):
    print(f"  {ACTION_NAMES.get(a, f'?{a}')}: {c} ({c/len(actions)*100:.1f}%)")

demo = {
    "frames": np.array(frames, dtype=np.uint8),
    "actions": np.array(actions, dtype=np.int64),
    "track": args.track,
    "run": args.run,
}
with open(demo_file, "wb") as f:
    pickle.dump(demo, f)

print(f"Saved to {demo_file}")
