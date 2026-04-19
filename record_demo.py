# record_demo.py
# Records your gameplay: frames + key inputs.
# Run this, switch to the game, play the track.
# Press ESC to stop recording.
#
# Usage: .venv/bin/python record_demo.py --track summer_1 --run 1

import argparse
import os
import time
import cv2
import numpy as np
import pickle
from mss import mss
from pynput.keyboard import Listener, Key, KeyCode

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="summer_1")
parser.add_argument("--run", type=int, default=1)
args = parser.parse_args()

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

# Track which keys are currently held
keys_held = set()
recording = True

def on_press(key):
    global recording
    if key == Key.esc:
        recording = False
        return False
    try:
        keys_held.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        keys_held.discard(key.char)
    except AttributeError:
        pass

def keys_to_action():
    """Convert currently held keys to our action space."""
    w = 'w' in keys_held
    a = 'a' in keys_held
    d = 'd' in keys_held

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
        return 3  # coast (no gas, no steering)

# Create demos directory
os.makedirs("demos", exist_ok=True)
demo_file = f"demos/{args.track}_run{args.run}.pkl"

print(f"Recording demo for track: {args.track}, run: {args.run}")
print("You have 3 seconds — switch to the game!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)
print("GO! Play the track. Press ESC when done.")

frames = []
actions = []
sct = mss()

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

while recording:
    # Grab frame
    raw = np.array(sct.grab(REGION))[:, :, :3]
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    action = keys_to_action()
    frames.append(small)
    actions.append(action)

    time.sleep(0.05)  # ~20 FPS to match training

listener.stop()

print(f"\nRecorded {len(frames)} frames ({len(frames)/20:.1f} seconds)")

# Save as pickle
demo = {
    "frames": np.array(frames, dtype=np.uint8),
    "actions": np.array(actions, dtype=np.int64),
    "track": args.track,
    "run": args.run,
}
with open(demo_file, "wb") as f:
    pickle.dump(demo, f)

print(f"Saved to {demo_file}")
