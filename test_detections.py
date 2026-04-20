# test_detections.py
# Run this, switch to the game, and drive around.
# It prints what it detects in real-time without sending any keys.

import cv2
import numpy as np
import time
from mss import mss

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

# Wall detection
WALL_HSV_LOW = np.array([90, 40, 0])
WALL_HSV_HIGH = np.array([120, 100, 64])
WALL_REGION = (0.57, 0.59, 0.30, 0.70)
WALL_THRESHOLD = 0.15

# Track detection
TRACK_HSV_LOW = np.array([92, 33, 90])
TRACK_HSV_HIGH = np.array([122, 93, 170])
TRACK_REGION = (0.57, 0.59, 0.45, 0.54)
TRACK_THRESHOLD = 0.3

# OOB detection
OOB_HSV_LOW = np.array([55, 40, 80])
OOB_HSV_HIGH = np.array([90, 150, 180])
OOB_THRESHOLD = 0.75

print("You have 3 seconds — switch to the game!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)
print("Watching... press Ctrl+C to stop.\n")

sct = mss()
step = 0

try:
    while True:
        raw = np.array(sct.grab(REGION))[:, :, :3]
        h, w = raw.shape[:2]

        # Wall check
        t, b, l, r = WALL_REGION
        wall_area = raw[int(t*h):int(b*h), int(l*w):int(r*w)]
        wall_hsv = cv2.cvtColor(wall_area, cv2.COLOR_BGR2HSV)
        wall_mask = cv2.inRange(wall_hsv, WALL_HSV_LOW, WALL_HSV_HIGH)
        wall_ratio = np.count_nonzero(wall_mask) / wall_mask.size

        # Track check
        t, b, l, r = TRACK_REGION
        track_area = raw[int(t*h):int(b*h), int(l*w):int(r*w)]
        track_hsv = cv2.cvtColor(track_area, cv2.COLOR_BGR2HSV)
        track_mask = cv2.inRange(track_hsv, TRACK_HSV_LOW, TRACK_HSV_HIGH)
        track_ratio = np.count_nonzero(track_mask) / track_mask.size

        # OOB check
        bottom = raw[h // 2:, :]
        oob_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        oob_mask = cv2.inRange(oob_hsv, OOB_HSV_LOW, OOB_HSV_HIGH)
        oob_ratio = np.count_nonzero(oob_mask) / oob_mask.size

        # Print detections
        flags = []
        if wall_ratio >= WALL_THRESHOLD:
            flags.append(f"WALL({wall_ratio*100:.0f}%)")
        if track_ratio >= TRACK_THRESHOLD:
            flags.append(f"TRACK({track_ratio*100:.0f}%)")
        if oob_ratio >= OOB_THRESHOLD:
            flags.append(f"OOB({oob_ratio*100:.0f}%)")

        if step % 5 == 0:  # print every 5th frame to avoid spam
            status = " | ".join(flags) if flags else "nothing detected"
            print(f"  step={step:4d} | wall={wall_ratio*100:4.0f}% | track={track_ratio*100:4.0f}% | oob={oob_ratio*100:4.0f}% | {status}", flush=True)

        step += 1
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped.")
