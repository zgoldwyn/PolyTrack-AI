# capture_track.py
# 1. Get the car on the track so grey road is visible in front of it
# 2. Run this script
# 3. Click TOP-LEFT then BOTTOM-RIGHT of the area right in front of the car
#    (the grey track surface)
# It samples the color and tells you the HSV range + region coordinates.

import cv2
import numpy as np
import time
from mss import mss
from pynput.mouse import Listener

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

clicks = []

def on_click(x, y, button, pressed):
    if not pressed:
        return
    clicks.append((x, y))
    print(f"Click {len(clicks)}: ({x}, {y})")
    if len(clicks) == 2:
        return False

print("Click TOP-LEFT then BOTTOM-RIGHT of the grey track area in front of the car.")

with Listener(on_click=on_click) as listener:
    listener.join()

with mss() as sct:
    full = np.array(sct.grab(REGION))[:, :, :3]

x1 = int(clicks[0][0] - REGION["left"])
y1 = int(clicks[0][1] - REGION["top"])
x2 = int(clicks[1][0] - REGION["left"])
y2 = int(clicks[1][1] - REGION["top"])

x1 = max(0, min(x1, REGION["width"]))
x2 = max(0, min(x2, REGION["width"]))
y1 = max(0, min(y1, REGION["height"]))
y2 = max(0, min(y2, REGION["height"]))

patch = full[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

h_mean = int(np.mean(hsv_patch[:, :, 0]))
s_mean = int(np.mean(hsv_patch[:, :, 1]))
v_mean = int(np.mean(hsv_patch[:, :, 2]))

# Region as fraction of full frame (so it works regardless of resolution)
frame_h, frame_w = full.shape[:2]
ry1 = round(min(y1, y2) / frame_h, 2)
ry2 = round(max(y1, y2) / frame_h, 2)
rx1 = round(min(x1, x2) / frame_w, 2)
rx2 = round(max(x1, x2) / frame_w, 2)

print(f"\nPatch size: {patch.shape[1]}x{patch.shape[0]}")
print(f"Sampled HSV: H={h_mean}, S={s_mean}, V={v_mean}")
print(f"\nSuggested HSV range:")
print(f"  TRACK_HSV_LOW  = ({max(0, h_mean-15)}, {max(0, s_mean-30)}, {max(0, v_mean-40)})")
print(f"  TRACK_HSV_HIGH = ({min(180, h_mean+15)}, {min(255, s_mean+30)}, {min(255, v_mean+40)})")
print(f"\nRegion (as fraction of frame):")
print(f"  TRACK_CHECK_REGION = ({ry1}, {ry2}, {rx1}, {rx2})  # (top%, bottom%, left%, right%)")
print(f"\nPixel coordinates: y={min(y1,y2)}-{max(y1,y2)}, x={min(x1,x2)}-{max(x1,x2)}")

cv2.imwrite("track_sample.png", patch)
print("Saved track_sample.png for reference")
