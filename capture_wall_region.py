# capture_wall_region.py
# Drive into a wall, then run this.
# Click TOP-LEFT then BOTTOM-RIGHT of where the wall appears in front of the car.
# It will sample the color AND give you the region coordinates for env.py.

import cv2
import numpy as np
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

print("Click TOP-LEFT then BOTTOM-RIGHT of where the wall shows in front of the car.")

with Listener(on_click=on_click) as listener:
    listener.join()

with mss() as sct:
    full = np.array(sct.grab(REGION))[:, :, :3]

x1 = int(clicks[0][0] - REGION["left"])
y1 = int(clicks[0][1] - REGION["top"])
x2 = int(clicks[1][0] - REGION["left"])
y2 = int(clicks[1][1] - REGION["top"])

x1, x2 = max(0, min(x1, x2)), min(REGION["width"], max(x1, x2))
y1, y2 = max(0, min(y1, y2)), min(REGION["height"], max(y1, y2))

frame_h, frame_w = full.shape[:2]
t = round(y1 / frame_h, 2)
b = round(y2 / frame_h, 2)
l = round(x1 / frame_w, 2)
r = round(x2 / frame_w, 2)

region = full[y1:y2, x1:x2]
hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

h_mean = int(np.mean(hsv[:, :, 0]))
s_mean = int(np.mean(hsv[:, :, 1]))
v_mean = int(np.mean(hsv[:, :, 2]))

print(f"\nRegion: {region.shape[1]}x{region.shape[0]} pixels")
print(f"Average HSV: H={h_mean}, S={s_mean}, V={v_mean}")
print(f"\nWALL_CHECK_REGION = ({t}, {b}, {l}, {r})")
print(f"WALL_HSV_LOW  = ({max(0, h_mean-15)}, {max(0, s_mean-30)}, {max(0, v_mean-40)})")
print(f"WALL_HSV_HIGH = ({min(180, h_mean+15)}, {min(255, s_mean+30)}, {min(255, v_mean+40)})")

# Test current range against this region
low = np.array([87, 7, 0])
high = np.array([122, 86, 86])
mask = cv2.inRange(hsv, low, high)
ratio = np.count_nonzero(mask) / mask.size
print(f"\nCurrent wall range match: {ratio*100:.1f}%")

cv2.imwrite("wall_region_sample.png", region)
print("Saved wall_region_sample.png")
