# capture_template.py
# 1. Get the game into the "stuck" state so the popup is showing
# 2. Run this script
# 3. Click the TOP-LEFT corner of the popup text/letter
# 4. Click the BOTTOM-RIGHT corner of the popup text/letter
# It saves the cropped region as stuck_template.png

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
        return False  # stop listener

print("Click the TOP-LEFT then BOTTOM-RIGHT corners of the stuck popup text.")

with Listener(on_click=on_click) as listener:
    listener.join()

# Grab the screen and crop to the clicked region
with mss() as sct:
    full = np.array(sct.grab(REGION))[:, :, :3]

x1 = int(clicks[0][0] - REGION["left"])
y1 = int(clicks[0][1] - REGION["top"])
x2 = int(clicks[1][0] - REGION["left"])
y2 = int(clicks[1][1] - REGION["top"])

# Clamp to region bounds
x1 = max(0, min(x1, REGION["width"]))
x2 = max(0, min(x2, REGION["width"]))
y1 = max(0, min(y1, REGION["height"]))
y2 = max(0, min(y2, REGION["height"]))

template = full[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
cv2.imwrite("stuck_template.png", template)
print(f"Saved stuck_template.png ({template.shape[1]}x{template.shape[0]} pixels)")
print("This will be used by env.py to detect the stuck popup.")
