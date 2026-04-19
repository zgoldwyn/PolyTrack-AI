# capture_oob.py
# Drive the car onto the green out-of-bounds area, then run this.
# Click on a patch of the green ground. It will sample the color
# and save the HSV range for detection.

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
    print(f"Click: ({x}, {y})")
    return False

print("Click on a patch of the GREEN out-of-bounds ground.")

with Listener(on_click=on_click) as listener:
    listener.join()

with mss() as sct:
    full = np.array(sct.grab(REGION))[:, :, :3]

x = int(clicks[0][0] - REGION["left"])
y = int(clicks[0][1] - REGION["top"])

# Sample a 20x20 patch around the click
x1 = max(0, x - 10)
y1 = max(0, y - 10)
x2 = min(REGION["width"], x + 10)
y2 = min(REGION["height"], y + 10)

patch = full[y1:y2, x1:x2]
hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

h_mean = int(np.mean(hsv_patch[:, :, 0]))
s_mean = int(np.mean(hsv_patch[:, :, 1]))
v_mean = int(np.mean(hsv_patch[:, :, 2]))

print(f"\nSampled HSV: H={h_mean}, S={s_mean}, V={v_mean}")
print(f"Suggested range:")
print(f"  OOB_HSV_LOW  = ({max(0, h_mean-15)}, {max(0, s_mean-40)}, {max(0, v_mean-40)})")
print(f"  OOB_HSV_HIGH = ({min(180, h_mean+15)}, {min(255, s_mean+40)}, {min(255, v_mean+40)})")
print(f"\nPaste these into env.py")

# Save the patch for reference
cv2.imwrite("oob_sample.png", patch)
print("Saved oob_sample.png for reference")
