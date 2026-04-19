# capture_checkpoint.py
# Captures the checkpoint counter region from the HUD.
# 
# Run this 4 times — once for each checkpoint state (0/3, 1/3, 2/3, 3/3).
# Each time, get the game into that state first, then run this script
# and click the corners around the checkpoint number/text.
#
# It will ask you which state you're capturing.

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

state = input("Which checkpoint state is showing? (0, 1, 2, or 3): ").strip()
if state not in ("0", "1", "2", "3"):
    print("Must be 0, 1, 2, or 3")
    exit(1)

print(f"Capturing checkpoint state {state}/3")
print("Click TOP-LEFT then BOTTOM-RIGHT of the checkpoint number/counter.")

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

template = full[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
filename = f"checkpoint_{state}.png"
cv2.imwrite(filename, template)
print(f"Saved {filename} ({template.shape[1]}x{template.shape[0]} pixels)")
