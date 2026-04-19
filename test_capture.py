import cv2
import numpy as np
from mss import mss

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}
with mss() as sct:
    img = np.array(sct.grab(REGION))[:, :, :3]
    cv2.imwrite("capture_test.png", img)

print("saved capture_test.png")