# debug_oob.py
# Drive onto the green area, then run this.
# It grabs a frame and shows you exactly what the OOB detector sees.

import cv2
import numpy as np
import time
from mss import mss

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

print("You have 3 seconds — switch to the game!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss() as sct:
    raw = np.array(sct.grab(REGION))[:, :, :3]

print(f"Frame shape: {raw.shape}")
print(f"Frame dtype: {raw.dtype}")

# Check what color format mss gives us
# mss returns BGRA, we slice to BGR — but let's verify
h = raw.shape[0]
bottom = raw[h // 2:, :]

# Sample center pixel of bottom half
cy, cx = bottom.shape[0] // 2, bottom.shape[1] // 2
pixel_bgr = bottom[cy, cx]
print(f"\nCenter pixel of bottom half (BGR): {pixel_bgr}")

# Convert to HSV
hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
pixel_hsv = hsv[cy, cx]
print(f"Center pixel of bottom half (HSV): {pixel_hsv}")

# Print average HSV of bottom half
print(f"\nBottom half average HSV: H={np.mean(hsv[:,:,0]):.1f}, S={np.mean(hsv[:,:,1]):.1f}, V={np.mean(hsv[:,:,2]):.1f}")

# Test with current range
low = np.array([40, 30, 60])
high = np.array([95, 160, 200])
mask = cv2.inRange(hsv, low, high)
ratio = np.count_nonzero(mask) / mask.size
print(f"\nGreen ratio with current range: {ratio:.3f} ({ratio*100:.1f}%)")

# Test with very wide range to see what's there
low_wide = np.array([30, 10, 30])
high_wide = np.array([100, 255, 255])
mask_wide = cv2.inRange(hsv, low_wide, high_wide)
ratio_wide = np.count_nonzero(mask_wide) / mask_wide.size
print(f"Green ratio with WIDE range:    {ratio_wide:.3f} ({ratio_wide*100:.1f}%)")

# Save debug images
cv2.imwrite("debug_bottom_half.png", bottom)
cv2.imwrite("debug_green_mask.png", mask)
cv2.imwrite("debug_green_mask_wide.png", mask_wide)
print("\nSaved debug_bottom_half.png, debug_green_mask.png, debug_green_mask_wide.png")
print("Open them to see what the detector is seeing.")
