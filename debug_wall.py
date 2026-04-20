# debug_wall.py
# Drive into a wall so it's in front of the car, then run this.

import cv2
import numpy as np
import time
from mss import mss

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

print("You have 3 seconds — switch to the game with the car against a wall!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss() as sct:
    raw = np.array(sct.grab(REGION))[:, :, :3]

h, w = raw.shape[:2]

# Check the region in front of the car (same as env.py)
t, b, l, r = 0.45, 0.55, 0.40, 0.60
region = raw[int(t*h):int(b*h), int(l*w):int(r*w)]

print(f"Check region size: {region.shape}")

hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

# Center pixel
cy, cx = region.shape[0]//2, region.shape[1]//2
print(f"Center pixel BGR: {region[cy, cx]}")
print(f"Center pixel HSV: {hsv[cy, cx]}")
print(f"Region average HSV: H={np.mean(hsv[:,:,0]):.1f}, S={np.mean(hsv[:,:,1]):.1f}, V={np.mean(hsv[:,:,2]):.1f}")

# Test with current wall range
low = np.array([87, 7, 0])
high = np.array([122, 86, 86])
mask = cv2.inRange(hsv, low, high)
ratio = np.count_nonzero(mask) / mask.size
print(f"\nWall ratio with current range: {ratio:.3f} ({ratio*100:.1f}%)")

# Test with very wide range
low_wide = np.array([80, 0, 0])
high_wide = np.array([130, 100, 100])
mask_wide = cv2.inRange(hsv, low_wide, high_wide)
ratio_wide = np.count_nonzero(mask_wide) / mask_wide.size
print(f"Wall ratio with WIDE range:    {ratio_wide:.3f} ({ratio_wide*100:.1f}%)")

cv2.imwrite("debug_wall_region.png", region)
cv2.imwrite("debug_wall_mask.png", mask)
print("\nSaved debug_wall_region.png and debug_wall_mask.png")
