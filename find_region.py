# find_region.py
# Run this, then click the TOP-LEFT and BOTTOM-RIGHT corners of the
# PolyTrack game area. It prints the region dict you need.

from pynput.mouse import Listener

clicks = []

def on_click(x, y, button, pressed):
    if pressed:
        clicks.append((x, y))
        print(f"Click {len(clicks)}: ({x}, {y})")
        if len(clicks) == 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            region = {
                "top": min(y1, y2),
                "left": min(x1, x2),
                "width": abs(x2 - x1),
                "height": abs(y2 - y1),
            }
            print(f"\nYour region:\nREGION = {region}")
            print("\nPaste this into env.py, train.py, infer.py, smoke_test.py, and test_capture.py")
            return False  # stop listener

print("Click the TOP-LEFT corner of the game area, then the BOTTOM-RIGHT corner.")

with Listener(on_click=on_click) as listener:
    listener.join()
