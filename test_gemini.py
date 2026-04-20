# test_gemini.py
# Grabs a screenshot and asks Gemini if the car is hitting a wall.
# Set GEMINI_API_KEY env var first.

import os
import time
import cv2
import numpy as np
from mss import mss
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY first: export GEMINI_API_KEY='your-key'")
    exit(1)

client = genai.Client(api_key=api_key)

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

print("You have 3 seconds — switch to the game!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

with mss() as sct:
    raw = np.array(sct.grab(REGION))[:, :, :3]

_, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 30])
image_bytes = buf.tobytes()

print("Sending to Gemini Flash...")
start = time.time()

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=[
        "Is this racing car currently hitting, touching, or scraping against a wall or barrier? Answer only 'yes' or 'no'.",
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    ],
)

elapsed = time.time() - start
print(f"Response ({elapsed:.2f}s): {response.text.strip()}")
