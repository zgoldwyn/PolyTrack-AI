# infer.py
import argparse
import time
from stable_baselines3 import PPO
from env import PolytrackEnv

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="summer_1", help="Track name to load the model for")
args = parser.parse_args()

model_name = f"polytrack_{args.track}"

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

env = PolytrackEnv(REGION)
model = PPO.load(model_name)

print(f"Running model for track: {args.track}")
print("Starting in 3 seconds — click on the game window now!")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)
print("Go!")

obs, _ = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, _ = env.reset()
except KeyboardInterrupt:
    print("\nStopped.")
    env.close()
