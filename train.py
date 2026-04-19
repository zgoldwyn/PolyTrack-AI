# train.py
import argparse
import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import PolytrackEnv

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="summer_1", help="Track name for the model file (e.g. summer_1, winter_3)")
args = parser.parse_args()

model_name = f"polytrack_{args.track}"
model_file = f"{model_name}.zip"

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

def make_env():
    return PolytrackEnv(REGION)

env = DummyVecEnv([make_env])

# Resume from saved model if it exists, otherwise start fresh
if os.path.exists(model_file):
    print(f"Found existing model ({model_file}) — resuming training.")
    model = PPO.load(model_name, env=env, learning_rate=1e-4)
else:
    print(f"No saved model found — starting fresh for track: {args.track}")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=None,  # set to "./tb_logs/" to enable tensorboard
    )

def handle_interrupt(sig, frame):
    print(f"\n\nInterrupted — saving model to {model_file}...")
    model.save(model_name)
    env.close()
    print(f"Saved to {model_file}. You can resume later.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

print(f"Training [{args.track}]. Press Ctrl+C to stop and save.")
model.learn(total_timesteps=2_000_000)
model.save(model_name)
print(f"Training complete. Saved to {model_file}")
