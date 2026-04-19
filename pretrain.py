# pretrain.py
# Behavioral cloning: trains the PPO policy to mimic recorded demos.
# Run this after recording demos, before RL fine-tuning.
#
# Usage: .venv/bin/python pretrain.py --track summer_1

import argparse
import glob
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import PolytrackEnv

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="summer_1")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

# Load all demos for this track
demo_files = sorted(glob.glob(f"demos/{args.track}_run*.pkl"))
if not demo_files:
    print(f"No demos found for track {args.track} in demos/")
    exit(1)

all_frames = []
all_actions = []
for f in demo_files:
    print(f"Loading {f}")
    with open(f, "rb") as fh:
        demo = pickle.load(fh)
    all_frames.append(demo["frames"])
    all_actions.append(demo["actions"])

frames = np.concatenate(all_frames)
actions = np.concatenate(all_actions)
print(f"Total: {len(frames)} frames from {len(demo_files)} demo(s)")

# Build stacked observations (4 frames like the env does)
obs_list = []
act_list = []
for i in range(3, len(frames)):
    stacked = np.stack(frames[i-3:i+1], axis=0)  # shape: (4, 84, 84)
    obs_list.append(stacked)
    act_list.append(actions[i])

obs_array = np.array(obs_list, dtype=np.float32) / 255.0  # normalize
act_array = np.array(act_list, dtype=np.int64)

print(f"Training samples: {len(obs_array)}")

# Create or load model
model_name = f"polytrack_{args.track}"
model_file = f"{model_name}.zip"

def make_env():
    return PolytrackEnv(REGION)

env = DummyVecEnv([make_env])

if os.path.exists(model_file):
    print(f"Loading existing model: {model_file}")
    model = PPO.load(model_name, env=env)
else:
    print("Creating new model for pretraining")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=None,
    )

# Extract the policy network
policy = model.policy
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Create dataloader
obs_tensor = torch.tensor(obs_array)
act_tensor = torch.tensor(act_array)
dataset = TensorDataset(obs_tensor, act_tensor)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Train
policy.train()
for epoch in range(args.epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_obs, batch_act in loader:
        batch_obs = batch_obs.to(policy.device)
        batch_act = batch_act.to(policy.device)

        # Get action logits from the policy
        features = policy.extract_features(batch_obs, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)

        loss = loss_fn(logits, batch_act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_act)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_act).sum().item()
        total += len(batch_act)

    acc = correct / total * 100
    avg_loss = total_loss / total
    print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | accuracy={acc:.1f}%")

# Save the pretrained model
model.save(model_name)
print(f"\nPretrained model saved to {model_file}")
print(f"Now run: .venv/bin/python train.py --track {args.track}")
print("to fine-tune with RL")
