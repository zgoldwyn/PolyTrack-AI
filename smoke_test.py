from env import PolytrackEnv

REGION = {"top": 41, "left": 3, "width": 1795, "height": 1125}

env = PolytrackEnv(REGION)
obs, info = env.reset()
print("reset obs shape:", obs.shape)

for i in range(20):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    print(i, obs.shape, reward, terminated, truncated)

env.close()