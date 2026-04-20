# debug_demo.py
# Shows what actions were recorded in a demo file
import pickle
import sys
import numpy as np

ACTION_NAMES = {
    0: "forward",
    1: "forward+left",
    2: "forward+right",
    3: "coast",
    4: "coast+left",
    5: "coast+right",
}

for f in sys.argv[1:]:
    with open(f, "rb") as fh:
        demo = pickle.load(fh)
    actions = demo["actions"]
    print(f"\n{f}: {len(actions)} frames")
    unique, counts = np.unique(actions, return_counts=True)
    for a, c in zip(unique, counts):
        print(f"  {ACTION_NAMES.get(a, f'action_{a}')}: {c} ({c/len(actions)*100:.1f}%)")
