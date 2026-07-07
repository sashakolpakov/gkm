import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from legs import detect_noise_mask

env = A.Arena('ls20')
env.reset()
with open("checkpoint.json") as f:
    ck = json.load(f)
for a in ck["final_path"]:
    env.step(a)
f0 = env.frame()
mask = detect_noise_mask(env)
print("noise mask cells:", int(mask.sum()))
ys,xs = np.where(mask)
if mask.sum()>0:
    print("mask rows range", ys.min(), ys.max(), "cols", xs.min(), xs.max())
    print("mask locations:", list(zip(ys.tolist(),xs.tolist()))[:80])

# Now look at each action's diff EXCLUDING mask
for act in (1,2,3,4):
    c = env.clone()
    f1 = c.step(act)
    diff = np.argwhere((f0 != f1) & (~mask))
    print(f"\naction {act}: {len(diff)} non-noise cells changed")
    for r,cc in diff[:40]:
        print(f"   ({r},{cc}) {f0[r,cc]} -> {f1[r,cc]}")
