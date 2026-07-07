import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
np.set_printoptions(linewidth=200, threshold=10000)

env = A.Arena('ls20')
env.reset()
with open("checkpoint.json") as f:
    ck = json.load(f)
for a in ck["final_path"]:
    env.step(a)
print("levels_completed:", env.levels_completed, "terminal:", env.terminal())
f0 = env.frame()
print("shape", f0.shape)
# color histogram
vals, cnts = np.unique(f0, return_counts=True)
print("colors:", dict(zip(vals.tolist(), cnts.tolist())))
# print the grid compactly
for r in range(f0.shape[0]):
    print("".join(f"{v:X}" if v!=0 else "." for v in f0[r]))
