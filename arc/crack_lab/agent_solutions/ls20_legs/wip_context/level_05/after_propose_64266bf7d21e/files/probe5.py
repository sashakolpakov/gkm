import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

env = A.Arena('ls20')
env.reset()
with open("checkpoint.json") as f:
    ck = json.load(f)
for a in ck["final_path"]:
    env.step(a)
print("levels_completed after replay:", env.levels_completed)
print("terminal:", env.terminal())

f = env.frame()
print("frame shape:", f.shape)
np.save("/tmp/level5_start.npy", f)
print(f)
