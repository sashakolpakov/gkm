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

f0 = env.frame()

for act in (1,2,3,4):
    c = env.clone()
    f1 = c.step(act)
    diff = np.argwhere(f0 != f1)
    print(f"action {act}: {len(diff)} cells changed, levels_completed={c.levels_completed}")
    if len(diff) > 0 and len(diff) < 60:
        for r,cc in diff:
            print(f"   ({r},{cc}) {f0[r,cc]} -> {f1[r,cc]}")
