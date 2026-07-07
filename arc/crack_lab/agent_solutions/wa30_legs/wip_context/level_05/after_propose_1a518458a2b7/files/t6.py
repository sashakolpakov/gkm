import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck = json.load(open("checkpoint.json"))
env = A.Arena('wa30')
path = ck["final_path"]
for i, a in enumerate(path):
    env.step(a)
    if env.levels_completed == 3:
        print("level 3 completed at path index", i, "of", len(path)-1)
        break
f = np.array(env.frame())
print("timer 7s:", (f[63]==7).sum())
