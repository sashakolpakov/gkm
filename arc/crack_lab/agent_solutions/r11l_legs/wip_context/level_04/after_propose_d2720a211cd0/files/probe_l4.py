import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

env = A.Arena('r11l')
ck = json.load(open('checkpoint.json'))
for act in ck['final_path']:
    env.step(act)
print("levels_completed:", env.levels_completed)
f = env.frame()
print("shape:", f.shape)
vals, cnts = np.unique(f, return_counts=True)
print("colors:", dict(zip(vals.tolist(), cnts.tolist())))
np.save('l4_frame.npy', f)
