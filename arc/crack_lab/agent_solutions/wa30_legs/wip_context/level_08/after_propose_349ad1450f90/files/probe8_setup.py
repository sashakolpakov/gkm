import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

env = A.Arena('wa30')
ck = json.load(open('checkpoint.json'))
path = ck['final_path']
for a in path:
    if env.terminal(): break
    env.step(a)
print("levels_completed after path:", env.levels_completed)
print("terminal:", env.terminal())
f = np.asarray(env.frame())
print("shape", f.shape, "colors", sorted(set(int(v) for v in np.unique(f))))
np.save('/tmp/l8_frame.npy', f)
