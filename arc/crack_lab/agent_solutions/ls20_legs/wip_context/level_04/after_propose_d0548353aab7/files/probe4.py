import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

env = A.Arena('ls20')
with open('checkpoint.json') as f:
    ck = json.load(f)
for a in ck['final_path']:
    env.step(a)
print('after replay levels_completed=', env.levels_completed, 'terminal=', env.terminal())

f = env.frame()
print('frame shape', f.shape)
np.set_printoptions(linewidth=200, threshold=100000)
print(f)
