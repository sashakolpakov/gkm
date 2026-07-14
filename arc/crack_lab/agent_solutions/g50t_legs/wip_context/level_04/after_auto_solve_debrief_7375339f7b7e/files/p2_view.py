from l2env import get_l2_env
import numpy as np
env = get_l2_env()
f = np.asarray(env.frame())
print("actions", env.actions)
print("colors", dict(zip(*[a.tolist() for a in np.unique(f, return_counts=True)])))
# print grid compactly with chars
chars = {0:'.',1:'1',5:'#',8:'8',9:'9'}
for r in range(64):
    print(''.join(chars.get(int(f[r,c]),'?') for c in range(64)))
