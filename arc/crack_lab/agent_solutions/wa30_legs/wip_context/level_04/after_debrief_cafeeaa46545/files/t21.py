from probe import *
from helpers import avatar
env = get_env()
c = env.clone()
c.step(4); c.step(2); c.step(5)  # grab B4 below avatar; avatar(32,32), box(36,32)
c.step(4)  # move right while box attached below
f = np.array(c.frame())
print("avatar:", avatar(f))
for r in range(28,48):
    print(''.join(CH[int(v)] for v in f[r,24:48]))
