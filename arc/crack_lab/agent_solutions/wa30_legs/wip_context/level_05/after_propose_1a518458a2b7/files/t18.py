from probe import *
from helpers import avatar
env = get_env()
c = env.clone()
c.step(4); c.step(2); c.step(5)  # grab box below; avatar(32,32), box(36,32)
# move down: box leads toward bottom fence (rows 44-47)
for i in range(4):
    c.step(2)
    f = np.array(c.frame())
    print(i, "avatar:", avatar(f))
for r in range(32,64):
    print(''.join(CH[int(v)] for v in f[r,20:48]))
