from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
# grab the box below-right, carry up to fence, face up at fence, USE
c.step(4); c.step(2); c.step(5)  # attach box below
# avatar (32,32). carry up: box follows below. Move up to fence row: avatar can reach row 24 (fence rows 20-23)
c.step(1); c.step(1)  # avatar rows 24-27
f0 = np.array(c.frame())
print("avatar:", avatar(f0))
for r in range(20,44):
    print(''.join(CH[int(v)] for v in f0[r,20:48]))
c.step(5)  # USE while carrying, facing up at fence
f1 = np.array(c.frame())
ys,xs = np.where(f0!=f1)
ch={}
for y,x in zip(ys,xs): ch.setdefault((int(f0[y,x]),int(f1[y,x])),[]).append((int(y),int(x)))
print("USE:", {k:(len(v),v[0]) for k,v in ch.items()})
