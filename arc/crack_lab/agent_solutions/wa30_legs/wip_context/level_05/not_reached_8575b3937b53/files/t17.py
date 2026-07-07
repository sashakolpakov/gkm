from probe import *
from helpers import avatar
env = get_env()
c = env.clone()
# grab box below (at 36,32): right, down(bump), use
c.step(4); c.step(2); c.step(5)
# now face up (move up once - box follows below), then face up again... box trails.
c.step(1)  # up; avatar (28,32), box (32,32)
f0 = np.array(c.frame())
print("avatar:", avatar(f0))
c.step(5)  # USE while facing up (away from box)
f1 = np.array(c.frame())
ys,xs = np.where(f0!=f1)
ch={}
for y,x in zip(ys,xs): ch.setdefault((int(f0[y,x]),int(f1[y,x])),[]).append((int(y),int(x)))
print("USE facing away:", {k:(len(v),v[0],v[-1]) for k,v in ch.items()})
