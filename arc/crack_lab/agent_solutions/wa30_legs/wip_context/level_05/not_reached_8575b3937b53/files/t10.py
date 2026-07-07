from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
c.step(4); c.step(2); c.step(5)  # face box below, USE
f0 = np.array(c.frame())
# show local area rows 30-45 cols 24-44
for r in range(30,46):
    print(''.join(CH[int(v)] for v in f0[r,24:44]))
print("avatar:", avatar(f0))
# now move up: does box follow?
c.step(1)
f1 = np.array(c.frame())
ys,xs = np.where(f0!=f1)
ch = {}
for y,x in zip(ys,xs): ch.setdefault((int(f0[y,x]),int(f1[y,x])),[]).append((int(y),int(x)))
print("after UP:", {k:(len(v),v[0]) for k,v in ch.items()})
for r in range(26,42):
    print(''.join(CH[int(v)] for v in f1[r,24:44]))
