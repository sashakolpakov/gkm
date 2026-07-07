from probe import *
from helpers import avatar, timer
env = get_env()
f = np.array(env.frame())
print("avatar:", avatar(f), "timer:", timer(f))
# Try USE at start
c = env.clone(); c.step(5)
f1 = np.array(c.frame())
ys, xs = np.where(f != f1)
print("USE diffs:", [(int(y),int(x),int(f[y,x]),int(f1[y,x])) for y,x in zip(ys,xs)][:20])
