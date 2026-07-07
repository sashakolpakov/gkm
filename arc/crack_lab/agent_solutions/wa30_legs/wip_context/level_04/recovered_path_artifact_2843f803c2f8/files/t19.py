from probe import *
from helpers import avatar, timer, objects
env = get_env()
c = env.clone()
c.step(4); c.step(2); c.step(5); c.step(2); c.step(2)  # box embedded in bottom fence at (44,32)
c.step(5)  # release
prevf = np.array(c.frame())
for i in range(50):
    c.step(1)  # idle-ish: up (will re-enter arena)
    f = np.array(c.frame())
    ys,xs = np.where(f!=prevf)
    ch={}
    for y,x in zip(ys,xs):
        if y==63: continue
        a,b=int(prevf[y,x]),int(f[y,x])
        if 14 in (a,b): continue
        ch.setdefault((a,b),[]).append((int(y),int(x)))
    if ch: print(i, {k:(len(v),v[0],v[-1]) for k,v in ch.items()})
    prevf=f
    if c.terminal(): print("TERM", c.levels_completed); break
print("timer:", timer(prevf))
