from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
prevf = np.array(c.frame())
for i in range(60):
    c.step(3)
    f = np.array(c.frame())
    # summarize changes excluding avatar(14/0) and timer row
    ys,xs = np.where(f!=prevf)
    ch={}
    for y,x in zip(ys,xs):
        if y==63: continue
        a,b = int(prevf[y,x]), int(f[y,x])
        if 14 in (a,b) or 0 in (a,b): continue
        ch.setdefault((a,b),[]).append((int(y),int(x)))
    if ch:
        print(i, {k:(len(v),v[0],v[-1]) for k,v in ch.items()})
    prevf = f
    if c.terminal():
        print("TERMINAL at", i, "levels:", c.levels_completed); break
