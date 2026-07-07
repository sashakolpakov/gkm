from probe import *
from helpers import avatar, timer
env = get_env()
c = env.clone()
PLAN = (
    [4,2,5,2,2,5] +        # B4 -> bottom (44,32)
    [1,4,2,5,2,5] +        # B6 -> bottom (44,36)
    [3,3,3,5,3,5] +        # B5 -> left (40,20)
    [1,1,1,4,1,3,5,3,5] +  # B1 -> left (24,20)
    [2,4,4,4,1,5,2,2,5,3,1,4,5,4,4,5] +  # B3 -> right (32,44)
    [1,3,3,3,1,4,5,4,4,4,5]              # B2 -> right (24,44)
)
for i, a in enumerate(PLAN):
    c.step(a)
    if c.terminal():
        print("TERM during plan at", i, "lv", c.levels_completed); break
f = np.array(c.frame())
print("plan done. avatar:", avatar(f), "timer:", timer(f), "lv:", c.levels_completed)
prevf = f
for i in range(40):
    c.step(5)
    f = np.array(c.frame())
    ys,xs = np.where(f!=prevf)
    ch={}
    for y,x in zip(ys,xs):
        if y==63: continue
        a,b=int(prevf[y,x]),int(f[y,x])
        if 14 in (a,b): continue
        ch.setdefault((a,b),[]).append((int(y),int(x)))
    if ch: print(i, {k:(len(v),v[0],v[-1]) for k,v in ch.items()}, "lv:", c.levels_completed, "t:", timer(f))
    prevf=f
    if c.terminal(): print("TERM lv:", c.levels_completed, "t:", timer(f)); break
