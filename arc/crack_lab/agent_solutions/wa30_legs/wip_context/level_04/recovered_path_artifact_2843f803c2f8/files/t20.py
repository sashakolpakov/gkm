from probe import *
from helpers import avatar, timer, objects
env = get_env()
c = env.clone()
# box A: at (36,32) -> bottom fence at cols 32-35
c.step(4); c.step(2); c.step(5); c.step(2); c.step(2); c.step(5)
# now avatar (40,32). box B: at (40,36) right of avatar -> bump right, grab, push into right fence? 
# grab: face right
c.step(4)  # bump into box at (40,36)
c.step(5)  # grab
# carry right: box leads right toward right fence (cols 44-47)
c.step(4); c.step(4)
f = np.array(c.frame())
print("avatar:", avatar(f))
c.step(5)  # release in right fence
for r in range(36,52):
    print(''.join(CH[int(v)] for v in f[r,32:60]))
prevf = np.array(c.frame())
for i in range(60):
    c.step(1)
    f = np.array(c.frame())
    ys,xs = np.where(f!=prevf)
    ch={}
    for y,x in zip(ys,xs):
        if y==63: continue
        a,b=int(prevf[y,x]),int(f[y,x])
        if 14 in (a,b): continue
        ch.setdefault((a,b),[]).append((int(y),int(x)))
    if ch: print(i, {k:(len(v),v[0],v[-1]) for k,v in ch.items()}, "lv:", c.levels_completed)
    prevf=f
    if c.terminal(): print("TERM lv:", c.levels_completed); break
print("timer:", timer(prevf), "levels:", c.levels_completed)
