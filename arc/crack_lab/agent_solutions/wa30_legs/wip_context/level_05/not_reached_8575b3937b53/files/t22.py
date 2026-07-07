from probe import *
from helpers import avatar, timer
env = get_env()
c = env.clone()
# deliver B4 bottom, B6 right as in t20 but minimal, then immediately B2 top, then watch
c.step(4); c.step(2); c.step(5); c.step(2); c.step(2); c.step(5)      # B4 -> (44,32); avatar (40,32)
c.step(4); c.step(5); c.step(4); c.step(4); c.step(5)                # B6 -> (40,44); avatar (40,40)
# B2 at (24,32): avatar to (28,32): up? avatar at (40,40): up->(36,40), up->(32,40), left->(32,36)?... plan: up,up,left,left? cells: (40,40)->(36,40)->(32,36)? no, one move at a time.
for a in [1,1,3,3,1]:  # (36,40),(32,40),(32,36),(32,32),(28,32)
    c.step(a)
print("avatar before grab B2:", avatar(np.array(c.frame())))
c.step(1)  # bump B2 (24,32)
c.step(5)  # grab (box on top)
c.step(1)  # push up: avatar (24,32)? box embeds (20,32)
f=np.array(c.frame()); print("after push:", avatar(f))
c.step(5)  # release
prevf = np.array(c.frame())
print("timer:", timer(prevf))
for i in range(60):
    c.step(2)
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
    if c.terminal(): print("TERM lv:", c.levels_completed); break
