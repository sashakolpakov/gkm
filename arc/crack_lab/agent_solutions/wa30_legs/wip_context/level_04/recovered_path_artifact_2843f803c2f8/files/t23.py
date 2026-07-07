from probe import *
from helpers import avatar, timer
env = get_env()
c = env.clone()
c.step(4); c.step(2); c.step(5); c.step(2); c.step(2); c.step(5)      # B4 bottom
c.step(4); c.step(5); c.step(4); c.step(4); c.step(5)                # B6 -> (40,40) inside! courier grabs anyway
# B5 (40,24): avatar (40,40): left moves: cells row4: (40,36),(40,32),(40,28) then bump (40,24)
for a in [3,3,3]:
    c.step(a)
c.step(3)  # bump B5
c.step(5)  # grab on left side
c.step(2)  # move down? box on left: avatar (44?) no avatar at (40,28) can't go down (fence). push left: 
c.step(3)  # avatar (40,24), box embeds (40,20)? 
f=np.array(c.frame()); print("avatar:", avatar(f), "timer:", timer(f))
c.step(5)
prevf = np.array(c.frame())
for i in range(70):
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
    if c.terminal(): print("TERM lv:", c.levels_completed, "t:", timer(f)); break
