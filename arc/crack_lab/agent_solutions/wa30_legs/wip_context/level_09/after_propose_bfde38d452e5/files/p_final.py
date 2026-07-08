import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def scan(f):
    boxes=set(); av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 9 in u and 4 in u: boxes.add((R,C))
    return av,boxes
def feat(e):
    f=np.asarray(e.frame()); av,bx=scan(f)
    m=int((f==15).sum())
    rfill=int((f[8:16,44:60]==4).sum())+int((f[48:60,48:60]==4).sum())
    lfill=int((f[8:20,4:16]==4).sum())+int((f[48:60,12:24]==4).sum())
    # stage: first remove movers (m->0), then maximize both fills
    return (e.levels_completed, -m, rfill+lfill)
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f)
    def cl(c):
        ys,xs=np.where(f==c);return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
    return (av,frozenset(bx),cl(15))
env=l8();base=env.levels_completed
beam=[(feat(env),env.clone(),[])];seen={sig(env)};best=None
t0=time.time()
for d in range(130):
    if time.time()-t0>220:print("time",d);break
    cand=[]
    for sc,node,pth in beam:
        for a in (1,2,3,4,5):
            ch=node.clone();ch.step(a)
            if ch.levels_completed>base:best=pth+[a];break
            if ch.terminal():continue
            k=sig(ch)
            if k in seen:continue
            seen.add(k);cand.append((feat(ch),ch,pth+[a]))
        if best:break
    if best:break
    if not cand:break
    cand.sort(key=lambda x:x[0],reverse=True);beam=cand[:220]
    if d%8==0:print('d',d,'best',beam[0][0],'t',round(time.time()-t0,1))
print('FOUND',None if not best else len(best))
if best:json.dump(best,open('/tmp/l8_win.json','w'));print(best)
