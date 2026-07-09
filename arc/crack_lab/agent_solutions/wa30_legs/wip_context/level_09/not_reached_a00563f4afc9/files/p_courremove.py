import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
# beam to minimize courier(12) px, keeping movers
def scan(f):
    av=None
    for R in range(16):
        for C in range(16):
            if 14 in set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4])): av=(R,C)
    return av
def feat(e):
    f=np.asarray(e.frame()); return (e.levels_completed,-int((f==12).sum()))
def sig(e):
    f=np.asarray(e.frame())
    def cl(c):
        ys,xs=np.where(f==c);return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
    return (scan(f),cl(12),cl(15))
env=l8();base=env.levels_completed
beam=[(feat(env),env.clone(),[])];seen={sig(env)};best=None;mn=99;bn=None
t0=time.time()
for d in range(60):
    if time.time()-t0>90:break
    cand=[]
    for sc,node,pth in beam:
        for a in (1,2,3,4,5):
            ch=node.clone();ch.step(a)
            if ch.levels_completed>base:best=pth+[a];break
            if ch.terminal():continue
            k=sig(ch)
            if k in seen:continue
            seen.add(k);fe=feat(ch);cand.append((fe,ch,pth+[a]))
            if -fe[1]<mn:mn=-fe[1];bn=(ch.clone(),pth+[a])
        if best:break
    if best:break
    if not cand:break
    cand.sort(key=lambda x:x[0],reverse=True);beam=cand[:200]
    if d%6==0:print('d',d,'best',beam[0][0],'mn12',mn)
print('FOUND',best,'min12',mn)
if bn:
    # from min-courier state, yield and check win + left fills
    c=bn[0];b2=c.levels_completed
    TL=(slice(8,20),slice(4,16));BL=(slice(48,60),slice(12,24))
    for i in range(60):
        if c.terminal() or c.levels_completed>b2:break
        c.step(1 if i%2==0 else 2)
    f=np.asarray(c.frame())
    print("min12 pathlen",len(bn[1]),"after yield lvl",c.levels_completed,"TLfill",int((f[TL]==4).sum()),"BLfill",int((f[BL]==4).sum()))
