import sys,time; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def scan(f):
    boxes=set(); movers=[]; av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 15 in u: movers.append((R,C))
            if 9 in u and 4 in u: boxes.add((R,C))
    return av,boxes,tuple(sorted(movers))
def seated(f):
    return int((f[8:16,44:60]==4).sum())+int((f[48:60,48:60]==4).sum())
def sig(e):
    f=np.asarray(e.frame()); av,bx,mv=scan(f); return (av,frozenset(bx),mv)
env=l8(); base=env.levels_completed
beam=[(0,env.clone(),[])]; seen={sig(env)}; bestnode=None; bestsc=-1
t0=time.time()
for depth in range(40):
    if time.time()-t0>60: break
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k)
            s=seated(np.asarray(ch.frame()))
            cand.append((s,ch,path+[a]))
            if s>bestsc: bestsc=s; bestnode=(ch,path+[a])
    if not cand: break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:150]
print("best seated px",bestsc,"pathlen",len(bestnode[1]))
f=np.asarray(bestnode[0].frame())
print("TOP rows8-16 cols44-60:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
print("BOT rows48-60 cols44-60:")
for r in f[48:60,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
