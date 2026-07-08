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
    top=int((f[8:16,44:60]==4).sum())
    bot=int((f[48:60,48:60]==4).sum())
    return top,bot
def feat(e):
    f=np.asarray(e.frame()); av,bx,mv=scan(f); t,b=seated(f)
    # distance of free boxes (outside container bbox) to nearest container-open side
    return (e.levels_completed, t+b)
def sig(e):
    f=np.asarray(e.frame()); av,bx,mv=scan(f)
    return (av,frozenset(bx),mv)
env=l8(); base=env.levels_completed
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None
t0=time.time(); TIME=100
for depth in range(130):
    if time.time()-t0>TIME: print("time"); break
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base: best=path+[a]; break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); cand.append((feat(ch),ch,path+[a]))
        if best: break
    if best: break
    if not cand: print("dead",depth); break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:150]
    if depth%4==0: print('d',depth,'best',beam[0][0],'ncand',len(cand),'t',round(time.time()-t0,1))
print('FOUND',best)
