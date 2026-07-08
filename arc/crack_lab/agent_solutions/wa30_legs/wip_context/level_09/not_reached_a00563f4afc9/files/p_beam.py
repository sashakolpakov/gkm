import sys,time; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
TOP={(r,c) for r in (2,3) for c in (11,12,13,14)}
BOT={(r,c) for r in (12,13,14) for c in (12,13,14)}
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
    t=len(bx&TOP); b=len(bx&BOT)
    return (e.levels_completed, t+b, -(len(bx)-t-b))
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f)
    return (av,frozenset(bx))
env=l8(); base=env.levels_completed
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None
t0=time.time()
for depth in range(40):
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
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:120]
    if not beam: break
    print('d',depth,'best',beam[0][0],'ncand',len(cand),'t',round(time.time()-t0,1))
print('FOUND',best)
