import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
REGS=[(slice(8,16),slice(44,60)),(slice(48,60),slice(48,60)),(slice(48,60),slice(12,24)),(slice(8,20),slice(4,16))]
def scan(f):
    boxes=set(); av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 9 in u and 4 in u: boxes.add((R,C))
    return av,boxes
def fill(f): return sum(int((f[r]==4).sum()) for r in REGS)
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f); return (av,frozenset(bx))
env=l8(); base=env.levels_completed
beam=[(0,env.clone(),[])]; seen={sig(env)}; bestnode=None; mx=-1
t0=time.time()
for depth in range(45):
    if time.time()-t0>90: break
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); fl=fill(np.asarray(ch.frame()))
            cand.append((fl,ch,path+[a]))
            if fl>mx: mx=fl; bestnode=(ch.clone(),path+[a])
    if not cand: break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:200]
print("mx fill",mx,"pathlen",len(bestnode[1]))
# idle from bestnode
c=bestnode[0]; base2=c.levels_completed
for i in range(50):
    if c.terminal(): print("terminal at idle",i); break
    c.step(1 if i%2==0 else 2)
    if c.levels_completed>base2: print("WIN after idle",i); break
print("final lvl",c.levels_completed)
json.dump(bestnode[1],open('/tmp/l8_fillpath.json','w'))
