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
def feat(e):
    f=np.asarray(e.frame()); av,bx=scan(f)
    return (e.levels_completed, fill(f))
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f); return (av,frozenset(bx))
env=l8(); base=env.levels_completed
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None; mx=-1
t0=time.time()
for depth in range(130):
    if time.time()-t0>170: print("time",depth); break
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base: best=path+[a]; break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); fe=feat(ch); cand.append((fe,ch,path+[a]))
            if fe[1]>mx: mx=fe[1]
        if best: break
    if best: break
    if not cand: print("dead",depth); break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:200]
    if depth%8==0: print('d',depth,'best',beam[0][0],'mx',mx,'t',round(time.time()-t0,1))
print('FOUND',None if not best else len(best))
if best: json.dump(best,open('/tmp/l8_win.json','w')); print(best)
