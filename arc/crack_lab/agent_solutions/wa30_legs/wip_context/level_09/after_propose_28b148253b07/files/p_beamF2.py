import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
def scan(f):
    boxes=set(); av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 9 in u and 4 in u: boxes.add((R,C))
    return av,boxes
def inTOP(c): return 2<=c[0]<=3 and 11<=c[1]<=14
def inBOT(c): return 12<=c[0]<=14 and 12<=c[1]<=14
def feat(e):
    f=np.asarray(e.frame()); av,bx=scan(f)
    fill=int((f[8:16,44:60]==4).sum())+int((f[48:60,48:60]==4).sum())
    free=sum(1 for b in bx if not inTOP(b) and not inBOT(b))
    return (-free, fill)
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f); return (av,frozenset(bx))
env=l8()
for a in path[:20]: env.step(a)
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; saved=None
t0=time.time()
for depth in range(100):
    if time.time()-t0>120 or saved: break
    cand=[]
    for sc,node,pth in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); fe=feat(ch); cand.append((fe,ch,pth+[a]))
            if fe[0]==0: saved=(ch.clone(),pth+[a]); break
        if saved: break
    if saved: break
    if not cand: break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:150]
if saved:
    f=np.asarray(saved[0].frame())
    print("reached free=0, beam depth",len(saved[1]))
    # full cell map
    for R in range(16):
        row=""
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=sorted(set(int(v) for v in np.unique(blk)))
            vals,cnts=np.unique(blk,return_counts=True); dom=int(vals[np.argmax(cnts)])
            row+=f"{dom:3d}"
        print(f"{R:2d}{row}")
    for c in [0,2,3,4,5,7,9,12,14,15]:
        print("c",c,int((f==c).sum()))
else: print("did not reach free=0")
