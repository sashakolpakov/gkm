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
    return (e.levels_completed, -free, fill)
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f); return (av,frozenset(bx))
env=l8()
for a in path[:20]: env.step(a)
base=env.levels_completed; start_moves=len(env.path)-466
print("start after prefix, level8 moves",start_moves,"feat",feat(env))
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None
t0=time.time()
for depth in range(110):
    if time.time()-t0>170: print("time",depth);break
    cand=[]
    for sc,node,pth in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base: best=pth+[a];break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); cand.append((feat(ch),ch,pth+[a]))
        if best: break
    if best: break
    if not cand: print("dead",depth);break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:150]
    if depth%6==0: print('d',depth,'best',beam[0][0],'t',round(time.time()-t0,1))
print('FOUND',None if not best else len(best))
if best:
    full=path[:20]+best
    json.dump(full,open('/tmp/l8_win.json','w'))
    print("total level8 moves",len(full))
