import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def scan(f):
    av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            if 14 in set(int(v) for v in np.unique(blk)): av=(R,C)
    return av
def agents(f): return int((f==15).sum())+int((f==12).sum())
def feat(e):
    f=np.asarray(e.frame()); return (e.levels_completed, -agents(f))
def sig(e):
    f=np.asarray(e.frame())
    def cells(c):
        ys,xs=np.where(f==c); return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
    return (scan(f),cells(15),cells(12))
env=l8(); base=env.levels_completed
beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None; mn=999; bestnode=None
t0=time.time()
for depth in range(120):
    if time.time()-t0>160: print("time",depth);break
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base: best=path+[a];break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); fe=feat(ch); cand.append((fe,ch,path+[a]))
            ag=-fe[1]
            if ag<mn: mn=ag; bestnode=(ch.clone(),path+[a])
        if best: break
    if best: break
    if not cand: print("dead",depth);break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:250]
    if depth%6==0: print('d',depth,'best',beam[0][0],'mn',mn,'t',round(time.time()-t0,1))
print('FOUND',None if not best else len(best),'minagents',mn)
if best: json.dump(best,open('/tmp/l8_win.json','w')); print(best)
elif bestnode:
    json.dump(bestnode[1],open('/tmp/l8_minagent.json','w'))
    print("minagent pathlen",len(bestnode[1]))
    c=bestnode[0]; b2=c.levels_completed
    for i in range(40):
        if c.terminal() or c.levels_completed>b2: break
        c.step(1 if i%2==0 else 2)
    print("after idle from minagent lvl",c.levels_completed)
