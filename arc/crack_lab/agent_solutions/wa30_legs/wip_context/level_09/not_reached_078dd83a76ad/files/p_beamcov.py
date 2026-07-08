import sys,time,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
GOALS=[tuple(g) for g in json.load(open('/tmp/l8_goals.json'))]
def scan(f):
    boxes=set(); av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 9 in u and (4 in u or 3 in u): boxes.add((R,C))
    return av,boxes
def cover(f):
    c=0
    for (R,C) in GOALS:
        u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
        if 9 in u and (3 in u or 4 in u): c+=1
    return c
def feat(e):
    f=np.asarray(e.frame()); return (e.levels_completed, cover(f))
def sig(e):
    f=np.asarray(e.frame()); av,bx=scan(f); return (av,frozenset(bx))
env=l8();base=env.levels_completed
beam=[(feat(env),env.clone(),[])];seen={sig(env)};best=None;mx=-1
t0=time.time()
for d in range(120):
    if time.time()-t0>150:print("time",d);break
    cand=[]
    for sc,node,pth in beam:
        for a in (1,2,3,4,5):
            ch=node.clone();ch.step(a)
            if ch.levels_completed>base:best=pth+[a];break
            if ch.terminal():continue
            k=sig(ch)
            if k in seen:continue
            seen.add(k);fe=feat(ch);cand.append((fe,ch,pth+[a]))
            if fe[1]>mx:mx=fe[1]
        if best:break
    if best:break
    if not cand:break
    cand.sort(key=lambda x:x[0],reverse=True);beam=cand[:200]
    if d%8==0:print('d',d,'best',beam[0][0],'maxcover',mx,'t',round(time.time()-t0,1))
print('FOUND',None if not best else len(best),'maxcover',mx,'of',len(GOALS))
if best:json.dump(best,open('/tmp/l8_win.json','w'));print(best)
