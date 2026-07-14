import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def key(env):
    return np.asarray(env.frame()).tobytes()

def prog(env):
    start=env.clone(); seen={key(start):[]}; q=deque([(start,[])])
    found=[]
    maxlvl=0
    while q:
        n,pth=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen[k]=pth+[a]; q.append((c,pth+[a]))
            if c.levels_completed>0 or c.terminal():
                found.append((c.levels_completed,c.terminal(),pth+[a]))
            if c.levels_completed>maxlvl: maxlvl=c.levels_completed
        if len(seen)>20000: 
            print("cap"); break
    print("states",len(seen),"maxlvl",maxlvl,"found",found[:5])
    raise SystemExit
A.run_program('g50t', prog)
