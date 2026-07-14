import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def key(e):
    f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
def prog(env):
    start=env.clone(); seen={key(start)}; q=deque([(start,[])])
    win=None; has2=0; lowest=0; lowpath=None
    n=0
    while q:
        node,pth=q.popleft()
        for a in (1,2,3,4,5):
            c=node.clone(); c.step(a)
            if c.levels_completed>0:
                win=pth+[a]; print("WIN",len(win),win); raise SystemExit
            if c.terminal(): continue
            k=key(c)
            if k in seen: continue
            seen.add(k); q.append((c,pth+[a]))
            f=np.asarray(c.frame())
            if (f==2).any(): has2+=1
            for b in P.connected_components(f,colors=[9]):
                if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15:
                    if b.bbox[0]>lowest: lowest=b.bbox[0]; lowpath=pth+[a]
        n+=1
        if len(seen)>60000:
            print("CAP",len(seen)); break
    print("total",len(seen),"states_with_color2",has2,"lowest_avatar_row",lowest,"win",win)
    print("lowpath",lowpath)
    raise SystemExit
A.run_program('g50t', prog)
