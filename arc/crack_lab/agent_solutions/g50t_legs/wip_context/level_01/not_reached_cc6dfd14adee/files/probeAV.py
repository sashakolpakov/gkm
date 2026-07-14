import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
def prog(env):
    start=env.clone()
    f0=np.asarray(start.frame())
    def maskkey(e):
        f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
    seen={maskkey(start)}; q=deque([start])
    # track which cells EVER differ from initial (excluding counter row63)
    everchanged=np.zeros((64,64),bool)
    while q:
        n=q.popleft()
        f=np.asarray(n.frame())
        diff=(f!=f0); diff[63,:]=False
        everchanged|=diff
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a)
            if c.terminal(): continue
            k=maskkey(c)
            if k in seen: continue
            seen.add(k); q.append(c)
    ys,xs=np.where(everchanged)
    if len(ys):
        print("changed region bbox rows",ys.min(),ys.max(),"cols",xs.min(),xs.max())
    print("goal box (rows49-56 cols42-51) ever changed?", everchanged[49:57,42:52].any())
    print("num ever-changed cells", int(everchanged.sum()))
    # print a compact map of changed area rows7-45
    for r in range(7,45):
        print(f"{r:2d} "+''.join('X' if everchanged[r,c] else '.' for c in range(13,52)))
    raise SystemExit
A.run_program('g50t', prog)
