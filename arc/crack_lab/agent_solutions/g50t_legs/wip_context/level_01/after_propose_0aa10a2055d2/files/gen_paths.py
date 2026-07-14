import sys, numpy as np, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
def key(e):
    f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
def prog(env):
    start=env.clone(); seen={key(start):[]}; q=deque([(start,[])]); paths=[[]]
    while q:
        node,pth=q.popleft()
        for a in (1,2,3,4,5):
            c=node.clone(); c.step(a)
            if c.terminal(): continue
            k=key(c)
            if k in seen: continue
            seen[k]=pth+[a]; q.append((c,pth+[a])); paths.append(pth+[a])
    json.dump(paths, open("paths25.json","w"))
    print("saved",len(paths))
    prog.done=True
A.run_program('g50t', prog)
