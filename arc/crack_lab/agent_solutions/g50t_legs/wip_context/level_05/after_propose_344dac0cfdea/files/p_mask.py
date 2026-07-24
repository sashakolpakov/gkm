import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
import numpy as np
def mkey(e):
    f=e.frame().copy(); f[63,:]=0
    return (f.tobytes(), int(e.levels_completed))
env=A.Arena('g50t')
base=0
q=deque([(env.clone(),[])]); seen={mkey(env)}; found=None
while q and len(seen)<60000:
    n,p=q.popleft()
    if len(p)>=160: continue
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a)
        if c.levels_completed>base: found=p+[a]; q.clear(); break
        k=mkey(c)
        if k in seen: continue
        seen.add(k); q.append((c,p+[a]))
print("found",found,"states",len(seen))
if found:
    e=A.Arena('g50t')
    for a in found: e.step(a)
    print("validate levels",e.levels_completed)
