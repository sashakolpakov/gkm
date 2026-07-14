import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def key(e):
    f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
def prog(env):
    start=env.clone(); seen={key(start)}; q=deque([(start,[])]); configs=[(start,[])]
    while q:
        n,p=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen.add(k); q.append((c,p+[a])); configs.append((c,p+[a]))
    for c,p in configs:
        f=np.asarray(c.frame())
        av=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63]
        # left block cells rows38-42 col14
        leftblock=''.join(str(int(f[r,14])) for r in range(38,43))
        print("len",len(p),"av",av,"c8",P.color_counts(f).get(8),"leftcol14[38..42]",leftblock)
    raise SystemExit
A.run_program('g50t', prog)
