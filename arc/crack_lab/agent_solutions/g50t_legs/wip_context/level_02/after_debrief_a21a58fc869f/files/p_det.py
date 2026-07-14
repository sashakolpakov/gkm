import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
# determinism check
e1=A.Arena('g50t'); e2=A.Arena('g50t')
seq=[2,4,2,4,5,2,3]
for a in seq: e1.step(a); e2.step(a)
print("deterministic:",(e1.frame()==e2.frame()).all())
# what changes on a single no-op action (UP at start = blocked)
e=A.Arena('g50t')
b=e.frame().copy(); e.step(1); f=e.frame()
print("UP@start diff",int((b!=f).sum()))
# after one DOWN then UP (noop) 
e=A.Arena('g50t'); e.step(2); b=e.frame().copy(); e.step(1); 
import numpy as np
ys,xs=np.where(b!=e.frame())
print("noop UP after DOWN changes:",[(int(y),int(x),int(b[y,x]),int(e.frame()[y,x])) for y,x in zip(ys,xs)])
# bottom row over several toggles
e=A.Arena('g50t')
def brow(e): return ''.join(str(int(v)) for v in e.frame()[63,55:64])
print("brow start",brow(e))
for i in range(5):
    for a in [4,4,4,4,5]: e.step(a)
    print("brow after toggle",i+1,brow(e))
