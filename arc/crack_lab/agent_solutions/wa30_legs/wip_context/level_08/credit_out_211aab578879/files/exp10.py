from l8env import l8
from chase import clear_both, movers
from legs import carry_box_to, _grid_scan
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
reg=[(2,3),(3,3),(4,1),(4,2),(4,3),(12,3),(12,4),(12,5),(13,3),(13,4),(13,5),(14,3),(14,4),(14,5)]
regset=set(reg)
def covered(f): return [r for r in reg if {4,3,0}&cc(f,r[0],r[1])]
def allboxes(c):
    av,boxes,walls=_grid_scan(c); return av,boxes
c=l8().clone()
clear_both(c)
print('cleared movers=%d mu=%d'%(len(movers(np.asarray(c.frame()))),mu(c)))
fail={}
for it in range(14):
    if c.terminal() or c.levels_completed>7: break
    f=np.asarray(c.frame())
    av,boxes=allboxes(c)
    # boxes not already on a region cell
    src=[b for b in boxes if b not in regset and fail.get(b,0)<2]
    tgts=[r for r in reg if not({4,3,0}&cc(f,r[0],r[1]))]  # uncovered region cells
    if not src or not tgts: print('done it',it,'src',len(src),'tgts',len(tgts)); break
    # nearest region cell to avatar, then nearest box to it
    tgts.sort(key=lambda t:abs(t[0]-av[0])+abs(t[1]-av[1]))
    tgt=tgts[0]
    src.sort(key=lambda b:abs(b[0]-tgt[0])+abs(b[1]-tgt[1]))
    ok=carry_box_to(c,src[0],tgt)
    if not ok: fail[src[0]]=fail.get(src[0],0)+1
    print('box%s->%s ok=%s lvl=%d mu=%d cov=%d'%(src[0],tgt,ok,c.levels_completed,mu(c),len(covered(np.asarray(c.frame())))))
print('FINAL lvl',c.levels_completed,'cov',len(covered(np.asarray(c.frame()))),'of',len(reg))
