from l8env import l8
from chase import clear_both, movers
from legs import carry_box_to, _grid_scan
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (12,13,14)]
cont=set(topc+botc)
def empty(f,cells):
    return [(R,C) for (R,C) in cells if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def free(c):
    av,boxes,walls=_grid_scan(c); return av,[b for b in boxes if b not in cont]
def occ(f): return len([(R,C) for (R,C) in cont if {4,3,0}&cc(f,R,C)])
c=l8().clone()
clear_both(c)
print('cleared, movers',len(movers(np.asarray(c.frame()))),'mu',mu(c))
fail={}
for it in range(12):
    if c.terminal() or c.levels_completed>7: break
    f=np.asarray(c.frame())
    av,fb=free(c)
    fb=[b for b in fb if fail.get(b,0)<2]
    if not fb: print('no free'); break
    fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=fb[0]
    cells=topc if box[0]<6 else botc
    empt=empty(f,cells) or empty(f,topc+botc)
    if not empt: break
    empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
    ok=any(carry_box_to(c,box,t) for t in empt[:2])
    if not ok: fail[box]=fail.get(box,0)+1
    print('carry box%s ok=%s lvl=%d mu=%d occ=%d'%(box,ok,c.levels_completed,mu(c),occ(np.asarray(c.frame()))))
print('FINAL lvl',c.levels_completed,'mu',mu(c))
