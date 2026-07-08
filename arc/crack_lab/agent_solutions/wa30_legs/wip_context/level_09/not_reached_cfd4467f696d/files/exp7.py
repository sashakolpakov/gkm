from l8env import l8
from mvcarry2 import carry
from legs import _grid_scan
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (12,13,14)]
cont=topc+botc
def empty_cont(f,cells):
    return [(R,C) for (R,C) in cells if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def occ(f): return len([(R,C) for (R,C) in cont if {4,3,0}&cc(f,R,C)])
def free_boxes(c):
    av,boxes,walls=_grid_scan(c)
    return av,[b for b in boxes if b not in cont]
c=l8().clone()
fail={}
for it in range(30):
    if c.terminal() or c.levels_completed>7: break
    f=np.asarray(c.frame())
    av,fb=free_boxes(c)
    fb=[b for b in fb if fail.get(b,0)<3]
    if not fb: break
    # nearest box to avatar
    fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=fb[0]
    # target: empty container cell in SAME band, nearest to box
    cells=topc if box[0]<6 else botc
    empt=empty_cont(f,cells)
    if not empt:
        empt=empty_cont(f,cont)
    if not empt: break
    empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
    ok=False
    for tgt in empt[:3]:
        ok=carry(c,box,tgt)
        if ok: break
    if not ok: fail[box]=fail.get(box,0)+1
    f2=np.asarray(c.frame())
    print('it%d box%s ok=%s lvl=%d mu=%d occ=%d fb=%d'%(it,box,ok,c.levels_completed,mu(c),occ(f2),len(free_boxes(c)[1])))
for _ in range(15):
    if c.terminal() or c.levels_completed>7: break
    c.step(5)
f=np.asarray(c.frame())
print('FINAL lvl',c.levels_completed,'mu',mu(c),'occ',occ(f),'fb',len(free_boxes(c)[1]))
