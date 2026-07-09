from l8env import l8
from legs import carry_box_to, _grid_scan
import numpy as np

def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (11,12,13,14)]
cont=topc+botc
def empty_cont(f):
    return [(R,C) for (R,C) in cont if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def free_boxes(c):
    av,boxes,walls=_grid_scan(c)
    fb=[b for b in boxes if not(2<=b[0]<=3 and b[1]>=11) and not(12<=b[0]<=14 and b[1]>=11)]
    return av,fb

c=l8().clone()
fail={}
for it in range(40):
    if c.terminal() or c.levels_completed>7: break
    f=np.asarray(c.frame())
    av,fb=free_boxes(c)
    fb=[b for b in fb if fail.get(b,0)<2]
    empt=empty_cont(f)
    if not fb or not empt:
        break
    fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=fb[0]
    empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
    ok=False
    for tgt in empt[:2]:
        ok=carry_box_to(c,box,tgt)
        if ok: break
    if not ok:
        fail[box]=fail.get(box,0)+1
    print('it%d box%s ok=%s lvl=%d mu=%d freeleft=%d'%(it,box,ok,c.levels_completed,mu(c),len(free_boxes(c)[1])))
# idle tail
for _ in range(30):
    if c.terminal() or c.levels_completed>7: break
    c.step(5)
print('FINAL lvl',c.levels_completed,'mu',mu(c),'term',c.terminal(),'freeleft',len(free_boxes(c)[1]))
