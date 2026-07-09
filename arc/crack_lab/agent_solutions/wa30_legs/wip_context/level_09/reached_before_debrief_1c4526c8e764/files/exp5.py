from l8env import l8
from legs import carry_box_to, _grid_scan
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (12,13,14)]
cont=topc+botc
def empty_cont(f,cells):
    return [(R,C) for (R,C) in cells if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def occ(f):
    return len([(R,C) for (R,C) in cont if {4,3,0}&cc(f,R,C)])
def free_boxes(c,region):
    av,boxes,walls=_grid_scan(c)
    if region=='bot': fb=[b for b in boxes if b[0]>9 and b not in cont]
    else: fb=[b for b in boxes if b[0]<6 and b not in cont]
    return av,fb

c=l8().clone()
fail={}
def run_region(region,cells,iters):
    global c,fail
    for it in range(iters):
        if c.terminal() or c.levels_completed>7: return
        f=np.asarray(c.frame())
        av,fb=free_boxes(c,region)
        fb=[b for b in fb if fail.get(b,0)<3]
        empt=empty_cont(f,cells)
        if not fb or not empt: return
        fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
        box=fb[0]
        empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
        ok=False
        for tgt in empt[:3]:
            ok=carry_box_to(c,box,tgt)
            if ok: break
        if not ok: fail[box]=fail.get(box,0)+1
        f2=np.asarray(c.frame())
        print('%s it%d box%s ok=%s lvl=%d mu=%d occ=%d'%(region,it,box,ok,c.levels_completed,mu(c),occ(f2)))
run_region('bot',botc,12)
run_region('top',topc,12)
for _ in range(15):
    if c.terminal() or c.levels_completed>7: break
    c.step(5)
f=np.asarray(c.frame())
print('FINAL lvl',c.levels_completed,'mu',mu(c),'occ',occ(f))
