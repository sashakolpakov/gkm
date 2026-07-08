from l8env import l8
from legs import carry_box_to, _grid_scan
import numpy as np

def cellcolors(f,R,C):
    return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))

def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())

topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (11,12,13,14)]
cont=topc+botc

def empty_cont(f):
    out=[]
    for (R,C) in cont:
        u=cellcolors(f,R,C)
        if 2 in u and not (4 in u or 3 in u or 0 in u):
            out.append((R,C))
    return out

def free_boxes(c):
    av,boxes,walls=_grid_scan(c)
    fb=[]
    for b in boxes:
        if 2<=b[0]<=3 and b[1]>=11: continue
        if 12<=b[0]<=14 and b[1]>=11: continue
        fb.append(b)
    return av,fb

c=l8().clone()
for it in range(30):
    if c.terminal() or c.levels_completed>7: break
    f=np.asarray(c.frame())
    av,fb=free_boxes(c)
    empt=empty_cont(f)
    if not fb:
        print('no free boxes at it',it); break
    if not empt:
        print('no empty cont'); break
    fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=fb[0]
    empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
    ok=False
    for tgt in empt[:3]:
        ok=carry_box_to(c,box,tgt)
        if ok: break
    print('it%d box%s ok=%s lvl=%d mu=%d freeleft=%d'%(it,box,ok,c.levels_completed,mu(c),len(free_boxes(c)[1])))
    if not ok:
        # skip: nudge to avoid infinite loop
        c.step(2)
print('FINAL lvl',c.levels_completed,'mu',mu(c),'term',c.terminal())
