from l8env import l8
from legs import carry_box_to, _grid_scan
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
topc=[(r,co) for r in (2,3) for co in (11,12,13,14)]
botc=[(r,co) for r in (12,13,14) for co in (12,13,14)]
cont=set(topc+botc)
def empty(f,cells):
    return [(R,C) for (R,C) in cells if 2 in cc(f,R,C) and not({4,3,0}&cc(f,R,C))]
def free(c):
    av,boxes,walls=_grid_scan(c); return av,[b for b in boxes if b not in cont]
c=l8().clone()
cnt=[0]; o=c.step
def cs(a): 
    if cnt[0]>=134: raise RuntimeError('cap')
    cnt[0]+=1; return o(a)
c.step=cs
fail={}
try:
    while cnt[0]<108:
        f=np.asarray(c.frame())
        av,fb=free(c)
        fb=[b for b in fb if fail.get(b,0)<2]
        if not fb: break
        fb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
        box=fb[0]
        cells=topc if box[0]<6 else botc
        empt=empty(f,cells) or empty(f,topc+botc)
        if not empt: break
        empt.sort(key=lambda t:abs(t[0]-box[0])+abs(t[1]-box[1]))
        ok=False
        for tgt in empt[:2]:
            b=cnt[0]
            ok=carry_box_to(c,box,tgt)
            if ok: break
        if not ok: fail[box]=fail.get(box,0)+1
        print('box%s ok=%s steps=%d lvl=%d'%(box,ok,cnt[0],c.levels_completed))
        if c.levels_completed>7: print('WIN'); break
    while cnt[0]<134 and c.levels_completed==7 and not c.terminal():
        c.step(5)
except RuntimeError:
    pass
print('FINAL lvl',c.levels_completed,'steps',cnt[0])
