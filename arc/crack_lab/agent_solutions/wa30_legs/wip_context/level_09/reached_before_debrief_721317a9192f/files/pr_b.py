import l9lib as L
import numpy as np
e = L.fresh()
SOCK = [(2,13),(2,14),(6,13),(6,14),(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
def stat(e):
    f=np.asarray(e.frame())
    seated=0
    for (R,C) in SOCK:
        blk=f[R*4:R*4+4,C*4:C*4+4]
        s=set(int(v) for v in blk.ravel())
        if 4 in s and 9 in s: seated+=1
    ys,xs=np.where(f==15)
    mv=(int(ys.min())//4,int(xs.min())//4) if len(ys) else None
    return seated, int((f==7).sum()), mv, e.levels_completed, e.terminal()
for i in range(70):
    if e.terminal(): break
    e.step(1)
    if i%5==4 or e.terminal(): print(i+1, stat(e))
print('final', stat(e))
