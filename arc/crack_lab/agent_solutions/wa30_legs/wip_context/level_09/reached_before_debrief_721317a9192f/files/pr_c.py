import l9lib as L
import numpy as np
e = L.fresh()
def snap(e):
    f=np.asarray(e.frame())
    boxes=set(); seated=set(); cour=[]
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            s=set(int(v) for v in blk.ravel())
            if s=={4,9}: 
                # seated if this is a socket loc
                pass
            if 12 in s: cour.append((R,C))
            if 9 in s and (4 in s or 5 in s): boxes.add((R,C))
    return boxes, cour
SOCK={(2,13),(2,14),(6,13),(6,14),(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)}
prev=None
for i in range(46):
    e.step(1)
    b,c=snap(e)
    if i%3==2:
        print(i+1,'cour',c,'unseated',sorted(x for x in b if x not in SOCK),'seated',sorted(x for x in b if x in SOCK))
