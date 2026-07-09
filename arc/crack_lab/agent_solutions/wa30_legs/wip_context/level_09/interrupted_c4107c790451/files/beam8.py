import numpy as np, time
from l8env import l8
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def score(f):
    int9=int((f[9:14,45:58]==9).sum())+int((f[49:58,49:58]==9).sum())
    seated=0
    for R in range(16):
        for C in range(16):
            b=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(b))
            if 9 in u and 5 in u and 2 not in u and int((b==9).sum())>=4 and int((b==5).sum())>=8: seated+=1
    return int9+4*seated
start=l8()
beam=[(score(start.frame()),start)]
best=None; t0=time.time()
for depth in range(140):
    cand=[]
    for sc,env in beam:
        for a in (UP,DOWN,LEFT,RIGHT,USE):
            c=env.clone()
            if c.terminal(): continue
            c.step(a)
            if c.levels_completed>7:
                best=c; break
            cand.append((score(c.frame()),c))
        if best: break
    if best: print('WIN at depth',depth); break
    if not cand: break
    cand.sort(key=lambda x:-x[0])
    beam=cand[:10]
    if depth%20==0: print('depth',depth,'best score',beam[0][0],'t',round(time.time()-t0,1))
    if time.time()-t0>120: print('timeout'); break
print('best score reached', beam[0][0] if not best else 'WIN')
if best:
    import json
    print('winning path len', len(best.path))
