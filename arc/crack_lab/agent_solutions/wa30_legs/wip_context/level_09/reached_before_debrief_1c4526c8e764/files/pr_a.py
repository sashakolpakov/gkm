import l9lib as L
import numpy as np, perception as P
e = L.fresh()
f0 = np.asarray(e.frame())
def bar7(e): return int((np.asarray(e.frame())==7).sum())
def cells(e):
    f=np.asarray(e.frame()); out={}
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            s=tuple(sorted(set(int(v) for v in blk.ravel())))
            out[(R,C)]=s
    return out
print('bar7 start', bar7(e))
# idle 10 steps: watch mover + couriers + bar
prev=cells(e)
for i in range(12):
    e.step(1)
    cur=cells(e)
    ch={k:(prev[k],cur[k]) for k in cur if cur[k]!=prev[k]}
    print(i, 'bar', bar7(e), 'changes', ch)
    prev=cur
