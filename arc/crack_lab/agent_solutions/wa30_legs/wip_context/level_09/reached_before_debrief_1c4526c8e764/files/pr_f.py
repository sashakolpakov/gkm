import l9lib as L
import numpy as np
def mv(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==15)
    return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
e=L.fresh()
sched=[]
for i in range(46):
    e.step(5)  # USE as no-op idle (avatar at (8,8), nothing adjacent? box... check no side effect)
    sched.append(mv(e))
print(list(enumerate(sched,1)))
