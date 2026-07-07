import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
legs.carry_box_to(env,(2,2),(3,11),cap=80)
def cnt(f): 
    return {c:int((f==c).sum()) for c in [3,4,9,12,15,0,14]}
f=np.asarray(env.frame()); print("seat",cnt(f))
for i in range(10):
    env.step(2)  # idle DOWN (avatar in container, may move)
    f=np.asarray(env.frame())
    # where is color3
    ys,xs=np.where(f==3)
    loc=sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
    print(i,cnt(f),"c3loc",loc)
