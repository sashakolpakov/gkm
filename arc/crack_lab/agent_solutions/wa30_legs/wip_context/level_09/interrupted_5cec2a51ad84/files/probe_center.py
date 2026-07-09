import l9env, perception as P
import numpy as np
env=l9env.get_l9()
def center2(f): return int((f[13:23,21:31]==2).sum())  # interior of center container
def c12cells(f):
    ys,xs=np.where(f==12)
    return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
for t in range(22):
    f=np.asarray(env.frame())
    print(f"t{t} center2={center2(f)} c12cells={c12cells(f)}")
    env.step(5)
