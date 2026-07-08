import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path: env.step(a)
f=np.asarray(env.frame())
print("movers px",int((f==15).sum()),"couriers px",int((f==12).sum()),"lvl",env.levels_completed)
def c3(f): return int((f==3).sum())
# seat a penned box into top container
legs.carry_box_to(env,(2,2),(3,11),cap=80)
f=np.asarray(env.frame()); print("after seat c3px",c3(f),"lvl",env.levels_completed)
for i in range(10): env.step(2)
f=np.asarray(env.frame()); print("after idle c3px",c3(f),"lvl",env.levels_completed,"cour px",int((f==12).sum()))
print("TOP:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
