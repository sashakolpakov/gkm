import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
f=np.asarray(env.frame()); print("after prefix20: 15px",int((f==15).sum()),"moves",20)
base=env.levels_completed
def fill(f): return int((f[8:16,44:60]==4).sum()),int((f[48:60,48:60]==4).sum())
for i in range(115):
    if env.terminal(): print("terminal at",i); break
    env.step(1 if i%2==0 else 2)
    if env.levels_completed>base: print("WIN at idle step",i,"total",20+i+1); break
    if i%12==11: f=np.asarray(env.frame()); print(20+i+1,"fill",fill(f))
print("final",env.levels_completed)
