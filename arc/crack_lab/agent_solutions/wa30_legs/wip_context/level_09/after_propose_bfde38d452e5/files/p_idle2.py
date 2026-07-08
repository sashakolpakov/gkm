import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
env=l8()
base=env.levels_completed
def csum(f):
    return {c:int((f==c).sum()) for c in [0,2,4,7,9,12,14,15]}
f0=np.asarray(env.frame()); print("start",csum(f0),"lvl",env.levels_completed)
for step in range(160):
    a = 1 if step%2==0 else 2
    env.step(a)
    if env.levels_completed>base:
        print("LEVEL UP at",step); break
    if step%20==19:
        f=np.asarray(env.frame()); print(step,csum(f))
print("final lvl",env.levels_completed,"terminal",env.terminal())
