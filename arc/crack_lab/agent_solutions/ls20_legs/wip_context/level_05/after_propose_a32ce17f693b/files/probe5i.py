import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
f0=env.frame()
def tile12(f):
    ys,xs=np.where(f==12)
    # tile's 12 part: the block near cols49-58 rows35-44; filter that region
    m=(ys>=33)&(ys<=46)&(xs>=45)&(xs<=60)
    if m.sum()==0: return None
    return (ys[m].min(),xs[m].min(),ys[m].max(),xs[m].max(),int(m.sum()))
print("BASE tile12(local region):",tile12(f0))
for act in (1,2,3,4):
    c=env.clone(); f=c.step(act)
    print(f"act{act}: tile12={tile12(f)}")
# also from base, sequences to find down/left: try act1 then various
print("\n-- after act1 (tile up) then each action --")
base1=env.clone(); base1.step(1); f1=base1.frame()
print(" state tile:",tile12(f1))
for act in (1,2,3,4):
    c=base1.clone(); f=c.step(act)
    print(f"  act{act}: tile12={tile12(f)}")
