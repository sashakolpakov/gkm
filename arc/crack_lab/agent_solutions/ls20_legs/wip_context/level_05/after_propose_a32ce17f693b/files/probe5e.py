import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def setup():
    env = A.Arena('ls20'); env.reset()
    ck = json.load(open("checkpoint.json"))
    for a in ck["final_path"]: env.step(a)
    return env

def summarize(f):
    # report bounding boxes of interesting colors
    out={}
    for col in (0,1,8,9,11,12):
        ys,xs=np.where(f==col)
        if len(ys): out[col]=(ys.min(),ys.max(),xs.min(),xs.max(),len(ys))
    return out

base=setup()
f0=base.frame()
print("BASE:")
for k,v in summarize(f0).items(): print(f"  col{k}: {v}")

# apply action 2 repeatedly (candidate no-op) and watch
c=base.clone()
print("\n-- repeated action 2 --")
for i in range(4):
    f=c.step(2)
    s=summarize(f)
    print(f" step {i+1}: col0={s.get(0)} col1={s.get(1)} col9={s.get(9)} col12={s.get(12)} col11={s.get(11)}")
