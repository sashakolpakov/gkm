import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
def show(f,label,r0=33,r1=48,c0=44,c1=62):
    print(f"-- {label}")
    for r in range(r0,r1):
        print("".join(f"{v:X}" if v!=0 else "." for v in f[r,c0:c1]))
f0=env.frame(); show(f0,"BASE")
for act in (1,2,3,4):
    c=env.clone(); c.step(act); show(c.frame(),f"ACT{act}")
