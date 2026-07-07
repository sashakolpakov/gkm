import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def setup():
    env = A.Arena('ls20'); env.reset()
    ck = json.load(open("checkpoint.json"))
    for a in ck["final_path"]: env.step(a)
    return env
base=setup(); f0=base.frame()

def show(f, r0,r1,c0,c1,label):
    print(f"-- {label} rows{r0}:{r1} cols{c0}:{c1}")
    for r in range(r0,r1):
        print("".join(f"{v:X}" if v!=0 else "." for v in f[r,c0:c1]))

# avatar region and 5x5 tile region for base and each action
for act in (1,2,3,4):
    c=base.clone(); f=c.step(act)
    print(f"\n===== ACTION {act} =====")
    show(f,34,40,13,25,"avatar-area")
    show(f,34,46,47,60,"tile-area")
print("\n===== BASE =====")
show(f0,34,40,13,25,"avatar-area")
show(f0,34,46,47,60,"tile-area")
