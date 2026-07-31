import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH=json.load(f)["final_path"]

def sel(f):
    bp=list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else None

def geom(env,exclude=(0,2,4)):
    bf=arr(env.frame()).copy()
    bg=Counter(int(v) for v in bf.flat).most_common(1)[0][0]
    ctr=sel(bf); votes=[]
    for act in (1,2,3,4):
        m=env.clone();m.step(act);af=arr(m.frame())
        for r,c in zip(*((bf!=af).nonzero())):
            if int(af[r,c])==bg and int(bf[r,c]) not in exclude+(bg,):
                votes.append(int(bf[r,c]))
    col=Counter(votes).most_common(1)[0][0]
    offs=set()
    for act in (1,2,3,4):
        m=env.clone();m.step(act);af=arr(m.frame())
        for r,c in zip(*((bf!=af).nonzero())):
            if int(bf[r,c])==col and int(af[r,c])==bg:
                offs.add((int(r)-ctr[0],int(c)-ctr[1]))
    rr=[o[0] for o in offs];cc=[o[1] for o in offs]
    return ctr,col,(min(rr),max(rr),min(cc),max(cc)),len(offs)

def solve(env):
    for a in PATH: env.step(a)
    c=env.clone()
    c.step(5); c.step(5)   # cycle to plus (X->diamond->plus)
    print("plus at edge:", geom(c))
    # move plus left 5 steps (15 cells) and up 4 (12) to open area
    for _ in range(6): c.step(3)   # left 18
    for _ in range(2): c.step(1)   # up 6
    print("plus moved:", geom(c))
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
