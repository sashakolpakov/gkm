import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH=json.load(f)["final_path"]

def markers(f):
    rings=[b for b in connected_components(f,colors=(4,),min_area=8)
           if b.size==(3,3) and b.area==8]
    return sorted((b.bbox[0]+1,b.bbox[1]+1,int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings)

def sel(f):
    bp=list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else ('N',len(bp))

def solve(env):
    for i,a in enumerate(PATH):
        env.step(a)
        # focus on level 4: steps 103..170
        if 103<=i<=170 and (i%6==0 or i>=165):
            m=markers(arr(env.frame()))
            cc=Counter(col for _,_,col in m)
            print(f"step {i} lvl={env.levels_completed} sel={sel(arr(env.frame()))} nrings={len(m)} bycolor={dict(cc)}")
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
