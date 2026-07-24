import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH=json.load(f)["final_path"]
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5

def markers(f):
    rings=[b for b in connected_components(f,colors=(4,),min_area=8)
           if b.size==(3,3) and b.area==8]
    return set((b.bbox[0]+1,b.bbox[1]+1,int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings)

def sel(f):
    bp=list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else None

def solve(env):
    for a in PATH: env.step(a)
    base=env.clone()
    full=set(markers(arr(base.frame())))
    # For each shape index, move it around and union revealed markers
    for idx in range(3):
        c=env.clone()
        for _ in range(idx): c.step(USE)
        print("shape idx",idx,"center",sel(arr(c.frame())))
        # sweep it through several offsets, union markers seen
        for seq in ([UP]*4,[DOWN]*4,[LEFT]*6,[RIGHT]*6,[UP]*8,[LEFT]*10):
            cc=c.clone()
            for a in seq: cc.step(a)
            full|=markers(arr(cc.frame()))
    print("TOTAL markers (%d):"%len(full))
    from collections import defaultdict
    byc=defaultdict(list)
    for r,cc_,col in sorted(full): byc[col].append((r,cc_))
    for col in sorted(byc):
        print(f"  color {col} ({len(byc[col])}):", byc[col])
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
