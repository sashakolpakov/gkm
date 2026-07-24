import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter, defaultdict

with open("checkpoint.json") as f:
    PATH=json.load(f)["final_path"]

def markers(f):
    rings=[b for b in connected_components(f,colors=(4,),min_area=8)
           if b.size==(3,3) and b.area==8]
    return sorted((b.bbox[0]+1,b.bbox[1]+1,int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings)

def sel(f):
    bp=list(zip(*((f==0).nonzero())))
    return len(bp)

def solve(env):
    prev=env.levels_completed
    for i,a in enumerate(PATH):
        before=arr(env.frame()).copy()
        env.step(a)
        now=env.levels_completed
        if now!=prev:
            print(f"--- LEVEL {prev}->{now} at step {i} (action {a}) ---")
            print("markers just BEFORE completing:", markers(before))
            print("  nblack before:", sel(before))
            af=arr(env.frame())
            print("markers just AFTER (next level start):", markers(af))
            prev=now
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
