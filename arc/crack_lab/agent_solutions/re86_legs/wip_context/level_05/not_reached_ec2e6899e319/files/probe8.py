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
    return sorted((b.bbox[0]+1,b.bbox[1]+1,int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings)

def sel(f):
    bp=list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else None

def move(env,frm,to,step=3):
    dr=to[0]-frm[0]; dc=to[1]-frm[1]
    assert dr%step==0 and dc%step==0,(frm,to)
    for _ in range(abs(dr)//step): env.step(DOWN if dr>0 else UP)
    for _ in range(abs(dc)//step): env.step(RIGHT if dc>0 else LEFT)

def place(env, station, target):
    ctr=sel(arr(env.frame()))
    move(env,ctr,station)
    ctr=sel(arr(env.frame()))
    move(env,ctr,target)

def solve(env):
    for a in PATH: env.step(a)
    c=env.clone()
    # X selected at (42,24) -> station8 (54,57) -> target (30,48)
    place(c,(54,57),(30,48))
    print("after X:", "lvl",c.levels_completed, "markers", markers(arr(c.frame())))
    c.step(USE)  # -> diamond (18,30)
    print("sel now", sel(arr(c.frame())))
    place(c,(54,6),(6,30))
    print("after diamond:", "lvl",c.levels_completed, "markers", markers(arr(c.frame())))
    c.step(USE)  # -> plus (33,54)
    print("sel now", sel(arr(c.frame())))
    place(c,(54,6),(51,33))
    print("after plus:", "lvl",c.levels_completed, "markers", markers(arr(c.frame())))
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
