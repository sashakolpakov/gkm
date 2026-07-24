import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

def markers(f):
    rings=[b for b in connected_components(f,colors=(4,),min_area=8)
           if b.size==(3,3) and b.area==8]
    return sorted((b.bbox[0]+1,b.bbox[1]+1,int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings)

def sel_center(f):
    bp=list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else ('N',len(bp))

def offsets(env, color):
    bf=arr(env.frame())
    ctr=sel_center(bf)
    pts=[(int(r)-ctr[0],int(c)-ctr[1]) for r,c in zip(*((bf==color).nonzero()))]
    return ctr,pts

def solve(env):
    for a in PATH: env.step(a)
    c=env.clone()
    for _ in range(4): c.step(2)
    for _ in range(6): c.step(3)   # X recolored to 9 at (54,6)
    ctr,pts=offsets(c,9)
    drs=[p[0] for p in pts if (p[0],p[1]) not in [(0,0)]]
    print("X center",ctr,"npix",len(pts))
    print("row offset range",min(p[0] for p in pts),max(p[0] for p in pts),
          "col",min(p[1] for p in pts),max(p[1] for p in pts))
    # sample a few offsets
    print("sample offsets", sorted(pts)[:8], sorted(pts)[-8:])
    # check |dr|==|dc| fraction (is it an X?)
    xish=sum(abs(a)==abs(b) for a,b in pts)
    print("xish", xish, "of", len(pts))

    print("markers before:", markers(arr(c.frame())))
    # Now move X so a diagonal passes through a color-9 marker.
    # center (54,6). marker (60,33): dr=6 -> need dc=6 => col 39? not marker.
    # Try to cover (51,24): from center, want |51-r|==|24-c| and small.
    # Move center to (48,21): then (51,24): dr3 dc3 ok on diagonal. reach: up6 (54->48), right15(6->21)
    c2=c.clone()
    for _ in range(2): c2.step(1)   # up 6 -> row48
    for _ in range(5): c2.step(4)   # right15 -> col21
    print("after positioning: center", sel_center(arr(c2.frame())), "lvl", c2.levels_completed)
    print("markers now:", markers(arr(c2.frame())))
    # move away and recheck persistence
    c3=c2.clone()
    for _ in range(3): c3.step(1)
    print("after moving away: markers", markers(arr(c3.frame())), "lvl", c3.levels_completed)
    raise SystemExit

try: A.run_program('re86', solve)
except SystemExit: pass
except Exception: traceback.print_exc()
