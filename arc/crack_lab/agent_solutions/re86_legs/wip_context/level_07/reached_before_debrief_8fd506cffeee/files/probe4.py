import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

def markers(f):
    rings = [b for b in connected_components(f, colors=(4,), min_area=8)
             if b.size==(3,3) and b.area==8]
    return sorted(((b.bbox[0]+1, b.bbox[1]+1, int(f[b.bbox[0]+1,b.bbox[1]+1])) for b in rings))

def sel_center(f):
    bp = list(zip(*((f==0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp)==1 else ('N',len(bp))

def shape_color(env):
    # move clone one step, see which non-bg color pixels vanish to bg
    bf = arr(env.frame()).copy()
    bg = Counter(int(v) for v in bf.flat).most_common(1)[0][0]
    for act in (1,2,3,4):
        c=env.clone(); c.step(act); af=arr(c.frame())
        votes=[int(bf[r,cc]) for r,cc in zip(*((bf!=af).nonzero()))
               if int(af[r,cc])==bg and int(bf[r,cc]) not in (0,bg,2,4)]
        if votes:
            return Counter(votes).most_common(1)[0][0]
    return None

def drive(env, acts):
    for a in acts:
        env.step(a)

def solve(env):
    drive(env, PATH)
    print("start markers:", markers(arr(env.frame())))
    print("sel center:", sel_center(arr(env.frame())), "color:", shape_color(env))

    # Select X is at (42,24). Try to move its center DOWN into station9 interior.
    # station9 bbox (52,3,57,8) interior rows53-56. target lattice (54,6).
    c = env.clone()
    print("--- move X toward station9 (down 4, left 6) watching color ---")
    for i in range(4):
        c.step(2)
    for i in range(6):
        c.step(3)
    print("after move: sel center", sel_center(arr(c.frame())), "color", shape_color(c),
          "lvl", c.levels_completed)
    # did it recolor?
    raise SystemExit

try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
