import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components, frame_delta
import legs

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

def solve(env):
    for a in PATH:
        env.step(a)
    print("lvl0 =", env.levels_completed)

    # 1) Try the L4 leg on a clone, report outcome
    c = env.clone()
    try:
        legs.repaint_selected_shapes_to_cover_colored_ring_markers(c)
        print("L4 leg ran. lvl now:", c.levels_completed)
    except Exception as e:
        print("L4 leg raised:", repr(e))

    # 2) Experiment: move selected shape (X at ~42,24) and watch a marker.
    #    Move it and see what changes. Track reward each step.
    base = arr(env.frame())
    print("levels_completed attr:", env.levels_completed)
    # move X up toward marker at (45,33)? center (42,24). Let's just step each dir once and see delta.
    for act in (1,2,3,4,5):
        cc = env.clone()
        before = arr(cc.frame()).copy()
        cc.step(act)
        after = arr(cc.frame())
        d = frame_delta(before, after)
        print(f"act {act}: dcount={d['count']} bbox={d['bbox']} lvl={cc.levels_completed} sample={d['samples'][:4]}")
    raise SystemExit

try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
