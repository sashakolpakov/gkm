import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components, color_counts
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

GLY = {0: '.', 4: 'R', 5: 'b', 9: 'g', 11: 'B', 15: 'W'}

def show(f):
    for r in range(f.shape[0]):
        print(''.join(GLY.get(int(v), str(int(v) % 10)) for v in f[r]))

def solve(env):
    for a in PATH:
        env.step(a)
    print("levels_completed:", env.levels_completed, "terminal:", env.terminal())
    f = arr(env.frame())
    print("colors:", color_counts(f))
    print("actions:", getattr(env, 'actions', None))
    show(f)
    print("--- components (min_area=2) ---")
    for b in connected_components(f, min_area=2):
        if b.area > 400:
            print("  BG?", b.color, b.bbox, b.area)
        else:
            print(f"  c={b.color} bbox={b.bbox} sz={b.size} area={b.area}")
    raise SystemExit

try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
