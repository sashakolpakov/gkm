import json, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import (arr, color_counts, connected_components,
                        object_candidates, action_deltas)

with open("checkpoint.json") as f:
    CK = json.load(f)
PATH = CK["final_path"]

STORE = {}

def solve(env):
    # replay to start of level 5
    for a in PATH:
        env.step(a)
    STORE['lvl'] = env.levels_completed
    STORE['actions'] = list(env.actions)
    f = arr(env.frame())
    STORE['shape'] = f.shape
    STORE['colors'] = color_counts(f)
    # print a compact ascii of the frame
    STORE['frame'] = f.copy()
    raise SystemExit("__PROBE_DONE__")

try:
    A.run_program('re86', solve)
except SystemExit:
    pass

print("level_completed_before_L5:", STORE['lvl'])
print("actions:", STORE['actions'])
print("shape:", STORE['shape'])
print("colors:", STORE['colors'])
f = STORE['frame']
# ascii render with color->char
chars = {}
palette = " .oO#@%*+=xX<>?"
for i, c in enumerate(sorted(STORE['colors'])):
    chars[c] = palette[i] if i < len(palette) else str(c)
print("legend:", {chars[c]: c for c in sorted(STORE['colors'])})
for r in range(f.shape[0]):
    print("".join(chars[int(v)] for v in f[r]))
