import importlib.util, sys, os, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

# get env at start, advance past level 1 with checkpoint path
ck = json.load(open("checkpoint.json"))
path1 = ck["final_path"]

def player(env):
    for a in path1:
        env.step(a)
    print("after L1 path: levels=", env.levels_completed, "terminal=", env.terminal())
    f = np.asarray(env.frame())
    print("shape", f.shape, "colors", np.unique(f, return_counts=True))
    # print compact
    np.save("/tmp/l2frame.npy", f)
    raise SystemExit  # stop here

try:
    A.run_program('g50t', player)
except SystemExit:
    pass
