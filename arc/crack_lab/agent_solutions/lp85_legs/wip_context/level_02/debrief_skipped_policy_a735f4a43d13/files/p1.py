import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr, color_counts

def probe(env):
    print("actions", env.actions, "level", env.levels_completed)
    for _ in range(5):
        env.step(6, 5, 32)
    print("after L1: level", env.levels_completed, "terminal", env.terminal())
    f = arr(env.frame())
    print("shape", f.shape, "colors", color_counts(f))
    for r in range(f.shape[0]):
        print("%02d %s" % (r, "".join("%x" % v for v in f[r])))

A.run_program("lp85", probe)
