import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr, connected_components, frame_delta

def dump(f, tag):
    print("=== "+tag)
    for r in range(64):
        print("%02d %s" % (r, "".join("%x"%v for v in f[r])))

def probe(env):
    f0 = arr(env.frame()).copy()
    dump(f0, "L1 initial")
    for i in range(6):
        cl = env.clone()
        for _ in range(i): cl.step(6,5,32)
        print("clicks", i, "lvl", cl.levels_completed, "delta", frame_delta(f0, cl.frame())["count"])
    # what changes on the 1st click
    cl = env.clone(); cl.step(6,5,32)
    dump(arr(cl.frame()), "after 1 click")
A.run_program("lp85", probe)
