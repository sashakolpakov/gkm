import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr, connected_components

def probe(env):
    for _ in range(5):
        env.step(6, 5, 32)
    f = arr(env.frame())
    blobs = connected_components(f, min_area=1)
    for b in blobs:
        if b.color in (3,4): continue
        print("c=%2d bbox=%s area=%d" % (b.color, b.bbox, b.area))
A.run_program("lp85", probe)
