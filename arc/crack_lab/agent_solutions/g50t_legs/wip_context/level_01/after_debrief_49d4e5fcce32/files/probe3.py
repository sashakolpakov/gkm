import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    for b in P.connected_components(env.frame()):
        print(b.color, b.bbox, b.area, b.centroid)
    raise SystemExit
A.run_program('g50t', prog)
