import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None

def show(f):
    for row in f: print(''.join(str(int(v)) for v in row))

def prog(env):
    # navigate to (14,14): from (8,14) go DOWN
    c=env.clone(); c.step(2)  # down -> (14,14)
    print("pos",av(c))
    base=np.asarray(c.frame())
    c2=c.clone(); c2.step(5)
    f=np.asarray(c2.frame())
    d=P.frame_delta(base,f)
    print("delta samples:")
    for s in d['samples']: print(s)
    print("=== frame after USE ===")
    show(f)
    raise SystemExit
A.run_program('g50t', prog)
