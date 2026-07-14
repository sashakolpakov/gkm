import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def c1(e): return P.color_counts(e.frame()).get(1)
def av(e):
    for b in P.connected_components(e.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None
def prog(env):
    # oscillate right/left to see if counter grows and if state repeats
    c=env.clone()
    seen=set()
    for i in range(20):
        a=4 if i%2==0 else 3
        c.step(a)
        f=np.asarray(c.frame())
        k=f.tobytes()
        print(i,"act",a,"av",av(c),"c1",c1(c),"c9",P.color_counts(c.frame()).get(9),"term",c.terminal(),"seen_before",k in seen)
        seen.add(k)
    raise SystemExit
A.run_program('g50t', prog)
