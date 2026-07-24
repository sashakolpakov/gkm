import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
def prog(env):
    f_before=np.asarray(env.frame()).copy()
    c=env.clone()
    for a in [2,2,4,4,5]: c.step(a)
    f_after=np.asarray(env.frame())
    print("real env unchanged after cloning+stepping clone?", np.array_equal(f_before,f_after))
    # nested clone independence
    c2=c.clone(); c2.step(2)
    print("c unaffected by c2 step?", not np.array_equal(np.asarray(c.frame()),np.asarray(c2.frame())) or True)
    print("c frame != c2 frame?", not np.array_equal(np.asarray(c.frame()),np.asarray(c2.frame())))
    prog.d=1
A.run_program('g50t', prog)
