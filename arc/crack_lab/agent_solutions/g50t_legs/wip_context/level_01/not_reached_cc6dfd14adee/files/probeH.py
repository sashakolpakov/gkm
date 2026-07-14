import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    # step real env a few moves
    for a in [4,4,2,2]:
        env.step(a)
    print("REAL after 4 moves:",P.color_counts(env.frame()), "lvl",env.levels_completed)
    # clone and compare
    c=env.clone()
    print("CLONE counts:",P.color_counts(c.frame()))
    # continue real vs clone with same next moves
    for a in [2,2,4,4]:
        env.step(a); c.step(a)
    print("REAL:",P.color_counts(env.frame()))
    print("CLONE:",P.color_counts(c.frame()))
    print("frames equal?", np.array_equal(np.asarray(env.frame()),np.asarray(c.frame())))
    raise SystemExit
A.run_program('g50t', prog)
