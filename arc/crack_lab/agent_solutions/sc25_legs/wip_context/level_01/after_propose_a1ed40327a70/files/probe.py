import sys, os
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    print("actions:", env.actions)
    print("levels_completed:", env.levels_completed)
    f = np.asarray(env.frame())
    print("frame shape:", f.shape)
    print("colors:", P.color_counts(f))
    # print compact frame
    for row in f:
        print(''.join('.' if v==0 else format(v,'x') for v in row))

A.run_program('sc25', main)
