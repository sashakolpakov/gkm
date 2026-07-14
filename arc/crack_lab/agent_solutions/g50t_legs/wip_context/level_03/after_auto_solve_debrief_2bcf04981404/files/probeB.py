import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def show(f):
    for row in f: print(''.join(str(int(v)) for v in row[:12]))

def prog(env):
    c=env.clone()
    # do 6 USE with a move between to allow legend cycling; but USE at start=0.
    # Move down first then USE repeatedly (each USE moves up, so alternate down,use)
    seq=[]
    for i in range(8):
        f0=np.asarray(c.frame())
        c.step(2) # down
        c.step(5) # use
        f=np.asarray(c.frame())
        print("iter",i,"lvl",c.levels_completed,"legend row1:",''.join(str(int(v)) for v in f[1,:9]))
    print("=== legend region after ===")
    show(np.asarray(c.frame()))
    raise SystemExit
A.run_program('g50t', prog)
