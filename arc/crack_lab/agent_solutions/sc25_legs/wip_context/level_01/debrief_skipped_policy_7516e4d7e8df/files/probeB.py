import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def marker(f):
    # region rows19-22 cols39-42
    return [''.join(format(f[r,c],'x') for c in range(39,43)) for r in range(19,23)]
def timer(f):
    return int((f[:,62]==14).sum())

def main(env):
    c = env.clone()
    for i in range(8):
        f = np.asarray(c.frame())
        print(f"step {i}: marker={marker(f)} timer={timer(f)}")
        c.step(1)  # up
A.run_program('sc25', main)
