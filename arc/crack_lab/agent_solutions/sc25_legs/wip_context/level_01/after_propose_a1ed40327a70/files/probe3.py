import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    base = np.asarray(env.frame()).copy()
    # try clicking many coordinates with action 6
    coords = [(13,19),(19,13),(15,40),(40,15),(15,15),(52,15),(15,52),(55,30),(30,55),(50,30),(30,50)]
    for (x,y) in coords:
        clone = env.clone()
        clone.step(6,x,y)
        d = P.frame_delta(base, clone.frame())
        print(f"6 x={x} y={y} count={d['count']} bbox={d['bbox']}")

A.run_program('sc25', main)
