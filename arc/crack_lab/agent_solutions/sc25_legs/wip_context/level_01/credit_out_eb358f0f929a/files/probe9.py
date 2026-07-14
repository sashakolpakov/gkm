import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
import perception as P

def main(env):
    f = np.asarray(env.frame())
    # print region rows 15-25 cols 8-45 with col header
    print("   "+''.join(str(c%10) for c in range(8,45)))
    for r in range(15,25):
        print(f"{r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(8,45)))
    print("blobs (non-bg):")
    for b in P.connected_components(f):
        if b.color in (5,14): continue
        print(b.color, b.bbox, b.area)
A.run_program('sc25', main)
