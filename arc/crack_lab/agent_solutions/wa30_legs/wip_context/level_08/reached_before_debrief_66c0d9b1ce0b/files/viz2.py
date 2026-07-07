import numpy as np
f = np.load('/tmp/l8_frame.npy')
# per-cell signature (set of colors), only nontrivial
for R in range(16):
    for C in range(16):
        blk = f[R*4:R*4+4, C*4:C*4+4]
        u = sorted(set(int(v) for v in np.unique(blk)))
        if u != [1]:  # not pure background
            print(f"R{R:2d}C{C:2d} {u}")
