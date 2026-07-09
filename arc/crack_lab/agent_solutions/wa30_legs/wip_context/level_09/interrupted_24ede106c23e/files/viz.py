import numpy as np
f = np.load('/tmp/l8_frame.npy')
# print as 16x16 cell signature
for R in range(16):
    row=""
    for C in range(16):
        blk = f[R*4:R*4+4, C*4:C*4+4]
        u = sorted(set(int(v) for v in np.unique(blk)))
        # dominant non-zero
        vals,cnts=np.unique(blk,return_counts=True)
        dom=int(vals[np.argmax(cnts)])
        row += f"{dom:2d}"
    print(f"{R:2d} {row}")
