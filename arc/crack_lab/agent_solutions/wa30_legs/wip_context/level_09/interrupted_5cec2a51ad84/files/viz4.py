import numpy as np
f = np.load('/tmp/l8_frame.npy')
def region(r0,r1,c0,c1,label):
    print("===",label,f"rows{r0}-{r1} cols{c0}-{c1}")
    sub=f[r0:r1,c0:c1]
    for r in sub: print(" ".join(f"{int(v):2d}" for v in r))
# top-right socket block cells R2-3 C11-14 => pixels rows 8-15, cols 44-59
region(8,16,44,60,"top-right sockets")
print()
region(48,60,44,60,"bot-right sockets")  # R12-14 C12-14 => rows48-59 cols48-59
