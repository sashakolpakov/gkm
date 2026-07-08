import numpy as np
f = np.load('/tmp/l8_frame.npy')
def region(r0,r1,c0,c1,label):
    print("===",label)
    sub=f[r0:r1,c0:c1]
    for r in sub: print(" ".join(f"{int(v):2d}" for v in r))
# top-left: R2-4 C1-3 => rows8-19 cols4-15
region(8,20,4,16,"top-left")
print()
# bottom-left: R12-14 C3-5 => rows48-59 cols12-23
region(48,60,12,24,"bot-left")
