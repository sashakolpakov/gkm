import numpy as np
f = np.load('/tmp/l8_frame.npy')
def show(R,C,label):
    blk=f[R*4:R*4+4,C*4:C*4+4]
    print(label, f"R{R}C{C}")
    for r in blk: print("  "," ".join(f"{int(v):2d}" for v in r))
show(2,1,"box[4,9]")
show(2,11,"socket-right[2,9]")
show(4,1,"pure2 topleft")
show(12,3,"pure2 botleft")
show(2,3,"pure2 top")
show(13,13,"pure2 in botright")
