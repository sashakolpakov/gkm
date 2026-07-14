import numpy as np
f = np.load('l4_frame.npy')
# find all 6 pixels and their neighborhoods
print("6 pixels and 3x3 nbhd colors:")
for r,c in np.argwhere(f==6):
    blk = f[max(0,r-2):r+3, max(0,c-2):c+3]
    print((int(r),int(c)), sorted(set(int(v) for v in blk.flatten())))
