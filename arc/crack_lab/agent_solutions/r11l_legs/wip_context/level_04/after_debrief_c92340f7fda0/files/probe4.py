import numpy as np
f = np.load('l4_frame.npy')
for color in [0,1,3,6,7,8,9,11,12,13,14,15]:
    pts = np.argwhere(f==color)
    print(f"=== color {color}: {len(pts)} cells ===")
    if len(pts)>0:
        r0,c0 = pts.min(0); r1,c1=pts.max(0)
        print("bbox", (int(r0),int(c0),int(r1),int(c1)))
        # print small crop
    print(sorted((int(r),int(c)) for r,c in pts))
