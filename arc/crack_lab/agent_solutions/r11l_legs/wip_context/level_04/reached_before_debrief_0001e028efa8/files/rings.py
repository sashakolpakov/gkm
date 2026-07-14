import numpy as np
f = np.load('l4_frame.npy')
from legs import _isolated_cells, _neigh4_colors, _neigh8_colors, _centroid
# For each color, group isolated cells into clusters and report clusters that look like a hollow diamond (touch only bg 5)
from scipy import ndimage
for color in [7,8,9,11,12,13,14,15,3,1,6]:
    mask=(f==color)
    if mask.sum()==0: continue
    lbl,n=ndimage.label(mask, structure=np.ones((3,3)))
    for i in range(1,n+1):
        pts=np.argwhere(lbl==i)
        r0,c0=pts.min(0); r1,c1=pts.max(0)
        # check if hollow: bounding box interior mostly not this color
        h=r1-r0+1; w=c1-c0+1
        cen=_centroid([(int(r),int(c)) for r,c in pts])
        # neighbor colors overall
        nb=set()
        for r,c in pts:
            nb|=_neigh4_colors(f,r,c)
        nb-={color}
        print(f"color {color} comp size {len(pts)} bbox {(int(r0),int(c0),int(r1),int(c1))} center {cen} nbcolors {sorted(nb)}")
