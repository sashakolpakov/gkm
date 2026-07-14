import numpy as np
from scipy import ndimage
f=np.load('l4_frame.npy')

def box_fill_colors(f,r,c):
    # flood the connected non-bg region around the 6 (excluding rope color 1)
    from collections import deque
    seen=set(); q=deque([(r,c)]); cols=set()
    while q:
        y,x=q.popleft()
        if (y,x) in seen: continue
        seen.add((y,x))
        v=int(f[y,x])
        if v in (5,): continue
        cols.add(v)
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=y+dy,x+dx
            if 0<=ny<64 and 0<=nx<64 and (ny,nx) not in seen and int(f[ny,nx]) not in (5,):
                q.append((ny,nx))
    return cols,seen

# boxes: 6-pixels with a large multicolor fill
for r,c in np.argwhere(f==6):
    cols,seen=box_fill_colors(f,r,c)
    if len(seen)>=8:  # real box (chevron 6-cells are isolated size1)
        print("BOX at",(int(r),int(c)),"fill size",len(seen),"colors",sorted(cols-{6,1}))
