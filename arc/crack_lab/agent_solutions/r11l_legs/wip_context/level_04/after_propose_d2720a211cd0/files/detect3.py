import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from legs import _arr, click, active_pos, box_center
import numpy as np
from collections import deque
def load():
    env=A.Arena('r11l'); ck=json.load(open('checkpoint.json'))
    for a in ck['final_path']: env.step(a)
    return env

def flood_cols(f,r,c):
    seen=set();q=deque([(r,c)]);cols=set()
    while q:
        y,x=q.popleft()
        if (y,x) in seen:continue
        seen.add((y,x))
        if int(f[y,x])==5:continue
        cols.add(int(f[y,x]))
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=y+dy,x+dx
            if 0<=ny<64 and 0<=nx<64 and (ny,nx) not in seen and int(f[ny,nx])!=5:
                q.append((ny,nx))
    return cols,len(seen)

def find_boxes(f):
    out=[]
    for r,c in np.argwhere(f==6):
        cols,n=flood_cols(f,int(r),int(c))
        if n>=8:
            fill=sorted(cols-{0,1,3,5,6})
            out.append((int(r),int(c),fill))
    return out

def endpoint_markers(f):
    # cells that are single pixel wrapped by 0 or 3 diamond (center of marker)
    out=[]
    for r,c in np.argwhere(~np.isin(f,[0,1,3,5,6])):
        r,c=int(r),int(c)
        nb=set()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<64 and 0<=cc<64: nb.add(int(f[rr,cc]))
        if 0 in nb or 3 in nb:
            out.append((r,c))
    return out

env=load(); f=_arr(env)
boxes=find_boxes(f)
print("boxes:",boxes)
eps=endpoint_markers(f)
print("endpoint markers:",eps)
# probe each endpoint to see which box moves
bc0={tuple(b[:2]):b[2] for b in boxes}
box_pos0=[(b[0],b[1]) for b in boxes]
def box_centers(f):
    return set((int(r),int(c)) for r,c in np.argwhere(f==6) if flood_cols(f,int(r),int(c))[1]>=8)
for ep in eps:
    c=env.clone()
    click(c,ep[0],ep[1])  # select (harmless if already active)
    ap=active_pos(c)
    before=box_centers(_arr(c))
    click(c,ap[0],ap[1]+5)
    after=box_centers(_arr(c))
    moved=before-after
    print("ep",ep,"selected active",ap,"box moved from",sorted(moved))
