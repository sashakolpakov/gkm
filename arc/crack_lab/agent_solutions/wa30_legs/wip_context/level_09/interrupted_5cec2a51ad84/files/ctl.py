import numpy as np
from collections import deque
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}
FREE={0,1}
def blk(f,R,C): return f[R*4:R*4+4,C*4:C*4+4]
def cellfree(f,R,C):
    if R<0 or C<0 or R>=16 or C>=16: return False
    u=set(int(v) for v in np.unique(blk(f,R,C)))
    return u<= {0,1}
def av_cell(f):
    ys,xs=np.where((f==14)|(f==0))
    if not len(ys): return None
    return (int(ys.min())//4,int(xs.min())//4)
def grabbed_cell(f):
    for R in range(16):
        for C in range(16):
            u=set(int(v) for v in np.unique(blk(f,R,C)))
            if 0 in u and 9 in u and 2 not in u and 14 not in u:
                return (R,C)
    return None
def boxes(f):
    out=set()
    for R in range(16):
        for C in range(16):
            u=set(int(v) for v in np.unique(blk(f,R,C)))
            if 9 in u and (4 in u or 3 in u) and 2 not in u:
                out.add((R,C))
    return out
def sockets_interior(f):
    # cells with 9 and 2 (socket border/interior mix) -> approximate socket cells
    out=set()
    for R in range(16):
        for C in range(16):
            u=set(int(v) for v in np.unique(blk(f,R,C)))
            if 9 in u and 2 in u:
                out.add((R,C))
    return out
def interior9(f):
    return int((f[9:14,45:58]==9).sum())+int((f[49:58,49:58]==9).sum())
def bfs(f,start,goals,extra_free=set()):
    q=deque([start]); prev={start:None}
    goals=set(goals)
    while q:
        cur=q.popleft()
        if cur in goals:
            path=[];n=cur
            while prev[n]: p,a=prev[n];path.append(a);n=p
            return path[::-1],cur
        for a,(dr,dc) in DIRS.items():
            nb=(cur[0]+dr,cur[1]+dc)
            if nb in prev: continue
            if nb!=cur and (cellfree(f,nb[0],nb[1]) or nb in extra_free):
                prev[nb]=(cur,a); q.append(nb)
    return None,None
