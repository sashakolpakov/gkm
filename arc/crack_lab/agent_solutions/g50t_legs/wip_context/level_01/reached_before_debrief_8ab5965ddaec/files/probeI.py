import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque

def prog(env):
    f=np.asarray(env.frame())
    R,C=f.shape
    def free(r,c):
        if r<0 or c<0 or r+5>R or c+5>C: return False
        blk=f[r:r+5,c:c+5]
        return not (blk==0).any()
    # grid points on 6-lattice from (8,14)
    def gridpts():
        pts=[]
        r=8
        while r+5<=R:
            c=14
            while c+5<=C:
                pts.append((r,c)); c+=6
            r+=6
        return pts
    start=(8,14)
    # BFS with step 6 among free
    def neigh(p):
        r,c=p
        for dr,dc in ((6,0),(-6,0),(0,6),(0,-6)):
            n=(r+dr,c+dc)
            if free(*n): yield n
    seen={start}; q=deque([start])
    while q:
        p=q.popleft()
        for n in neigh(p):
            if n not in seen: seen.add(n); q.append(n)
    print("reachable(free-model) from start:",len(seen))
    for p in sorted(seen): print(p)
    # is goal-ish reachable? goal box rows49-55 col43-49 -> grid (50,44)
    print("free(50,44)?",free(50,44),"free(44,44)?",free(44,44))
    print("(50,44) in reach", (50,44) in seen, "(44,44) in reach",(44,44) in seen)
    raise SystemExit
A.run_program('g50t', prog)
