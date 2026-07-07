import numpy as np
from collections import deque
from lib6 import st

f0 = np.load('/tmp/l6_frame.npy')
walk = np.isin(f0, [3, 5])
for (r0,r1,c0,c1) in [(50,55,24,29),(31,34,25,28),(6,9,10,13),(6,9,40,43),(46,49,10,13),(11,14,15,18),(41,44,35,38)]:
    walk[r0:r1, c0:c1] = True
rows = list(range(5, 51, 5)); cols = list(range(9, 55, 5))
OK = {(r,c): bool(walk[r:r+5, c:c+5].all()) for r in rows for c in cols}
SPRINGS = {(5,49): (25,49), (20,49): (20,39)}
def nbrs(t):
    r, c = t
    for a,(dr,dc) in {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}.items():
        u = (r+dr, c+dc)
        if not OK.get(u):
            continue
        yield a, SPRINGS.get(u, u)
def path_to(a, b):
    q = deque([(a, [])]); seen = {a}
    while q:
        u, p = q.popleft()
        if u == b: return p
        for act, v in nbrs(u):
            if v not in seen:
                seen.add(v); q.append((v, p + [act]))
    return None
def goto(c, target):
    cur = st(c.frame())['av']
    p = path_to(cur, target)
    assert p is not None, (cur, target)
    for a in p:
        c.step(a)
    assert st(c.frame())['av'] == target, (st(c.frame())['av'], target)
    return len(p)
