import l9env, legs
import numpy as np
from collections import deque
env=l9env.get_l9()
av,head,boxes,cour,walls=legs._cells(env)
print("avatar cell",av,"head",head,"courier",cour)
print("boxes",sorted(boxes))
print("walls",sorted(walls))
# reachability from av avoiding walls+boxes
blocked=set(walls)|set(boxes)
G=16
q=deque([av]); seen={av}
while q:
    r,c=q.popleft()
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        n=(r+dr,c+dc)
        if 0<=n[0]<G and 0<=n[1]<G and n not in seen and n not in blocked:
            seen.add(n); q.append(n)
print("reachable cells:",len(seen))
grid=[['#' if (r,c) in blocked else ('.' if (r,c) in seen else ' ') for c in range(G)] for r in range(G)]
grid[av[0]][av[1]]='A'
for row in grid: print(''.join(row))
