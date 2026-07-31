from collections import deque
import model as M

start = frozenset([(0,5),(7,2)])
goal  = frozenset([(3,6),(6,6)])
print("start", sorted(start), "goal", sorted(goal))
q = deque([(start, [])]); seen = {start}
sol = None
while q:
    st, path = q.popleft()
    if st == goal:
        sol = path; break
    for b in M.ORDER:
        nxt = frozenset(M.move_cell(c, b) for c in st)
        if nxt in seen: continue
        seen.add(nxt); q.append((nxt, path+[b]))
print("reachable pair-states", len(seen))
print("solution len", len(sol) if sol else None, sol)
