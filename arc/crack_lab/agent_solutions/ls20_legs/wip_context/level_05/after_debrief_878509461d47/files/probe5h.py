import sys, json, time
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from collections import deque
from legs import detect_noise_mask, _masked_bytes
CK="/private/tmp/claude-501/-Users-sasha-gkm/e3e00be1-d1a5-4095-a6ef-4d720f42d84e/scratchpad/gkm_legs_ws_ls20_sonnetfresh/checkpoint.json"
def setup():
    env=A.Arena('ls20'); env.reset()
    for a in json.load(open(CK))["final_path"]: env.step(a)
    return env
env=setup()
mask=detect_noise_mask(env)
print("mask cells", int(mask.sum()))
# BFS with instrumentation, cap states, report depth reached
start=env.clone()
seen={_masked_bytes(start.frame(),mask)}
q=deque([(start,0)])
base=env.levels_completed
maxdepth=0; t0=time.time(); found=None; explored=0
CAP=6000
while q:
    node,d=q.popleft(); explored+=1; maxdepth=max(maxdepth,d)
    for a in (1,2,3,4):
        c=node.clone(); c.step(a)
        if c.levels_completed>base:
            found=d+1; break
        k=_masked_bytes(c.frame(),mask)
        if k not in seen:
            seen.add(k); q.append((c,d+1))
    if found: break
    if len(seen)>=CAP: break
print(f"explored={explored} seen={len(seen)} maxdepth={maxdepth} found_at_depth={found} time={time.time()-t0:.1f}s")
