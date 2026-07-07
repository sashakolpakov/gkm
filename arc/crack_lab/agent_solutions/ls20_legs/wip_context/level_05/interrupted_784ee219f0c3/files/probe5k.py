import sys, json, time
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
from legs import detect_noise_mask, _masked_bytes
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
mask=detect_noise_mask(env)
start=env.clone()
seen={_masked_bytes(start.frame(),mask)}
q=deque([(start,[])]); base=env.levels_completed
t0=time.time(); found=None; CAP=20000; nstep=0
while q:
    node,path=q.popleft()
    for a in (1,2,3,4):
        c=node.clone()
        try: c.step(a); nstep+=1
        except Exception: continue
        if c.levels_completed>base:
            found=path+[a]; break
        try: k=_masked_bytes(c.frame(),mask)
        except Exception: continue
        if k not in seen:
            seen.add(k); q.append((c,path+[a]))
    if found: break
    if len(seen)>=CAP: break
    if time.time()-t0>520: print("TIMEOUT"); break
print(f"seen={len(seen)} steps={nstep} time={time.time()-t0:.0f}s found_len={len(found) if found else None}")
if found: print("PATH",found)
