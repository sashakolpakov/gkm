import sys, json, time
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('ls20'); env.reset()
for a in json.load(open("checkpoint.json"))["final_path"]: env.step(a)
base=env.levels_completed
def avpos(f):
    ys,xs=np.where(f==9)
    m=(ys>=30)  # avatar 9 is below row30; fixed 9 at row26 excluded
    if m.sum()==0: return None
    return (int(ys[m].min()),int(xs[m].min()))
# DFS deduping by avatar position, explore maze, watch for reward
start=env.clone()
seen={avpos(start.frame())}
stack=[(start,[])]; t0=time.time(); nstep=0; found=None; positions=set(seen)
maxreach=None
while stack:
    node,path=stack.pop()
    for a in (1,2,3,4):
        c=node.clone()
        try: c.step(a); nstep+=1
        except Exception: continue
        if c.levels_completed>base: found=path+[a]; break
        p=avpos(c.frame())
        if p is None: continue
        if p not in seen:
            seen.add(p); positions.add(p); stack.append((c,path+[a]))
    if found: break
    if nstep>8000: break
    if time.time()-t0>200: break
print(f"distinct avatar positions={len(positions)} steps={nstep} time={time.time()-t0:.0f}s found={found}")
ys=[p[0] for p in positions]; xs=[p[1] for p in positions]
print(f"pos row range {min(ys)}-{max(ys)} col range {min(xs)}-{max(xs)}")
