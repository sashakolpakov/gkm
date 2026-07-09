import l9env
import numpy as np, time, json, heapq
start=l9env.get_l9(); base=start.levels_completed
def metrics(f):
    filled=0
    for r in (3,4,5):
        for c in (5,6,7):
            blk=f[r*4:r*4+4,c*4:c*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 9 in u and (4 in u or 3 in u) and 2 not in u: filled+=1
    ci2=int((f[13:23,21:31]==2).sum())
    return filled,ci2
def score(f):
    fl,ci=metrics(f)
    return fl*200 - ci   # higher better
f0=np.asarray(start.frame())
# priority queue: (-score, tie, env, path)
h=[]; cnt=0
heapq.heappush(h,(-score(f0),cnt,start,[])); cnt+=1
seen=set(); best=None; t0=time.time(); gbest=-999
IT=0
while h and time.time()-t0<180:
    negs,_,e,path=heapq.heappop(h)
    IT+=1
    for a in (1,2,3,4,5):
        c=e.clone(); c.step(a)
        if c.levels_completed>base:
            best=path+[a]; print("WIN",len(best)); break
        if c.terminal(): continue
        f=np.asarray(c.frame())
        fl,ci=metrics(f)
        k=(fl,ci,f[::6,::6].tobytes())
        if k in seen: continue
        seen.add(k)
        s=fl*200-ci
        if s>gbest: gbest=s; 
        heapq.heappush(h,(-s,cnt,c,path+[a])); cnt+=1
    if best: break
    if IT%500==0:
        print(f"it{IT} gbest{gbest} qsize{len(h)} seen{len(seen)} t{time.time()-t0:.0f}")
print("done win",best is not None,"gbest",gbest,"it",IT,"t",time.time()-t0)
if best: json.dump(best,open("l9_win.json","w")); print("SAVED")
