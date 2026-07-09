import l9env, legs
import numpy as np, time, heapq
start=l9env.get_l9()
base=start.levels_completed
def feats(e):
    f=np.asarray(e.frame())
    c2=int((f==2).sum())
    av,boxes,walls=legs._grid_scan(e)
    loose=len(boxes)
    return c2,loose
def h(e):
    c2,loose=feats(e)
    return c2 - 3*loose
W=50; DEPTH=64
beam=[(h(start),0,start,[])]
best=None; t0=time.time()
seen=set()
for d in range(DEPTH):
    nxt=[]
    for _,_,e,path in beam:
        for a in (1,2,3,4,5):
            c=e.clone(); c.step(a)
            if c.levels_completed>base:
                print("WIN path len",len(path)+1,path+[a]); best=path+[a]; break
            if c.terminal(): continue
            f=np.asarray(c.frame())
            k=f[::4,::4].tobytes()
            if k in seen: continue
            seen.add(k)
            nxt.append((h(c),np.random.rand(),c,path+[a]))
        if best: break
    if best: break
    nxt.sort(key=lambda x:-x[0])
    beam=nxt[:W]
    if d%8==0:
        print(f"d{d} bestH{beam[0][0] if beam else None} nbeam{len(beam)} t{time.time()-t0:.0f}")
print("done best",best is not None,"t",time.time()-t0)
