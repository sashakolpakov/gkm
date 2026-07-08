import l9env
import numpy as np, time, json
start=l9env.get_l9(); base=start.levels_completed
def metrics(f):
    filled=0; ci2=0
    for r in (3,4,5):
        for c in (5,6,7):
            blk=f[r*4:r*4+4,c*4:c*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 9 in u and (4 in u or 3 in u) and 2 not in u: filled+=1
    ci2=int((f[13:23,21:31]==2).sum())
    return filled,ci2
W=500; DEPTH=66
f0=np.asarray(start.frame())
m0=metrics(f0)
beam=[(-m0[0],m0[1],0.0,start,[])]  # sort by (-filled, ci2)
best=None; t0=time.time(); seen=set(); gbest=(0,100)
for d in range(DEPTH):
    nxt=[]
    for _,_,_,e,path in beam:
        for a in (1,2,3,4,5):
            c=e.clone(); c.step(a)
            if c.levels_completed>base:
                best=path+[a]; print("WIN len",len(best)); break
            if c.terminal(): continue
            f=np.asarray(c.frame())
            fl,ci=metrics(f)
            k=(fl,ci,f[::6,::6].tobytes())
            if k in seen: continue
            seen.add(k)
            if (fl,-ci)>(gbest[0],-gbest[1]): gbest=(fl,ci)
            nxt.append((-fl,ci,np.random.rand(),c,path+[a]))
        if best: break
    if best: break
    nxt.sort(key=lambda x:(x[0],x[1],x[2]))
    beam=nxt[:W]
    if d%5==0:
        print(f"d{d} top(fl,ci2)=({-beam[0][0]},{beam[0][1]}) gbest{gbest} nb{len(beam)} t{time.time()-t0:.0f}")
print("done win",best is not None,"gbest",gbest,"t",time.time()-t0)
if best: json.dump(best,open("l9_win.json","w")); print("saved")
