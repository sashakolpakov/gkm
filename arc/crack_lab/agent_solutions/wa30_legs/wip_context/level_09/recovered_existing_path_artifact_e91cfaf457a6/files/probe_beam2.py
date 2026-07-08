import l9env
import numpy as np, time
start=l9env.get_l9(); base=start.levels_completed
def sockets2(f):
    c=int((f[13:23,21:31]==2).sum())      # center interior
    rt=int((f[9:11,53:59]==2).sum())       # right top
    rb=int((f[25:27,53:59]==2).sum())      # right bottom
    return c+rt+rb, c, rt, rb
W=250; DEPTH=68
f0=np.asarray(start.frame())
beam=[(sockets2(f0)[0],0.0,start,[])]
best=None; t0=time.time(); seen=set(); gmin=9999; gstate=None
for d in range(DEPTH):
    nxt=[]
    for _,_,e,path in beam:
        for a in (1,2,3,4,5):
            c=e.clone(); c.step(a)
            if c.levels_completed>base:
                best=path+[a]; print("WIN len",len(best)); break
            if c.terminal(): continue
            f=np.asarray(c.frame())
            tot,cc,rt,rb=sockets2(f)
            k=(cc,rt,rb,f[32:36,:].tobytes(),f[::8,::8].tobytes())
            if k in seen: continue
            seen.add(k)
            if tot<gmin: gmin=tot; gstate=(cc,rt,rb)
            nxt.append((tot,np.random.rand(),c,path+[a]))
        if best: break
    if best: break
    nxt.sort(key=lambda x:(x[0],x[1]))
    beam=nxt[:W]
    if d%6==0:
        print(f"d{d} minTot{beam[0][0] if beam else None} gmin{gmin}{gstate} nb{len(beam)} t{time.time()-t0:.0f}")
print("done win",best is not None,"gmin",gmin,gstate,"t",time.time()-t0)
if best:
    import json; json.dump(best,open("l9_win.json","w"))
