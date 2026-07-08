from probe9 import *
def interior2(f): return int((f[13:23,21:31]==2).sum())
def avcell(f):
    ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
def score(env):
    f=np.asarray(env.frame())
    i2=interior2(f)
    return i2
def beam(start, width=250, depth=70):
    base=start.levels_completed
    frontier=[(score(start), start.clone(), [])]
    global_best=(score(start),[])
    for d in range(depth):
        cand=[]
        seen=set()
        for h,env,path in frontier:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base:
                    return ('WIN', path+[a])
                f=np.asarray(c.frame())
                s=interior2(f)
                k=(avcell(f), s, f[12:24,20:32].tobytes())
                if k in seen: continue
                seen.add(k)
                cand.append((s,c,path+[a]))
                if s<global_best[0]: global_best=(s,path+[a])
        if not cand: break
        cand.sort(key=lambda x:x[0])
        frontier=cand[:width]
        if d%8==0: print('d',d,'bestI2',frontier[0][0],'nfront',len(frontier))
    return ('BEST', global_best)
if __name__=='__main__':
    env=fresh()
    r=beam(env)
    print('RES',r[0], r[1][0] if r[0]=='BEST' else len(r[1]))
    if r[0]=='WIN': print('WINPATH',r[1])
