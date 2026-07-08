from probe9 import *
def interior2(f): return int((f[13:23,21:31]==2).sum())
def avcell(f):
    ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
def beam(start, width=700, depth=72):
    base=start.levels_completed
    frontier=[(interior2(np.asarray(start.frame())), start.clone(), [])]
    gbest=(999,[])
    for d in range(depth):
        cand=[]; seen=set()
        for h,env,path in frontier:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base:
                    return ('WIN', path+[a])
                f=np.asarray(c.frame()); s=interior2(f)
                k=(avcell(f), f[12:24,20:32].tobytes())
                if k in seen: continue
                seen.add(k)
                cand.append((s,c,path+[a]))
                if s<gbest[0]: gbest=(s,path+[a])
        if not cand: break
        cand.sort(key=lambda x:x[0])
        frontier=cand[:width]
        if d%8==0: print('d',d,'bestI2',frontier[0][0],'gbest',gbest[0],flush=True)
    return ('BEST', gbest)
if __name__=='__main__':
    env=fresh(); r=beam(env)
    print('RES',r[0])
    if r[0]=='WIN':
        print('WINLEN',len(r[1])); print('WINPATH',r[1])
    else:
        print('bestI2',r[1][0],'len',len(r[1][1]))
        # save best path
        import json; json.dump(r[1][1], open('best_l9_path.json','w'))
