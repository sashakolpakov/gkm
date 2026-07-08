from probe9 import *
# container interior pixels: rows13-22, cols21-30 (inside the 12x12 frame at rows12-23 cols20-31)
def interior2(env):
    f=np.asarray(env.frame())
    reg=f[13:23,21:31]
    return int((reg==2).sum())
def key(env):
    f=np.asarray(env.frame())
    # coarse container 3x3 fill signature + avatar cell + held
    sig=[]
    for R in range(3,6):
        for C in range(5,8):
            u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
            sig.append(1 if (4 in u and 2 not in u) else 0)
    ys,xs=np.where(f==14); av=(int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
    return (tuple(sig),av)
import heapq
def beam(start, width=40, depth=72):
    base=start.levels_completed
    beam=[(interior2(start), start.clone(), [])]
    seen={}
    best=(interior2(start),None)
    for d in range(depth):
        cand=[]
        for h,env,path in beam:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base:
                    return ('WIN',path+[a])
                hi=interior2(c)
                k=key(c)
                if k in seen and seen[k]<=hi: 
                    continue
                seen[k]=hi
                cand.append((hi,c,path+[a]))
                if hi<best[0]: best=(hi,path+[a])
        if not cand: break
        cand.sort(key=lambda x:x[0])
        beam=cand[:width]
        if d%10==0: print('depth',d,'best interior2',beam[0][0])
    return ('BEST',best)
if __name__=='__main__':
    env=fresh()
    print('start interior2',interior2(env))
    r=beam(env)
    print('RESULT',r[0], 'val', (r[1] if r[0]=='WIN' else r[1]))
