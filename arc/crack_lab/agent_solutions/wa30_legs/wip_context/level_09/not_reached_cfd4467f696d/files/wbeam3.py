import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ck=json.load(open('checkpoint.json'))
def fresh():
    e=A.Arena('wa30', _budget=A._Budget(300000))
    for a in ck['final_path']: e.step(a)
    return e
CELLS=[(r,c) for r in range(3,6) for c in range(5,8)]
def feats(env):
    f=np.asarray(env.frame())
    reg=f[13:23,21:31]
    i2=int((reg==2).sum())
    filled=0
    for (R,C) in CELLS:
        b=f[R*4:R*4+4,C*4:C*4+4]
        if not (b==2).any(): filled+=1
    ys,xs=np.where(f==14); av=(int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
    return filled,i2,av,f[12:24,20:32].tobytes()
def beam(start, width=300, depth=72):
    base=start.levels_completed
    fl,i2,av,key=feats(start)
    frontier=[((-fl,i2), start.clone(), [])]
    gbest=(-fl,i2,[])
    for d in range(depth):
        cand=[]; seen=set()
        for h,env,path in frontier:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base:
                    return ('WIN', path+[a])
                fl,i2,av,key=feats(c)
                k=(av,key)
                if k in seen: continue
                seen.add(k)
                cand.append(((-fl,i2),c,path+[a]))
                cur=(-fl,i2)
                if cur<(gbest[0],gbest[1]): gbest=(-fl,i2,path+[a])
        if not cand: break
        cand.sort(key=lambda x:x[0])
        frontier=cand[:width]
        if d%6==0: print('d',d,'best(-fl,i2)',frontier[0][0],'gbest',gbest[0],gbest[1],flush=True)
    return ('BEST', gbest)
if __name__=='__main__':
    env=fresh(); r=beam(env)
    if r[0]=='WIN':
        print('WIN len',len(r[1])); json.dump(r[1],open('win_l9.json','w')); print(r[1])
    else:
        print('BEST filled',-r[1][0],'i2',r[1][1],'len',len(r[1][2]))
        json.dump(r[1][2],open('best_l9_path.json','w'))
