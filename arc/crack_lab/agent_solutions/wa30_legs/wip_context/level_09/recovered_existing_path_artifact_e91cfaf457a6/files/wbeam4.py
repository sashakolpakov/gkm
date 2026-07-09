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
    filled=sum(1 for (R,C) in CELLS if not (f[R*4:R*4+4,C*4:C*4+4]==2).any())
    i2=int((f[13:23,21:31]==2).sum())
    alive=1 if (f==15).any() else 0
    ys,xs=np.where(f==14); av=(int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
    return filled,alive,i2,av,f[12:24,20:32].tobytes()
def beam(start,width=350,depth=74):
    base=start.levels_completed
    fl,al,i2,av,key=feats(start)
    frontier=[((-fl,al,i2),start.clone(),[])]; gbest=((-fl,al,i2),[])
    for d in range(depth):
        cand=[]; seen=set()
        for h,env,path in frontier:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base: return ('WIN',path+[a])
                fl,al,i2,av,key=feats(c); k=(av,key,al)
                if k in seen: continue
                seen.add(k); sc=(-fl,al,i2); cand.append((sc,c,path+[a]))
                if sc<gbest[0]: gbest=(sc,path+[a])
        if not cand: break
        cand.sort(key=lambda x:x[0]); frontier=cand[:width]
        if d%6==0: print('d',d,'best',frontier[0][0],'gbest',gbest[0],flush=True)
    return ('BEST',gbest)
if __name__=='__main__':
    env=fresh(); r=beam(env)
    if r[0]=='WIN':
        print('WIN len',len(r[1])); json.dump(r[1],open('win_l9.json','w')); print(r[1])
    else:
        print('BEST',r[1][0]); json.dump(r[1][1],open('best_l9_path.json','w'))
