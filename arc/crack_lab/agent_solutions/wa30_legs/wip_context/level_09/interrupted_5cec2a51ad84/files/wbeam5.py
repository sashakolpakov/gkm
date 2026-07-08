import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from legs import carry_box_to
ck=json.load(open('checkpoint.json'))
def fresh():
    e=A.Arena('wa30', _budget=A._Budget(300000))
    for a in ck['final_path']: e.step(a)
    return e
CELLS=[(r,c) for r in range(3,6) for c in range(5,8)]
def feats(env):
    f=np.asarray(env.frame())
    open_cells=[(R,C) for (R,C) in CELLS if (f[R*4:R*4+4,C*4:C*4+4]==2).any()]
    filled=9-len(open_cells)
    i2=int((f[13:23,21:31]==2).sum())
    ys,xs=np.where(f==14); av=(int(ys.min())//4,int(xs.min())//4) if len(ys) else (0,0)
    return filled,i2,av,f[12:24,20:32].tobytes(),open_cells
def looseboxes(env):
    f=np.asarray(env.frame()); s=set(CELLS); out=[]
    for R in range(16):
      for C in range(16):
        u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
        if 9 in u and 4 in u and 2 not in u and (R,C) not in s: out.append((R,C))
    return out
def try_finish(env, base):
    # from an 8-cell state, try to fill remaining open cells; return path if wins
    cl=env.clone(); n0=len(cl.path)
    for _ in range(4):
        fl,i2,av,key,opens=feats(cl)
        if cl.levels_completed>base: return []
        if not opens: break
        lb=looseboxes(cl)
        if not lb: 
            # yield a few for courier
            for _ in range(6):
                if cl.terminal() or cl.levels_completed>base: break
                cl.step(5)
            if cl.levels_completed>base: return [5]  # marker
            continue
        s=opens[0]
        lb.sort(key=lambda b:abs(b[0]-s[0])+abs(b[1]-s[1]))
        done=False
        for b in lb[:3]:
            try:
                if carry_box_to(cl,b,s): done=True; break
            except Exception: pass
        if cl.levels_completed>base: return []
        if not done:
            for _ in range(4):
                if cl.terminal() or cl.levels_completed>base: break
                cl.step(5)
    return None
def beam(start,width=300,depth=70):
    base=start.levels_completed
    fl,i2,av,key,opens=feats(start)
    frontier=[((-fl,i2),start.clone(),[])]
    for d in range(depth):
        cand=[]; seen=set()
        for h,env,path in frontier:
            if env.terminal(): continue
            for a in [1,2,3,4,5]:
                c=env.clone(); c.step(a)
                if c.levels_completed>base: return ('WIN',path+[a])
                fl,i2,av,key,opens=feats(c); k=(av,key)
                if k in seen: continue
                seen.add(k)
                if fl>=8 and c7_ok(c):
                    r=try_finish(c,base)
                    if r is not None:
                        print('FINISH from filled',fl,'depth',d,flush=True)
                        return ('WIN', path+[a])  # the finish confirms winnable state
                cand.append(((-fl,i2),c,path+[a]))
        if not cand: break
        cand.sort(key=lambda x:x[0]); frontier=cand[:width]
        if d%6==0: print('d',d,'best',frontier[0][0],flush=True)
    return ('BEST',frontier[0][2] if frontier else [])
def c7_ok(env):
    return int((np.asarray(env.frame())==7).sum())>3
if __name__=='__main__':
    env=fresh(); r=beam(env)
    print('RES',r[0],'len',len(r[1]))
    json.dump(r[1],open('l9_result_path.json','w'))
