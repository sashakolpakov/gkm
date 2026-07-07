from probe6 import fresh_at_level6
import numpy as np, time
from collections import deque

def key(f):
    g=f[:60,:64].reshape(15,4,16,4)
    def has(col): return (g==col).any(axis=(1,3))  # 15x16 bool
    h9=has(9); h4=has(4); h3=has(3); h2=has(2); h0=has(0); h14=has(14); h15=has(15)
    box=h9&(h4|h3)&(~h2)
    grab=h9&h0
    av=None; 
    ys,xs=np.where(h14)
    if len(ys): av=(int(ys[0]),int(xs[0]))
    cour=None
    ys,xs=np.where(h15)
    if len(ys): cour=(int(ys[0]),int(xs[0]))
    bx=tuple(sorted(zip(*[a.tolist() for a in np.where(box)])))
    gr=tuple(sorted(zip(*[a.tolist() for a in np.where(grab)])))
    return (av,cour,bx,gr)

def solve(prefix, max_states=400000, time_cap=250, max_depth=70):
    base=fresh_at_level6(budget=200_000_000)
    b0=base.clone()
    for a in prefix: b0.step(a)
    start_lvl=b0.levels_completed
    t0=time.time()
    k0=key(np.asarray(b0.frame()))
    q=deque([(b0,[])]); seen={k0}
    while q and len(seen)<max_states and time.time()-t0<time_cap:
        node,path=q.popleft()
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>start_lvl:
                return prefix+path+[a], len(seen), time.time()-t0, 'WIN'
            if ch.terminal(): continue
            k=key(np.asarray(ch.frame()))
            if k in seen: continue
            seen.add(k)
            if len(path)+1<max_depth: q.append((ch,path+[a]))
    status='EXHAUSTED' if not q else ('CAP' if len(seen)>=max_states else 'TIME')
    return None, len(seen), time.time()-t0, status

if __name__=='__main__':
    prefix=[1]*7+[4]*6+[5]
    res=solve(prefix, time_cap=250)
    print('courier-removed:',('WIN len %d'%len(res[0]) if res[0] else res[3]), res[1], '%.0fs'%res[2])
    if res[0]: print(res[0])
