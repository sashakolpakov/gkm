from probe6 import fresh_at_level6
import numpy as np, time
from collections import deque

def cellkey(f):
    out=[]
    for R in range(15):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=frozenset(int(v) for v in np.unique(blk))
            out.append(u)
    # compress to a signature: represent avatar(14/0), boxes(4/9 or 3/9), grabbed(0-border box)
    sig=[]
    for u in out:
        if 14 in u: sig.append('A')
        elif (4 in u and 9 in u) or (3 in u and 9 in u): sig.append('b')
        elif 0 in u and 9 in u: sig.append('g')  # grabbed box
        elif 15 in u: sig.append('T')
        elif 2 in u: sig.append('2')
        elif 5 in u: sig.append('#')
        elif u<= {1,7}: sig.append('.')
        else: sig.append('?')
    return ''.join(sig)

def run_bfs(prefix, max_depth=40, max_states=60000, time_cap=200):
    env=fresh_at_level6(); base=env.clone()
    for a in prefix: base.step(a)
    start_lvl=base.levels_completed
    t0=time.time()
    startkey=cellkey(np.asarray(base.frame()))
    q=deque([(base,[])])
    seen={startkey}
    best=None
    while q and len(seen)<max_states and time.time()-t0<time_cap:
        node,path=q.popleft()
        for a in (1,2,3,4,5):
            ch=node.clone()
            try: ch.step(a)
            except Exception: continue
            if ch.terminal():
                if ch.levels_completed>start_lvl:
                    return path+[a], len(seen), time.time()-t0
                continue
            if ch.levels_completed>start_lvl:
                return path+[a], len(seen), time.time()-t0
            k=cellkey(np.asarray(ch.frame()))
            if k in seen: continue
            seen.add(k)
            if len(path)+1<max_depth:
                q.append((ch,path+[a]))
    return None, len(seen), time.time()-t0

if __name__=='__main__':
    prefix=[1]*7+[4]*6+[5]  # remove courier
    res,ns,dt=run_bfs(prefix,max_depth=45,max_states=80000,time_cap=240)
    print('result',res)
    print('states',ns,'time',dt)
