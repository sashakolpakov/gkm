import probe7
import numpy as np

env=probe7.get_env_at_L7()
base=env.levels_completed
LEFT=[(7,3),(8,3)]; RIGHT=[(7,12),(8,12)]

def feat(e):
    f=np.asarray(e.frame())
    def hasbox(c):
        R,C=c; u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4])); return 9 in u and (4 in u or 3 in u or 5 in u)
    lc=e.levels_completed
    lcov=sum(hasbox(c) for c in LEFT)
    rcov=sum(hasbox(c) for c in RIGHT)
    return (lc,lcov,rcov)

def sig(e):
    f=np.asarray(e.frame()); parts=[]
    for R in range(5,10):
        for C in range(16):
            u=frozenset(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]) if v in (0,2,3,4,5,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)

beam=[(feat(env),env.clone(),[])]; seen={sig(env)}; best=None
for depth in range(45):
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base: best=path+[a]; break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k); cand.append((feat(ch),ch,path+[a]))
        if best: break
    if best: break
    cand.sort(key=lambda x:x[0],reverse=True); beam=cand[:80]
    if not beam: break
    if depth%5==0: print('d',depth,'best',beam[0][0])
print('FOUND',best)
