import probe7
import numpy as np

env=probe7.get_env_at_L7()

def score(e):
    f=np.asarray(e.frame())
    lc=e.levels_completed
    two=int((f==2).sum())
    # covered target = boxes (9-core) sitting on col3 or col12 target cells
    return (lc, -two)

def sig(e):
    f=np.asarray(e.frame())
    parts=[]
    for R in range(5,10):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=frozenset(int(v) for v in np.unique(blk) if v in (0,2,3,4,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)

base=env.levels_completed
beam=[(score(env), env.clone(), [])]
seen={sig(env)}
best=None
for depth in range(70):
    cand=[]
    for sc,node,path in beam:
        for a in (1,2,3,4,5):
            ch=node.clone(); ch.step(a)
            if ch.levels_completed>base:
                best=path+[a]; break
            if ch.terminal(): continue
            k=sig(ch)
            if k in seen: continue
            seen.add(k)
            cand.append((score(ch),ch,path+[a]))
        if best: break
    if best: break
    cand.sort(key=lambda x:x[0],reverse=True)
    beam=cand[:60]
    if not beam: break
    if depth%10==0:
        print('depth',depth,'best score',beam[0][0],'beamlen',len(beam))
print('FOUND',best)
