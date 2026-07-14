import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    # exhaustively enumerate the 25 counter-masked configs on real-faithful clones,
    # and for each, list the path; then we can test on real env.
    from collections import deque
    def key(e):
        f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
    start=env.clone(); seen={key(start):[]}; q=deque([start])
    paths=[]
    while q:
        n=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen[k]=seen[key(n)]+[a] if key(n) in seen else [a]
            q.append(c)
    # recompute paths properly
    seen={key(start):[]}; q=deque([(start,[])])
    configs=[]
    while q:
        n,p=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen[k]=p+[a]; q.append((c,p+[a])); configs.append((p+[a],c))
    print("num configs",len(configs))
    # for each config, print avatar bbox, c8, phase(legend row1), lvl
    for p,c in configs:
        f=np.asarray(c.frame())
        avb=[b.bbox for b in P.connected_components(f,colors=[9]) if b.area==24]
        print(len(p),avb,"c8",P.color_counts(f).get(8),"legrow1",''.join(str(int(v)) for v in f[1,1:8]),"lvl",c.levels_completed)
    raise SystemExit
A.run_program('g50t', prog)
