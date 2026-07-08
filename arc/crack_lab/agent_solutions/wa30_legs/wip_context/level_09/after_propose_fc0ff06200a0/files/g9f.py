from probe9 import *
from legs import carry_box_to, _grid_scan
SOCK8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
SOCK9=SOCK8+[(4,6)]
TSET=set(SOCK9)
def boxes_open(env):
    f=np.asarray(env.frame()); box=set(); openc=[]
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u and 2 not in u: box.add((R,C))
    for s in SOCK9:
        u=set(int(v) for v in np.unique(f[s[0]*4:s[0]*4+4,s[1]*4:s[1]*4+4]))
        if 2 in u: openc.append(s)
    return box,openc
def sfill(env):
    f=np.asarray(env.frame())
    return sum(1 for s in SOCK8 if (lambda u:9 in u and 4 in u and 2 not in u)(set(int(v) for v in np.unique(f[s[0]*4:s[0]*4+4,s[1]*4:s[1]*4+4]))))
def rollout_yield(env):
    cl=env.clone(); base=cl.levels_completed; mx=sfill(cl)
    for _ in range(80):
        if cl.terminal() or cl.levels_completed>base: break
        cl.step(5); mx=max(mx,sfill(cl))
    return max(mx,sfill(cl)), (cl.levels_completed>base)
def solve(env, verbose=False):
    base=env.levels_completed
    while not env.terminal() and env.levels_completed<=base and c7(env)>2:
        box,openc=boxes_open(env)
        if not openc: break
        free=[b for b in box if b not in TSET]
        if not free: env.step(5); continue
        av=_grid_scan(env)[0]
        pairs=sorted([(abs(av[0]-b[0])+abs(av[1]-b[1])+abs(b[0]-s[0])+abs(b[1]-s[1]),b,s) for s in openc for b in free])[:10]
        best=None
        for man,b,s in pairs:
            try:
                cl=env.clone()
                if not carry_box_to(cl,b,s): continue
                score,won=rollout_yield(cl)
                key=(won, score, -(len(cl.path)))
                if best is None or key>best[0]: best=(key,b,s)
            except Exception: continue
        if best is None:
            env.step(5); continue
        _,b,s=best
        carry_box_to(env,b,s)
        if verbose: print('carry',b,'->',s,'score',best[0],'sockets',sfill(env),'c7',c7(env),'lvl',env.levels_completed)
    # final yield
    b0=env.levels_completed
    while not env.terminal() and env.levels_completed<=b0 and c7(env)>0:
        env.step(5)
    return sfill(env)
if __name__=='__main__':
    env=fresh(); n=solve(env,verbose=True)
    print('FINAL sockets',n,'lvl',env.levels_completed,'term',env.terminal(),'c7',c7(env))
