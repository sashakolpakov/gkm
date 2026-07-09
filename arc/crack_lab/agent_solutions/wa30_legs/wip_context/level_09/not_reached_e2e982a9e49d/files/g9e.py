from probe9 import *
from legs import carry_box_to, _grid_scan
SOCK9=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7),(4,6)]
SOCK8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
TSET=set(SOCK9)
def state(env):
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
def cost_clone(env,b,s):
    try:
        cl=env.clone(); n0=len(cl.path); f0=sfill(cl)
        if carry_box_to(cl,b,s) and sfill(cl)>=f0: return len(cl.path)-n0
    except Exception: return None
    return None
def solve(env, verbose=False):
    base=env.levels_completed
    while not env.terminal() and env.levels_completed<=base:
        if c7(env)<=1: break
        box,openc=state(env)
        if not openc: 
            if verbose: print('FULL9'); break
        free=[b for b in box if b not in TSET]
        if not free: env.step(5); continue
        av=_grid_scan(env)[0]
        pairs=sorted([(abs(av[0]-b[0])+abs(av[1]-b[1])+abs(b[0]-s[0])+abs(b[1]-s[1]),b,s) for s in openc for b in free])
        best=None
        for man,b,s in pairs:
            if best is not None and man>=best[0]+6: break  # manhattan lower bound prune
            c=cost_clone(env,b,s)
            if c is not None and (best is None or c<best[0]): best=(c,b,s)
        if best is None:
            env.step(5); continue
        _,b,s=best
        carry_box_to(env,b,s)
        if verbose: print('carry',b,'->',s,'cost',best[0],'sockets',sfill(env),'c7',c7(env),'lvl',env.levels_completed)
    return sfill(env)
if __name__=='__main__':
    env=fresh(); n=solve(env,verbose=True)
    print('FINAL sockets',n,'lvl',env.levels_completed,'term',env.terminal(),'c7',c7(env))
