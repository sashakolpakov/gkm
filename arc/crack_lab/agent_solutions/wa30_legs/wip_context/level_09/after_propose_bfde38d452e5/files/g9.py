from probe9 import *
from legs import carry_box_to, _grid_scan
LEFT=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7),(4,6)]
SOCK=set(LEFT)
def state(env):
    f=np.asarray(env.frame()); box=set(); openc=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u and 2 not in u: box.add((R,C))
    for s in LEFT:
        blk=f[s[0]*4:s[0]*4+4,s[1]*4:s[1]*4+4]; u=set(int(v) for v in np.unique(blk))
        if 2 in u: openc.add(s)
    return box,openc
def fills(env): return 9-len(state(env)[1])
def try_carry_clone(env,b,s):
    """Return real-step cost if carry feasible on a clone, else None."""
    try:
        cl=env.clone(); n0=len(cl.path)
        ok=carry_box_to(cl,b,s)
        if ok and fills(cl)>fills(env): return len(cl.path)-n0
    except Exception:
        return None
    return None
def solve(env, verbose=False, ncand=6):
    base=env.levels_completed
    while not env.terminal() and env.levels_completed<=base:
        if c7(env)<=2: break
        box,openc=state(env)
        if not openc:
            if verbose: print('FULL'); break
        free=[b for b in box if b not in SOCK]
        if not free: env.step(5); continue
        av=_grid_scan(env)[0]
        cands=[]
        for s in openc:
            for b in free:
                d=abs(av[0]-b[0])+abs(av[1]-b[1])+abs(b[0]-s[0])+abs(b[1]-s[1])
                cands.append((d,b,s))
        cands.sort()
        chosen=None
        for d,b,s in cands[:ncand]:
            cost=try_carry_clone(env,b,s)
            if cost is not None:
                if chosen is None or cost<chosen[0]: chosen=(cost,b,s)
        if chosen is None:
            env.step(5); continue
        _,b,s=chosen
        f0=fills(env)
        carry_box_to(env,b,s)
        if verbose: print('carry',b,'->',s,'fills',fills(env),'(was',f0,')','c7',c7(env),'lvl',env.levels_completed)
    return fills(env)
if __name__=='__main__':
    env=fresh(); n=solve(env,verbose=True)
    print('FINAL fills',n,'lvl',env.levels_completed,'term',env.terminal(),'c7',c7(env))
