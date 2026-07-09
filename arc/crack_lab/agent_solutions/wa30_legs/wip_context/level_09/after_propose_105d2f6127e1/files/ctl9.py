from probe9 import *
from legs import carry_box_to, _grid_scan, _movers, _bfs_path, _avatar_nav
DIRS={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
CELLS=[(r,c) for r in range(3,6) for c in range(5,8)]
def av(env):
    f=np.asarray(env.frame()); ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
def opens(env):
    f=np.asarray(env.frame()); return [(R,C) for (R,C) in CELLS if (f[R*4:R*4+4,C*4:C*4+4]==2).any()]
def loose(env):
    f=np.asarray(env.frame()); s=set(CELLS); out=[]
    for R in range(16):
      for C in range(16):
        u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
        if 9 in u and 4 in u and 2 not in u and (R,C) not in s: out.append((R,C))
    return out
def kill_if_parked(env, prev_m):
    ms=_movers(env,15)
    if not ms: return None
    m=ms[0]
    # in/near container and stationary
    if prev_m==m and 2<=m[0]<=6 and 4<=m[1]<=8:
        a=av(env)
        if a is None: return m
        d=abs(a[0]-m[0])+abs(a[1]-m[1])
        if d==1:
            env.step(DIRS[(m[0]-a[0],m[1]-a[1])]); env.step(5)
            return _movers(env,15)[0] if _movers(env,15) else None
        # navigate adjacent (short)
        av2,boxes,walls=_grid_scan(env)
        blocked=(set(walls)|set(boxes))-{m}
        # target a cell adjacent to m
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            adj=(m[0]+dr,m[1]+dc)
            if adj in blocked: continue
            p=_bfs_path(a,adj,blocked)
            if p is not None and len(p)<=3:
                for st in p: env.step(st)
                a=av(env)
                if abs(a[0]-m[0])+abs(a[1]-m[1])==1:
                    env.step(DIRS[(m[0]-a[0],m[1]-a[1])]); env.step(5)
                return _movers(env,15)[0] if _movers(env,15) else None
    return m
def solve(env, verbose=False):
    base=env.levels_completed; prev_m=None
    guard=0
    while not env.terminal() and env.levels_completed<=base and c7(env)>2 and guard<40:
        guard+=1
        op=opens(env)
        if not op:
            break
        # try kill parked mover first (cheap)
        prev_m2=kill_if_parked(env, prev_m)
        if env.terminal() or env.levels_completed>base: break
        ms=_movers(env,15); prev_m=ms[0] if ms else None
        op=opens(env)
        if not op: break
        lb=loose(env)
        if not lb:
            for _ in range(4):
                if env.terminal() or env.levels_completed>base: break
                env.step(5)
            continue
        a=av(env)
        # choose (box,cell) min cost; avatar prefers deep/east, but take cheapest
        cands=sorted([(abs(a[0]-b[0])+abs(a[1]-b[1])+abs(b[0]-s[0])+abs(b[1]-s[1]),b,s) for s in op for b in lb])
        did=False
        for _,b,s in cands:
            try: ok=carry_box_to(env,b,s)
            except Exception: ok=False; break
            if ok: did=True; break
        ms=_movers(env,15); prev_m=ms[0] if ms else None
        if not did: env.step(5)
        if verbose:
            f=np.asarray(env.frame()); fl=sum(1 for (R,C) in CELLS if not (f[R*4:R*4+4,C*4:C*4+4]==2).any())
            print('fills',fl,'open',opens(env),'mover',_movers(env,15),'c7',c7(env),'lvl',env.levels_completed)
    # endgame yield
    while not env.terminal() and env.levels_completed<=base and c7(env)>0:
        env.step(5)
    f=np.asarray(env.frame()); return sum(1 for (R,C) in CELLS if not (f[R*4:R*4+4,C*4:C*4+4]==2).any())
if __name__=='__main__':
    env=fresh(); n=solve(env,verbose=True)
    print('FINAL fills',n,'lvl',env.levels_completed,'term',env.terminal())
