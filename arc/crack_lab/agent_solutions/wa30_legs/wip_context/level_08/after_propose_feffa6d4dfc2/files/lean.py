import numpy as np
from legs import _grid_scan, _grid_grabbed, _bfs_path, _bfs_pair, _avatar_nav
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
FACE={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
def lean_ferry(env, box, drop, cap=40):
    av,boxes,walls=_grid_scan(env)
    # nav to a reachable adjacent grab cell
    best=None
    blk=set(walls)|(boxes-{box})
    for (dr,dc),fa in FACE.items():
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16) or ac in blk: continue
        p=_bfs_path(av,ac,blk)
        if p is not None and (best is None or len(p)<best[0]): best=(len(p),ac,fa)
    if best is None: return False
    _,ac,fa=best
    if not _avatar_nav(env,ac,cap): return False
    if env.terminal(): return False
    env.step(fa); env.step(USE)
    if not _grid_grabbed(env): return False
    rav=_grid_scan(env)[0]
    hb=sorted(_grid_grabbed(env))[0]; off=(hb[0]-rav[0],hb[1]-rav[1])
    gav=(drop[0]-off[0],drop[1]-off[1])
    for _ in range(cap):
        if env.terminal(): return False
        av,boxes,walls=_grid_scan(env)
        if av==gav: break
        held=_grid_grabbed(env)
        b2=set(walls)
        for b in boxes:
            if b not in held: b2.add(b)
        b2.discard(gav); b2.discard(av)
        p=_bfs_pair(av,off,gav,b2)
        if not p: break
        before=av; env.step(p[0])
        if _grid_scan(env)[0]==before: break
    # release + stepaway
    env.step(USE)
    av,boxes,walls=_grid_scan(env); held=_grid_grabbed(env)
    for a,(dr,dc) in FACE.items():
        nb=(av[0]+dr,av[1]+dc)
        if nb in held or nb in walls or nb in boxes or not(0<=nb[0]<16 and 0<=nb[1]<16): continue
        env.step(a); break
    return True
