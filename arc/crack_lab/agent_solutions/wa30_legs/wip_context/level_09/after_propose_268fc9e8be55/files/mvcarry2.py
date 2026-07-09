import numpy as np
from legs import _grid_scan, _grid_grabbed, _bfs_path, _bfs_pair
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}
FACE={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}

def movers(env):
    f=np.asarray(env.frame())
    ys,xs=np.where(f==15)
    return set((int(y)//4,int(x)//4) for y,x in zip(ys,xs))

def _wait(env):
    # advance one turn without moving: step into a wall/box neighbor if possible
    av,boxes,walls=_grid_scan(env)
    blk=set(walls)|set(boxes)
    for a,(dr,dc) in DIRS.items():
        nb=(av[0]+dr,av[1]+dc)
        if nb in blk or not(0<=nb[0]<16 and 0<=nb[1]<16):
            env.step(a); return
    env.step(UP)

def nav(env, goal, cap=60):
    for _ in range(cap):
        if env.terminal(): return False
        av,boxes,walls=_grid_scan(env)
        if av==goal: return True
        hard=set(walls)|set(boxes); hard.discard(goal); hard.discard(av)
        mv=movers(env)
        blk=hard|mv
        p=_bfs_path(av,goal,blk)
        if p:
            nb=(av[0]+DIRS[p[0]][0],av[1]+DIRS[p[0]][1])
            if nb in mv:
                _wait(env); continue
            env.step(p[0]); continue
        # movers block: plan ignoring movers
        p=_bfs_path(av,goal,hard)
        if not p: return False
        nb=(av[0]+DIRS[p[0]][0],av[1]+DIRS[p[0]][1])
        if nb in mv:
            _wait(env); continue
        env.step(p[0])
    return _grid_scan(env)[0]==goal

def grab(env, box, cap=60):
    # navigate adjacent and grab; robust to mover on box/adjacent
    for attempt in range(8):
        av,boxes,walls=_grid_scan(env)
        if box not in boxes:
            # box may be carried by mover; give up
            return False
        hard=set(walls)|set(boxes)
        mv=movers(env)
        # pick reachable adjacent cell
        best=None
        for (dr,dc),fa in FACE.items():
            ac=(box[0]+dr,box[1]+dc)
            if not(0<=ac[0]<16 and 0<=ac[1]<16): continue
            if ac in walls or ac in (boxes-{box}): continue
            p=_bfs_path(av,ac,(hard-{box})|mv)
            if p is None:
                p=_bfs_path(av,ac,hard-{box})
                extra=100
            else: extra=0
            if p is not None and (best is None or len(p)+extra<best[0]):
                best=(len(p)+extra,ac,fa)
        if best is None: return False
        _,ac,fa=best
        if not nav(env,ac,cap): 
            continue
        # face and grab, waiting if mover on box
        if box in movers(env):
            _wait(env); continue
        env.step(fa); 
        if box in movers(env):
            continue
        env.step(USE)
        if _grid_grabbed(env):
            return True
    return False

def carry_to(env, drop, cap=80):
    # box currently held; rigid-carry so held box lands on drop
    held=_grid_grabbed(env)
    if not held: return False
    av=_grid_scan(env)[0]
    # offset from avatar to held box (use first held cell)
    hb=sorted(held)[0]
    off=(hb[0]-av[0],hb[1]-av[1])
    gav=(drop[0]-off[0],drop[1]-off[1])
    stuck=0
    for _ in range(cap):
        if env.terminal(): return False
        av,boxes,walls=_grid_scan(env)
        if av==gav:
            env.step(USE); return True
        held=_grid_grabbed(env)
        hard=set(walls)
        for b in boxes:
            if b not in held: hard.add(b)
        hard.discard(gav); hard.discard(av)
        mv=movers(env)
        p=_bfs_pair(av,off,gav,hard|mv)
        usemv=True
        if not p:
            p=_bfs_pair(av,off,gav,hard); usemv=False
        if not p:
            stuck+=1
            if stuck>10: env.step(USE); return False
            _wait(env); continue
        nb=(av[0]+DIRS[p[0]][0],av[1]+DIRS[p[0]][1])
        nbb=(nb[0]+off[0],nb[1]+off[1])
        if nb in mv or nbb in mv:
            _wait(env); stuck+=1
            if stuck>12: env.step(USE); return False
            continue
        before=av
        env.step(p[0])
        if _grid_scan(env)[0]==before:
            stuck+=1
            if stuck>12: env.step(USE); return False
        else: stuck=0
    env.step(USE); return _grid_scan(env)[0]==gav

def carry(env, box, drop):
    if not grab(env, box): return False
    return carry_to(env, drop)
