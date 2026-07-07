import numpy as np
from collections import deque
from legs import _grid_scan, _grid_grabbed, _bfs_path, _bfs_pair, _mover_cell
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}

def movers(env):
    f=np.asarray(env.frame())
    ys,xs=np.where(f==15)
    cells=set()
    # cluster into 4x4 cells
    for y,x in zip(ys,xs):
        cells.add((y//4,x//4))
    return cells

def obstacles(env, exclude_box=None):
    av,boxes,walls=_grid_scan(env)
    blk=set(walls)|set(boxes)
    if exclude_box in blk: blk.discard(exclude_box)
    blk|=movers(env)
    return av,blk

def nav(env, goal, cap=40):
    for _ in range(cap):
        if env.terminal(): return False
        av,blk=obstacles(env)
        if av==goal: return True
        b=set(blk); b.discard(goal); b.discard(av)
        p=_bfs_path(av,goal,b)
        if not p:
            # maybe mover transiently blocks; try step toward goal ignoring movers
            av2,boxes,walls=_grid_scan(env)
            b2=set(walls)|set(boxes); b2.discard(goal); b2.discard(av2)
            p=_bfs_path(av2,goal,b2)
            if not p: return False
        before=av
        env.step(p[0])
        # if didn't move (mover blocked), just continue/retry
    return obstacles(env)[0]==goal

def carry(env, box, drop, cap=60):
    av,boxes,walls=_grid_scan(env)
    if box not in boxes: return False
    faces={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
    # choose grab cell reachable
    best=None
    _,blk=obstacles(env,exclude_box=box)
    for (dr,dc),fa in faces.items():
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16) or ac in blk: continue
        p=_bfs_path(av,ac,set(blk)-{box}|set())
        if p is not None and (best is None or len(p)<best[0]):
            best=(len(p),ac,fa)
    if best is None: return False
    _,ac,fa=best
    if not nav(env,ac,cap): return False
    env.step(fa); env.step(USE)
    held=_grid_grabbed(env)
    if not held: 
        env.step(USE); return False
    rav=_grid_scan(env)[0]
    off=(box[0]-rav[0],box[1]-rav[1])
    gav=(drop[0]-off[0],drop[1]-off[1])
    stuck=0
    for _ in range(cap):
        if env.terminal(): return False
        av,boxes,walls=_grid_scan(env)
        if av==gav:
            env.step(USE); return True
        held=_grid_grabbed(env)
        blk=set(walls)
        for b in boxes:
            if b not in held: blk.add(b)
        blk|=movers(env)
        blk.discard(gav); blk.discard(av)
        p=_bfs_pair(av,off,gav,blk)
        if not p:
            # try ignoring movers
            blk2=set(walls)
            for b in boxes:
                if b not in held: blk2.add(b)
            blk2.discard(gav); blk2.discard(av)
            p=_bfs_pair(av,off,gav,blk2)
            if not p:
                env.step(USE); return False
        before=av
        env.step(p[0])
        if _grid_scan(env)[0]==before:
            stuck+=1
            if stuck>6:
                env.step(USE); return False
        else:
            stuck=0
    env.step(USE)
    return _grid_scan(env)[0]==gav
