import numpy as np
from legs import _grid_scan, _grid_grabbed, _avatar_nav, _bfs_pair, _bfs_path
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
FACE={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
def boxes_all(f):
    out=set()
    for R in range(16):
        for C in range(16):
            u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
            if 9 in u and ({4,5,3,0}&u) and 2 not in u: out.add((R,C))
    return out
def ferry_free(env, box, drop, cap=60):
    av,boxes,walls=_grid_scan(env)
    faces={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
    for (dr,dc) in faces:
        if env.terminal(): return False
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16): continue
        if ac in walls or ac in (boxes-{box}): continue
        face=faces[(box[0]-ac[0],box[1]-ac[1])]
        trial=env.clone()
        if not _avatar_nav(trial,ac,cap): continue
        trial.step(face); trial.step(USE)
        if not _grid_grabbed(trial): continue
        tav=_grid_scan(trial)[0]; off=(box[0]-tav[0],box[1]-tav[1])
        gav=(drop[0]-off[0],drop[1]-off[1])
        probe=trial.clone()
        # rigid carry probe
        ok=_rigid(probe,off,gav,cap)
        if not ok: continue
        # commit
        if not _avatar_nav(env,ac,cap): return False
        env.step(face); env.step(USE)
        if not _grid_grabbed(env): return False
        rav=_grid_scan(env)[0]; off=(box[0]-rav[0],box[1]-rav[1]); gav=(drop[0]-off[0],drop[1]-off[1])
        if not _rigid(env,off,gav,cap):
            env.step(USE); _stepaway(env); return False
        env.step(USE); _stepaway(env); return True
    return False
def _rigid(env,off,gav,cap):
    for _ in range(cap):
        if env.terminal(): return False
        av,boxes,walls=_grid_scan(env)
        if av==gav: return True
        held=_grid_grabbed(env)
        blk=set(walls)
        for b in boxes:
            if b not in held: blk.add(b)
        bx=(av[0]+off[0],av[1]+off[1])
        if not(0<=av[0]<16 and 0<=av[1]<16 and 0<=bx[0]<16 and 0<=bx[1]<16 and av not in blk and bx not in blk): return False
        blk.discard(gav); blk.discard(av)
        p=_bfs_pair(av,off,gav,blk)
        if not p: return False
        before=av; env.step(p[0])
        if _grid_scan(env)[0]==before: return False
    return _grid_scan(env)[0]==gav
def _stepaway(env):
    av,boxes,walls=_grid_scan(env)
    held=_grid_grabbed(env)
    for a,(dr,dc) in {UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}.items():
        nb=(av[0]+dr,av[1]+dc)
        if nb in held: continue
        if nb in walls or nb in boxes: continue
        if not(0<=nb[0]<16 and 0<=nb[1]<16): continue
        env.step(a); return
    env.step(DOWN)
