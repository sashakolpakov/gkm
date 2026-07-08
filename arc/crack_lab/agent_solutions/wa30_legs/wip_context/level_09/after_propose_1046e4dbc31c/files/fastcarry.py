from pl9 import *
from legs import _grid_scan, _grid_grabbed, _bfs_path, _bfs_pair
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
def fast_carry(env, box, drop):
    av,boxes,walls=_grid_scan(env)
    if box not in boxes: return False
    blocked=set(walls)|(set(boxes)-{box})
    best=None
    for (dr,dc) in DIRS:
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16) or ac in blocked: continue
        p1=_bfs_path(av,ac,blocked)
        if p1 is None: continue
        offset=(box[0]-ac[0],box[1]-ac[1])
        goal_av=(drop[0]-offset[0],drop[1]-offset[1])
        if not(0<=goal_av[0]<16 and 0<=goal_av[1]<16): continue
        p2=_bfs_pair(ac,offset,goal_av,blocked)
        if p2 is None: continue
        cost=len(p1)+len(p2)
        if best is None or cost<best[0]:
            best=(cost,ac,offset,goal_av,p1,p2,DIRS[(box[0]-ac[0],box[1]-ac[1])])
    if best is None: return False
    _,ac,offset,goal_av,p1,p2,face=best
    # execute p1 (nav) directly, with 1-retry on block
    for a in p1:
        before=_grid_scan(env)[0]
        env.step(a)
        if env.terminal(): return False
        if _grid_scan(env)[0]==before:  # blocked (courier); wait then retry once
            env.step(a)
            if _grid_scan(env)[0]==before: return False
    env.step(face); env.step(USE)
    if not _grid_grabbed(env): return False
    for a in p2:
        before=_grid_scan(env)[0]
        env.step(a)
        if env.terminal(): return False
        if _grid_scan(env)[0]==before:
            env.step(a)
            if _grid_scan(env)[0]==before:
                env.step(USE); return False
    env.step(USE)
    return True
if __name__=='__main__':
    import perception as P
    def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
    env=fresh(); seq=[]; orig=env.step
    env.step=lambda a:(seq.append(a),orig(a))[1]
    for b,d in [((8,14),(5,7)),((7,12),(4,7)),((5,11),(3,7))]:
        s0=len(seq); ok=fast_carry(env,b,d)
        print('fast_carry',b,'->',d,'ok',ok,'steps',len(seq)-s0,'c7',c7(env))
    big8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
    g=grid(np.asarray(env.frame()))
    print('total',len(seq),'edges',sum(1 for s in big8 if g[s[0]][s[1]] in('4c','?39')))
