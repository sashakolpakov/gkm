from pl9 import *
from legs import _grid_scan, _grid_grabbed, _bfs_path, _bfs_pair, _rigid_carry
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
def carry_fast(env, box, drop, cap=40):
    av,boxes,walls=_grid_scan(env)
    if box not in boxes: return False
    blocked=set(walls)|(set(boxes)-{box})
    best=None
    for (dr,dc) in DIRS:
        ac=(box[0]+dr,box[1]+dc)  # approach/grab cell
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
            best=(cost,ac,offset,goal_av,DIRS[(box[0]-ac[0],box[1]-ac[1])])
    if best is None: return False
    _,ac,offset,goal_av,face=best
    # nav to ac
    for _ in range(cap):
        av=_grid_scan(env)[0]
        if av==ac: break
        p=_bfs_path(av,ac,set(walls)|(set(_grid_scan(env)[1])-{box}))
        if not p: return False
        env.step(p[0])
        if env.terminal(): return False
    env.step(face); env.step(USE)
    if not _grid_grabbed(env): return False
    av=_grid_scan(env)[0]; off=(box[0]-av[0],box[1]-av[1])
    # recompute goal
    g=(drop[0]-off[0],drop[1]-off[1])
    if not _rigid_carry(env,off,g,cap): 
        env.step(USE); return False
    env.step(USE)
    return True
if __name__=='__main__':
    import perception as P
    def c7(env): return P.color_counts(np.asarray(env.frame())).get(7,0)
    env=fresh()
    seq=[]; orig=env.step; env.step=lambda a:(seq.append(a),orig(a))[1]
    total0=len(seq)
    for (b,d) in [((5,3),(4,6)),((8,14),(5,7)),((7,12),(4,7)),((5,11),(3,7))]:
        s0=len(seq); ok=carry_fast(env,b,d)
        print('carry_fast',b,'->',d,'ok',ok,'steps',len(seq)-s0,'c7',c7(env))
    g=grid(np.asarray(env.frame()))
    big9=[(4,6),(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
    print('total steps',len(seq),'big',sum(1 for s in big9 if g[s[0]][s[1]] in('4c','?39')),'lvl',env.levels_completed)
