import numpy as np
from legs import _grid_scan, _bfs_path
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}
FACE={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}
def movers(f):
    return set((int(y)//4,int(x)//4) for y,x in zip(*np.where(f==15)))
def chase_clear_one(env, band, cap=40):
    # band: 'top' rows<6, 'bot' rows>9
    for t in range(cap):
        if env.terminal(): return False
        f=np.asarray(env.frame())
        mv=[m for m in movers(f) if (m[0]<6 if band=='top' else m[0]>9)]
        if not mv: return True
        av,boxes,walls=_grid_scan(env)
        target=min(mv,key=lambda m:abs(m[0]-av[0])+abs(m[1]-av[1]))
        # if adjacent, face+USE
        d=(target[0]-av[0],target[1]-av[1])
        if abs(d[0])+abs(d[1])==1:
            env.step(FACE[d]); env.step(USE)
            continue
        # else path toward an adjacent cell of target (avoid walls, movers as soft)
        blk=set(walls)|set(boxes)|(set(mv)-{target})
        best=None
        for (dr,dc),fa in FACE.items():
            adj=(target[0]+dr,target[1]+dc)
            if adj in blk or not(0<=adj[0]<16 and 0<=adj[1]<16): continue
            p=_bfs_path(av,adj,blk)
            if p and (best is None or len(p)<len(best)): best=p
        if not best:
            # step toward target ignoring boxes
            dr=np.sign(target[0]-av[0]); dc=np.sign(target[1]-av[1])
            a=DOWN if dr>0 else (UP if dr<0 else (RIGHT if dc>0 else LEFT))
            env.step(a); continue
        env.step(best[0])
    return not [m for m in movers(np.asarray(env.frame())) if (m[0]<6 if band=='top' else m[0]>9)]
def clear_both(env):
    # clear whichever band avatar is closer to first
    av=_grid_scan(env)[0]
    if av[0]<8:
        chase_clear_one(env,'top'); chase_clear_one(env,'bot')
    else:
        chase_clear_one(env,'bot'); chase_clear_one(env,'top')
