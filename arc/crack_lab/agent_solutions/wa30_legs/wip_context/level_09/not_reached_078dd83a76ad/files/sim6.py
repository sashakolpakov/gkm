from probe6 import fresh_at_level6
import numpy as np
from collections import deque
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}

def parse(f):
    av=None; boxes=set(); walls=set(); grabbed=set(); walk=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if (4 in u and 9 in u and 2 not in u) or (3 in u and 9 in u and 2 not in u):
                boxes.add((R,C))
            if 0 in u and 9 in u: grabbed.add((R,C))
            if (blk==5).sum()>=8: walls.add((R,C))
    return av,boxes,grabbed,walls

def blocked_set(walls,boxes,exclude=()):
    b=set(walls)|set(boxes)
    for e in exclude: b.discard(e)
    return b

def bfs(start,goal,blocked):
    if start==goal: return []
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        for a,(dr,dc) in DIRS.items():
            nb=(cur[0]+dr,cur[1]+dc)
            if not(0<=nb[0]<16 and 0<=nb[1]<16): continue
            if nb in prev: continue
            if nb in blocked and nb!=goal: continue
            prev[nb]=(cur,a)
            if nb==goal:
                path=[];n=nb
                while prev[n]:
                    p,ac=prev[n]; path.append(ac); n=p
                return path[::-1]
            q.append(nb)
    return None

def nav_to(c,goal,cap=60):
    for _ in range(cap):
        if c.terminal(): return False
        f=np.asarray(c.frame()); av,boxes,gr,walls=parse(f)
        if av==goal: return True
        bl=blocked_set(walls,boxes,exclude=(goal,av))
        p=bfs(av,goal,bl)
        if not p: return False
        c.step(p[0])
    return False

def carry_pair(c,offset,goal_av,cap=80):
    for _ in range(cap):
        if c.terminal(): return False
        f=np.asarray(c.frame()); av,boxes,gr,walls=parse(f)
        # grabbed box is 'gr'; treat other boxes as obstacles
        carried = gr
        if av==goal_av: return True
        bl=set(walls)
        for b in boxes: 
            if b not in carried: bl.add(b)
        def ok(a):
            bcell=(a[0]+offset[0],a[1]+offset[1])
            if not(0<=a[0]<16 and 0<=a[1]<16 and 0<=bcell[0]<16 and 0<=bcell[1]<16): return False
            if a in bl or bcell in bl: return False
            return True
        # BFS for avatar
        if not ok(av): return False
        q=deque([av]);prev={av:None}
        found=None
        while q:
            cur=q.popleft()
            if cur==goal_av: found=cur;break
            for a,(dr,dc) in DIRS.items():
                nb=(cur[0]+dr,cur[1]+dc)
                if nb in prev: continue
                if not ok(nb): continue
                prev[nb]=(cur,a); q.append(nb)
        if found is None: return False
        # reconstruct first step
        path=[];n=found
        while prev[n]:
            p,ac=prev[n];path.append(ac);n=p
        path=path[::-1]
        if not path: return True
        c.step(path[0])
    return False

def deliver(c, box, drop):
    """grab box cell, carry to drop cell, release."""
    f=np.asarray(c.frame()); av,boxes,gr,walls=parse(f)
    if box not in boxes: return False
    bl=blocked_set(walls,boxes,exclude=())
    # find adjacent-to-box reachable cell
    best=None
    for a,(dr,dc) in DIRS.items():
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16): continue
        if ac in bl: continue
        p=bfs(av,ac,blocked_set(walls,boxes,exclude=(ac,av)))
        if p is not None and (best is None or len(p)<best[0]): best=(len(p),ac,(dr,dc))
    if best is None: return False
    _,ac,d=best
    if not nav_to(c,ac): return False
    face={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}[(box[0]-ac[0],box[1]-ac[1])]
    c.step(face); c.step(USE)  # grab
    f=np.asarray(c.frame()); av2,boxes2,gr2,walls2=parse(f)
    if not gr2: return False
    gcell=list(gr2)[0]
    offset=(gcell[0]-av2[0],gcell[1]-av2[1])
    goal_av=(drop[0]-offset[0],drop[1]-offset[1])
    if not carry_pair(c,offset,goal_av): 
        c.step(USE); return False
    c.step(USE)  # release
    return True

def deliver2(c, box, drop, cap_nav=60, cap_carry=120):
    """Robust deliver: try each approach side; pick one whose carry succeeds (simulated on clone)."""
    import numpy as np
    f=np.asarray(c.frame()); av,boxes,gr,walls=parse(f)
    if box not in boxes: return False
    order=[]
    for a,(dr,dc) in DIRS.items():
        ac=(box[0]+dr,box[1]+dc)
        if not(0<=ac[0]<16 and 0<=ac[1]<16): continue
        if ac in walls or ac in (boxes-{box}): continue
        order.append((a,ac,(dr,dc)))
    for a,ac,d in order:
        trial=c.clone()
        if not nav_to(trial,ac,cap_nav): continue
        face={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}[(box[0]-ac[0],box[1]-ac[1])]
        trial.step(face); trial.step(USE)
        ff=np.asarray(trial.frame()); av2,b2,gr2,w2=parse(ff)
        if not gr2: continue
        gcell=list(gr2)[0]; offset=(gcell[0]-av2[0],gcell[1]-av2[1])
        goal_av=(drop[0]-offset[0],drop[1]-offset[1])
        t2=trial.clone()
        if carry_pair(t2,offset,goal_av,cap_carry):
            t2.step(USE)
            # commit: replicate on real c
            nav_to(c,ac,cap_nav); c.step(face); c.step(USE)
            fc=np.asarray(c.frame()); avc_,bc,grc,wc=parse(fc)
            gc=list(grc)[0]; off=(gc[0]-avc_[0],gc[1]-avc_[1])
            gav=(drop[0]-off[0],drop[1]-off[1])
            if carry_pair(c,off,gav,cap_carry):
                c.step(USE); return True
            else:
                c.step(USE); return False
    return False
