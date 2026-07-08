# Standalone experimental solver for wa30 L8 (pixel level). Will refactor into legs if it wins.
import numpy as np
from collections import deque
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
DIRS={UP:(-1,0),DOWN:(1,0),LEFT:(0,-1),RIGHT:(0,1)}
def blk(f,R,C): return f[R*4:R*4+4,C*4:C*4+4]
def uset(f,R,C): return set(int(v) for v in np.unique(blk(f,R,C)))
def cell_empty(f,R,C):
    if not(0<=R<16 and 0<=C<16): return False
    return uset(f,R,C)<= {0,1}
def av_cell(f):
    ys,xs=np.where((f==14)|(f==0))
    if not len(ys): return None
    return (int(ys.min())//4,int(xs.min())//4)
def grabbed_cell(f):
    for R in range(16):
        for C in range(16):
            u=uset(f,R,C)
            if 0 in u and 9 in u and 2 not in u and 14 not in u:
                return (R,C)
    return None
def box_cells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            u=uset(f,R,C)
            if 9 in u and (4 in u or 3 in u) and 2 not in u:
                out.add((R,C))
    return out
def socket_interior_cells(f):
    # cells that are entirely inside a 9-container interior: block is all 2, and has 9-neighbors forming a border region
    # approximate: block all-2 AND within bbox of a 9 component. We'll just detect all-2 cells adjacent (within 3) to 9.
    out=set()
    nine=np.argwhere(f==9)
    if len(nine)==0: return out
    for R in range(16):
        for C in range(16):
            u=uset(f,R,C)
            if u=={2}:
                out.add((R,C))
    return out
def bfs(f,start,goals,occ=cell_empty):
    goals=set(goals)
    if start in goals: return [],start
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        for a,(dr,dc) in DIRS.items():
            nb=(cur[0]+dr,cur[1]+dc)
            if nb in prev: continue
            if nb in goals or occ(f,nb[0],nb[1]):
                prev[nb]=(cur,a)
                if nb in goals:
                    path=[];n=nb
                    while prev[n]: p,ac=prev[n];path.append(ac);n=p
                    return path[::-1],nb
                q.append(nb)
    return None,None

def try_solve(env, prefer='both', budget=140, verbose=False):
    base=env.levels_completed
    def S(a):
        if env.terminal(): return False
        env.step(a); return True
    for step in range(budget):
        if env.terminal() or env.levels_completed>base: break
        f=env.frame(); av=av_cell(f); g=grabbed_cell(f)
        socks=socket_interior_cells(f)
        if g is not None:
            # push toward nearest socket interior: if grabbed adjacent to a 2-socket cell, push into it and release
            tgt=min(socks,key=lambda s:abs(s[0]-g[0])+abs(s[1]-g[1])) if socks else None
            if tgt is None: S(USE); continue
            if abs(g[0]-tgt[0])+abs(g[1]-tgt[1])==1:
                dr,dc=tgt[0]-g[0],tgt[1]-g[1]
                face={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}[(dr,dc)]
                S(face); S(USE); continue
            # move avatar so grabbed cell approaches socket: step avatar toward tgt if both cells traversable
            order=[]
            if tgt[0]!=g[0]: order.append(DOWN if tgt[0]>g[0] else UP)
            if tgt[1]!=g[1]: order.append(RIGHT if tgt[1]>g[1] else LEFT)
            done=False
            for a in order:
                dr,dc=DIRS[a]; nav=(av[0]+dr,av[1]+dc); ng=(g[0]+dr,g[1]+dc)
                if cell_empty(f,nav[0],nav[1]) and (cell_empty(f,ng[0],ng[1]) or ng in socks):
                    S(a); done=True; break
            if not done: S(USE)
            continue
        # empty-handed: grab nearest reachable box
        bs=list(box_cells(f))
        if prefer=='top': bs=[b for b in bs if b[0]<8]
        elif prefer=='bottom': bs=[b for b in bs if b[0]>=8]
        if not bs: S(USE); continue
        bs.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
        grabbed=False
        for target in bs[:4]:
            adj={}
            for a,(dr,dc) in DIRS.items():
                ac=(target[0]-dr,target[1]-dc)  # cell from which facing (dr,dc) points to target
                if cell_empty(f,ac[0],ac[1]): adj[ac]=a
            if not adj: continue
            path,reached=bfs(f,av,set(adj.keys()))
            if not path and av not in adj: continue
            if path: S(path[0])
            f2=env.frame(); av2=av_cell(f2)
            if av2 in adj:
                S(adj[av2]); S(USE)
            grabbed=True; break
        if not grabbed: S(USE)
    return env.levels_completed

if __name__=='__main__':
    from l8env import l8
    for pref in ['both','top','bottom']:
        env=l8()
        lvl=try_solve(env,prefer=pref,budget=140)
        f=env.frame(); int9=int((f[9:14,45:58]==9).sum())+int((f[49:58,49:58]==9).sum())
        print('prefer',pref,'-> lvl',lvl,'int9',int9,'steps',len(env.path)-466,'term',env.terminal())
