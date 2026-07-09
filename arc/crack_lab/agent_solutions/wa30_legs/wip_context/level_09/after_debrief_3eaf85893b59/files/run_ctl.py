import numpy as np, ctl
from ctl import *
from l8env import l8
env=l8()
def nearest_socket_cell(f, cell, socks):
    if not socks: return None
    return min(socks, key=lambda s: abs(s[0]-cell[0])+abs(s[1]-cell[1]))
maxint=0
for step in range(140):
    if env.terminal() or env.levels_completed>7:
        break
    f=env.frame()
    g=grabbed_cell(f)
    av=av_cell(f)
    socks=sockets_interior(f)
    maxint=max(maxint,interior9(f))
    if g is not None:
        # carry toward nearest socket: move so grabbed cell is adjacent to a socket cell, then push+release
        target=nearest_socket_cell(f,g,socks)
        if target is None:
            env.step(USE); continue
        # if grabbed adjacent to a socket cell, push toward it and release
        if abs(g[0]-target[0])+abs(g[1]-target[1])==1:
            dr,dc=target[0]-g[0],target[1]-g[1]
            face={(-1,0):UP,(1,0):DOWN,(0,-1):LEFT,(0,1):RIGHT}[(dr,dc)]
            env.step(face)  # push into socket
            env.step(USE)   # release
            continue
        # else move avatar toward target (greedy), keeping simple
        dr=np.sign(target[0]-av[0]); dc=np.sign(target[1]-av[1])
        moved=False
        for a in ([DOWN if dr>0 else UP] if dr!=0 else [])+([RIGHT if dc>0 else LEFT] if dc!=0 else []):
            nb=(av[0]+DIRS[a][0],av[1]+DIRS[a][1])
            gb=(g[0]+DIRS[a][0],g[1]+DIRS[a][1])
            if cellfree(f,nb[0],nb[1]) and (cellfree(f,gb[0],gb[1]) or (gb in socks)):
                b=env.frame(); env.step(a); moved=True; break
        if not moved:
            env.step(USE)  # drop if stuck
    else:
        bs=[b for b in boxes(f) if b not in socks]
        if not bs:
            env.step(USE); continue
        # nearest box; find free adjacent cell to grab
        bs.sort(key=lambda b: abs(b[0]-av[0])+abs(b[1]-av[1]))
        target=bs[0]
        adj=[(target[0]+dr,target[1]+dc,a) for a,(dr,dc) in {UP:(1,0),DOWN:(-1,0),LEFT:(0,1),RIGHT:(0,-1)}.items()]
        goals={(r,c) for r,c,a in adj if cellfree(f,r,c)}
        if not goals:
            env.step(USE); continue
        path,reached=bfs(f,av,goals)
        if not path:
            # box unreachable; try next
            env.step(USE); continue
        if path:
            env.step(path[0])
            # if now adjacent, face+grab
            f2=env.frame(); av2=av_cell(f2)
            for r,c,a in adj:
                if av2==(r,c):
                    env.step(a); env.step(USE); break
    if step%10==0:
        print('step',step,'lvl',env.levels_completed,'int9',interior9(env.frame()),'nboxes',len(boxes(env.frame())))
print('DONE lvl',env.levels_completed,'maxint9',maxint,'steps',len(env.path)-466)
