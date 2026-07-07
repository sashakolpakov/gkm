import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
from collections import deque
DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
def full_scan(f):
    walls=set();b4=set();b3=set();movers=set();cours=set();av=None;held=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 0 in u and 9 in u: held.add((R,C))
            if 15 in u: movers.add((R,C))
            if 12 in u: cours.add((R,C))
            if int((blk==5).sum())>=8: walls.add((R,C))
            if 9 in u and 4 in u and 0 not in u: b4.add((R,C))
            if 9 in u and 3 in u: b3.add((R,C))
    return av,walls,b4,b3,movers,cours,held
def c3px(f): return int((f==3).sum())
env=l8(); base=env.levels_completed; n0=len(env.path)
TOP=[(3,11),(3,12),(3,13),(2,11),(2,12),(2,13)]
BOT=[(13,12),(14,12),(12,12),(13,13),(14,13),(12,13)]
def next_tgt(b3):
    for t in TOP:
        if t not in b3: return t,'T'
    for t in BOT:
        if t not in b3: return t,'B'
    return None,None
for it in range(20):
    f=np.asarray(env.frame()); av,walls,b4,b3,movers,cours,held=full_scan(f)
    if env.levels_completed>base: print("WIN it",it); break
    tgt,side=next_tgt(b3)
    if tgt is None: print("all targets seated"); break
    # pick nearest grabbable box (color4) to tgt, prefer penned (static)
    src=sorted(b4,key=lambda b:abs(b[0]-tgt[0])+abs(b[1]-tgt[1]))
    ok=False
    for s in src[:4]:
        if legs.carry_box_to(env,s,tgt,cap=60):
            ok=True; break
    f=np.asarray(env.frame())
    print(f"it{it} tgt{tgt}{side} ok{ok} moves{len(env.path)-n0} c3px{c3px(f)} lvl{env.levels_completed}")
    if not ok:
        # nudge to escape / let movers move
        for _ in range(2): env.step(2)
print("FINAL lvl",env.levels_completed,"moves",len(env.path)-n0,"c3px",c3px(np.asarray(env.frame())))
