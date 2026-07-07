# Determine top-container capacity by force-seating boxes via courier handoff.
import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
from collections import deque
DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
def scan(f):
    walls=set();boxes=set();cont=set();movers=set();cours=set();av=None;held=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 0 in u and 9 in u: held.add((R,C))
            if 15 in u: movers.add((R,C))
            if 12 in u: cours.add((R,C))
            if int((blk==5).sum())>=8: walls.add((R,C))
            if 9 in u and 4 in u and 2 not in u and 0 not in u: boxes.add((R,C))
            elif (9 in u and 2 in u) or (2 in u and 4 not in u and 9 not in u): cont.add((R,C))
    return av,walls,boxes,cont,movers,cours,held
TOP={(r,c) for r in (2,3) for c in (11,12,13,14)}
BOT={(r,c) for r in (12,13,14) for c in (12,13,14)}
env=l8()
# just report the container cell classification
f=np.asarray(env.frame());av,walls,boxes,cont,movers,cours,held=scan(f)
print("TOP cells classified cont?",{c:(c in cont) for c in TOP})
print("BOT cells classified cont?",{c:(c in cont) for c in BOT})
print("boxes",sorted(boxes))
