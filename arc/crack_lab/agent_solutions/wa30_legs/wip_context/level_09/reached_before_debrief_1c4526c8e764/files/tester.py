from probe6 import fresh_at_level6
from nav import cellmap, avc, pos
import numpy as np
def run(seq, show=True):
    env=fresh_at_level6(); c=env.clone()
    for a in seq:
        if c.terminal(): break
        c.step(a)
    f=np.asarray(c.frame())
    if show:
        print('moves',len(seq),'lvl',c.levels_completed,'term',c.terminal(),'A',avc(f),'T',pos(f,15))
        print(cellmap(f))
    return c, f
