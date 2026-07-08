import probe7, legs
import numpy as np

env=probe7.get_env_at_L7()
LEFT=[(8,3),(7,3)]

def boxes(e): return sorted(legs._grid_scan(e)[1])
def hasbox(e,c):
    f=np.asarray(e.frame()); R,C=c; u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
    return 9 in u and (4 in u or 3 in u)

c=env.clone()
for it in range(20):
    if c.levels_completed>6:
        print('WON iter',it); break
    empty=[t for t in LEFT if not hasbox(c,t)]
    if not empty:
        print('both left filled at iter',it,'lc',c.levels_completed)
        # nudge / idle to check win
        for _ in range(5):
            c.step(legs.USE)
            if c.levels_completed>6: print('WON after idle'); break
        break
    bx=boxes(c)
    # pick a box not already on a left cell
    src=[b for b in bx if b not in LEFT]
    if not src:
        c.step(legs.USE); continue
    tgt=empty[0]
    # nearest box to target
    src.sort(key=lambda b: abs(b[0]-tgt[0])+abs(b[1]-tgt[1]))
    ok=legs.carry_box_to(c,src[0],tgt)
    print('iter',it,'moved',src[0],'->',tgt,'ok',ok,'lc',c.levels_completed,'boxes',boxes(c))
print('final lc',c.levels_completed)
