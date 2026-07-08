import l9env, legs
import numpy as np
env=l9env.get_l9()
def center_boxes(env):
    av,boxes,walls=legs._grid_scan(env)
    return sum(1 for b in boxes if 3<=b[0]<=5 and 5<=b[1]<=7)
def full2(f,G=16):
    return sum(1 for R in range(G) for C in range(G) if np.all(f[R*4:R*4+4,C*4:C*4+4]==2))
for t in range(0,66,3):
    f=np.asarray(env.frame())
    print(f"t{t} centerboxes={center_boxes(env)} full2tiles={full2(f)} loose={len(legs._grid_scan(env)[1])} lvl{env.levels_completed}")
    for _ in range(3):
        if env.terminal():break
        env.step(1)
