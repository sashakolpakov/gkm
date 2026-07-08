import l9env, legs
import numpy as np
def cell_is_box(f,R,C):
    blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
    return (9 in u and (4 in u or 3 in u) and 2 not in u)
def fc(f): return sum(1 for r in (3,4,5) for c in (5,6,7) if cell_is_box(f,r,c))
# park at (0,15) top-right-ish left of wall, far from center
env=l9env.get_l9()
ok=legs._avatar_nav(env,(0,0),cap=30)
mx=0
for t in range(40):
    if env.terminal():break
    env.step(1)
    f=np.asarray(env.frame()); mx=max(mx,fc(f))
    if t%4==0: print(f"t{t} filled{fc(f)} max{mx} n7{int((f==7).sum())}")
print("max filled",mx)
