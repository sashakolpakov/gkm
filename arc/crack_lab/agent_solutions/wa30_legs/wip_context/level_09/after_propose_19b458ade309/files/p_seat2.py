import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8(); base=env.levels_completed; n0=len(env.path)
def c3(f): return int((f==3).sum())
targets=[(3,11),(3,12),(3,13),(2,11)]
srcs=[(2,2),(2,1),(3,2),(3,1)]  # penned
for src,tgt in zip(srcs,targets):
    ok=legs.carry_box_to(env,src,tgt,cap=80)
    f=np.asarray(env.frame())
    print(f"seat {src}->{tgt} ok={ok} moves={len(env.path)-n0} lvl={env.levels_completed} c3px={c3(f)}")
    if env.levels_completed>base: print("WIN"); break
f=np.asarray(env.frame())
print("TOP:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
