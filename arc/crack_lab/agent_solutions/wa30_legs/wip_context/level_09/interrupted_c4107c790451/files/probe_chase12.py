import l9env, legs
import numpy as np
env=l9env.get_l9()
def c(col,f): return int((f==col).sum())
# left c12 roams cells ~ rows3-7 cols3-6. band_pred: cell col<9 (left side) and it's the moving one
# but there are two c12; stationary at (1,11). Use band left: col<9
band=lambda m: m[1]<9 and m[0]<9
before=c(12,np.asarray(env.frame()))
legs.chase_and_clear(env,12,band,cap=40)
f=np.asarray(env.frame())
print("c12 before",before,"after",c(12,f),"lvl",env.levels_completed,"term",env.terminal(),"steps",len(env.path))
print("movers12 now",legs._movers(env,12))
