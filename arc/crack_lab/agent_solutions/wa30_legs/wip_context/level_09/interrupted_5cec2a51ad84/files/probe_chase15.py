import l9env, legs
import numpy as np
env=l9env.get_l9()
def c(col,f): return int((f==col).sum())
before=c(15,np.asarray(env.frame()))
legs.chase_and_clear(env,15,lambda m:True,cap=50)
f=np.asarray(env.frame())
print("c15 before",before,"after",c(15,f),"lvl",env.levels_completed,"term",env.terminal(),"steps used",len(env.path)-588)
print("movers15 now",legs._movers(env,15),"avatar",legs._grid_scan(env)[0])
