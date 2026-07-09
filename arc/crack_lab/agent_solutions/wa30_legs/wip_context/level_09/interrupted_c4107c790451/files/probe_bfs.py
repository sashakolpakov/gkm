import l9env, perception as P
import numpy as np
env=l9env.get_l9()
base=env.levels_completed
def key(e):
    f=np.asarray(e.frame())
    # coarse: avatar cell + downsample to 16x16 dominant color
    ds=f[::4,::4]
    return (ds.tobytes(),)
def goal(e,path):
    return e.levels_completed>base
path=P.bounded_bfs(env,goal,key_fn=key,max_states=40000,max_depth=13)
print("bfs path:",path)
