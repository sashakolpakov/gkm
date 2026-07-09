import l9env, legs
import numpy as np
env=l9env.get_l9()
legs.chase_and_clear(env,15,lambda m:True,cap=50)
f=np.asarray(env.frame())
print("after clear: n7",int((f==7).sum()),"steps",len(env.path)-588)
for t in range(8):
    env.step(1)  # idle UP
    f=np.asarray(env.frame())
    print(f"idle{t} n7={int((f==7).sum())} c15={int((f==15).sum())} term={env.terminal()} lvl={env.levels_completed}")
