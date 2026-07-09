import l9env, perception as P
import numpy as np
env=l9env.get_l9()
for t in range(0,160,10):
    f=np.asarray(env.frame())
    cc=P.color_counts(f)
    print(f"t{t} lvl{env.levels_completed} c2={cc.get(2,0)} c4={cc.get(4,0)} c9={cc.get(9,0)} c12={cc.get(12,0)} c15={cc.get(15,0)} c7={cc.get(7,0)}")
    for _ in range(10):
        env.step(5)
