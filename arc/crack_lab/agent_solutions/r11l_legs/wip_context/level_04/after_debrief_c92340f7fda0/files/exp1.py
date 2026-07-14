import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from legs import _arr, click
import numpy as np
def load():
    env = A.Arena('r11l')
    ck = json.load(open('checkpoint.json'))
    for act in ck['final_path']:
        env.step(act)
    return env

env=load()
f0=_arr(env).copy()
# active at (20,23). move it to (20,30)
c=env.clone()
click(c, 20, 30)
f1=_arr(c)
d=np.argwhere(f0!=f1)
print("clicked (20,30) -> changed cells:", len(d))
for r,cc in d:
    print((int(r),int(cc)), int(f0[r,cc]),'->',int(f1[r,cc]))
