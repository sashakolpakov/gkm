import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from legs import _arr, click, active_pos
import numpy as np
def load():
    env = A.Arena('r11l')
    ck = json.load(open('checkpoint.json'))
    for act in ck['final_path']:
        env.step(act)
    return env
env=load()
f0=_arr(env).copy()
for pt in [(6,39),(36,17),(36,46),(47,10),(52,27),(52,46)]:
    c=env.clone()
    click(c, pt[0], pt[1])
    f1=_arr(c)
    d=np.argwhere(f0!=f1)
    ap=active_pos(c)
    print(f"click {pt}: {len(d)} changed, active_pos now {ap}")
