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
def boxes(f):
    out={}
    for r,c in np.argwhere(f==6):
        blk=f[max(0,r-2):r+3,max(0,c-2):c+3]
        cols=set(int(v) for v in blk.flatten())-{5,6}
        # only real boxes: those with a 2-color fill
        out[(int(r),int(c))]=sorted(cols)
    return out

env=load()
f0=_arr(env)
print("initial boxes:", boxes(f0))
endpoints=[(20,23),(6,39),(36,17),(36,46),(47,10),(52,27),(52,46)]
for ep in endpoints:
    c=env.clone()
    if ep!=(20,23):
        click(c,ep[0],ep[1])  # select
    b_before=boxes(_arr(c))
    ap=active_pos(c)
    # move it by +6 col
    click(c, ap[0], ap[1]+6)
    b_after=boxes(_arr(c))
    # find which box moved
    moved=[]
    ba_centers=list(b_after.keys())
    for k in b_before:
        if k not in b_after:
            moved.append(k)
    print(f"ep {ep}: box centers before {sorted(b_before.keys())} after {sorted(b_after.keys())}")
