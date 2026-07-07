import sys, json
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def setup():
    env = A.Arena('ls20'); env.reset()
    ck = json.load(open("checkpoint.json"))
    for a in ck["final_path"]: env.step(a)
    return env
base=setup()

def avatar_pos(f):
    ys,xs=np.where(f==0)  # avatar core = color 0
    return (round(ys.mean(),1),round(xs.mean(),1)) if len(ys) else None
def tile_pos(f):
    # the 5x5 tile: find the 99999 block (3x5 of color 9) -- use color9 centroid
    ys,xs=np.where(f==9)
    return (round(ys.mean(),1),round(xs.mean(),1),len(ys)) if len(ys) else None

f0=base.frame()
print("BASE avatar(col0)=",avatar_pos(f0)," tile(col9)=",tile_pos(f0))
for act in (1,2,3,4):
    c=base.clone(); f=c.step(act)
    print(f"act{act}: avatar={avatar_pos(f)} tile9={tile_pos(f)} lvl={c.levels_completed}")

# two-step: apply act then act again
print("\n-- double steps --")
for act in (1,2,3,4):
    c=base.clone(); c.step(act); f=c.step(act)
    print(f"act{act}x2: avatar={avatar_pos(f)} tile9={tile_pos(f)}")
