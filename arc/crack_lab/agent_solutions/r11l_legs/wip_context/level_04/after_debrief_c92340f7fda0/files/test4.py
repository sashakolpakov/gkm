import sys, json, time
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from legs import multicolor_systems, box_center
import players
def load():
    env=A.Arena('r11l'); ck=json.load(open('checkpoint.json'))
    for a in ck['final_path']: env.step(a)
    return env
env=load()
print("systems:")
for s in multicolor_systems(env):
    print(s)
print("start level", env.levels_completed)
t=time.time()
players.play_level_4(env)
print("after level", env.levels_completed, "elapsed", round(time.time()-t,1))
for col in (12,11,8):
    print("box",col,"->",box_center(env,col))
