import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from legs import *
env = A.Arena('r11l')
ck = json.load(open('checkpoint.json'))
for act in ck['final_path']:
    env.step(act)
print("level", env.levels_completed)
syss = ring_systems(env)
for s in syss:
    print(s)
print("active_pos", active_pos(env))
