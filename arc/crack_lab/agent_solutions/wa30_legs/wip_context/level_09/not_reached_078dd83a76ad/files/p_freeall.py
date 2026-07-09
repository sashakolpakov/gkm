import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)   # remove movers
base=env.levels_completed
# free all penned boxes into open top region (courier-reachable)
for b,d in [((2,1),(1,5)),((2,2),(1,6)),((3,1),(0,5)),((3,2),(0,6))]:
    legs.carry_box_to(env,b,d,cap=45)
print("after free-all moves",len(env.path)-466,"lvl",env.levels_completed)
# park middle band
legs._walk_avatar_to(env,(8,0),legs._obstacles_grid,30)
def fill(f): return int((f[8:16,44:60]==4).sum()),int((f[48:60,48:60]==4).sum())
mx=(0,0)
for i in range(120):
    if env.terminal(): print("crash",len(env.path)-466);break
    env.step(3 if i%2==0 else 4)
    if env.levels_completed>base: print("WIN",len(env.path)-466);break
    fs=fill(np.asarray(env.frame())); mx=(max(mx[0],fs[0]),max(mx[1],fs[1]))
print("maxfill",mx,"final",env.levels_completed)
