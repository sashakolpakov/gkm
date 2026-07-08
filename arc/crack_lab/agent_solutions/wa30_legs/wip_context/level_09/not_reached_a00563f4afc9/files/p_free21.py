import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
base=env.levels_completed
# free (2,1) -> deliver to open top region (1,6)
ok=legs.carry_box_to(env,(2,1),(1,6),cap=45)
print("free (2,1) ok",ok,"level8 moves",len(env.path)-466)
# park avatar low-left out of way and yield till crash
for a in [2,3,3,3]:
    if not env.terminal(): env.step(a)
won=False
for i in range(200):
    if env.terminal(): print("term/crash at level8 moves",len(env.path)-466);break
    env.step(3 if i%2==0 else 4)
    if env.levels_completed>base: print("WIN level8 moves",len(env.path)-466);won=True;break
print("final lvl",env.levels_completed)
