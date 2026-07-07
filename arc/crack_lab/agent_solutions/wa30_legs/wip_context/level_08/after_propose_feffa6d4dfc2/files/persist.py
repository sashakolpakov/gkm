import numpy as np
from collections import deque
from ctl import *
from l8env import l8
env=l8()
# Navigate avatar to bottom arena: right to gap cols36-43, down, to near bottom socket
f=env.frame()
def nav(env,goalset):
    for _ in range(60):
        f=env.frame(); a=av_cell(f)
        if a in goalset: return True
        path,_=bfs(f,a,goalset)
        if not path: return False
        env.step(path[0])
    return av_cell(env.frame()) in goalset
# goal: cell just left of bottom socket. socket left border col48=cell12. left-adjacent cell11 (cols44-47). pick rows 49..58 -> cell rows 12..14
goals={(r,11) for r in range(12,15)}
ok=nav(env,goals)
print('nav to socket-left ok',ok,'av',av_cell(env.frame()),'steps',len(env.path)-466)
# find a box adjacent-ish; look for any box in bottom arena reachable, grab and push into socket
def bot_boxes(f): return [b for b in boxes(f) if b[0]>=11]
f=env.frame()
print('bottom boxes',sorted(bot_boxes(f)))
# region view
for r in range(44,60):
    print(''.join('%2d'%int(f[r,c]) for c in range(40,62)))
print('--- now grab & push ---')
def viz(env,tag):
    f=env.frame(); print(tag,'int9',int((f[49:58,49:58]==9).sum()),'lvl',env.levels_completed,'av',av_cell(f))
    for r in range(44,60):
        print(''.join('%2d'%int(f[r,c]) for c in range(40,62)))
env.step(5)  # grab highlighted box(12,12)
viz(env,'after grab')
env.step(4)  # push right into interior
viz(env,'after push right')
env.step(5)  # release
viz(env,'after release')
# move avatar far away (left) and idle a bit, check persistence
for a in [3,3,3,1,1]: env.step(a)
for _ in range(8): env.step(5)
viz(env,'after leave+idle')
