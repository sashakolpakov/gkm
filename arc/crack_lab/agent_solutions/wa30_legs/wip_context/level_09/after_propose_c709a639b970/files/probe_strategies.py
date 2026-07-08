import l9env, legs
import numpy as np
def run(name, fn):
    env=l9env.get_l9(); base=env.levels_completed
    fn(env)
    print(name,"lvl",env.levels_completed,"term",env.terminal(),"steps",len(env.path)-588)

def yield_all(env):
    while not env.terminal() and env.levels_completed<=8:
        env.step(1)

# C: chase c15 then yield
def C(env):
    legs.chase_and_clear(env,15,lambda m:True,cap=45)
    yield_all(env)
run("C chase15+yield",C)

# A: chase c15 then fill center then yield
def A(env):
    legs.chase_and_clear(env,15,lambda m:True,cap=45)
    cells=[(4,6),(4,5),(4,7),(3,6),(5,6)]
    for t in cells:
        if env.terminal() or env.levels_completed>8: break
        av,boxes,walls=legs._grid_scan(env)
        free=[b for b in boxes if not(3<=b[0]<=5 and 5<=b[1]<=7)]
        if not free: env.step(1); continue
        free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
        legs.carry_box_to(env,free[0],t,cap=15)
    yield_all(env)
run("A chase15+fillcenter",A)

# D: deliver all boxes into center then yield (no chase)
def D(env):
    cells=[(4,6),(4,5),(4,7),(3,6),(5,6),(3,5),(5,5),(3,7),(5,7)]
    for t in cells:
        if env.terminal() or env.levels_completed>8: break
        av,boxes,walls=legs._grid_scan(env)
        free=[b for b in boxes if not(3<=b[0]<=5 and 5<=b[1]<=7)]
        if not free: env.step(1); continue
        free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
        legs.carry_box_to(env,free[0],t,cap=12)
    yield_all(env)
run("D fillcenter+yield",D)
