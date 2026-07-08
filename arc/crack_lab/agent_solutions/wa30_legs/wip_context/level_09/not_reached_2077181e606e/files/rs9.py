from probe9 import *
from legs import carry_box_to
import random
SOCK8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
def sfill(env):
    f=np.asarray(env.frame())
    return sum(1 for s in SOCK8 if (lambda u:9 in u and 4 in u and 2 not in u)(set(int(v) for v in np.unique(f[s[0]*4:s[0]*4+4,s[1]*4:s[1]*4+4]))))
BOXES=[(3,2),(5,1),(5,3),(5,11),(7,1),(7,2),(7,12),(8,1),(8,14)]
DEEP=[(3,5),(3,6),(3,7),(4,7),(5,6),(5,7)]
def run_plan(plan, extra_yield=True):
    env=fresh(); base=env.levels_completed; mx=sfill(env); won=False
    for b,s in plan:
        if env.terminal() or env.levels_completed>base: break
        try: carry_box_to(env,b,s)
        except Exception: break
        c=sfill(env); mx=max(mx,c)
        if env.levels_completed>base: won=True; break
    if extra_yield and not won:
        while not env.terminal() and env.levels_completed<=base and c7(env)>0:
            env.step(5); mx=max(mx,sfill(env))
            if env.levels_completed>base: won=True; break
    return mx, won, env.levels_completed
best=(0,None)
random.seed(1)
for it in range(40):
    k=random.choice([3,3,4,4])
    bs=random.sample(BOXES,k)
    cells=random.sample(DEEP,k)
    plan=list(zip(bs,cells))
    mx,won,lvl=run_plan(plan)
    if won: print('WON!',plan); break
    if mx>best[0]: best=(mx,plan); print('it',it,'mx',mx,'plan',plan)
print('BEST',best)
