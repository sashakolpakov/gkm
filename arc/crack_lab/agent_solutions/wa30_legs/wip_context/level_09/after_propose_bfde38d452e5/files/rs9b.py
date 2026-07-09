from probe9 import *
from legs import carry_box_to
import random, itertools
SOCK8=[(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
def sf(env):
    f=np.asarray(env.frame())
    return sum(1 for s in SOCK8 if (lambda u:9 in u and 4 in u and 2 not in u)(set(int(v) for v in np.unique(f[s[0]*4:s[0]*4+4,s[1]*4:s[1]*4+4]))))
WEST=[(3,2),(5,1),(5,3),(7,1),(7,2),(8,1)]
RIGHT=[(5,11),(7,12),(8,14)]
# avatar targets: cells courier is bad at (east + relay-block); leave west col to courier
ATARGETS=[(3,6),(3,7),(4,7),(5,6),(5,7)]
def run(plan):
    env=fresh(); base=env.levels_completed; mx=0
    for b,s in plan:
        if env.terminal() or env.levels_completed>base: break
        try: carry_box_to(env,b,s)
        except Exception: break
        mx=max(mx,sf(env))
        if env.levels_completed>base: return True,mx
    while not env.terminal() and env.levels_completed<=base and c7(env)>0:
        env.step(5); mx=max(mx,sf(env))
        if env.levels_completed>base: return True,mx
    return False, mx
best=0; random.seed(7)
allboxes=WEST+RIGHT
for it in range(120):
    k=random.choice([3,4,4,4])
    cells=random.sample(ATARGETS,min(k,len(ATARGETS)))
    # assign boxes preferring right boxes go to east-ish cells
    bs=random.sample(allboxes,len(cells))
    plan=list(zip(bs,cells))
    won,mx=run(plan)
    if won: print('WIN!! plan',plan); break
    if mx>best: best=mx; print('it',it,'mx',mx,'plan',plan)
print('best simultaneous sockets',best)
