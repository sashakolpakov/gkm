import l9env, legs
import numpy as np
env=l9env.get_l9(); base=env.levels_completed
def r2(f):
    return int((f[9:11,53:59]==2).sum()), int((f[25:27,53:59]==2).sum())
# fill bottom-right then top-right, using nearest boxes
targets=[(6,14),(6,13),(2,14),(2,13)]
for t in targets:
    if env.terminal() or env.levels_completed>base: break
    av,boxes,walls=legs._grid_scan(env)
    free=[b for b in boxes]
    free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
    ok=False
    for b in free[:5]:
        if legs.carry_box_to(env,b,t,cap=22):
            ok=True; break
    f=np.asarray(env.frame())
    print(f"seat {t} ok{ok} r2{r2(f)} lvl{env.levels_completed} steps{len(env.path)-588} n7{int((f==7).sum())}")
print("FINAL lvl",env.levels_completed,"term",env.terminal())
