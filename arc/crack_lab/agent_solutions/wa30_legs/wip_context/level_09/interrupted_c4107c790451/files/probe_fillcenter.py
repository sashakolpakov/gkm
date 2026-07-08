import l9env, legs
import numpy as np
env=l9env.get_l9()
base=env.levels_completed
cells=[(3,5),(4,5),(5,5),(3,6),(4,6),(5,6),(3,7),(4,7),(5,7)]
def cb(env):
    av,boxes,walls=legs._grid_scan(env)
    return sum(1 for b in boxes if 3<=b[0]<=5 and 5<=b[1]<=7)
for t in cells:
    if env.terminal() or env.levels_completed>base: break
    av,boxes,walls=legs._grid_scan(env)
    free=[b for b in boxes if not(3<=b[0]<=5 and 5<=b[1]<=7)]
    if not free: 
        env.step(1); continue
    free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
    legs.carry_box_to(env,free[0],t,cap=25)
    print("target",t,"centerboxes",cb(env),"steps",len(env.path)-588,"lvl",env.levels_completed,"n7",int((np.asarray(env.frame())==7).sum()))
print("FINAL lvl",env.levels_completed,"term",env.terminal(),"steps",len(env.path)-588)
