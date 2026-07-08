import l9env, legs
import numpy as np
def cell_is_box(f,R,C):
    blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
    return (9 in u and (4 in u or 3 in u) and 2 not in u)
def filled_center(f):
    return sum(1 for r in (3,4,5) for c in (5,6,7) if cell_is_box(f,r,c))
def ci2(f): return int((f[13:23,21:31]==2).sum())
order=[(4,6),(3,6),(5,6),(4,5),(4,7),(3,5),(3,7),(5,5),(5,7)]
env=l9env.get_l9(); base=env.levels_completed
for t in order:
    if env.terminal() or env.levels_completed>base: break
    f=np.asarray(env.frame())
    if cell_is_box(f,t[0],t[1]): 
        continue
    av,boxes,walls=legs._grid_scan(env)
    free=[b for b in boxes if not(3<=b[0]<=5 and 5<=b[1]<=7)]
    free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
    for b in free[:4]:
        if legs.carry_box_to(env,b,t,cap=16): break
    f=np.asarray(env.frame())
    print(f"tgt{t} filled{filled_center(f)} ci2{ci2(f)} steps{len(env.path)-588} n7{int((f==7).sum())} lvl{env.levels_completed}")
f=np.asarray(env.frame())
print("FINAL filled",filled_center(f),"ci2",ci2(f),"lvl",env.levels_completed,"term",env.terminal(),"steps",len(env.path)-588)
