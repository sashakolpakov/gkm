import l9env, legs
import numpy as np
def ci2(e): 
    f=np.asarray(e.frame()); return int((f[13:23,21:31]==2).sum())
def cell2(f,R,C): return int((f[R*4:R*4+4,C*4:C*4+4]==2).sum())
env=l9env.get_l9(); base=env.levels_completed
cells=[(r,c) for r in (3,4,5) for c in (5,6,7)]
for it in range(12):
    if env.terminal() or env.levels_completed>base:break
    f=np.asarray(env.frame())
    # target center cell with most 2 remaining
    open_cells=[t for t in cells if cell2(f,t[0],t[1])>0]
    if not open_cells: break
    open_cells.sort(key=lambda t:-cell2(f,t[0],t[1]))
    av,boxes,walls=legs._grid_scan(env)
    free=[b for b in boxes if not(3<=b[0]<=5 and 5<=b[1]<=7)]
    if not free: env.step(1); continue
    t=open_cells[0]
    free.sort(key=lambda b:abs(b[0]-t[0])+abs(b[1]-t[1]))
    legs.carry_box_to(env,free[0],t,cap=14)
    print(f"it{it} tgt{t} ci2{ci2(env)} steps{len(env.path)-588} n7{int((np.asarray(env.frame())==7).sum())} lvl{env.levels_completed}")
print("FINAL ci2",ci2(env),"lvl",env.levels_completed,"term",env.terminal(),"steps",len(env.path)-588)
