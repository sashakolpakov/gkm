import l9env, legs
import numpy as np
def ci2(f): return int((f[13:23,21:31]==2).sum())   # center interior 2
def rt2(f): return int((f[9:11,53:59]==2).sum())     # top-right container interior
def rb2(f): return int((f[25:27,53:59]==2).sum())    # bottom-right container interior
# seat into center
env=l9env.get_l9()
f=np.asarray(env.frame()); print("start center_i2",ci2(f),"rtop",rt2(f),"rbot",rb2(f))
av,boxes,walls=legs._grid_scan(env)
free=sorted(boxes,key=lambda b:abs(b[0]-4)+abs(b[1]-6))
ok=legs.carry_box_to(env,free[0],(4,6),cap=20)
f=np.asarray(env.frame()); print("after seat center (4,6): ok",ok,"center_i2",ci2(f),"steps",len(env.path)-588)
# seat into right-top container
env=l9env.get_l9()
ok=legs.carry_box_to(env,(8,14),(2,14),cap=25)
f=np.asarray(env.frame()); print("after seat rtop (2,14): ok",ok,"rtop",rt2(f),"steps",len(env.path)-588)
ok2=legs.carry_box_to(env,(7,12),(2,13),cap=25)
f=np.asarray(env.frame()); print("after seat rtop (2,13): ok",ok2,"rtop",rt2(f),"steps",len(env.path)-588)
print("term",env.terminal(),"lvl",env.levels_completed)
