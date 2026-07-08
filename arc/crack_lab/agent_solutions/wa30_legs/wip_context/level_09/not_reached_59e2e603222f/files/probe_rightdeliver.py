import l9env, legs
import numpy as np
env=l9env.get_l9()
def kp2(f): return int((f[12:16,40:64]==2).sum())
def loose(env): return len(legs._grid_scan(env)[1])
print("start loose",loose(env),"keypad2",kp2(np.asarray(env.frame())))
# deliver right box (8,14) onto right container lower cell region (7,13) just below container(6,x)
# actually try to place onto the container interior by targeting cell (2,13)/(6,13)
for box,tgt in [((8,14),(7,14)),((7,12),(7,13)),((5,11),(4,11))]:
    b=env.levels_completed
    ok=legs.carry_box_to(env,box,tgt,cap=30)
    f=np.asarray(env.frame())
    print(f"box{box}->{tgt} ok{ok} loose{loose(env)} keypad2{kp2(f)} lvl{env.levels_completed} steps{len(env.path)-588}")
