import l9env, legs
import numpy as np
# candidate drop cells: keypad cells (row3, cols10-15), right container frames, center edges
env0=l9env.get_l9()
av,boxes,walls=legs._grid_scan(env0)
print("boxes",sorted(boxes))
# nearest box to avatar
targets=[(3,10),(3,11),(3,12),(3,13),(3,14),(3,15),  # keypad
         (2,13),(2,14),(6,13),(6,14),                # right containers
         (3,6),(4,6),(5,6),(3,5),                     # center container interior
         (8,2),(7,2)]                                 # 2222 payload area
for box in [(8,14),(7,12),(5,11)]:  # right boxes (safe from c12)
    for t in targets:
        env=env0.clone()
        base=env.levels_completed
        ok=legs.carry_box_to(env,box,t,cap=40)
        if env.levels_completed>base:
            print("WIN! box",box,"->",t,"lvl",env.levels_completed)
        elif ok:
            pass
    print("done box",box)
