import sys
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import arr, frame_delta
from collections import Counter

def probe(env):
    for _ in range(5): env.step(6, 5, 32)
    bf = arr(env.frame()).copy()
    cl = env.clone(); cl.step(6,20,17)
    af = arr(cl.frame())
    ys,xs = (bf!=af).nonzero()
    outside = [(int(y),int(x),int(bf[y,x]),int(af[y,x])) for y,x in zip(ys,xs) if not (11<=y<=51 and 11<=x<=51)]
    print("changes outside panel:", outside)
    # counter over repeated clicks
    for n in (1,2,3,5,10,20,40,80):
        cl = env.clone()
        for _ in range(n): cl.step(6,20,17)
        a2 = arr(cl.frame())
        d = [(int(y),int(x),int(bf[y,x]),int(a2[y,x])) for y,x in zip(*(bf!=a2).nonzero()) if not (11<=int(y)<=51 and 11<=int(x)<=51)]
        print("n=%d outside=%s lvl=%d term=%s" % (n, d, cl.levels_completed, cl.terminal()))
    # click on tokens / other spots
    spots = {"tok(17,23)":(23,17),"tok(26,17)":(17,26),"bracket":(34,25),
             "panel_edge_top":(30,12),"panel_edge_bot":(30,50),"left_of_row0":(17,17),
             "right_of_row0":(44,17),"above_col2":(23,14),"below_col2":(23,49),
             "above_col6":(35,14),"below_col6":(35,49),"row9_left":(14,44),"row9_right":(48,44),
             "row0_far_left":(14,17),"row0_far_right":(48,17)}
    for name,(x,y) in spots.items():
        c = env.clone(); c.step(6,x,y)
        d = frame_delta(bf, c.frame())
        print("click", name, (x,y), "delta", d["count"], d["bbox"])
A.run_program("lp85", probe)
