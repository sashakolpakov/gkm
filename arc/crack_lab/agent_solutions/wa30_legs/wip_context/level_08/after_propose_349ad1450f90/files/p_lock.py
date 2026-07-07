import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
def c3loc(f):
    ys,xs=np.where(f==3); return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
# test releasing a box on various target cells; check if color-3 persists 3 steps later
tests=[(3,13),(2,13),(3,14),(12,13),(13,13),(14,13),  # container interiors
        (4,1),(4,2),(2,3),(3,3),  # top-left fence cells
        (12,3),(13,4),(14,5)]     # bottom-left 3x3 cells
for tgt in tests:
    env=l8()
    for a in path[:20]: env.step(a)  # movers off
    # grab a penned box and carry to tgt
    ok=legs.carry_box_to(env,(2,2),tgt,cap=45)
    if not ok:
        # try other penned
        ok=legs.carry_box_to(env,(2,1),tgt,cap=45)
    f=np.asarray(env.frame()); imm=c3loc(f)
    # step away (avatar move) and check persistence
    for _ in range(3): env.step(2)
    f2=np.asarray(env.frame()); pers=c3loc(f2)
    print(f"tgt{tgt} ok{ok} c3_immediate{imm} c3_after3{pers}")
