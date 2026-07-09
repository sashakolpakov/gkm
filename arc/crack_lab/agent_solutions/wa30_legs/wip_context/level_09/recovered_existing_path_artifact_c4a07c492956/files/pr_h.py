import l9lib as L
import numpy as np
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def mv(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==15)
    return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
def av(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4)

# V1: idle near (8,3), step DOWN into (9,3) at step 40, USE at 41
e=L.fresh()
# path to (8,3): DOWN (9,8), LEFT*4 (9,4), UP (8,4), LEFT (8,3): 7 steps
for a in [DOWN]+[LEFT]*4+[UP,LEFT]:
    e.step(a)
print('avatar', av(e), 'mover', mv(e))
n=7
# idle with UP/DOWN wiggle? UP would move to (7,3) open; wiggle up/down keeps us near.
# use LEFT/RIGHT wiggle: (8,2) open? row8 col2 has 2-pad... avatar on pad? try RIGHT/LEFT wiggle
while n < 39:
    e.step(RIGHT); n+=1   # (8,4)
    if n>=39: break
    e.step(LEFT); n+=1    # (8,3)
print('pre', n, av(e), mv(e))
# ensure at (8,3) at n=39
if av(e)!=(8,3):
    e.step(LEFT); n+=1
e.step(DOWN); n+=1  # step 40: move into (9,3), mover should be at (10,3)
print('after down', n, av(e), mv(e))
e.step(USE); n+=1
print('after use', n, av(e), mv(e))
for i in range(3):
    e.step(USE); n+=1
    print(n, av(e), mv(e))
print('--- descend to adjacency')
e.step(DOWN); n+=1
print(n, av(e), mv(e))
e.step(USE); n+=1
print('after use', n, av(e), mv(e))
e.step(USE); n+=1
print('after use2', n, av(e), mv(e))
