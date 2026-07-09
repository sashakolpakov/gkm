import l9lib as L
import numpy as np
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def mv(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==15)
    return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
def av(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4)
e=L.fresh()
# walk avatar to (9,3) early: (8,8)->(9,8) DOWN, LEFT x5 -> (9,3)
for a in [DOWN]+[LEFT]*5:
    e.step(a)
print('avatar', av(e))  # expect (9,3), facing LEFT
# face down: press DOWN would move into (10,3)! (10,3) open. So wiggle: DOWN moves us in. 
# Instead spam USE and see if kill works regardless of facing.
n=6
for i in range(40):
    n+=1
    e.step(USE)
    m=mv(e)
    if i>30 or m is None or (m and abs(m[0]-9)+abs(m[1]-3)<=2):
        print('step',n,'mover',m,'avatar',av(e))
    if m is None: break
