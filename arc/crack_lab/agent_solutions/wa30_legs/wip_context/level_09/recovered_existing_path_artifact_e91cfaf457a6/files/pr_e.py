import l9lib as L
import numpy as np
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def av(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4)
def cellsig(e,R,C):
    f=np.asarray(e.frame()); blk=f[R*4:R*4+4,C*4:C*4+4]
    return sorted(set(int(v) for v in blk.ravel()))
e=L.fresh()
# nav to below box (5,11): stand (6,11), face UP, grab
for a in [DOWN,RIGHT,RIGHT,RIGHT,UP,UP,UP]:
    e.step(a)
print('avatar',av(e), 'box sig', cellsig(e,5,11))
e.step(UP); e.step(USE)  # face box? avatar at (6,11) after? check
print('after grab: avatar',av(e), 'box', cellsig(e,5,11))
# carry up: avatar (6,11)->(5,11)? box to (4,11)
e.step(UP)
print('avatar',av(e), '4,11', cellsig(e,4,11), '5,11', cellsig(e,5,11))
e.step(UP)  # box would push into fence (3,11)
print('avatar',av(e), '3,11', cellsig(e,3,11), '4,11', cellsig(e,4,11))
e.step(UP)
print('avatar',av(e), '2,11', cellsig(e,2,11), '3,11', cellsig(e,3,11))
e.step(USE)  # release on fence
print('released:', cellsig(e,3,11))
import numpy as np
for i in range(16):
    e.step(DOWN if i%2 else UP)
    f=np.asarray(e.frame()); ys,xs=np.where(f==12)
    cs=sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
    print(i, 'couriers', cs, 'fence box', cellsig(e,3,11), 'socks', cellsig(e,2,13), cellsig(e,2,14))
