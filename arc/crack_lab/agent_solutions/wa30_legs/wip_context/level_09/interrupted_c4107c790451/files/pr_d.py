import l9lib as L
import numpy as np
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def av(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4)
e=L.fresh()
# go down to row9, right along row9 to col10, then up
seq=[DOWN, RIGHT, RIGHT, UP,UP,UP,UP,UP]
for a in seq:
    e.step(a); 
print('avatar', av(e))
# now try stepping up into strip
e.step(UP)
print('after UP', av(e))
e.step(UP)
print('after UP2', av(e))
