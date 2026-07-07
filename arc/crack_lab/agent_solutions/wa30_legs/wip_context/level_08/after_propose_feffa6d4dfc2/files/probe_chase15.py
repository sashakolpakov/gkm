import numpy as np, perception as P, legs
from l8env import l8

def cellof(f,color):
    ys,xs=np.where(f==color)
    if not len(ys): return None
    return (int(ys.min())//4,int(xs.min())//4)
def avcell(f):
    return cellof(f,14)
def m15cells(f):
    return [(int(b.bbox[0])//4,int(b.bbox[1])//4) for b in P.connected_components(f,colors=[15])]

env=l8()
# bring avatar into top arena near mover patrol: RIGHTx3 then UP into arena to row cell ~3
for a in [4,4,4,1,1,1,1,1]:  # up through gap to ~row cell 3
    env.step(a)
print('avatar cell', avcell(env.frame()), 'm15', m15cells(env.frame()))
# now reactive chase top mover (row<6)
n15_start=len(m15cells(env.frame()))
for t in range(60):
    f=env.frame(); a=avcell(f); ms=[m for m in m15cells(f) if m[0]<6]
    if not ms:
        print('top mover gone at t',t); break
    m=min(ms,key=lambda mm:abs(mm[0]-a[0])+abs(mm[1]-a[1]))
    dr=m[0]-a[0]; dc=m[1]-a[1]
    adj = abs(dr)+abs(dc)==1
    if adj:
        face={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}[(dr,dc)]
        env.step(face); env.step(5)
    else:
        # move toward
        if abs(dr)>=abs(dc): env.step(1 if dr<0 else 2)
        else: env.step(3 if dc<0 else 4)
    if env.terminal(): print('term',t); break
print('final m15', m15cells(env.frame()), 'lvl', env.levels_completed)
