import l9lib as L
import numpy as np
UP,DOWN,LEFT,RIGHT,USE=1,2,3,4,5
def mv(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==15)
    return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
def av(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4)
def seated(e):
    f=np.asarray(e.frame()); SOCK=[(2,13),(2,14),(6,13),(6,14),(3,5),(3,6),(3,7),(4,5),(4,7),(5,5),(5,6),(5,7)]
    n=0
    for R,C in SOCK:
        s=set(int(v) for v in f[R*4:R*4+4,C*4:C*4+4].ravel())
        if 4 in s and 9 in s: n+=1
    return n
e=L.fresh()
plan=[]
# fence box (5,11): 7 nav + face/grab + 2 up + release = 12
plan += [DOWN,RIGHT,RIGHT,RIGHT,UP,UP,UP]   # -> (6,11)
plan += [UP,USE,UP,UP,USE]                   # box to fence (3,11), avatar (4,11)  t=12
# box (7,12)->(6,13): nav to (8,12): DOWN*4, RIGHT
plan += [DOWN,DOWN,DOWN,DOWN,RIGHT]          # t=17 avatar (8,12)
plan += [UP,USE,RIGHT,UP,USE]                # grab (7,12), carry to (6,13), release; avatar (7,13) t=22
# box (8,14)->(6,14): 
plan += [DOWN,DOWN,RIGHT]                    # (9,14) t=25
plan += [UP,USE,UP,UP,USE]                   # grab (8,14), push up to (6,14); avatar (7,14) t=30
# travel toward (9,3)
plan += [DOWN,DOWN]+[LEFT]*10                # t=42 -> (9,4)
n=0
for a in plan:
    e.step(a); n+=1
print('t',n,'avatar',av(e),'mover',mv(e),'seated',seated(e),'lv',e.levels_completed)
# now mover should be near (9,4)/(9,5) coming toward us; USE spam / step logic
for i in range(20):
    m=mv(e); a=av(e)
    if m is None: break
    if abs(m[0]-a[0])+abs(m[1]-a[1])==1:
        e.step(USE); n+=1
    else:
        # step toward it along row 9
        e.step(LEFT if m[1]<a[1] else RIGHT); n+=1
    print(n, av(e), mv(e), 'seated', seated(e))
# then idle until win
while not e.terminal() and e.levels_completed<=8 and n<75:
    e.step(USE); n+=1
print('final t',n,'lv',e.levels_completed,'terminal',e.terminal(),'seated',seated(e))
