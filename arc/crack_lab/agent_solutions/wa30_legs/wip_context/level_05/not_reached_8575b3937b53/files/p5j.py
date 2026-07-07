from probe5 import *
env = get_env()
c = env.clone()
def state(c):
    f=np.array(c.frame())
    ys,xs=np.where(f==14); a=(int(ys.min())//4,int(xs.min())//4)
    ys,xs=np.where(f==12); b=(int(ys.min())//4,int(xs.min())//4) if len(ys) else None
    return a,b
phase1 = [4] + [2]*5 + [3] + [5] + [1]*7 + [3]*8 + [5]
phase2 = [4]*10 + [1]*5 + [3] + [5] + [2]*5 + [3]*9 + [5]
phase3 = [4]*9 + [2]*5 + [2] + [5] + [1]*5 + [3]*10 + [5]
n=0
for name,seq in [("P1",phase1),("P2",phase2),("P3",phase3)]:
    for a in seq:
        c.step(a); n+=1
        if c.levels_completed>4: break
    print(name,"done at step",n,"state",state(c),"levels",c.levels_completed)
# idle: retreat right and bump wall
for a in [4]*3:
    c.step(a); n+=1
while n<130 and c.levels_completed==4 and not c.terminal():
    c.step(2); n+=1
print("end at step",n,"levels",c.levels_completed,"terminal",c.terminal())
show(c.frame())
