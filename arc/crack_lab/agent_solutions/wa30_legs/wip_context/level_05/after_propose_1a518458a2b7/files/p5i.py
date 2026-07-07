from probe5 import *
env = get_env()
c = env.clone()
def state(c):
    f=np.array(c.frame())
    ys,xs=np.where(f==14); a=(int(ys.min())//4,int(xs.min())//4)
    ys,xs=np.where(f==12); b=(int(ys.min())//4,int(xs.min())//4) if len(ys) else None
    # boxes: 9-cores
    ys,xs=np.where(f==9)
    cores=set()
    for y,x in zip(ys,xs):
        cores.add((y//4,x//4))
    return a,b,sorted(cores)
seq = [4] + [2]*5 + [3] + [5] + [1]*7 + [3]*8
for i,a in enumerate(seq):
    c.step(a)
av,co,cores = state(c)
print("avatar",av,"courier",co)
print("cores",cores)
print("levels",c.levels_completed)
show(c.frame())
