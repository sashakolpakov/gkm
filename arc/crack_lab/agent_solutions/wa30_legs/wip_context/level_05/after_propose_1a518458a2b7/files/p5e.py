from probe5 import *
env = get_env()
c = env.clone()
for i in range(16):
    c.step(2)
c.step(5)  # grab B6 below
def av(f):
    f=np.array(f); ys,xs=np.where(f==14); return (ys.min(),xs.min(),ys.max(),xs.max())
def box0(f):
    f=np.array(f); ys,xs=np.where(f==0)
    # exclude row63
    m = ys<63
    ys,xs=ys[m],xs[m]
    return (ys.min(),xs.min(),ys.max(),xs.max()) if len(ys) else None
f=c.frame(); print("after grab: avatar",av(f),"attached0",box0(f))
c.step(3)  # move left once
f=c.frame(); print("after left: avatar",av(f),"attached0",box0(f))
c.step(1)
f=c.frame(); print("after up: avatar",av(f),"attached0",box0(f))
