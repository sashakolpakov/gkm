from probe5 import *
env = get_env()
c = env.clone()
def st(c):
    f=np.array(c.frame())
    ys,xs=np.where(f==12); b=(int(ys.min())//4,int(xs.min())//4) if len(ys) else None
    # count boxes in zone: 9-core inside zone interior (rows 25-38, cols 9-14)
    return b
prev=None
for i in range(126):
    c.step(1)
    b=st(c)
    if b!=prev:
        print(i+1, b)
        prev=b
    if c.terminal(): print("terminal at", i+1); break
