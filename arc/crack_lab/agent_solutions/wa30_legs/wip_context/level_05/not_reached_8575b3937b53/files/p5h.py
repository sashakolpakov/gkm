from probe5 import *
env = get_env()
def avbb(c):
    f=np.array(c.frame()); ys,xs=np.where(f==14); return (int(ys.min()),int(ys.max()),int(xs.min()),int(xs.max()))
for a,name in [(1,'up'),(2,'down'),(3,'left'),(4,'right')]:
    c = env.clone()
    pos=[avbb(c)]
    for i in range(5):
        c.step(a); pos.append(avbb(c))
    print(name, pos)
