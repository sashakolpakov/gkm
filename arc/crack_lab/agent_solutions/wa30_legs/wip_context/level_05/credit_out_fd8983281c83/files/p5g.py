from probe5 import *
env = get_env()
c = env.clone()
def av(c):
    f=np.array(c.frame()); ys,xs=np.where(f==14); return (ys.min(),xs.min())
seq = [4]*4 + [2]*20 + [3] + [5]  # right4, down20, bump left, grab
for a in seq: c.step(a)
print("av", av(c))
for i in range(28): c.step(1)
print("after up28 av", av(c))
for i in range(22):
    prev=np.array(c.frame()); c.step(3); cur=np.array(c.frame())
    if (prev==cur).all():
        print("blocked at left step",i, "av",av(c)); break
print("av",av(c), "levels", c.levels_completed)
show(c.frame())
