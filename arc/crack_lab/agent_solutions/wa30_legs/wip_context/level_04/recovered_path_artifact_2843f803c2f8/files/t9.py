from probe import *
from helpers import objects, avatar, timer
env = get_env()

def snap(e):
    f = np.array(e.frame())
    return f

def report(f0, f1, label):
    ys, xs = np.where(f0 != f1)
    changes = {}
    for y,x in zip(ys,xs):
        changes.setdefault((int(f0[y,x]),int(f1[y,x])), []).append((int(y),int(x)))
    print(label, {k: (len(v), v[0]) for k,v in changes.items()})

# experiment A: right then down (bump box below-right)
c = env.clone()
c.step(4)  # right -> avatar (32,32)
f0 = snap(c)
print("A avatar:", avatar(f0))
c.step(2)  # down: bump box at rows36-39 cols32-35
f1 = snap(c)
report(f0, f1, "A bump down:")
# press down again
c.step(2)
f2 = snap(c)
report(f1, f2, "A bump down x2:")
# then USE
c.step(5)
f3 = snap(c)
report(f2, f3, "A then USE:")
