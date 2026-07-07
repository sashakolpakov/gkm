from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
for i in range(14):
    c.step(5)
    f = np.array(c.frame())
    bs = objects(f, 12)
    print(i, "b:", bs, "timer:", timer(f))
