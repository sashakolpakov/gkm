from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
prev = None
for i in range(40):
    c.step(3)  # left; will hit fence soon and become idle-ish
    f = np.array(c.frame())
    bs = tuple(objects(f, 12))
    if bs != prev:
        print(i, "b:", bs, "timer:", timer(f), "term:", c.terminal())
        prev = bs
    if c.terminal(): break
