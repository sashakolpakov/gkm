from probe import *
from helpers import avatar
env = get_env()
# test fence crossing empty-handed in each direction
for name, moves in [("left", [3]*4), ("up", [1]*4), ("down",[2]*4), ("right",[4]*5)]:
    c = env.clone()
    for a in moves: c.step(a)
    print(name, "avatar:", avatar(np.array(c.frame())))
