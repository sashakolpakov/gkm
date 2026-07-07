from probe import *
from helpers import objects, avatar, timer
env = get_env()
c = env.clone()
for i in range(30):
    c.step(3)
    f = np.array(c.frame())
    border = int(f[4,24])
    print(i, "outer box border:", border, "b12 top-right:", int(f[4,56]), "b bottom:", int(f[56,24]))
    if border != 4:
        break
