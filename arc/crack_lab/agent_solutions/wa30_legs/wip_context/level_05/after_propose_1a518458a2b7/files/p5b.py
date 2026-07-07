from probe5 import *
env = get_env()
c = env.clone()
for i in range(30):
    c.step(1)
show(c.frame())
print("levels", c.levels_completed, "terminal", c.terminal())
