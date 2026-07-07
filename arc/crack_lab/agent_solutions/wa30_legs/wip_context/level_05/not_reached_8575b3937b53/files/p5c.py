from probe5 import *
env = get_env()
c = env.clone()
for i in range(200):
    c.step(1)
    if c.terminal() or c.levels_completed > 4:
        print("stopped at step", i+1, "levels", c.levels_completed, "terminal", c.terminal())
        break
f = np.array(c.frame())
print("timer 7s:", (f[63]==7).sum())
print("levels", c.levels_completed, "terminal", c.terminal())
show(f)
