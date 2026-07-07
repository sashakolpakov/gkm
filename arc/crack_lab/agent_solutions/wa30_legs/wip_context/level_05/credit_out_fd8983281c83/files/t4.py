from probe import *
env = get_env()
c = env.clone()
for i in range(30):
    c.step(2)
    f = np.array(c.frame())
    n7 = (f[63]==7).sum()
    if n7 <= 1 or c.terminal():
        print("step",i,"timer:",n7,"term:",c.terminal(),"levels:",c.levels_completed)
    if c.terminal():
        break
print("final n7:", n7, "term:", c.terminal())
show(np.array(c.frame()))
