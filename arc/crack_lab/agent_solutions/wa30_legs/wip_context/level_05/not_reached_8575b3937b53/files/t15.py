from probe import *
env = get_env()
c = env.clone()
for i in range(20):
    c.step(3)
f = np.array(c.frame())
for r in range(0,36):
    print(''.join(CH[int(v)] for v in f[r,0:32]))
