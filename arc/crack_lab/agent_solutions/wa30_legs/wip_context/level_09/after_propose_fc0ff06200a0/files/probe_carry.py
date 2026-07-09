import l9env
import numpy as np
def region(f,r0,r1,c0,c1,tag=""):
    print(tag)
    for r in range(r0,r1):
        print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(c0,c1)))
env=l9env.get_l9()
for a in [4,4,4,4,4]: env.step(a)  # approach, box turns 3
f=np.asarray(env.frame()); region(f,32,36,48,64,"adjacent(facing right), box=3?")
env.step(1) # UP
f=np.asarray(env.frame()); region(f,28,40,48,64,"after UP")
