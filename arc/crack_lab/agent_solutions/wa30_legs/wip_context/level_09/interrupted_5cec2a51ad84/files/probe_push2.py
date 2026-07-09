import l9env
import numpy as np
env=l9env.get_l9()
for a in [4,4,4,4,4]: env.step(a)
f=np.asarray(env.frame())
def region(f,r0,r1,c0,c1,tag=""):
    print(tag)
    print("    "+''.join(str(c%10) for c in range(c0,c1)))
    for r in range(r0,r1):
        print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(c0,c1)))
region(f,30,40,48,64,"before USE")
# now push right into box
env.step(4)
f=np.asarray(env.frame())
region(f,30,40,48,64,"after one more RIGHT")
# try USE
env.step(5)
f=np.asarray(env.frame())
region(f,30,40,48,64,"after USE")
