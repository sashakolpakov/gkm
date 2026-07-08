import l9env
import numpy as np
env=l9env.get_l9()
f=np.asarray(env.frame())
print("    "+''.join(str(c%10) for c in range(40,64)))
for r in range(0,40):
    print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(40,64)))
