import l9env, perception as P
import numpy as np
env = l9env.get_l9()
f = env.frame()
for r in range(64):
    print(''.join(f'{int(v):x}' if v else '.' for v in f[r]))
