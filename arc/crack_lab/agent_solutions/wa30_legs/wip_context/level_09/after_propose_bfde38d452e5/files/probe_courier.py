import l9env
import numpy as np
env=l9env.get_l9()
for t in range(14):
    f=np.asarray(env.frame())
    # region rows 8-24 cols 12-32 (left courier + center container left edge)
    print(f"--- t{t} ---")
    for r in range(11,24):
        print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(12,34)))
    env.step(1)
