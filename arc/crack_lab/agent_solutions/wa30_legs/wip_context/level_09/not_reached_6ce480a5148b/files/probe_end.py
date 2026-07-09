import l9env
import numpy as np
env=l9env.get_l9()
for _ in range(60): env.step(5)
f=np.asarray(env.frame())
print("=== t60 full ===")
print("    "+''.join(str(c%10) for c in range(64)))
for r in range(64):
    print(f'{r:2d}: '+''.join(f'{int(v):x}' if v else '.' for v in f[r]))
