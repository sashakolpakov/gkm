import numpy as np
f=np.load("l7start.npy")
print("    "+''.join(str(c%10) for c in range(64)))
for r in range(64):
    print(f'{r:2d}: '+''.join(f'{int(v):x}' if v else '.' for v in f[r]))
