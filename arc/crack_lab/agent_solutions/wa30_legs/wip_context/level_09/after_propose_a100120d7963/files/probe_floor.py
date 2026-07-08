import l9env
import numpy as np
env=l9env.get_l9()
for t in range(0,66,4):
    f=np.asarray(env.frame())
    row=''.join(f'{int(v):x}' for v in f[63])
    n7=int((f[63]==7).sum())
    print(f"t{t:2d} n7={n7:2d} {row}")
    for _ in range(4):
        if env.terminal(): break
        env.step(5)
