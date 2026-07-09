import l9env, legs
import numpy as np
env=l9env.get_l9()
av,boxes,walls=legs._grid_scan(env)
# nearest box to (8,2)
free=sorted(boxes,key=lambda b:abs(b[0]-8)+abs(b[1]-2))
print("boxes near target:",free[:3])
base=env.levels_completed
ok=legs.carry_box_to(env,free[0],(8,2),cap=40)
f=np.asarray(env.frame())
print("carry ok",ok,"lvl",env.levels_completed,"steps",len(env.path)-588,"term",env.terminal())
for r in range(30,36):
    print(f'{r}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(4,18)))
