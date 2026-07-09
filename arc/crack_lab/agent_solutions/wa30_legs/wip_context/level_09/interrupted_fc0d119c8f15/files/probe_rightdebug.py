import l9env, legs
import numpy as np
env=l9env.get_l9()
av,boxes,walls=legs._grid_scan(env)
print("avatar",av)
print("boxes",sorted(boxes))
print("walls has (2,13)?",(2,13) in walls,"(2,14)?",(2,14) in walls,"(6,13)?",(6,13) in walls)
# is (2,14) reachable? nav there
ok=legs._avatar_nav(env.clone(),(3,14),cap=40)
print("nav to (3,14) below top container ok",ok)
ok=legs._avatar_nav(env.clone(),(7,14),cap=40)
print("nav to (7,14) below bot container ok",ok)
# try carry a right box (8,14) to (6,14) (bottom container)
env2=l9env.get_l9()
ok=legs.carry_box_to(env2,(8,14),(6,14),cap=30)
print("carry (8,14)->(6,14):",ok,"steps",len(env2.path)-588)
f=np.asarray(env2.frame())
for r in range(24,36):
    print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(50,64)))
