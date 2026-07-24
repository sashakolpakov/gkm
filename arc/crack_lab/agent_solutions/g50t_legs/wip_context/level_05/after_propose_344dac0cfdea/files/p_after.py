import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def show(env,r,c):
    for row in env.frame()[r[0]:r[1]]:
        print(''.join(str(int(v)) for v in row[c[0]:c[1]]))

env=A.Arena('g50t')
for a in [4,4,4,4]: env.step(a)   # to (8,38)
env.step(5)  # USE stamp
print("after USE at (8,38): 9 count",P.color_counts(env.frame()).get(9))
print("top-left region:")
show(env,(0,15),(0,45))
print("action deltas now:")
for a,v in P.action_deltas(env).items():
    print(P.ACTION_NAME[a],v['count'],v['bbox'])
