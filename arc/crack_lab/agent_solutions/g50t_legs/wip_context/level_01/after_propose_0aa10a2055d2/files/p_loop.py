import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def legend(f):
    for row in f[0:7]:
        print(''.join(str(int(v)) for v in row[0:10]))
env=A.Arena('g50t')
for i in range(6):
    for a in [4,4,4,4]: env.step(a)
    env.step(5)
    print(f"--- after delivery {i+1}: lvl={env.levels_completed} 8count={P.color_counts(env.frame()).get(8)} ---")
    legend(env.frame())
