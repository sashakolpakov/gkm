import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

env = A.Arena('g50t')
d = P.action_deltas(env)
for a,v in d.items():
    print(P.ACTION_NAME[a], v['count'], v['bbox'], v['samples'][:6])
