import sys; sys.path.insert(0,'.')
from mkstate import l8
import perception as P
env=l8()
d=P.action_deltas(env)
for a,info in d.items():
    print(P.ACTION_NAME[a], info['count'], info['bbox'])
