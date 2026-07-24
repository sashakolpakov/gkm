import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

env=A.Arena('g50t')
key=lambda e: e.frame().tobytes()
goal=lambda e,p: e.levels_completed>0
path=P.bounded_bfs(env,goal,key_fn=key,max_states=4000,max_depth=30)
print("PATH",path)
