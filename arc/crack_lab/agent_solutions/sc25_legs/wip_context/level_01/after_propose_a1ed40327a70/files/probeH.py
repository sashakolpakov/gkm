import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    goal=lambda e,path: e.levels_completed>0
    key=lambda e: np.asarray(e.frame()).tobytes()
    path=P.bounded_bfs(env,goal,actions=(1,2,3,4),key_fn=key,max_states=4000,max_depth=40)
    print("BFS path:",path)
A.run_program('sc25', main)
