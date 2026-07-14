import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    goal=lambda e,path: e.levels_completed>0
    def key(e):
        f=np.asarray(e.frame()).copy()
        f[:,60:]=0  # mask timer
        return f.tobytes()
    path=P.bounded_bfs(env,goal,actions=(1,2,3,4),key_fn=key,max_states=8000,max_depth=60)
    print("BFS path:",path)
    if path is None:
        # report reachable distinct states count
        seen=set()
        from collections import deque
        q=deque([(env.clone(),0)]); seen.add(key(env))
        maxc=0
        while q:
            n,d=q.popleft()
            if d>60: continue
            for a in (1,2,3,4):
                ch=n.clone(); ch.step(a); k=key(ch)
                if k not in seen:
                    seen.add(k); q.append((ch,d+1))
            if len(seen)>8000: break
        print("distinct states reachable(masked):",len(seen))
A.run_program('sc25', main)
