import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import legs, time
env=A.Arena('g50t')
t=time.time()
path=legs.bfs_to_reward(env)
print("path len",len(path),"time",round(time.time()-t,1),"path",path)
# validate
e=A.Arena('g50t')
legs.run_path(e,path)
print("levels",e.levels_completed)
