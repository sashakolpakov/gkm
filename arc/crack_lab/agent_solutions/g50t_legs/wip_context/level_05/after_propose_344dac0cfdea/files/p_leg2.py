import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import legs, time
env=A.Arena('g50t')
t=time.time()
plan=legs.plan_unlock_reach(env)
print("plan",plan,"len",len(plan) if plan else None,"time",round(time.time()-t,1))
if plan:
    e=A.Arena('g50t')
    legs.run_path(e,plan)
    print("validate levels",e.levels_completed,"real moves",len(e.path))
