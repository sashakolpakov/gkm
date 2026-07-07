import sys; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A, json
import players
def solve(env):
    # replay checkpoint like resumed_solve
    for act in json.load(open('checkpoint.json'))['final_path']:
        if env.terminal(): break
        env.step(act)
    print("after checkpoint moves=",len(env.path),"level=",env.levels_completed)
    # now spam moves for level 8
    cnt=0
    while not env.terminal():
        env.step(1 if cnt%2==0 else 2); cnt+=1
    print("level8 spam moves=",cnt)
levels,path,err=A.run_program('wa30',solve)
print("levels",levels,"total moves",len(path),"err",err)
