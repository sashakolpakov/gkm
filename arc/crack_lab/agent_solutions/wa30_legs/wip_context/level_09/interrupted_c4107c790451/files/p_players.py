import sys; sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players
def solve(env):
    while not env.terminal():
        k=env.levels_completed+1
        fn=getattr(players,f'play_level_{k}',None)
        if fn is None: 
            print("no player for",k); return
        before=env.levels_completed
        n0=len(env.path)
        fn(env)
        print(f"level {k}: +{len(env.path)-n0} moves, now completed={env.levels_completed}")
        if env.levels_completed<=before:
            print("no progress at",k); return
levels,path,err=A.run_program('wa30',solve,step_cap=1200)
print("levels",levels,"moves",len(path),"err",err)
