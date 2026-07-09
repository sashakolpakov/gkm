import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players
env=A.Arena('wa30')
import time
def solve(env):
    while not env.terminal():
        k=env.levels_completed+1
        fn=getattr(players,f'play_level_{k}',None)
        if fn is None: return
        before=env.levels_completed
        n0=len(env.path)
        try:
            fn(env)
        except Exception as e:
            print("exc lvl",k,e); return
        print(f"level {k}: completed={env.levels_completed} moves_this_level={len(env.path)-n0} total={len(env.path)}")
        if env.levels_completed<=before:
            print("no progress at",k); return
solve(env)
print("FINAL levels",env.levels_completed,"total moves",len(env.path))
