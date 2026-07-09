import importlib.util, json, os, sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import players
def probe_player(env):
    n=0
    try:
        while not env.terminal():
            env.step(5); n+=1
    except Exception as e:
        print('stopped at n=%d err=%s lvl=%d'%(n,e,env.levels_completed))
        return
    print('reached terminal at n=%d lvl=%d'%(n,env.levels_completed))
players.play_level_8=probe_player
def resumed(env):
    ck=json.load(open('checkpoint.json'))
    for a in ck['final_path']: env.step(a)
    from solve import solve as s
    s(env)
import solve
levels,path,err=A.run_program('wa30', resumed)
print('RESULT levels=%d moves=%d err=%s'%(levels,len(path),err))
