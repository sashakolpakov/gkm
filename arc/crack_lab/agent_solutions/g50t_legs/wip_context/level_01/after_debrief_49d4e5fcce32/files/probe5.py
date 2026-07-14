import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def av_tl(env):
    best=None
    for b in P.connected_components(env.frame(),colors=[9]):
        if 18<=b.area<=26 and b.bbox[0]<45:  # exclude goal(area19 rows49) & bottom line
            # avatar area24; goal area19 at row49
            if b.bbox[0]<45:
                best=b
    return best

def key(env):
    b=av_tl(env)
    return b.bbox[:2] if b else None

def prog(env):
    base=env.levels_completed
    goal=lambda e,p: e.levels_completed>base
    path=P.bounded_bfs(env,goal,actions=(1,2,3,4),key_fn=key,max_states=5000,max_depth=40)
    print("path",path)
    if path:
        r=P.path_result(env,path)
        print("levels",r['levels_completed'],"terminal",r['terminal'])
    raise SystemExit
A.run_program('g50t', prog)
