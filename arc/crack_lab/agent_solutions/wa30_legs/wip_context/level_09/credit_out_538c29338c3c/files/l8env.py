import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

def l8():
    env = A.Arena('wa30', _budget=A._Budget(50000*400))
    ck = json.load(open('checkpoint.json'))
    for a in ck['final_path']:
        env.step(a)
    assert env.levels_completed == 7
    return env
