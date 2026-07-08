# Utility: build a level-8 clone and pickle path for reuse
import sys, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
def l8():
    env=A.Arena('wa30')
    for a in json.load(open('checkpoint.json'))['final_path']:
        if env.terminal(): break
        env.step(a)
    return env
